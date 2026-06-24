#!/usr/bin/env python3
"""Local MinerU2.5-Pro model implementation without Transformers imports.

This is intentionally a narrow eager inference implementation for the
MinerU2.5-Pro-2605-1.2B checkpoint. It mirrors the Qwen2-VL model structure
used by the checkpoint while keeping the model class itself free of
Transformers dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch import nn

from config import MinerUConfig, MinerUTextConfig, MinerUVisionConfig


FRACTAL_NZ = 29
DECODE_WEIGHT_FORMAT_NONE = "none"
DECODE_WEIGHT_FORMAT_NZ = "decode_nz"
DECODE_WEIGHT_FORMAT_CHOICES = (DECODE_WEIGHT_FORMAT_NONE, DECODE_WEIGHT_FORMAT_NZ)


def _resolve_model_dir(model_dir: str | Path) -> Path:
    path = Path(model_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            "LocalMinerU2_5ForConditionalGeneration requires a local model directory. "
            f"Path does not exist: {path}"
        )
    if not path.is_dir():
        raise NotADirectoryError(f"model path is not a directory: {path}")
    return path


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "quick_gelu":
        return x * torch.sigmoid(1.702 * x)
    if name == "gelu":
        return F.gelu(x)
    if name == "gelu_pytorch_tanh":
        return F.gelu(x, approximate="tanh")
    raise ValueError(f"unsupported activation: {name!r}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    mrope_section = [int(value) for value in mrope_section] * 2
    cos = torch.cat([part[i % 3] for i, part in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
    sin = torch.cat([part[i % 3] for i, part in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def build_causal_mask(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_length: int = 0,
) -> torch.Tensor:
    batch_size, query_length = inputs_embeds.shape[:2]
    if attention_mask is None:
        kv_length = int(past_length + query_length)
        attention_mask = torch.ones(batch_size, kv_length, device=inputs_embeds.device, dtype=torch.long)
    else:
        kv_length = int(attention_mask.shape[-1])
    kv_positions = torch.arange(kv_length, device=inputs_embeds.device, dtype=cache_position.dtype)
    allowed = kv_positions.unsqueeze(0) <= cache_position.reshape(-1, 1)
    allowed = allowed.reshape(1, 1, query_length, kv_length).expand(batch_size, 1, query_length, kv_length)
    padding_allowed = attention_mask[:, None, None, :kv_length].to(device=inputs_embeds.device, dtype=torch.bool)
    allowed = allowed & padding_allowed
    mask = torch.zeros((batch_size, 1, query_length, kv_length), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    return mask.masked_fill(~allowed, torch.finfo(inputs_embeds.dtype).min)


def build_static_decode_mask(
    inputs_embeds: torch.Tensor,
    cache_position: torch.Tensor,
    cache_length: int,
) -> torch.Tensor:
    cache_position = cache_position.reshape(-1).to(device=inputs_embeds.device, dtype=torch.int64)
    kv_positions = torch.arange(int(cache_length), device=inputs_embeds.device, dtype=torch.int64)
    allowed = kv_positions.unsqueeze(0) <= cache_position.unsqueeze(1)
    allowed = allowed.view(inputs_embeds.shape[0], 1, 1, int(cache_length))
    mask = torch.zeros(
        (inputs_embeds.shape[0], 1, 1, int(cache_length)),
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    return mask.masked_fill(~allowed, torch.finfo(inputs_embeds.dtype).min)


def update_prefill_kv_cache_(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    sequence_length = int(key_states.shape[2])
    key_cache[:, :, :sequence_length, :].copy_(key_states.contiguous())
    value_cache[:, :, :sequence_length, :].copy_(value_states.contiguous())


def update_decode_kv_cache_(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    positions = cache_position.reshape(-1).to(device=key_cache.device, dtype=torch.int64).contiguous()
    if key_cache.device.type == "npu":
        import torch_npu

        torch_npu.scatter_update_(key_cache, positions, key_states.contiguous(), 2)
        torch_npu.scatter_update_(value_cache, positions, value_states.contiguous(), 2)
        return
    key_cache.index_copy_(2, positions, key_states.contiguous())
    value_cache.index_copy_(2, positions, value_states.contiguous())


def configure_decode_weight_format(
    model: "LocalMinerU2_5ForConditionalGeneration",
    mode: str,
) -> dict[str, object]:
    """Configure decoder weights for the one-token decode path.

    The checkpoint ties the LM head to ``embed_tokens.weight``. For NZ decode we
    keep the embedding table in normal layout for token lookup and create a
    decode-only NZ copy for the final logits projection.
    """

    if mode not in DECODE_WEIGHT_FORMAT_CHOICES:
        raise ValueError(f"unsupported decode weight format {mode!r}; expected {DECODE_WEIGHT_FORMAT_CHOICES}")

    model.decode_lm_head_weight = None
    metadata: dict[str, object] = {
        "requested_mode": str(mode),
        "effective_mode": DECODE_WEIGHT_FORMAT_NONE,
        "target_format": None,
        "target_format_id": None,
        "decoder_linear_count": 0,
        "converted_decoder_linear_count": 0,
        "already_nz_decoder_linear_count": 0,
        "lm_head_copy": None,
        "skipped_reason": None,
    }
    if mode == DECODE_WEIGHT_FORMAT_NONE:
        return metadata

    device = model.device
    if device.type != "npu":
        metadata["skipped_reason"] = f"decode_nz requires NPU tensors, got device={device.type}"
        return metadata

    import torch_npu

    metadata["target_format"] = "FRACTAL_NZ"
    metadata["target_format_id"] = FRACTAL_NZ
    converted: list[str] = []
    already_nz: list[str] = []
    before_formats: dict[str, int | None] = {}
    after_formats: dict[str, int | None] = {}

    decoder_linears: list[tuple[str, nn.Linear]] = []
    for layer_idx, layer in enumerate(model.model.layers):
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                decoder_linears.append((f"model.layers.{layer_idx}.{name}", module))

    for name, module in decoder_linears:
        weight = module.weight
        if weight.device.type != "npu":
            raise RuntimeError(f"decode_nz requested but {name}.weight is on {weight.device}")
        before_format = int(torch_npu.get_npu_format(weight))
        before_formats[name] = before_format
        if before_format == FRACTAL_NZ:
            already_nz.append(name)
        else:
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, FRACTAL_NZ)
            converted.append(name)
        after_formats[name] = int(torch_npu.get_npu_format(module.weight))

    lm_head_source = model.model.embed_tokens.weight
    if lm_head_source.device.type != "npu":
        raise RuntimeError(f"decode_nz requested but embed_tokens.weight is on {lm_head_source.device}")
    lm_head_before = int(torch_npu.get_npu_format(lm_head_source))
    lm_head_copy = torch_npu.npu_format_cast(lm_head_source.detach(), FRACTAL_NZ)
    lm_head_copy.requires_grad_(False)
    model.decode_lm_head_weight = lm_head_copy
    lm_head_after = int(torch_npu.get_npu_format(model.decode_lm_head_weight))

    metadata.update(
        {
            "effective_mode": DECODE_WEIGHT_FORMAT_NZ,
            "decoder_linear_count": len(decoder_linears),
            "converted_decoder_linear_count": len(converted),
            "already_nz_decoder_linear_count": len(already_nz),
            "converted_decoder_linear_names": converted,
            "already_nz_decoder_linear_names": already_nz,
            "before_formats": before_formats,
            "after_formats": after_formats,
            "all_decoder_linears_nz": all(value == FRACTAL_NZ for value in after_formats.values()),
            "lm_head_copy": {
                "source": "model.embed_tokens.weight",
                "reason": "tied LM head needs NZ matmul weight, but embedding lookup must keep the original ND table",
                "source_format_before": lm_head_before,
                "copy_format_after": lm_head_after,
                "copy_is_target_format": lm_head_after == FRACTAL_NZ,
                "shape": [int(dim) for dim in lm_head_copy.shape],
                "dtype": str(lm_head_copy.dtype),
                "device": str(lm_head_copy.device),
            },
        }
    )
    return metadata


@dataclass
class LocalMinerUOutput:
    logits: torch.Tensor
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    rope_deltas: torch.Tensor | None = None


@dataclass
class LocalMinerUStaticOutput:
    logits: torch.Tensor
    cache: "LocalMinerUStaticCache"
    rope_deltas: torch.Tensor
    next_cache_position: torch.Tensor


@dataclass
class LocalMinerUStaticCache:
    key_caches: tuple[torch.Tensor, ...]
    value_caches: tuple[torch.Tensor, ...]
    cache_length: int

    @classmethod
    def allocate(
        cls,
        config: MinerUTextConfig,
        *,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        init_mode: str = "zeros",
    ) -> "LocalMinerUStaticCache":
        cache_shape = (
            int(batch_size),
            int(config.num_key_value_heads),
            int(cache_length),
            int(config.head_dim),
        )
        key_caches = []
        value_caches = []
        for _layer_idx in range(config.num_hidden_layers):
            if init_mode == "zeros":
                key_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
                value_cache = torch.zeros_like(key_cache)
            elif init_mode == "empty":
                key_cache = torch.empty(cache_shape, device=device, dtype=dtype)
                value_cache = torch.empty_like(key_cache)
            else:
                raise ValueError(f"unknown static cache init_mode: {init_mode!r}")
            key_caches.append(key_cache)
            value_caches.append(value_cache)
        return cls(tuple(key_caches), tuple(value_caches), int(cache_length))

    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_caches[int(layer_idx)], self.value_caches[int(layer_idx)]

    def flat_tensors(self) -> tuple[torch.Tensor, ...]:
        return (*self.key_caches, *self.value_caches)


class MinerURMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MinerURotaryEmbedding(nn.Module):
    def __init__(self, config: MinerUTextConfig):
        super().__init__()
        rope_scaling = config.rope_scaling or {}
        self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if self.rope_type != "default":
            raise ValueError(f"Only default RoPE is implemented for now, got {self.rope_type!r}")
        self.base = float(config.rope_theta)
        self.dim = int(config.head_dim)
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)
        self.attention_scaling = 1.0

    def _compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))

    def reset_inv_freq(self, device: torch.device | None = None) -> None:
        self.register_buffer("inv_freq", self._compute_inv_freq().to(device=device), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids = position_ids[:, :, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MinerUMLP(nn.Module):
    def __init__(self, config: MinerUTextConfig):
        super().__init__()
        self.hidden_act = config.hidden_act
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_activation(self.hidden_act, self.gate_proj(x)) * self.up_proj(x))


class MinerUAttention(nn.Module):
    def __init__(self, config: MinerUTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.mrope_section = list((config.rope_scaling or {}).get("mrope_section", [8, 12, 12]))
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, query_length, _hidden = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return query_states, key_states, value_states

    def apply_rotary(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = position_embeddings
        return apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.mrope_section,
        )

    def attend(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, _heads, query_length, _dim = query_states.shape
        key_for_attn = repeat_kv(key_states, self.num_key_value_groups)
        value_for_attn = repeat_kv(value_states, self.num_key_value_groups)
        scores = torch.matmul(query_states, key_for_attn.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : key_for_attn.shape[-2]]
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(probs, value_for_attn)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, query_length, -1)
        return self.o_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states, key_states = self.apply_rotary(query_states, key_states, position_embeddings)
        if past_key_values is not None:
            past_key, past_value = past_key_values[self.layer_idx]
            key_states = torch.cat((past_key, key_states), dim=2)
            value_states = torch.cat((past_value, value_states), dim=2)
        new_past = (key_states, value_states) if use_cache else None
        return self.attend(query_states, key_states, value_states, attention_mask), new_past

    def forward_prefill_static(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states, key_states = self.apply_rotary(query_states, key_states, position_embeddings)
        return self.attend(query_states, key_states, value_states, attention_mask), key_states, value_states

    def forward_decode_static(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states, key_states = self.apply_rotary(query_states, key_states, position_embeddings)
        update_decode_kv_cache_(
            key_cache,
            value_cache,
            cache_position,
            key_states,
            value_states,
        )
        return self.attend(query_states, key_cache, value_cache, attention_mask)


class MinerUDecoderLayer(nn.Module):
    def __init__(self, config: MinerUTextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MinerUAttention(config, layer_idx)
        self.mlp = MinerUMLP(config)
        self.input_layernorm = MinerURMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MinerURMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_past = self.self_attn(
            hidden_states,
            attention_mask,
            position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, new_past

    def apply_blocks(self, residual: torch.Tensor, attn_output: torch.Tensor) -> torch.Tensor:
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_prefill_static(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: LocalMinerUStaticCache | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, key_states, value_states = self.self_attn.forward_prefill_static(
            hidden_states,
            attention_mask,
            position_embeddings,
        )
        if cache is not None:
            key_cache, value_cache = cache.layer(self.self_attn.layer_idx)
            update_prefill_kv_cache_(key_cache, value_cache, key_states, value_states)
        return self.apply_blocks(residual, attn_output)

    def forward_decode_static(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn.forward_decode_static(
            hidden_states,
            attention_mask,
            position_embeddings,
            key_cache,
            value_cache,
            cache_position,
        )
        return self.apply_blocks(residual, attn_output)


class MinerUTextModel(nn.Module):
    def __init__(self, config: MinerUTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([MinerUDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = MinerURMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MinerURotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.embed_tokens(input_ids)
        past_length = 0 if past_key_values is None else int(past_key_values[0][0].shape[2])
        cache_position = torch.arange(
            past_length,
            past_length + inputs_embeds.shape[1],
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, -1, -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        causal_mask = build_causal_mask(inputs_embeds, attention_mask, cache_position, past_length=past_length)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        new_past_key_values = [] if use_cache else None
        for layer in self.layers:
            hidden_states, new_past = layer(
                hidden_states,
                causal_mask,
                position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(new_past)
        return self.norm(hidden_states), new_past_key_values

    def forward_prefill_static(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache: LocalMinerUStaticCache | None = None,
    ) -> torch.Tensor:
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.int64)
        causal_mask = build_causal_mask(inputs_embeds, attention_mask, cache_position)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer.forward_prefill_static(
                hidden_states,
                causal_mask,
                position_embeddings,
                cache=cache,
            )
        return self.norm(hidden_states)

    def forward_decode_static(
        self,
        inputs_embeds: torch.Tensor,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        cache_length: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _hidden = inputs_embeds.shape
        if seq_length != 1:
            raise ValueError(f"static decode expects exactly one token, got seq_length={seq_length}")
        cache_position = cache_position.reshape(-1).to(device=inputs_embeds.device, dtype=torch.int64)
        if cache_position.numel() == 1:
            cache_position = cache_position.expand(batch_size)
        if cache_position.numel() != batch_size:
            raise ValueError(f"cache_position must be scalar or batch-shaped, got {tuple(cache_position.shape)}")
        if attention_mask is None:
            attention_mask = build_static_decode_mask(inputs_embeds, cache_position, cache_length)
        position_ids = cache_position.view(batch_size, 1) + rope_deltas.to(device=inputs_embeds.device, dtype=torch.int64)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward_decode_static(
                hidden_states,
                attention_mask,
                position_embeddings,
                key_caches[layer_idx],
                value_caches[layer_idx],
                cache_position,
            )
        return self.norm(hidden_states)


class MinerUVisionPatchEmbed(nn.Module):
    def __init__(self, config: MinerUVisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.embed_dim
        kernel_size = [config.temporal_patch_size, config.patch_size, config.patch_size]
        self.proj = nn.Conv3d(config.in_channels, config.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class MinerUVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = int(dim)
        self.theta = float(theta)
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))

    def reset_inv_freq(self, device: torch.device | None = None) -> None:
        self.register_buffer("inv_freq", self._compute_inv_freq().to(device=device), persistent=False)

    def forward(self, seqlen: int | torch.Tensor) -> torch.Tensor:
        seq = torch.arange(int(seqlen), device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class MinerUVisionAttention(nn.Module):
    def __init__(self, config: MinerUVisionConfig):
        super().__init__()
        self.dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits, k_splits, v_splits = [
            torch.split(tensor, lengths, dim=2) for tensor in (query_states, key_states, value_states)
        ]
        outputs = []
        for q, k, v in zip(q_splits, k_splits, v_splits):
            scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            outputs.append(torch.matmul(probs, v).transpose(1, 2).contiguous())
        attn_output = torch.cat(outputs, dim=1).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class MinerUVisionMLP(nn.Module):
    def __init__(self, config: MinerUVisionConfig):
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.hidden_act = config.hidden_act
        self.fc1 = nn.Linear(config.embed_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, config.embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(_activation(self.hidden_act, self.fc1(x)))


class MinerUVisionBlock(nn.Module):
    def __init__(self, config: MinerUVisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.attn = MinerUVisionAttention(config)
        self.mlp = MinerUVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MinerUPatchMerger(nn.Module):
    def __init__(self, config: MinerUVisionConfig):
        super().__init__()
        self.hidden_size = config.embed_dim * (config.spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class MinerUVisionTransformer(nn.Module):
    def __init__(self, config: MinerUVisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = MinerUVisionPatchEmbed(config)
        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = MinerUVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([MinerUVisionBlock(config) for _ in range(config.depth)])
        self.merger = MinerUPatchMerger(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            t_int = int(t.item())
            h_int = int(h.item())
            w_int = int(w.item())
            hpos_ids = torch.arange(h_int, device=grid_thw.device).unsqueeze(1).expand(-1, w_int)
            hpos_ids = hpos_ids.reshape(
                h_int // self.spatial_merge_size,
                self.spatial_merge_size,
                w_int // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w_int, device=grid_thw.device).unsqueeze(0).expand(h_int, -1)
            wpos_ids = wpos_ids.reshape(
                h_int // self.spatial_merge_size,
                self.spatial_merge_size,
                w_int // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t_int, 1))
        pos_ids_tensor = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids_tensor].flatten(1)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        return self.merger(hidden_states)


class LocalMinerU2_5ForConditionalGeneration(nn.Module):
    """MinerU2.5-Pro core VLM with local eager model code."""

    def __init__(self, config: MinerUConfig):
        super().__init__()
        self.config = config
        self.visual = MinerUVisionTransformer(config.vision_config)
        self.model = MinerUTextModel(config.text_config)
        self.rope_deltas: torch.Tensor | None = None
        self.decode_lm_head_weight: torch.Tensor | None = None

    @property
    def dtype(self) -> torch.dtype:
        return self.model.embed_tokens.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.model.embed_tokens.weight.device

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        dtype: torch.dtype | None = torch.float16,
        device: str | torch.device | None = None,
    ) -> "LocalMinerU2_5ForConditionalGeneration":
        model_path = _resolve_model_dir(model_dir)
        config = MinerUConfig.from_model_dir(model_path)
        model = cls(config)
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device)

        from safetensors.torch import load_file

        safetensors_paths = sorted(model_path.glob("*.safetensors"))
        if not safetensors_paths:
            raise FileNotFoundError(f"no .safetensors files found in {model_path}")
        state_dict: dict[str, torch.Tensor] = {}
        for shard_path in safetensors_paths:
            state_dict.update(load_file(shard_path, device=str(device or "cpu")))
        if "lm_head.weight" in state_dict:
            # MinerU2.5-Pro ties embeddings; keep a separate lm_head out of this local module.
            state_dict.pop("lm_head.weight")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"missing checkpoint keys: {missing}")
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
        model._reset_rope_buffers()
        return model.eval()

    def _reset_rope_buffers(self) -> None:
        for module in self.modules():
            if isinstance(module, (MinerURotaryEmbedding, MinerUVisionRotaryEmbedding)):
                module.reset_inv_freq(device=module.inv_freq.device)

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.type(self.visual.dtype)
        return self.visual(pixel_values, grid_thw=image_grid_thw)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        vision_start_token_id = self.config.vision_start_token_id
        if image_grid_thw is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
            mrope_position_deltas = []
            image_index = 0
            for batch_idx, sample_input_ids in enumerate(input_ids):
                visible_input_ids = sample_input_ids[attention_mask[batch_idx].to(sample_input_ids.device) == 1]
                vision_start_indices = torch.argwhere(visible_input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = visible_input_ids[vision_start_indices + 1]
                image_nums = int((vision_tokens == image_token_id).sum().item())
                input_tokens = visible_input_ids.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images = image_nums
                for _ in range(image_nums):
                    ed = input_tokens.index(image_token_id, st) if remain_images > 0 else len(input_tokens) + 1
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    llm_grid_t = int(t.item())
                    llm_grid_h = int(h.item()) // spatial_merge_size
                    llm_grid_w = int(w.item()) // spatial_merge_size
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[:, batch_idx, attention_mask[batch_idx] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids[batch_idx]))
            return position_ids, torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0)[0].max(-1, keepdim=True)[0]
            return position_ids, max_position_ids + 1 - attention_mask.shape[-1]
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        return position_ids, torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)

    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is None:
            return inputs_embeds
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required when pixel_values is provided")
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_token_count = int((input_ids == self.config.image_token_id).sum().item())
        if image_token_count * inputs_embeds.shape[-1] != image_embeds.numel():
            raise ValueError(
                "image features and image tokens do not match: "
                f"tokens={image_token_count} "
                f"features={int(image_embeds.shape[0])}"
            )
        image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

    def allocate_static_cache(
        self,
        *,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        init_mode: str = "zeros",
    ) -> LocalMinerUStaticCache:
        return LocalMinerUStaticCache.allocate(
            self.config.text_config,
            batch_size=batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            init_mode=init_mode,
        )

    def forward_static_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        *,
        cache_length: int,
        cache: LocalMinerUStaticCache | None = None,
        cache_init_mode: str = "zeros",
        logits_to_keep: int = 0,
    ) -> LocalMinerUStaticOutput:
        inputs_embeds = self.build_inputs_embeds(input_ids, pixel_values, image_grid_thw)
        batch_size, sequence_length, _hidden = inputs_embeds.shape
        if int(sequence_length) > int(cache_length):
            raise ValueError(f"prefill sequence length {sequence_length} exceeds static cache length {cache_length}")
        position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, attention_mask)
        if cache is None:
            cache = self.allocate_static_cache(
                batch_size=int(batch_size),
                cache_length=int(cache_length),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                init_mode=cache_init_mode,
            )
        hidden_states = self.model.forward_prefill_static(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = F.linear(hidden_states[:, slice_indices, :], self.model.embed_tokens.weight)
        next_cache_position = torch.full(
            (int(batch_size),),
            int(sequence_length),
            device=inputs_embeds.device,
            dtype=torch.int64,
        )
        self.rope_deltas = rope_deltas
        return LocalMinerUStaticOutput(
            logits=logits,
            cache=cache,
            rope_deltas=rope_deltas,
            next_cache_position=next_cache_position,
        )

    def forward_static_decode(
        self,
        input_ids: torch.Tensor,
        cache: LocalMinerUStaticCache,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> LocalMinerUOutput:
        inputs_embeds = self.model.embed_tokens(input_ids)
        hidden_states = self.model.forward_decode_static(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            key_caches=cache.key_caches,
            value_caches=cache.value_caches,
            cache_length=cache.cache_length,
            attention_mask=attention_mask,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = F.linear(hidden_states[:, slice_indices, :], self.model.embed_tokens.weight)
        return LocalMinerUOutput(logits=logits, rope_deltas=rope_deltas)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        rope_deltas: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> LocalMinerUOutput:
        inputs_embeds = self.build_inputs_embeds(input_ids, pixel_values, image_grid_thw)
        if position_ids is None:
            if past_key_values is None:
                position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, attention_mask)
                self.rope_deltas = rope_deltas
            else:
                past_length = int(past_key_values[0][0].shape[2])
                batch_size, seq_length, _hidden = inputs_embeds.shape
                delta = rope_deltas if rope_deltas is not None else self.rope_deltas
                if delta is None:
                    raise ValueError("rope_deltas are required for cached decode")
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                position_ids = position_ids + (past_length + delta.to(inputs_embeds.device)).view(1, batch_size, 1)
        hidden_states, past = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = F.linear(hidden_states[:, slice_indices, :], self.model.embed_tokens.weight)
        return LocalMinerUOutput(
            logits=logits,
            past_key_values=past,
            rope_deltas=rope_deltas if rope_deltas is not None else self.rope_deltas,
        )

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "npu":
            import torch_npu

            torch_npu.npu.synchronize()

    def _prefill_generation_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        *,
        eos_token_id: int,
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True,
            logits_to_keep=1,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
        finished = next_token.squeeze(1) == eos_token_id
        current_attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        return outputs.past_key_values, outputs.rope_deltas, next_token, current_attention_mask, finished

    def _decode_generation_step(
        self,
        next_token: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        rope_deltas: torch.Tensor,
        finished: torch.Tensor,
        *,
        eos_token_id: int,
        pad_token_id: int,
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        outputs = self.forward(
            input_ids=next_token,
            attention_mask=attention_mask,
            pixel_values=None,
            image_grid_thw=None,
            past_key_values=past_key_values,
            use_cache=True,
            rope_deltas=rope_deltas,
            logits_to_keep=1,
        )
        new_next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
        new_next_token = torch.where(finished.view(-1, 1), torch.full_like(new_next_token, pad_token_id), new_next_token)
        new_finished = finished | (new_next_token.squeeze(1) == eos_token_id)
        new_attention_mask = torch.cat([attention_mask, torch.ones_like(new_next_token)], dim=1)
        return outputs.past_key_values, new_next_token, new_attention_mask, new_finished

    @torch.inference_mode()
    def benchmark_decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        *,
        warmup_steps: int = 8,
        measure_steps: int = 64,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> dict[str, float | int | bool]:
        eos_token_id = int(self.config.eos_token_id if eos_token_id is None else eos_token_id)
        pad_token_id = int(self.config.pad_token_id if pad_token_id is None else pad_token_id)
        warmup_steps = max(0, int(warmup_steps))
        measure_steps = max(0, int(measure_steps))

        if warmup_steps > 0:
            warm_past, warm_rope_deltas, warm_next, warm_attention_mask, warm_finished = self._prefill_generation_state(
                input_ids,
                attention_mask,
                pixel_values,
                image_grid_thw,
                eos_token_id=eos_token_id,
            )
            for _ in range(warmup_steps):
                warm_past, warm_next, warm_attention_mask, warm_finished = self._decode_generation_step(
                    warm_next,
                    warm_attention_mask,
                    warm_past,
                    warm_rope_deltas,
                    warm_finished,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
            self._sync_device()

        self._sync_device()
        prefill_start = time.perf_counter()
        past, rope_deltas, next_token, current_attention_mask, finished = self._prefill_generation_state(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            eos_token_id=eos_token_id,
        )
        self._sync_device()
        prefill_s = time.perf_counter() - prefill_start

        self._sync_device()
        decode_start = time.perf_counter()
        for _ in range(measure_steps):
            past, next_token, current_attention_mask, finished = self._decode_generation_step(
                next_token,
                current_attention_mask,
                past,
                rope_deltas,
                finished,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
        self._sync_device()
        decode_s = time.perf_counter() - decode_start

        return {
            "enabled": True,
            "warmup_decode_steps": int(warmup_steps),
            "measured_decode_steps": int(measure_steps),
            "prefill_s": float(prefill_s),
            "decode_s": float(decode_s),
            "decode_tok_s": float(measure_steps / decode_s) if decode_s > 0 else 0.0,
            "raw_decode_forward_calls": int(measure_steps),
            "stop_on_eos": False,
        }

    @torch.inference_mode()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        *,
        max_new_tokens: int = 512,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        eos_token_id = int(self.config.eos_token_id if eos_token_id is None else eos_token_id)
        pad_token_id = int(self.config.pad_token_id if pad_token_id is None else pad_token_id)
        past, rope_deltas, next_token, current_attention_mask, finished = self._prefill_generation_state(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            eos_token_id=eos_token_id,
        )
        generated = [next_token]
        for _ in range(max(0, int(max_new_tokens) - 1)):
            if bool(finished.all().item()):
                break
            past, next_token, current_attention_mask, finished = self._decode_generation_step(
                next_token,
                current_attention_mask,
                past,
                rope_deltas,
                finished,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            generated.append(next_token)
        return torch.cat(generated, dim=1)

    @torch.inference_mode()
    def generate_ids_static(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        *,
        max_new_tokens: int = 512,
        cache_length: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        eos_token_id = int(self.config.eos_token_id if eos_token_id is None else eos_token_id)
        pad_token_id = int(self.config.pad_token_id if pad_token_id is None else pad_token_id)
        cache_length = int(cache_length or (int(input_ids.shape[1]) + int(max_new_tokens)))
        outputs = self.forward_static_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cache_length=cache_length,
            logits_to_keep=1,
        )
        cache = outputs.cache
        rope_deltas = outputs.rope_deltas
        cache_position = outputs.next_cache_position.clone()
        next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
        generated = [next_token]
        finished = next_token.squeeze(1) == eos_token_id
        for _ in range(max(0, int(max_new_tokens) - 1)):
            if bool(finished.all().item()):
                break
            outputs_decode = self.forward_static_decode(
                input_ids=next_token,
                cache=cache,
                cache_position=cache_position,
                rope_deltas=rope_deltas,
                logits_to_keep=1,
            )
            next_token = torch.argmax(outputs_decode.logits[:, -1, :].float(), dim=-1, keepdim=True)
            next_token = torch.where(finished.view(-1, 1), torch.full_like(next_token, pad_token_id), next_token)
            generated.append(next_token)
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            cache_position.add_(1)
        return torch.cat(generated, dim=1)

    def make_flat_static_decode_module(self, *, cache_length: int) -> "MinerUFlatStaticDecodeModule":
        return MinerUFlatStaticDecodeModule(self, cache_length=cache_length)


class MinerUFlatStaticDecodeModule(nn.Module):
    def __init__(self, model: LocalMinerU2_5ForConditionalGeneration, *, cache_length: int):
        super().__init__()
        self.model = model
        self.num_layers = int(model.config.text_config.num_hidden_layers)
        if self.num_layers != 24:
            raise ValueError(f"MinerU2.5-Pro static decode expects 24 decoder layers, got {self.num_layers}")
        self.cache_length = int(cache_length)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        k0: torch.Tensor,
        k1: torch.Tensor,
        k2: torch.Tensor,
        k3: torch.Tensor,
        k4: torch.Tensor,
        k5: torch.Tensor,
        k6: torch.Tensor,
        k7: torch.Tensor,
        k8: torch.Tensor,
        k9: torch.Tensor,
        k10: torch.Tensor,
        k11: torch.Tensor,
        k12: torch.Tensor,
        k13: torch.Tensor,
        k14: torch.Tensor,
        k15: torch.Tensor,
        k16: torch.Tensor,
        k17: torch.Tensor,
        k18: torch.Tensor,
        k19: torch.Tensor,
        k20: torch.Tensor,
        k21: torch.Tensor,
        k22: torch.Tensor,
        k23: torch.Tensor,
        v0: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
        v3: torch.Tensor,
        v4: torch.Tensor,
        v5: torch.Tensor,
        v6: torch.Tensor,
        v7: torch.Tensor,
        v8: torch.Tensor,
        v9: torch.Tensor,
        v10: torch.Tensor,
        v11: torch.Tensor,
        v12: torch.Tensor,
        v13: torch.Tensor,
        v14: torch.Tensor,
        v15: torch.Tensor,
        v16: torch.Tensor,
        v17: torch.Tensor,
        v18: torch.Tensor,
        v19: torch.Tensor,
        v20: torch.Tensor,
        v21: torch.Tensor,
        v22: torch.Tensor,
        v23: torch.Tensor,
    ) -> torch.Tensor:
        key_caches = (
            k0,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            k7,
            k8,
            k9,
            k10,
            k11,
            k12,
            k13,
            k14,
            k15,
            k16,
            k17,
            k18,
            k19,
            k20,
            k21,
            k22,
            k23,
        )
        value_caches = (
            v0,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
            v16,
            v17,
            v18,
            v19,
            v20,
            v21,
            v22,
            v23,
        )
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        hidden_states = self.model.model.forward_decode_static(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            key_caches=key_caches,
            value_caches=value_caches,
            cache_length=self.cache_length,
            attention_mask=None,
        )
        lm_head_weight = self.model.decode_lm_head_weight
        if lm_head_weight is None:
            lm_head_weight = self.model.model.embed_tokens.weight
        return F.linear(hidden_states[:, -1:, :], lm_head_weight)
