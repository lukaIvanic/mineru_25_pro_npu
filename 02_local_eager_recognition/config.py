#!/usr/bin/env python3
"""Small dependency-free config objects for the MinerU2.5-Pro core VLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def default_rope_scaling() -> dict[str, Any]:
    return {"mrope_section": [8, 12, 12], "rope_type": "default", "type": "default"}


@dataclass(frozen=True)
class MinerUVisionConfig:
    depth: int = 32
    embed_dim: int = 1280
    hidden_size: int = 896
    hidden_act: str = "quick_gelu"
    mlp_ratio: int = 4
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    initializer_range: float = 0.02

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "MinerUVisionConfig":
        raw = dict(raw or {})
        return cls(
            depth=int(raw.get("depth", cls.depth)),
            embed_dim=int(raw.get("embed_dim", cls.embed_dim)),
            hidden_size=int(raw.get("hidden_size", cls.hidden_size)),
            hidden_act=str(raw.get("hidden_act", cls.hidden_act)),
            mlp_ratio=int(raw.get("mlp_ratio", cls.mlp_ratio)),
            num_heads=int(raw.get("num_heads", cls.num_heads)),
            in_channels=int(raw.get("in_channels", raw.get("in_chans", cls.in_channels))),
            patch_size=int(raw.get("patch_size", raw.get("spatial_patch_size", cls.patch_size))),
            spatial_merge_size=int(raw.get("spatial_merge_size", cls.spatial_merge_size)),
            temporal_patch_size=int(raw.get("temporal_patch_size", cls.temporal_patch_size)),
            initializer_range=float(raw.get("initializer_range", cls.initializer_range)),
        )


@dataclass(frozen=True)
class MinerUTextConfig:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: dict[str, Any] = field(default_factory=default_rope_scaling)
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "MinerUTextConfig":
        raw = dict(raw or {})
        rope_scaling = dict(raw.get("rope_scaling") or default_rope_scaling())
        rope_scaling.setdefault("rope_type", rope_scaling.get("type", "default"))
        rope_scaling.setdefault("type", rope_scaling.get("rope_type", "default"))
        rope_scaling.setdefault("mrope_section", [8, 12, 12])
        return cls(
            vocab_size=int(raw.get("vocab_size", cls.vocab_size)),
            hidden_size=int(raw.get("hidden_size", cls.hidden_size)),
            intermediate_size=int(raw.get("intermediate_size", cls.intermediate_size)),
            num_hidden_layers=int(raw.get("num_hidden_layers", cls.num_hidden_layers)),
            num_attention_heads=int(raw.get("num_attention_heads", cls.num_attention_heads)),
            num_key_value_heads=int(raw.get("num_key_value_heads", cls.num_key_value_heads)),
            hidden_act=str(raw.get("hidden_act", cls.hidden_act)),
            max_position_embeddings=int(raw.get("max_position_embeddings", cls.max_position_embeddings)),
            rms_norm_eps=float(raw.get("rms_norm_eps", cls.rms_norm_eps)),
            rope_theta=float(raw.get("rope_theta", cls.rope_theta)),
            rope_scaling=rope_scaling,
            attention_dropout=float(raw.get("attention_dropout", cls.attention_dropout)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", cls.tie_word_embeddings)),
            bos_token_id=int(raw.get("bos_token_id") or cls.bos_token_id),
            eos_token_id=int(raw.get("eos_token_id") or cls.eos_token_id),
            pad_token_id=int(raw.get("pad_token_id") or cls.pad_token_id),
        )


@dataclass(frozen=True)
class MinerUConfig:
    text_config: MinerUTextConfig
    vision_config: MinerUVisionConfig
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    tie_word_embeddings: bool = True

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def max_position_embeddings(self) -> int:
        return self.text_config.max_position_embeddings

    @classmethod
    def default(cls) -> "MinerUConfig":
        text_config = MinerUTextConfig()
        vision_config = MinerUVisionConfig()
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            bos_token_id=text_config.bos_token_id,
            eos_token_id=text_config.eos_token_id,
            pad_token_id=text_config.pad_token_id,
            tie_word_embeddings=text_config.tie_word_embeddings,
        )

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "MinerUConfig":
        path = Path(model_dir).expanduser() / "config.json"
        if not path.exists():
            return cls.default()
        return cls.from_json_file(path)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "MinerUConfig":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MinerUConfig":
        raw = dict(raw)
        text_raw = dict(raw.get("text_config") or {})
        for key in ("rope_scaling", "rope_theta", "tie_word_embeddings"):
            if key in raw and key not in text_raw:
                text_raw[key] = raw[key]
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if key in raw and key not in text_raw:
                text_raw[key] = raw[key]
        text_config = MinerUTextConfig.from_dict(text_raw)
        vision_config = MinerUVisionConfig.from_dict(raw.get("vision_config"))
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=int(raw.get("image_token_id", cls.image_token_id)),
            video_token_id=int(raw.get("video_token_id", cls.video_token_id)),
            vision_start_token_id=int(raw.get("vision_start_token_id", cls.vision_start_token_id)),
            vision_end_token_id=int(raw.get("vision_end_token_id", cls.vision_end_token_id)),
            bos_token_id=int(raw.get("bos_token_id") or text_config.bos_token_id),
            eos_token_id=int(raw.get("eos_token_id") or text_config.eos_token_id),
            pad_token_id=int(raw.get("pad_token_id") or text_config.pad_token_id),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", text_config.tie_word_embeddings)),
        )
