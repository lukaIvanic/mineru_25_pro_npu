# Experiment 04: Compiled Batch Decode

Goal: measure absolute throughput of compiled batched recognition decode for
MinerU2.5-Pro. This is not a serving scheduler and not an EOS experiment. It
uses real, distinct crop images for each batch row, builds a ready batch with
sequential eager prefill, then runs a fixed number of compiled one-token decode
forwards.

Entrypoint:

```text
bench_compiled_batch_decode.py
```

The rest of the folder is copied from experiment 03 so this experiment has no
runtime dependency on exp03.

## Decode Contract

For batch size `B`, cache length `L`, and measured decode steps `N`:

```text
1. Select B distinct crop images.
2. For each crop independently:
   crop -> AutoProcessor -> input_ids/pixels/grid
   eager static prefill -> one [1, 2, L, 64] K cache and V cache per layer
   first next_token, rope_deltas, next_cache_position
3. Concatenate the B prefilled cache rows:
   K_l: [1, 2, L, 64] x B -> [B, 2, L, 64]
   V_l: [1, 2, L, 64] x B -> [B, 2, L, 64]
   next_token: [B, 1]
   cache_position: [B]
   rope_deltas: [B, 1]
4. Compile one static decode graph:
   forward(input_ids[B,1], cache_position[B], rope_deltas[B,1], 24 K, 24 V)
     -> logits[B,1,vocab]
5. Run exactly N compiled decode forwards:
   logits -> argmax per row -> next input_ids
   cache_position += 1 for every row
   ignore EOS
```

Throughput math:

```text
raw_batch_tokens = B * N
decode_calls_per_s = N / decode_s
raw_batch_tok_s = (B * N) / decode_s
```

Prefill, processor work, cache assembly, compile, and warmup are reported but
excluded from `raw_batch_tok_s`.

Decode rotary can be selected with `--decode-rotary-impl`:

```text
manual          original PyTorch rotate_half math
npu_rotary_mul  NPU-native torch_npu.npu_rotary_mul(..., rotary_mode="half")
```

The native path is decode-only. Prefill still uses the manual implementation so
the ready cache contract does not change while we measure the compiled decode
graph. Validation compares batched compiled decode against single-item static
eager decode with manual rotary.

## Correctness Check

The script validates the batched compiled decode against single-item static
eager decode for `--validation-steps` fixed steps. This check ignores EOS just
like the benchmark loop. If `validation.token_match_all=false`, do not trust the
throughput result.

## NPU Commands

Set `MODEL_DIR` to the local MinerU2.5-Pro snapshot. Use fp16, JIT compile off,
slow processor, and the NPU Conv3D patch like exp03.

Start with B=1 to preserve the exp03 baseline shape:

```sh
python 04_compiled_batch_decode/bench_compiled_batch_decode.py \
  --model "$MODEL_DIR" \
  --device npu:0 \
  --dtype float16 \
  --npu-jit-compile off \
  --npu-conv3d-mode auto \
  --no-use-fast \
  --batch-size 1 \
  --cache-length 512 \
  --measure-steps 64 \
  --warmup-steps 8 \
  --validation-steps 8 \
  --decode-weight-format decode_nz \
  --decode-rotary-impl manual \
  --torchair-cache-dir outputs/exp04_torchair_cache_decode_nz \
  --hash-model-files \
  --output outputs/exp04_batch1_decode_nz.json
```

Then run B=2, B=4, and B=8:

```sh
python 04_compiled_batch_decode/bench_compiled_batch_decode.py \
  --model "$MODEL_DIR" \
  --device npu:0 \
  --dtype float16 \
  --npu-jit-compile off \
  --npu-conv3d-mode auto \
  --no-use-fast \
  --batch-size 2 \
  --cache-length 512 \
  --measure-steps 64 \
  --warmup-steps 8 \
  --validation-steps 8 \
  --decode-weight-format decode_nz \
  --decode-rotary-impl manual \
  --torchair-cache-dir outputs/exp04_torchair_cache_decode_nz \
  --hash-model-files \
  --output outputs/exp04_batch2_decode_nz.json

python 04_compiled_batch_decode/bench_compiled_batch_decode.py \
  --model "$MODEL_DIR" \
  --device npu:0 \
  --dtype float16 \
  --npu-jit-compile off \
  --npu-conv3d-mode auto \
  --no-use-fast \
  --batch-size 4 \
  --cache-length 512 \
  --measure-steps 64 \
  --warmup-steps 8 \
  --validation-steps 8 \
  --decode-weight-format decode_nz \
  --decode-rotary-impl manual \
  --torchair-cache-dir outputs/exp04_torchair_cache_decode_nz \
  --hash-model-files \
  --output outputs/exp04_batch4_decode_nz.json

python 04_compiled_batch_decode/bench_compiled_batch_decode.py \
  --model "$MODEL_DIR" \
  --device npu:0 \
  --dtype float16 \
  --npu-jit-compile off \
  --npu-conv3d-mode auto \
  --no-use-fast \
  --batch-size 8 \
  --cache-length 512 \
  --measure-steps 64 \
  --warmup-steps 8 \
  --validation-steps 8 \
  --decode-weight-format decode_nz \
  --decode-rotary-impl manual \
  --torchair-cache-dir outputs/exp04_torchair_cache_decode_nz \
  --hash-model-files \
  --output outputs/exp04_batch8_decode_nz.json
```

The first run for each `(batch_size, cache_length, decode_weight_format)` may
pay a fresh TorchAir compile. The cache key also includes `decode_rotary_impl`,
so manual and `npu_rotary_mul` use separate graphs. A warm rerun should avoid
the cold compile.

After the manual baseline passes, repeat the same B=1/2/4/8 commands with:

```text
--decode-rotary-impl npu_rotary_mul
--torchair-cache-dir outputs/exp04_torchair_cache_decode_nz_rotary_npu
--output outputs/exp04_batch{B}_decode_nz_rotary_npu.json
```

Do not compare speed unless both modes pass validation for the same batch size.

Report these fields for every batch size:

```text
model_identity
decode_weight_format
decode_rotary_impl
compile
timing_s
throughput
validation
selected_crops
```

The key comparisons are:

```text
throughput.raw_batch_tok_s
throughput.decode_calls_per_s
timing_s.decode_s
validation.token_match_all
decode_rotary_impl.effective_mode
compile.compiled_first_call_s
compile.torchair_cache_dir
```

Do not add helper scripts on the NPU lane. The benchmark already emits the JSON
needed for the comparison.

## CUDA Smoke

CUDA is only a development check for the batch plumbing:

```sh
python 04_compiled_batch_decode/bench_compiled_batch_decode.py \
  --model "$MODEL_DIR" \
  --device cuda:0 \
  --dtype float16 \
  --no-use-fast \
  --batch-size 2 \
  --cache-length 512 \
  --measure-steps 8 \
  --warmup-steps 2 \
  --validation-steps 4 \
  --decode-weight-format none \
  --decode-rotary-impl manual \
  --output outputs/exp04_batch2_cuda_smoke.json
```

CUDA results are not NPU throughput evidence.
