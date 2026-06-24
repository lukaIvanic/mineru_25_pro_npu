# Experiment 03: Compiled Single-Batch Decode

Goal: keep the MinerU2.5-Pro two-step flow from experiment 02, but make the
recognition decode path static-cache compiled. Layout detection remains dynamic
eager for now. The model class still has no Transformers imports; `AutoProcessor`
is still used by the runner for tokenization and image preprocessing.

Current entrypoint:

```text
run_local_model_two_step_extract.py
```

Local model files:

```text
config.py
local_modeling_mineru.py
```

## Pipeline

```text
RGB input image
  -> resize to 1036 x 1036 for layout
  -> dynamic eager layout generation with "\nLayout Detection:"
  -> parse <|box_start|>...<|ref_start|>... layout output
  -> crop one selected block from the original RGB image
  -> choose the block recognition prompt
  -> eager recognition prefill into a fixed KV cache
  -> TorchAir cache_compile / torch.compile one-token static decode
  -> greedy recognition text
  -> dynamic eager recognition reference validation
```

The compiled boundary is only the one-token recognition decode. Prefill,
vision, projector, layout generation, tokenization, and crop preprocessing are
outside the compiled graph.

Static decode details:

```text
batch size: 1
KV cache: [1, 2, cache_length, 64] per layer, 24 layers
cache update: torch_npu.scatter_update_ on NPU, index_copy_ elsewhere
attention: manual Qwen2 attention ops for now
compile: fullgraph=True, dynamic=False
TorchAir cache key: mineru_manual_attention_bs1_cache{cache_length}
compiled callable: explicit 24 K tensors + explicit 24 V tensors, no *args
```

The flat decode module is created per `(batch_size, cache_length)` and stores
`cache_length` as a Python attribute. The decode loops update `cache_position`
in-place with `add_(1)`. This is intentionally stricter than the earlier
varargs path because NPU/TorchAir recompile warnings are much easier to trigger
when the compiled callable owns Python tuple slicing or shape-derived cache
length logic.

## Work/NPU Smoke Command

Set `MODEL_DIR` to the local MinerU2.5-Pro snapshot directory on the NPU box.
The expected snapshot identity from the CUDA/Vast reference is:

```text
snapshot_revision: bff20d4ae2bf202df9f45284b4d43681555a97ed
config.json: size 2840, sha256 22097df08750242647a513043636a8dff16820a09757e9271e220bdea378df28
model.safetensors: size 2312126640, sha256 abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0
```

Run a short correctness and compile-cache smoke:

```sh
python 03_compiled_single_batch_decode/run_local_model_two_step_extract.py \
  --model "$MODEL_DIR" \
  --device npu:0 \
  --dtype float16 \
  --npu-jit-compile off \
  --npu-conv3d-mode auto \
  --no-use-fast \
  --image crops/crop_01_text_block_en.png \
  --max-new-tokens 128 \
  --cache-length 512 \
  --benchmark-decode \
  --decode-warmup-steps 4 \
  --decode-measure-steps 32 \
  --hash-model-files \
  --output outputs/exp03_crop_01_npu.json
```

Expected:

- stdout shows `jit_compile=False`
- stdout shows the Conv3D patch line
- `recognition.compiled_decode.enabled=true`
- `recognition.compiled_decode.compile.backend=torchair`
- `recognition.compiled_decode.compile.fullgraph=true`
- `recognition.compiled_decode.compile_warmup.ran_this_call=true` on the first
  cold process run for this shape
- `recognition.validation.trimmed_token_match=true`
- `recognition.canonical_reference.strict_match=true`
- `recognition.text` exactly equals:

```text
When an attempt is made to form the product BA, we discover that the dimensions are not compatible in this order because the rows of B are three-dimensional vectors and the columns of A are two-dimensional vectors. Hence the dot product of the jth row of B and the kth column of A is not defined.
```

- `recognition.decode_benchmark.scope=compiled_static_recognition_decode_only`
- `recognition.decode_benchmark.compile_warmup.ran_this_call=false` when it runs
  after recognition generation in the same process. If this is `true`, the
  benchmark paid a second compile and the decode compile cache contract is
  broken.
- warm cache reruns should have much smaller `compiled_first_call_s` than the
  first cold compile run

If the work/NPU run still prints repeated TorchAir/Dynamo recompilation
warnings, rerun the same command with guard logging enabled from process start:

```sh
TORCH_LOGS=recompiles,guards,graph_breaks \
TORCHDYNAMO_VERBOSE=1 \
python 03_compiled_single_batch_decode/run_local_model_two_step_extract.py \
  --model "$MODEL_DIR" \
  --device npu:0 \
  --dtype float16 \
  --npu-jit-compile off \
  --npu-conv3d-mode auto \
  --no-use-fast \
  --image crops/crop_01_text_block_en.png \
  --max-new-tokens 128 \
  --cache-length 512 \
  --benchmark-decode \
  --decode-warmup-steps 4 \
  --decode-measure-steps 8 \
  --hash-model-files \
  --output outputs/exp03_crop_01_npu_recompile_debug.json
```

The environment variables must be set before Python starts. Do not add inline
scripts. Report the exact `Recompiling function...`, `last reason`, guard
failure, and graph-break lines together with the JSON fields below.

Do not write helper scripts on the work/NPU lane. If the JSON is noisy because
TorchAir prints before the object, open the output file and read the final JSON
object from `outputs/exp03_crop_01_npu.json`.

Report these fields back:

```text
model_identity
timing_s
layout.raw_text
layout.decode_benchmark
recognition.text
recognition.compiled_decode
recognition.validation
recognition.canonical_reference
recognition.decode_benchmark
```

## CUDA/Vast Smoke Command

CUDA uses `torch.compile(fullgraph=True, dynamic=False)` instead of TorchAir:

```sh
python 03_compiled_single_batch_decode/run_local_model_two_step_extract.py \
  --model "$MODEL_DIR" \
  --device cuda:0 \
  --dtype float16 \
  --no-use-fast \
  --image crops/crop_01_text_block_en.png \
  --max-new-tokens 128 \
  --cache-length 512 \
  --benchmark-decode \
  --decode-warmup-steps 4 \
  --decode-measure-steps 32 \
  --output outputs/exp03_crop_01_cuda.json
```

CUDA is a development check only. Treat NPU output as authoritative for
TorchAir behavior and throughput.
