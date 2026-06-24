# Experiment 02: Local Eager Recognition

Goal: run the MinerU2.5-Pro two-step crop recognition flow with a local model
implementation. `AutoProcessor` is still used for tokenization and image
preprocessing, but the model class itself does not import Transformers.

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
  -> generate with "\nLayout Detection:"
  -> parse <|box_start|>...<|ref_start|>... layout output
  -> crop one selected block from the original RGB image
  -> choose the block recognition prompt
  -> generate recognition text
```

Implemented local model scope:

```text
Qwen2-VL Conv3D patch embed
32-layer vision tower, manual eager attention
2x2 patch merger
24-layer Qwen2 decoder, manual eager attention
dynamic KV cache
tied embedding LM head
greedy generate loop
```

The model loader requires a local checkpoint directory. It does not download
from Hugging Face. The MinerU2.5-Pro checkpoint currently has direct checkpoint
keys like `visual.*` and `model.layers.*`, so the local module names are chosen
to load those safetensor keys directly.

## Reusable Surfaces

`run_local_model_two_step_extract.py` owns the minimal protocol class:

```python
from run_local_model_two_step_extract import LocalMinerUTwoStepClient, LocalMinerUModelPredictor

predictor = LocalMinerUModelPredictor(model, processor, max_new_tokens=512)
client = LocalMinerUTwoStepClient(predictor)
result = client.two_step_extract(image, block_index=0)
```

`LocalMinerUTwoStepClient` exposes:

```text
layout_detect(image)
prepare_selected_block(image, blocks, block_index=...)
recognize_crop(crop, prompt, block)
two_step_extract(image, block_index=...)
```

## CUDA Smoke Command

Set `MODEL_DIR` to the local snapshot directory. On the current Vast CUDA box,
it is expected to look like:

```sh
export MODEL_DIR=/workspace/hf_cache/hub/models--opendatalab--MinerU2.5-Pro-2605-1.2B/snapshots/bff20d4ae2bf202df9f45284b4d43681555a97ed
```

Then run:

```sh
python 02_local_eager_recognition/run_local_model_two_step_extract.py \
  --model "$MODEL_DIR" \
  --device cuda:0 \
  --dtype float16 \
  --no-use-fast \
  --image crops/crop_01_text_block_en.png \
  --output outputs/local_model_crop_01_cuda.json
```

Expected: one parsed text block and the English BA matrix paragraph for
`recognition.text`.

Add decode-only timing:

```sh
python 02_local_eager_recognition/run_local_model_two_step_extract.py \
  --model "$MODEL_DIR" \
  --device cuda:0 \
  --dtype float16 \
  --no-use-fast \
  --image crops/crop_01_text_block_en.png \
  --benchmark-decode \
  --decode-warmup-steps 8 \
  --decode-measure-steps 64 \
  --output outputs/local_model_crop_01_cuda_decode_bench.json
```

This adds `layout.decode_benchmark` and `recognition.decode_benchmark` to the
JSON. `decode_tok_s` counts only cached decode forward calls after prefill;
`prefill_s` is reported separately and excluded from `decode_tok_s`.

## Work/NPU Smoke Command

Set `MODEL_DIR` to the local MinerU2.5-Pro snapshot directory on the NPU box,
then run:

```sh
python 02_local_eager_recognition/run_local_model_two_step_extract.py \
  --model "$MODEL_DIR" \
  --device npu:0 \
  --dtype float16 \
  --npu-jit-compile off \
  --npu-conv3d-mode auto \
  --no-use-fast \
  --image crops/crop_01_text_block_en.png \
  --benchmark-decode \
  --decode-warmup-steps 8 \
  --decode-measure-steps 64 \
  --output outputs/local_model_crop_01_npu.json
```

Expected: stdout shows `jit_compile=False` and the Conv3D patch line, then the
JSON result. `recognition.text` should be the English BA matrix paragraph.
Report both layout and recognition `decode_benchmark.decode_tok_s`.

For the first NPU report, include:

- exact command
- `MODEL_DIR` path
- whether stdout shows `jit_compile=False`
- whether stdout shows the Conv3D patch line
- layout raw text
- parsed block list
- recognition text
- layout and recognition decode_benchmark blocks
- timing_s
- output JSON path
- exact error text if it fails

Do not edit tracked files or write helper scripts on the NPU lane.

## Next Replacement Steps

1. Validate the local model on NPU.
2. Add optional logits/layer probes if generation text diverges.
3. Start experiment 03 by making decode cache/layout compile-friendly.
4. Replace processor/image preprocessing only after model parity is stable.
