# Experiment 02: Local Eager Recognition

Goal: replace the official MinerU runtime in small, testable layers before
replacing the model implementation itself.

## Step 01: Manual Protocol, Official HF Model

`run_manual_two_step_extract.py` does not import `MinerUClient` or
`mineru_vl_utils`. It still uses:

```text
AutoProcessor
Qwen2VLForConditionalGeneration
```

The script locally implements the minimal protocol surface needed for the first
crop/page tests:

```text
RGB input image
  -> resize to 1036 x 1036 for layout
  -> generate with "\nLayout Detection:"
  -> parse <|box_start|>...<|ref_start|>... layout output
  -> crop one selected block from the original RGB image
  -> choose the block recognition prompt
  -> generate recognition text
```

This is intentionally not the final local model. It is the first replacement
boundary: official HF model and processor, local MinerU-style protocol.

Reusable classes in the script:

```python
from run_manual_two_step_extract import LocalMinerUTwoStepClient, LocalTransformersPredictor

predictor = LocalTransformersPredictor(model, processor)
client = LocalMinerUTwoStepClient(predictor)
result = client.two_step_extract(image, block_index=0)
```

`LocalMinerUTwoStepClient` exposes the protocol surfaces we will replace/test
independently:

- `layout_detect(image)`
- `prepare_selected_block(image, blocks, block_index=...)`
- `recognize_crop(crop, prompt, block)`
- `two_step_extract(image, block_index=...)`

The CLI is only a wrapper around this class. Future experiment-2 scripts should
reuse the class instead of copying protocol logic into `main()`.

## Step 02: Local Model, External Processor

`run_local_model_two_step_extract.py` keeps the reusable two-step client from
step 01 and replaces the model backend with:

```text
config.py
local_modeling_mineru.py
```

The local model class does not import Transformers. It still expects
`AutoProcessor` for tokenization and image preprocessing, so this step isolates
the model implementation without changing preprocessing at the same time.

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

## CUDA Smoke Command

```sh
export HF_HOME=/workspace/hf_cache

python 02_local_eager_recognition/run_manual_two_step_extract.py \
  --device-map none \
  --device cuda:0 \
  --dtype float16 \
  --attn-implementation eager \
  --no-use-fast \
  --image crops/crop_01_text_block_en.png \
  --output outputs/manual_protocol_crop_01_cuda.json \
  --local-files-only
```

Expected: one parsed text block and a recognition string matching the official
experiment-01 output for `crop_01_text_block_en.png`.

## CUDA Local-Model Smoke Command

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

Expected: the same selected layout block and recognition text as the manual
protocol/HF-model smoke command.

## First Work/NPU Command

```sh
source /path/to/cann/set_env.sh
source /path/to/venv/bin/activate
export HF_HOME=/path/to/hf_cache
export ASCEND_RT_VISIBLE_DEVICES=0

python 02_local_eager_recognition/run_manual_two_step_extract.py \
  --device-map none \
  --device npu:0 \
  --dtype float16 \
  --attn-implementation eager \
  --npu-jit-compile off \
  --npu-conv3d-mode auto \
  --no-use-fast \
  --image crops/crop_01_text_block_en.png \
  --output outputs/manual_protocol_crop_01_npu.json
```

## First Work/NPU Local-Model Command

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
  --output outputs/local_model_crop_01_npu.json
```

Expected: stdout shows `jit_compile=False` and the Conv3D patch line, then the
JSON result. Compare `recognition.text` with the official/HF-model experiment
02 output for the same crop.

For the first NPU report, include:

- exact command
- whether stdout shows `jit_compile=False`
- whether stdout shows the Conv3D patch line
- layout raw text
- parsed block list
- recognition text
- output JSON path
- exact error text if it fails

Do not edit tracked files or write helper scripts on the NPU lane.

## Next Replacement Steps

1. Validate the local model on CUDA and NPU against the official/HF-model
   protocol output.
2. Add optional logits/layer probes if generation text diverges.
3. Start experiment 03 by making decode cache/layout compile-friendly.
4. Replace processor/image preprocessing only after model parity is stable.
