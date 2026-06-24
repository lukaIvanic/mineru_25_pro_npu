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

1. Replace `model.generate` with a manual greedy loop while still using the HF
   model.
2. Add local config and weight loading.
3. Implement local Qwen2-VL modules and compare logits layer by layer.
4. Replace processor/image preprocessing last, after model parity is stable.
