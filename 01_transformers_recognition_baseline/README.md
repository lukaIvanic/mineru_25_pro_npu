# Experiment 01: Transformers Baseline And MinerUClient Split Probe

Goal: establish the official MinerU2.5-Pro behavior before building local model code.

Default model:

```text
opendatalab/MinerU2.5-Pro-2605-1.2B
```

## First Script: Official High-Level Tutorial

`run_official_two_step_extract.py` mirrors the highest-level `mineru-vl-utils`
Transformers tutorial:

```text
Qwen2VLForConditionalGeneration.from_pretrained(...)
AutoProcessor.from_pretrained(...)
MinerUClient(backend="transformers", model=model, processor=processor)
client.two_step_extract(image)
```

Install the official tutorial dependencies:

```sh
pip install -U "mineru-vl-utils[transformers]"
```

Run on the default copied crop:

```sh
python3 01_transformers_recognition_baseline/run_official_two_step_extract.py \
  --output outputs/official_two_step_crop_01.json
```

Run on another image:

```sh
python3 01_transformers_recognition_baseline/run_official_two_step_extract.py \
  --image crops/crop_05_table_rwkv_dims.png \
  --output outputs/official_two_step_table.json
```

This is deliberately the fully official high-level path. It is expected to be
slow with the Transformers backend, but it gives us a reference output before
we split layout detection and recognition.

## Validated CUDA Smoke Environment

This was validated on Vast.ai instance `42360616`, alias `vast_mineru_cuda`:

```text
GPU: NVIDIA GeForce RTX 3090, 24576 MiB
driver: 580.126.18
base image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
python: 3.11.11
torch: 2.6.0+cu124
torchvision: 0.21.0+cu124
transformers: 4.57.6
mineru-vl-utils: 1.0.5
accelerate: 1.14.0
safetensors: 0.8.0
tokenizers: 0.22.2
huggingface-hub: 0.36.2
pillow: 12.2.0
numpy: 2.2.2
```

`mineru-vl-utils` is a PyPI package whose project homepage is:

```text
https://github.com/opendatalab/mineru-vl-utils
```

The validated setup used a venv with the CUDA image's system PyTorch:

```sh
python -m venv --system-site-packages /workspace/venvs/mineru25
source /workspace/venvs/mineru25/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -U "mineru-vl-utils[transformers]" accelerate safetensors pillow
```

The model cache was kept outside the repo:

```sh
export HF_HOME=/workspace/hf_cache
```

Validated command:

```sh
python 01_transformers_recognition_baseline/run_official_two_step_extract.py \
  --image crops/crop_01_text_block_en.png \
  --output outputs/official_two_step_crop_01.json \
  --no-use-fast \
  --no-tqdm
```

Observed output for `crop_01_text_block_en.png`: one detected `text` block with
reasonable English OCR content. First setup/load was about 30.5 s, and
`two_step_extract` was about 3.5 s on the RTX 3090.

`--use-fast` is off by default. Keep it off for NPU smoke tests because the
fast processor path has triggered torchvision-related problems in this project
family before. Turn it on only when explicitly testing that path.

## First Work/NPU Attempt

The CUDA command above uses `device_map=auto`, which is the official tutorial
shape. On Ascend, do not assume `device_map=auto` will choose the NPU correctly.
For the first NPU run, use explicit placement:

```sh
source /path/to/cann/set_env.sh
source /path/to/venv/bin/activate
export HF_HOME=/path/to/hf_cache
export ASCEND_RT_VISIBLE_DEVICES=0

python 01_transformers_recognition_baseline/run_official_two_step_extract.py \
  --device-map none \
  --device npu:0 \
  --dtype float16 \
  --attn-implementation eager \
  --npu-jit-compile off \
  --npu-conv3d-mode auto \
  --no-use-fast \
  --image crops/crop_01_text_block_en.png \
  --output outputs/npu_official_two_step_crop_01.json \
  --no-tqdm
```

Rationale:

- MinerU2.5-Pro config declares `bfloat16`, but small 310P-class devices are
  usually safer with `float16`.
- `--device-map none --device npu:0` loads normally and then moves the model
  explicitly, avoiding hidden CUDA-oriented placement assumptions.
- `--attn-implementation eager` avoids immediately betting on SDPA support in
  the first NPU smoke test.
- `--npu-jit-compile off` calls `torch.npu.set_compile_mode(jit_compile=False)`
  before loading/running the model. This avoids long eager-mode torch-npu JIT
  compiles.
- `--npu-conv3d-mode auto` patches `torch.nn.functional.conv3d` to call
  `torch_npu.npu_conv3d` for NPU tensors, matching the GLM-OCR/PaddleOCR-VL
  runtime workaround pattern.
- `--no-use-fast` keeps the processor on the conservative non-fast path.
- This is still the official high-level `MinerUClient.two_step_extract` path;
  it is only a device-placement adaptation.

For the NPU report, capture:

- `python --version`
- `pip show mineru-vl-utils transformers torch torch-npu accelerate safetensors tokenizers`
- `npu-smi info`
- exact command
- whether model loading succeeds
- whether the output JSON has one sensible text block for `crop_01_text_block_en.png`
- any dtype, unsupported-op, or memory error text

## What We Need To Reproduce

The official `mineru-vl-utils` flow is:

```text
page image
  -> MinerUClient.layout_detect(image)
      - resize page to layout_image_size, default 1036 x 1036
      - prompt: "\nLayout Detection:"
      - parse model output into ContentBlock objects with normalized bboxes
  -> MinerUClient.prepare_for_extract(image, blocks)
      - crop each selected block from the original page image
      - rotate by block angle if needed
      - resize small or extreme-aspect crops by client helper rules
      - choose prompt by block type
  -> batch content recognition
  -> post_process(blocks)
```

The default recognition prompts in `mineru-vl-utils` are:

```python
{
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
    "image": "\nImage Analysis:",
    "chart": "\nImage Analysis:",
    "[default]": "\nText Recognition:",
    "[layout]": "\nLayout Detection:",
}
```

## Simplified Baseline We Want

Do not start with full-page end-to-end throughput. Start with a split probe:

1. Load one input page/image.
2. Build a `MinerUClient(backend="transformers", ...)`.
3. Run `layout_detect(image)` and save the raw parsed blocks.
4. Select one block, preferably a text/table/equation block.
5. Run only that block through the official crop/prompt/content-recognition path.
6. Run `two_step_extract(image)` as the official reference.
7. Compare the selected block result against the corresponding block in `two_step_extract`.

The comparison should report:

- selected block index
- block type
- normalized bbox
- crop size
- recognition prompt
- raw generated text before postprocess, if available
- postprocessed content
- whether the selected-block split result matches the same block from `two_step_extract`

## Why This Shape

This keeps four things separate:

- layout detection
- crop preparation
- crop/content recognition
- postprocessing

That separation matters because experiments 02 and 03 will replace the model internals while still needing to check correctness against official MinerU behavior.

## Notes For Later

The pure model baseline should still use `Qwen2VLForConditionalGeneration` and `AutoProcessor` directly. `MinerUClient` is the protocol/reference layer, not the model architecture itself.
