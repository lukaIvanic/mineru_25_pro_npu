# AGENTS.md

## Operating Lanes

First classify where you are running from actual machine state, not memory:

- Authoring lane: Luka's local code-editing checkout. It may have no accelerator.
- Vast/CUDA lane: a rented Vast.ai GPU box, usually under `/workspace`; `nvidia-smi` works, Ascend tooling does not.
- Work/NPU lane: Ascend NPU tooling is present, such as `npu-smi` or `torch_npu`.

The Work/NPU lane is pull-only. Its job is to set up the environment, pull the repo, run scripts, inspect outputs, debug failures, and summarize exact findings. Do not edit tracked files, commit, push, or create branches from the Work/NPU lane. If a code change seems necessary, report the smallest proposed change, the command that failed, and the relevant logs.

CUDA/Vast results are smoke-test evidence only. Do not present CUDA throughput or correctness quirks as Ascend/NPU evidence.

## Project Direction

This folder is a standalone research workspace for MinerU2.5-Pro on Ascend/NPU. The immediate goal is to replicate experiments 01-03 from `paddle_ocr_vl_npu`, but for the MinerU architecture:

- `01_transformers_recognition_baseline`: run the public Transformers implementation directly.
- `02_local_eager_recognition`: implement the model locally with no Transformers dependency in the model code.
- `03_compiled_single_batch_decode`: make single-batch static decode compile-friendly for Ascend/TorchAir.

Default checkpoint:

```text
opendatalab/MinerU2.5-Pro-2605-1.2B
```

Use 2605 as the latest public model unless Luka explicitly asks for 2604 comparison. 2604 and 2605 have the same Qwen2-VL architecture shape; 2605 is the newer checkpoint.

## Current Architecture Understanding

MinerU2.5-Pro is a compact Qwen2-VL-style VLM rather than a custom backbone:

- Transformers class: `Qwen2VLForConditionalGeneration`
- model type: `qwen2_vl`
- dtype in config: `bfloat16`
- text decoder: 24 layers, hidden size 896, FFN intermediate 4864, 14 query heads, 2 KV heads, head dim 64
- vision tower: 32 layers, hidden size 1280, 16 heads, head dim 80, MLP ratio 4, `quick_gelu`
- patch embed: Qwen2-VL-style Conv3d with temporal patch size 2 and spatial patch size 14
- patch merger: 2x2 spatial merge, `4 * 1280 -> 5120 -> 896`
- vocab size: 151936
- image token IDs: `<|vision_start|>=151652`, `<|vision_end|>=151653`, `<|vision_pad|>=151654`, `<|image_pad|>=151655`, `<|video_pad|>=151656`
- image processor: `Qwen2VLImageProcessor`, `min_pixels=50176`, `max_pixels=1605632`, `patch_size=14`, `merge_size=2`

The vision encoder attends over pre-merge patch tokens. The decoder sees post-merge visual tokens.

## MinerUClient Boundary

The official `mineru-vl-utils` package provides `MinerUClient`, including `two_step_extract`, `layout_detect`, and batch/async layout APIs. Treat `MinerUClient` as an important reference for prompts, task routing, output formatting, and the two-stage page/crop workflow.

For experiments 01-03, do not hide the core model behind `MinerUClient` until the direct Transformers model path is understood. If a client probe is added, keep it separate from the pure model baseline so model architecture and wrapper/postprocessing behavior do not get mixed together.

Experiment 01 should reproduce the official two-step behavior in a simplified, inspectable form:

1. Run the official `MinerUClient.layout_detect(page_image)` on one page/image.
2. Select exactly one returned `ContentBlock`.
3. Run content recognition only for that selected block/crop, using the same prompt and sampling parameters that `MinerUClient.prepare_for_extract(...)` would choose.
4. Also run `MinerUClient.two_step_extract(page_image)` as the official reference.
5. Compare the selected block's type, bbox, crop, prompt, raw generated content, and postprocessed content against the corresponding block from `two_step_extract`.

This gives an accuracy/protocol anchor without waiting for full-page recognition. It also keeps layout detection, crop preparation, recognition, and postprocessing as separate surfaces for later local/compiled implementation work.

## Local Artifacts

- `crops/`: fresh copy of the PaddleOCR-VL project's OmniDocBench region crops and manifests. These are crops, not full pages.
- `01_transformers_recognition_baseline/`: planned minimal Transformers baseline.
- `02_local_eager_recognition/`: planned local eager model implementation.
- `03_compiled_single_batch_decode/`: planned static-cache compiled decode experiment.

Keep `crops/` in Git. Do not add model weights, HF caches, profiler dumps, generated outputs, or NPU compile artifacts.

## Source Links

- Model: https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B
- 2604 model for architecture comparison: https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B
- MinerU repo: https://github.com/opendatalab/MinerU
- MinerU VLM utilities: https://github.com/opendatalab/mineru-vl-utils
- Paper: https://arxiv.org/abs/2604.04771
