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
