# PIERMoE

This folder contains a submission-oriented public code package for PIERMoE.

Core implementation details in the model, description encoder, and routed expert module are intentionally hidden with the placeholder:

```text
accept will be available
```

## Files

- `config.py`: public configuration schema with generic default paths.
- `model.py`: public model interface and visual feature encoder.
- `description_fusion.py`: public description encoder interface.
- `emotion_grouped_moe.py`: public routed expert interface.
- `safe_audio_data_loader.py`: dataset loader compatibility wrapper.
- `train_piermoe.py`: training and evaluation entry point.


## Paths

Set local paths through environment variables before running the shell scripts:

```bash
PIERMoE_ROOT=/path/to/PIERMoE-main
DATA_ROOT=/path/to/dataset
DESC_JSONL_PATH=/path/to/descriptions.jsonl
TEXT_MODEL_NAME=/path/to/text-backbone
DESC_TEXT_MODEL_NAME=/path/to/text-backbone
AUDIO_MODEL_NAME_OR_PATH=/path/to/audio-backbone
```

## Examples

```bash
bash PIER-MoE/run_piermoe_mosi.sh
bash PIER-MoE/run_piermoe_mosei.sh
```
