# Post-training recipes

## OlympicCoder

To train the OlympicCoder models, run:

```
# 7B
sbatch --nodes=1 slurm/train.slurm --model OlympicCoder-7B --task sft --config v00.00 --accelerator zero3

# 32B
sbatch --nodes=16 slurm/train.slurm --model OlympicCoder-32B --task sft --config v00.00 --accelerator fsdp
```

Note that we found it necessary to switch to FSDP1 and paged AdamW 8-bit for the 32B model in order to fit the largest possible context size.

## Qwen3 ADPO (Anchored DPO)

The `recipes/Qwen3/adpo` folder contains a ready-to-run configuration for fine-tuning `Qwen/Qwen3-1.6B-Instruct` with the new ADPO trainer introduced in this repo. After installing `open-r1` and our local fork of `trl`, you can start a run locally with:

```
python -u src/open_r1/adpo.py \
	--config recipes/Qwen3/adpo/config_qwen3-1_6b.yaml \
	--report_to wandb \
	--wandb_project open-r1-ADPO \
	--run_name qwen3_1_6b_adpo_debug
```

This assumes GPUs with bfloat16 support and optionally a vLLM server if you enable `use_vllm` in the config.