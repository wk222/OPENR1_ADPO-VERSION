#!/usr/bin/env python3
"""
Debug script to check ADPO memory configuration
"""

from open_r1.configs import ADPOConfig, ADPOScriptArguments
from trl import ModelConfig, TrlParser

# Parse config
parser = TrlParser((ADPOScriptArguments, ADPOConfig, ModelConfig))
script_args, training_args, model_args = parser.parse_args_and_config()

print("=" * 80)
print("ADPO Configuration Debug")
print("=" * 80)

# Key memory-related settings
print(f"\n🔍 Memory-Critical Settings:")
print(f"  anchor_update_mode: {training_args.anchor_update_mode}")
print(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")
print(f"  gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
print(f"  num_generations: {training_args.num_generations}")
print(f"  gradient_checkpointing: {training_args.gradient_checkpointing}")

# vLLM settings
print(f"\n🚀 vLLM Settings:")
if hasattr(training_args, 'use_vllm'):
    print(f"  use_vllm: {training_args.use_vllm}")
if hasattr(training_args, 'vllm_mode'):
    print(f"  vllm_mode: {training_args.vllm_mode}")
if hasattr(training_args, 'vllm_gpu_memory_utilization'):
    print(f"  vllm_gpu_memory_utilization: {training_args.vllm_gpu_memory_utilization}")

# Model settings
print(f"\n🤖 Model Settings:")
print(f"  model_name_or_path: {model_args.model_name_or_path}")
print(f"  torch_dtype: {model_args.torch_dtype}")

# ADPO specific settings
print(f"\n📊 ADPO Specific:")
print(f"  tau: {training_args.tau}")
print(f"  beta_anchor_kl: {training_args.beta_anchor_kl}")
print(f"  beta_reward: {training_args.beta_reward}")
print(f"  use_q_centering: {training_args.use_q_centering}")
print(f"  use_adaptive_tau: {training_args.use_adaptive_tau}")

# Sequence lengths
print(f"\n📏 Sequence Lengths:")
print(f"  max_prompt_length: {training_args.max_prompt_length}")
print(f"  max_completion_length: {training_args.max_completion_length}")

print("\n" + "=" * 80)
print("❗ MEMORY ISSUE DIAGNOSIS:")
print("=" * 80)

# Check for common issues
issues = []

if training_args.anchor_update_mode != "on_policy":
    issues.append(f"⚠️  anchor_update_mode='{training_args.anchor_update_mode}' will load a FULL COPY of the model!")
    issues.append("   This DOUBLES memory usage. Set to 'on_policy' to fix.")

if hasattr(training_args, 'vllm_mode') and training_args.vllm_mode == "colocate":
    if hasattr(training_args, 'vllm_gpu_memory_utilization'):
        vllm_mem = training_args.vllm_gpu_memory_utilization
        if vllm_mem > 0.4:
            issues.append(f"⚠️  vllm_gpu_memory_utilization={vllm_mem} is high for colocate mode")
            issues.append(f"   Consider reducing to 0.3 or using vllm_mode='server'")

effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
if effective_batch_size * training_args.num_generations > 100:
    issues.append(f"⚠️  Effective batch size × num_generations = {effective_batch_size * training_args.num_generations}")
    issues.append("   This is quite large. Consider reducing batch_size or num_generations.")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✅ No obvious memory configuration issues found.")
    print("   The problem may be in recent code changes to TRL-ADPO trainer.")

print("\n" + "=" * 80)

