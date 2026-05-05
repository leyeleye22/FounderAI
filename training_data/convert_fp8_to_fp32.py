"""Convert FP8 quantized Qwen3 model to float32 for CPU training."""
import torch
import json
import os
from safetensors.torch import load_file, save_file
import shutil

MODEL_DIR = "C:/Users/Mr LEYE/Downloads/FounderAI"
OUTPUT_DIR = "C:/Users/Mr LEYE/Downloads/FounderAI/base_model_fp32"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copy config and tokenizer
for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
    src = os.path.join(MODEL_DIR, fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUTPUT_DIR, fname))

# Update config
config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
if "quantization_config" in config:
    del config["quantization_config"]
config["torch_dtype"] = "float32"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

def dequantize_fp8_block(weight_fp8, scale):
    """Dequantize FP8 weight with 128x128 block scales.
    
    Weight shape: [out_features, in_features]
    Scale shape: [out_blocks, in_blocks] where blocks = features // 128
    """
    block_size = 128
    out_features, in_features = weight_fp8.shape
    out_blocks = out_features // block_size
    in_blocks = in_features // block_size
    
    # Convert to float32
    weight_fp32 = weight_fp8.to(torch.float32)
    scale_fp32 = scale.to(torch.float32)
    
    # Reshape weight to blocks: [out_blocks, block_size, in_blocks, block_size]
    weight_blocked = weight_fp32.view(out_blocks, block_size, in_blocks, block_size)
    
    # Scale: [out_blocks, in_blocks] -> [out_blocks, 1, in_blocks, 1]
    scale_expanded = scale_fp32.unsqueeze(1).unsqueeze(3)
    
    # Dequantize
    dequantized = weight_blocked * scale_expanded
    
    # Reshape back
    return dequantized.view(out_features, in_features)

# Load and convert each shard
for shard_name in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
    print(f"Converting {shard_name}...")
    shard_path = os.path.join(MODEL_DIR, shard_name)
    tensors = load_file(shard_path)
    
    converted = {}
    dequantized_count = 0
    
    # Process all tensors
    weight_names = [k for k in tensors.keys() if k.endswith(".weight") and tensors[k].dtype == torch.float8_e4m3fn]
    
    for name in weight_names:
        scale_name = name.replace(".weight", ".weight_scale_inv")
        if scale_name in tensors:
            converted[name] = dequantize_fp8_block(tensors[name], tensors[scale_name])
            dequantized_count += 1
        else:
            print(f"  WARNING: No scale for {name}")
            converted[name] = tensors[name].to(torch.float32)
    
    # Copy non-FP8 tensors
    for name, tensor in tensors.items():
        if name not in converted and "weight_scale_inv" not in name:
            converted[name] = tensor.to(torch.float32)
    
    output_path = os.path.join(OUTPUT_DIR, shard_name)
    save_file(converted, output_path)
    print(f"  Dequantized {dequantized_count} weights, saved {len(converted)} tensors")

# Create new index
total_size = 0
new_weight_map = {}
for shard_name in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
    tensors = load_file(os.path.join(OUTPUT_DIR, shard_name))
    for name, tensor in tensors.items():
        new_weight_map[name] = shard_name
        total_size += tensor.numel() * tensor.element_size()

new_index = {
    "metadata": {"total_size": total_size},
    "weight_map": new_weight_map
}

with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(new_index, f, indent=2)

print(f"\nConversion complete!")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Total size: {total_size / 1e9:.1f} GB")
