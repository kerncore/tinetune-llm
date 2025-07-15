from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, required=True)
parser.add_argument("--adapters_base_dir", type=str, default="./adapters")
parser.add_argument("--output_path", type=str, default="./merged-mistral-full")
args = parser.parse_args()

adapter_dirs = sorted([
    os.path.join(args.adapters_base_dir, d)
    for d in os.listdir(args.adapters_base_dir)
    if os.path.isdir(os.path.join(args.adapters_base_dir, d))
])

# Merge adapters sequentially
print(f"Merging adapters: {adapter_dirs}")

# Start from the first adapter
model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
    trust_remote_code=True,
    local_files_only=True,
)
model = PeftModel.from_pretrained(model, adapter_dirs[0])
model = model.merge_and_unload()
model.save_pretrained("./temp-merged-step", safe_serialization=True)

# Merge rest
for adapter_path in adapter_dirs[1:]:
    model = AutoModelForCausalLM.from_pretrained(
        "./temp-merged-step",
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        trust_remote_code=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(
        "./temp-merged-step",
        safe_serialization=True,
        max_shard_size="10GB",
    )

# Save final merged
model.save_pretrained(args.output_path, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, local_files_only=True)
tokenizer.save_pretrained(args.output_path)

