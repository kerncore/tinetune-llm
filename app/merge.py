# release_llm.py

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

base_model_path = "/workspace/remove-refusals-with-transformer/mistral-small-24b"
adapter_paths = [
    "./adapters/adapter_00",
    "./adapters/adapter_01"
]
output_path = "./merged-mistral-full"

# Step 1: Merge adapter 1
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_paths[0])
model = model.merge_and_unload()
model.save_pretrained("./temp-merged1", safe_serialization=True)

# Step 2: Merge adapter 2 into step 1
model = AutoModelForCausalLM.from_pretrained("./temp-merged1", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_paths[1])
model = model.merge_and_unload()
model.save_pretrained(output_path, safe_serialization=True)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

