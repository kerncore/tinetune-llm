# testing_adapters.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_path = "/workspace/remove-refusals-with-transformer/mistral-small-24b"
adapter_path = "./adapters/adapter_00"  # Change as needed

model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_path)

model.eval()
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

prompt = "<s>[INST] <<SYS>> Test prompt for assistant <</SYS>> Hello there! [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))

