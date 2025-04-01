from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, required=True)
parser.add_argument("--adapter_path", type=str, required=True)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.base_model_path, device_map="auto", trust_remote_code=True, local_files_only=True)
model = PeftModel.from_pretrained(model, args.adapter_path)

model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, local_files_only=True)

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


