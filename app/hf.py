# hf.py

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

output_path = "./merged-mistral-full"
repo_id = "kerncore/mistral-qlora-merged"
hf_token = "hf_..."  # replace with your token

login(token=hf_token)

model = AutoModelForCausalLM.from_pretrained(output_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(output_path, trust_remote_code=True)

model.push_to_hub(repo_id, safe_serialization=True)
tokenizer.push_to_hub(repo_id)


