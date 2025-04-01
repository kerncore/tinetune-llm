from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default="./merged-mistral-full")
parser.add_argument("--repo_id", type=str, required=True)
parser.add_argument("--hf_token", type=str, required=True)
args = parser.parse_args()

login(token=args.hf_token)

model = AutoModelForCausalLM.from_pretrained(args.output_path, trust_remote_code=True, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(args.output_path, trust_remote_code=True, local_files_only=True)

model.push_to_hub(args.repo_id, safe_serialization=True)
tokenizer.push_to_hub(args.repo_id)

