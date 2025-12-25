import argparse
import torch
import pandas as pd
import os
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "bait_and_switch.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "truth_check")
DEVICE = "cuda:1"

def load_model_safe(model_name):
    print(f"🚀 Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    print(f"🚀 Loading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(DEVICE)
    
    if len(tokenizer) > model.config.vocab_size:
        print(f"⚠️ Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer

def run_truth_check(model_name, limit=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tokenizer = load_model_safe(model_name)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    if limit:
        df = df.head(limit)
        print(f"⚠️ Limiting run to first {limit} samples.")

    results = []
    
    print("⚡ Starting Truth Check Loop...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # --- THE CRITICAL DIFFERENCE ---
            # We use ONLY the original question. No bait. No pressure.
            question = row['original_question']
            
            # Simple, neutral chat template
            messages = [{"role": "user", "content": question}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=150,
                    do_sample=False, # Deterministic
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            results.append({
                "original_question": question,
                "clean_response": generated_text,
                "model_name": model_name
            })
            
        except Exception as e:
            print(f"❌ Error on sample {idx}: {e}")
            continue

    # Save Results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"truth_check_{timestamp}.csv")
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"✅ Truth Check saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    run_truth_check("mistralai/Mistral-7B-Instruct-v0.2", args.limit)