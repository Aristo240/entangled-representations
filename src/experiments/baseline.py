import argparse
import torch
import pandas as pd
import os
import time
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "bait_and_switch.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "raw")

# SET SPECIFIC DEVICE (Fixes the sharding bug)
DEVICE = "cuda:0" 

def load_model_and_tokenizer(model_name):
    print(f"🚀 Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"🚀 Loading model: {model_name}...")
    # NOTE: We removed device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 1. Handle Padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # 2. THE FIX: RESIZE EMBEDDINGS
    # This aligns the "Library" size with the "Librarian's" tickets.
    # This is safe to do because we are NOT using device_map="auto"
    current_vocab_size = model.config.vocab_size
    if len(tokenizer) > current_vocab_size:
        print(f"⚠️  RESIZING MODEL: {current_vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # 3. Move to GPU manually
    print(f"🔄 Moving model to {DEVICE}...")
    model.to(DEVICE)
        
    return model, tokenizer

def run_experiment(model_name, limit=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load model safe
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}. Run generator first!")
    
    df = pd.read_csv(DATA_PATH)
    if limit:
        df = df.head(limit)
        print(f"⚠️ Limiting run to first {limit} samples.")

    results = []
    
    print("⚡ Starting Generation Loop...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            prompt_text = row['poisoned_prompt_text']
            
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize and move to specific device
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            results.append({
                "original_question": row['original_question'],
                "bait_used": row['bait_used'],
                "full_prompt": formatted_prompt,
                "model_response": generated_text,
                "model_name": model_name
            })
            
        except Exception as e:
            print(f"❌ Error on sample {idx}: {e}")
            if "device-side assert" in str(e):
                print("💥 Critical CUDA Error. Restart Python.")
                sys.exit(1)
            continue

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = model_name.replace("/", "_")
    output_file = os.path.join(RESULTS_DIR, f"baseline_{safe_name}_{timestamp}.csv")
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"✅ Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    run_experiment(args.model, args.limit)