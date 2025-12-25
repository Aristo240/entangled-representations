import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.experiments.intervention import get_steering_vector, load_model_safe

DEVICE = "cuda:0"
LAYER_ID = 15 # The one you used

def run_logit_lens():
    print("🔮 Decrypting the Sycophancy Vector...")
    model, tokenizer = load_model_safe("mistralai/Mistral-7B-Instruct-v0.2")
    
    # 1. Get the vector
    # We need to temporarily patch the global variable in intervention.py or just recreate it
    # Ideally, save the vector to disk in intervention.py, but for now we regenerate it
    vector = get_steering_vector(model, tokenizer)
    
    # 2. Project to Vocabulary
    # The vector is size [4096]. The Unembedding matrix is [32000, 4096].
    # MatMul -> [32000] (Scores for every word in the dictionary)
    
    # Note: In Llama/Mistral, lm_head is the unembedding layer
    logits = torch.matmul(model.lm_head.weight, vector)
    
    # 3. Get Top and Bottom tokens
    top_tokens = torch.topk(logits, 10).indices
    bottom_tokens = torch.topk(logits, 10, largest=False).indices
    
    print("\n--- 🗣️ What does this vector say? ---")
    
    print("\n⬆️  Promoted Tokens (Sycophancy):")
    for t in top_tokens:
        print(f"  '{tokenizer.decode(t)}'")
        
    print("\n⬇️  Suppressed Tokens (Honesty/Refusal):")
    for t in bottom_tokens:
        print(f"  '{tokenizer.decode(t)}'")

if __name__ == "__main__":
    run_logit_lens()