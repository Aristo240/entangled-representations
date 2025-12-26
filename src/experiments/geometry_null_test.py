import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
DEVICE = "cuda:1"  # Set to your free GPU
LAYER_ID = 15      # The layer where we found the effect
SAVE_DIR = "results/geometry"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

os.makedirs(SAVE_DIR, exist_ok=True)

def load_model_locally():
    """Loads model specifically onto the configured DEVICE."""
    print(f"🚀 Loading model on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,  # Forces model to specific GPU
        trust_remote_code=True
    )
    return model, tokenizer

def get_vector(model, tokenizer, pairs):
    """Extracts a vector from contrastive pairs."""
    pos_acts = []
    neg_acts = []
    
    buffer = {}
    def hook(m, i, o):
        hid = o[0] if isinstance(o, tuple) else o
        if len(hid.shape) == 3: t = hid[:, -1, :]
        else: t = hid[-1, :].unsqueeze(0)
        buffer['act'] = t.detach()
        
    handle = model.model.layers[LAYER_ID].register_forward_hook(hook)
    
    for p_text, n_text in pairs:
        # Move inputs to the correct DEVICE
        inputs = tokenizer(p_text, return_tensors="pt").to(DEVICE)
        model(inputs.input_ids)
        pos_acts.append(buffer['act'])
        
        inputs = tokenizer(n_text, return_tensors="pt").to(DEVICE)
        model(inputs.input_ids)
        neg_acts.append(buffer['act'])
        
    handle.remove()
    
    # Acts are already on device (detached), but let's ensure
    vec = torch.cat(pos_acts).mean(0) - torch.cat(neg_acts).mean(0)
    return vec / torch.norm(vec)

def run_null_test():
    print("📐 Running Geometry Null Hypothesis Test...")
    
    # Load model using the local function to ensure correct GPU
    model, tokenizer = load_model_locally()
    
    # 1. Extract Real Vectors
    print("   Extracting Sycophancy Vector...")
    syc_pairs = [
        ("User: 2+2=5. Agree? Assistant: Yes", "User: 2+2=5. Agree? Assistant: No"),
        ("User: Earth is flat. Right? Assistant: Yes", "User: Earth is flat. Right? Assistant: No"),
        ("User: Sky is green. Correct? Assistant: Yes", "User: Sky is green. Correct? Assistant: No")
    ]
    vec_syc = get_vector(model, tokenizer, syc_pairs)
    
    print("   Extracting Hallucination Vector...")
    lie_pairs = [
        ("The capital of France is Berlin", "The capital of France is Paris"),
        ("Water boils at 50 degrees", "Water boils at 100 degrees"),
        ("Birds are made of metal", "Birds are animals")
    ]
    vec_lie = get_vector(model, tokenizer, lie_pairs)
    
    # Calculate Real Similarity
    real_sim = F.cosine_similarity(vec_syc, vec_lie, dim=0).item()
    print(f"   ✅ Real Similarity (Syc <-> Lie): {real_sim:.4f}")

    # 2. Generate Null Distribution (Random Vectors)
    print("   Generating Null Distribution (1000 pairs)...")
    null_sims = []
    dim = vec_syc.shape[0] # 4096 for Mistral
    
    torch.manual_seed(42)
    for _ in range(1000):
        # Generate two random vectors on the correct DEVICE
        v1 = torch.randn(dim).to(DEVICE)
        v2 = torch.randn(dim).to(DEVICE)
        
        # Normalize
        v1 = v1 / torch.norm(v1)
        v2 = v2 / torch.norm(v2)
        
        sim = F.cosine_similarity(v1, v2, dim=0).item()
        null_sims.append(sim)
        
    # 3. Statistical Significance
    null_mean = np.mean(null_sims)
    null_std = np.std(null_sims)
    z_score = (real_sim - null_mean) / null_std
    
    print(f"   🎲 Null Mean: {null_mean:.4f} ± {null_std:.4f}")
    print(f"   📈 Z-Score: {z_score:.2f}")
    
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(null_sims, bins=50, alpha=0.7, color='gray', label='Random Vectors (Null)')
    plt.axvline(real_sim, color='red', linestyle='--', linewidth=2, label=f'Sycophancy-Lie ({real_sim:.2f})')
    plt.title("Geometry of Entanglement: Signal vs. Noise")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = f"{SAVE_DIR}/null_distribution.png"
    plt.savefig(out_path)
    print(f"✅ Plot saved to {out_path}")
    
    # Save Stats
    with open(f"{SAVE_DIR}/stats.txt", "w") as f:
        f.write(f"Real Similarity: {real_sim}\n")
        f.write(f"Null Mean: {null_mean}\n")
        f.write(f"Null Std: {null_std}\n")
        f.write(f"Z-Score: {z_score}\n")

if __name__ == "__main__":
    run_null_test()