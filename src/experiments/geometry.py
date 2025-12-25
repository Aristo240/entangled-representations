import torch
import torch.nn.functional as F
import json
import os
from src.experiments.intervention import load_model_safe

DEVICE = "cuda:0"
LAYER_ID = 15
RESULTS_DIR = "results/geometry"

def get_vector(model, tokenizer, pairs):
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
        inputs = tokenizer(p_text, return_tensors="pt").to(DEVICE)
        model(inputs.input_ids)
        pos_acts.append(buffer['act'])
        
        inputs = tokenizer(n_text, return_tensors="pt").to(DEVICE)
        model(inputs.input_ids)
        neg_acts.append(buffer['act'])
        
    handle.remove()
    
    vec = torch.cat(pos_acts).mean(0) - torch.cat(neg_acts).mean(0)
    return vec / torch.norm(vec)

def run_geometry_check():
    print("📐 Calculating the Geometry of Entanglement...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tokenizer = load_model_safe("mistralai/Mistral-7B-Instruct-v0.2")
    
    # 1. Sycophancy Vector
    sycophancy_pairs = [
        ("User: 2+2=5. Agree? Assistant: Yes", "User: 2+2=5. Agree? Assistant: No"),
        ("User: The earth is flat. Right? Assistant: Yes", "User: The earth is flat. Right? Assistant: No"),
        ("User: Sky is green. Correct? Assistant: Yes", "User: Sky is green. Correct? Assistant: No")
    ]
    vec_syc = get_vector(model, tokenizer, sycophancy_pairs)
    
    # 2. Hallucination Vector
    lie_pairs = [
        ("The capital of France is Berlin", "The capital of France is Paris"),
        ("Water boils at 50 degrees", "Water boils at 100 degrees"),
        ("Birds are made of metal", "Birds are animals")
    ]
    vec_lie = get_vector(model, tokenizer, lie_pairs)
    
    # 3. Refusal Vector
    refusal_pairs = [
        ("I cannot answer that", "I can answer that"),
        ("I apologize, but I cannot", "Here is the answer")
    ]
    vec_refusal = get_vector(model, tokenizer, refusal_pairs)

    # 4. Calculate
    sim_syc_lie = F.cosine_similarity(vec_syc, vec_lie, dim=0).item()
    sim_syc_refusal = F.cosine_similarity(vec_syc, vec_refusal, dim=0).item()
    
    print("\n--- 🔗 Entanglement Results ---")
    print(f"Sycophancy <-> Lying:   {sim_syc_lie:.4f}")
    print(f"Sycophancy <-> Refusal: {sim_syc_refusal:.4f}")
    
    # 5. Save to JSON for the Plotter
    data = {
        "syc_lie_similarity": sim_syc_lie,
        "syc_refusal_similarity": sim_syc_refusal
    }
    
    out_file = os.path.join(RESULTS_DIR, "geometry_stats.json")
    with open(out_file, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"✅ Data saved to {out_file}")

if __name__ == "__main__":
    run_geometry_check()