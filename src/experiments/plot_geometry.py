import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse

# --- Configuration ---
RESULTS_DIR = "results/geometry"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_geometry():
    print("🎨 Generating Figure 3: The Geometry of Failure...")
    
    # 1. Load Data
    json_path = os.path.join(RESULTS_DIR, "geometry_stats.json")
    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found. Run geometry.py first.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    sim_lie = data["syc_lie_similarity"]
    sim_refusal = data["syc_refusal_similarity"]
    
    # 2. Calculate Angles
    angle_lie = np.arccos(sim_lie)
    angle_refusal = np.arccos(sim_refusal)
    
    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.2)
    
    # 4. Define Vectors
    vec_syc = np.array([0, 1]) # Reference (North)
    
    # Rotation: x = sin(theta), y = cos(theta)
    vec_lie = np.array([np.sin(angle_lie), np.cos(angle_lie)])
    
    # Refusal usually negative, flip X to put on left side
    vec_refusal = np.array([-np.sin(angle_refusal), np.cos(angle_refusal)])

    # 5. Draw
    def plot_arrow(vec, color, label, offset=(0, 0)):
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.05, head_length=0.1, fc=color, ec=color, linewidth=3)
        ax.text(vec[0] + offset[0], vec[1] + offset[1], label, 
                fontsize=14, fontweight='bold', color=color, ha='center')

    plot_arrow(vec_syc, '#D62728', "Sycophancy\n(Intent)", (0, 0.1)) 
    plot_arrow(vec_lie, '#FF7F0E', "Hallucination\n(Content)", (0.4, 0)) 
    plot_arrow(vec_refusal, '#1F77B4', "Refusal\n(Safety)", (-0.3, 0)) 

    # 6. Arcs & Annotations
    theta_deg = np.degrees(angle_lie)
    # Matplotlib arc starts from East (0). North is 90.
    arc_lie = plt.matplotlib.patches.Arc((0,0), 0.5, 0.5, theta1=90-theta_deg, theta2=90, color='gray', linestyle='--')
    ax.add_patch(arc_lie)
    
    info_text = (
        f"Cosine Sim: {sim_lie:.4f}\n"
        f"Angle: {theta_deg:.1f}°\n"
        "Status: ORTHOGONAL BUT COUPLED"
    )
    ax.text(0, -0.2, info_text, ha='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

    # Save
    out_path = os.path.join(PLOTS_DIR, "figure_3_geometry_dynamic.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {out_path}")

if __name__ == "__main__":
    plot_geometry()