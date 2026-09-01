"""
generate_robustness_chart.py

Generates a bar chart comparing plain vs. augmented model accuracy
across all 5 test conditions, using the final confirmed numbers.
Saves as eval/robustness_chart.png.

Run from the project root:
    python generate_robustness_chart.py
"""

import matplotlib.pyplot as plt
import numpy as np

conditions = ["Clean", "JPEG q=30", "Blur σ=2.0", "Crop 80%", "WildFake\n(Unseen Generator)"]
plain_acc = [87.25, 86.36, 89.69, 85.10, 50.62]
augmented_acc = [90.52, 88.65, 90.95, 88.93, 52.58]

x = np.arange(len(conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, plain_acc, width, label="Plain Model", color="#8899aa")
bars2 = ax.bar(x + width/2, augmented_acc, width, label="Augmented Model", color="#3d5a80")

ax.set_ylabel("Accuracy (%)")
ax.set_title("Robustness Comparison: Plain vs. Augmented Model")
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.legend()
ax.set_ylim(0, 100)
ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, label="Random chance")

# Add value labels on top of each bar
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("eval/robustness_chart.png", dpi=150)
print("Saved chart to eval/robustness_chart.png")
plt.close()