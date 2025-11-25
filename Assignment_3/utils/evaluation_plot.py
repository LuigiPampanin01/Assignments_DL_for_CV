import matplotlib.pyplot as plt

# ============================
#  Detection Rate Data
#  (from your experiments)
# ============================

# Number of proposals evaluated
proposals = [500, 1000, 1500, 2000]

# IoU threshold k = 0.5
k05 = [
    0.8599596065910239,   # 500 proposals
    0.8955070677780568,   # 1000
    0.8994025206431411,   # 1500
    0.8994025206431411    # 2000 (plateau)
]

# IoU threshold k = 0.6
k06 = [
    0.7566322857948174,   # 500 proposals
    0.8063024822034334,   # 1000
    0.8230711896616714,   # 1500
    0.8245749490061676    # 2000
]
plt.figure(figsize=(8,6))

# Plot k = 0.5
plt.plot(proposals, k05, marker='o', linewidth=2, label="IoU threshold k = 0.5")

# Plot k = 0.6
plt.plot(proposals, k06, marker='o', linewidth=2, label="IoU threshold k = 0.6")

# Labels and formatting
plt.title("Selective Search Detection Rate vs Number of Proposals")
plt.xlabel("Number of Proposals per Image")
plt.ylabel("Detection Rate")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(proposals)
plt.ylim(0.70, 0.95)
plt.legend()

# Save or show
plt.tight_layout()
plt.savefig("evaluation_plot.png")
