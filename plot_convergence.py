import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("doa_convergence_best1.csv")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['Iteration'][1:], df['BestCost'][1:], marker='o', linestyle='-', color='blue')
plt.title("Economic Load Dispatch using DREAM Algorithm")
plt.xlabel("Iteration")
plt.ylabel("Best Cost (Rs/hr)")
plt.grid(True)
plt.tight_layout()
plt.savefig("doa_convergence_best1.png")
plt.show()
