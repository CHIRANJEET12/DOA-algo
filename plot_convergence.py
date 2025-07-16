import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('doa_convergence.csv')

plt.figure(figsize=(10, 6))
plt.plot(data['Iteration'], data['BestCost'], linewidth=2)
plt.title('DOA Convergence for ELD Problem', fontsize=14)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Best Cost ($/hr)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('convergence_plot.png', dpi=300)
plt.show()