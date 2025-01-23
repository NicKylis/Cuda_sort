# All the code required to generate the plots in the documentation

import matplotlib.pyplot as plt
import numpy as np

# Data
N_exponents = [20, 23, 25, 28]  # Exponents of 2
execution_time = [20.121857, 179.788803, 799.303894, 7313.434570]

# Calculate N values as 2^exponent
N_values = [2 ** n for n in N_exponents]

# Convert exponents into labels like 2^20, 2^23, etc.
N_labels = [f'$2^{{{n}}}$' for n in N_exponents]

# Create the plot with Execution Time on the vertical axis and N on the horizontal axis
plt.figure(figsize=(8, 5))
plt.plot(N_values, execution_time, marker='o', linestyle='-', color='b')

# Set logarithmic scale for both axes
plt.xscale('log')
plt.yscale('log')

# Customize the x-axis with 2^ labels
plt.xticks(N_values, N_labels)

# Add labels and title
plt.xlabel('N', rotation=0, labelpad=20)  # Rotate 'N' for clarity
plt.ylabel('Execution Time (ms)')
plt.title('Execution Time vs N (Logarithmic Scale for Both Axes)')
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# Modify the y-ticks to show execution time values
yticks = plt.gca().get_yticks()

# Annotate execution times on the y-axis
ytick_labels = [f'{tick:.2f} ms' for tick in yticks]

# Update the y-ticks with the execution time labels
plt.yticks(yticks, ytick_labels)

# Show the plot
plt.show()