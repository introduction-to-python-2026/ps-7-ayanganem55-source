# main.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load and inspect the data
print("Loading and inspecting the data...")
# Load the 'penguins' dataset provided by seaborn
df = sns.load_dataset("penguins")

# Display basic information
print("\n--- DataFrame Head ---")
print(df.head())
print("\n--- DataFrame Info ---")
print(df.info())
print("\n--- DataFrame Description ---")
print(df.describe())

# 2. Select several features
# We will focus on numerical features: 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
# Drop rows with missing values for simplicity in visualization
df_cleaned = df.dropna(subset=features)
print(f"\nWorking with cleaned data focusing on features: {features}")

# 3. Create histograms
print("\nGenerating histograms...")
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Histograms of Penguin Features', fontsize=16)

sns.histplot(df_cleaned['bill_length_mm'], kde=True, ax=axes[0, 0], color='skyblue')
sns.histplot(df_cleaned['bill_depth_mm'], kde=True, ax=axes[0, 1], color='coral')
sns.histplot(df_cleaned['flipper_length_mm'], kde=True, ax=axes[1, 0], color='lightgreen')
sns.histplot(df_cleaned['body_mass_g'], kde=True, ax=axes[1, 1], color='salmon')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# You can save this histogram figure if you want, but the assignment asks only for the correlation plot.
# plt.savefig("histograms.png") 
plt.show()

# 4. Create scatter plots
print("\nGenerating individual scatter plots...")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_cleaned, x='flipper_length_mm', y='body_mass_g', hue='species')
plt.title('Flipper Length vs. Body Mass by Species')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_cleaned, x='bill_length_mm', y='bill_depth_mm', hue='species')
plt.title('Bill Length vs. Bill Depth by Species')
plt.show()


# 5. Generate and save a correlation scatter plot (using AI suggested enhancements - Seaborn style, regression line)
print("\nGenerating and saving the main correlation scatter plot...")

plt.figure(figsize=(10, 6))
# Use Seaborn's regplot for a scatter plot with a regression line for clearer correlation visualization
sns.regplot(data=df_cleaned, x='flipper_length_mm', y='body_mass_g', 
            ci=95, scatter_kws={'alpha':0.6, 's': 50}, line_kws={'color': 'red', 'lw': 3})

# Enhance readability with better fonts and titles
plt.title('Strong Positive Correlation: Flipper Length vs. Body Mass (with Regression Line)', fontsize=16, fontweight='bold')
plt.xlabel('Flipper Length (mm)', fontsize=12)
plt.ylabel('Body Mass (g)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Save the figure to a file
figure_filename = 'correlation_plot.png'
plt.savefig(figure_filename, bbox_inches='tight')
print(f"Successfully saved the correlation plot as {figure_filename}")
print("\nScript finished.")

