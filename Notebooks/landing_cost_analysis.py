import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Create detailed cost breakdown data
cost_data = {
  'Landing Outcome': ['Successful Landing', 'Failed Landing'],
  'Total Cost (M$)': [62, 165],
  'First Stage Cost (M$)': [45, 45],
  'Recovery Cost (M$)': [2, 0],
  'Replacement Cost (M$)': [0, 105],
  'Operation Cost (M$)': [15, 15]
}

# Create DataFrame
df = pd.DataFrame(cost_data)

# Save the raw data
df.to_csv('landing_costs_analysis.csv', index=False)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Plot 1: Simple comparison
bars = ax1.bar(df['Landing Outcome'], df['Total Cost (M$)'])
bars[0].set_color('#2ecc71')  # Green for success
bars[1].set_color('#e74c3c')  # Red for failure

# Add value labels
for bar in bars:
  height = bar.get_height()
  ax1.text(bar.get_x() + bar.get_width()/2., height,
           f'${int(height)}M',
           ha='center', va='bottom')

ax1.set_ylabel('Total Cost (Million $)')
ax1.set_title('Total Launch Cost Comparison')

# Plot 2: Stacked bar chart showing cost breakdown
cost_breakdown = df[['Landing Outcome', 'First Stage Cost (M$)', 'Recovery Cost (M$)', 
                  'Replacement Cost (M$)', 'Operation Cost (M$)']]

# Create stacked bars
bottom = np.zeros(2)
colors = ['#3498db', '#2ecc71', '#e74c3c', '#95a5a6']
labels = ['First Stage', 'Recovery', 'Replacement', 'Operations']

for idx, cost_type in enumerate(['First Stage Cost (M$)', 'Recovery Cost (M$)', 
                             'Replacement Cost (M$)', 'Operation Cost (M$)']):
  ax2.bar(cost_breakdown['Landing Outcome'], cost_breakdown[cost_type], 
          bottom=bottom, label=labels[idx], color=colors[idx])
  bottom += cost_breakdown[cost_type]

# Add value labels for total cost
for i, total in enumerate(df['Total Cost (M$)']):
  ax2.text(i, total + 5, f'${int(total)}M Total', ha='center')

ax2.set_ylabel('Cost Breakdown (Million $)')
ax2.set_title('Launch Cost Breakdown Analysis')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Calculate and display savings
savings = df['Total Cost (M$)'].iloc[1] - df['Total Cost (M$)'].iloc[0]
plt.figtext(0.5, -0.05, 
          f'Potential Savings per Successful Landing: ${int(savings)}M\n'
          f'ROI on Recovery System: {((savings/2)/.02):.1f}%',
          ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))

# Adjust layout and save
plt.tight_layout()
plt.savefig('landing_cost_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a summary statistics DataFrame
summary_stats = pd.DataFrame({
  'Metric': ['Total Potential Savings', 'ROI on Recovery System', 'Cost Difference %'],
  'Value': [f'${savings}M', f'{((savings/2)/.02):.1f}%', 
            f'{((df["Total Cost (M$)"].iloc[1]/df["Total Cost (M$)"].iloc[0])-1)*100:.1f}%']
})

# Save summary statistics
summary_stats.to_csv('landing_cost_summary.csv', index=False)

# Print created files
print("\nFiles created during execution:")
for file in ["landing_costs_analysis.csv", "landing_cost_analysis.png", "landing_cost_summary.csv"]:
  print(file)

# Display summary statistics
print("\nSummary Statistics:")
print(summary_stats.to_string(index=False))