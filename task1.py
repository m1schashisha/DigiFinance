# import libraries
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

# 1. Can you plot the default rate for each grade category?
# Create a dictionary to store grade statistics
grade_stats = defaultdict(lambda: {'total': 0, 'defaults': 0})

# import the dataset
with open('data/loan_data_2017.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    
    # Get the header and find relevant column indices
    header = next(plots, None)
    
    # Find the indices for grade and loan_status columns
    grade_idx = None
    loan_status_idx = None
    for idx, col in enumerate(header):
        if col.lower() == 'grade':
            grade_idx = idx
        elif col.lower() == 'loan_status':
            loan_status_idx = idx
    
    print(f"Found columns - Grade: {grade_idx}, Loan Status: {loan_status_idx}")
    print(f"Header: {header}")
    
    if grade_idx is None or loan_status_idx is None:
        raise ValueError("Could not find required columns 'grade' and 'loan_status'")
    
    # Process each loan
    row_count = 0
    for row in plots:
        row_count += 1
        if len(row) > max(grade_idx, loan_status_idx):
            grade = row[grade_idx]
            loan_status = row[loan_status_idx].lower()
            
            # Count total loans per grade
            grade_stats[grade]['total'] += 1
            
            # Count defaults (assuming 'Charged Off' means default)
            if loan_status in ['charged off', 'default']:
                grade_stats[grade]['defaults'] += 1

# Print statistics before plotting
print(f"\nProcessed {row_count} rows")
print("\nDefault rates by grade:")
for grade in sorted(grade_stats.keys()):
    total = grade_stats[grade]['total']
    defaults = grade_stats[grade]['defaults']
    rate = defaults / total if total > 0 else 0
    print(f"Grade {grade}: {rate:.2%} ({defaults}/{total} loans)")

# Calculate default rates and prepare plotting data
grade_categories = sorted(grade_stats.keys())  # Sort grades alphabetically
default_rates = []

for grade in grade_categories:
    total = grade_stats[grade]['total']
    defaults = grade_stats[grade]['defaults']
    default_rate = defaults / total if total > 0 else 0
    default_rates.append(default_rate)

# Create the plot
plt.figure(figsize=(10, 6))
bars = plt.bar(grade_categories, default_rates, color='skyblue', width=0.6)

# Add percentage labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1%}',
             ha='center', va='bottom')

# Customize the plot
plt.xlabel('Grade Category', fontsize=12)
plt.ylabel('Default Rate', fontsize=12)
plt.title('Default Rate by Loan Grade', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Format y-axis as percentage
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

plt.tight_layout()
plt.show()

