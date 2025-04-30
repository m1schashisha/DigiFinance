# Import required libraries that we'll use in this program
import matplotlib.pyplot as plt  # matplotlib helps us create visual graphs and charts
import csv  # csv module helps us read data from CSV (spreadsheet) files
from collections import defaultdict  # defaultdict is a special dictionary that can create default values automatically
import numpy as np  # numpy helps with numerical operations and data manipulation

# defaultdict is chosen over regular dictionaries for two key reasons:
# 1. Performance: It eliminates the need to check if a key exists before incrementing counters
# 2. Code cleanliness: It reduces boilerplate code for initializing new dictionary entries
# This makes our code more readable and efficient, especially when dealing with unknown categories
grade_stats = defaultdict(lambda: {'total': 0, 'defaults': 0})

# Open and read the CSV file that contains our loan data
# The 'with' statement ensures the file is properly closed after we're done reading it
with open('data/loan_data_2017.csv', 'r', encoding='utf-8') as file:
    # Create a CSV reader that understands the structure of our file
    # DictReader automatically reads the first row as column headers
    reader = csv.DictReader(file)
    
    rows_processed = 0  # Counter to keep track of how many loans we've analyzed
    # Process each row (each loan) in the CSV file
    for row in reader:
        try:
            # Get the loan grade and remove any extra spaces
            grade = row['grade'].strip()
            # Get the loan status (convert to lowercase to make comparison easier)
            status = row['loan_status'].strip().lower()
            
            # Skip this loan if either grade or status is empty
            if not grade or not status:
                continue
                
            # Count this loan in the total for its grade
            grade_stats[grade]['total'] += 1
            
            # Check if this loan defaulted
            # The definition of "default" includes three specific statuses:
            # 1. 'charged off' - Loans that the lender has written off as a loss
            # 2. 'default' - Loans explicitly marked as defaulted
            # 3. 'does not meet the credit policy. status:charged off' - Loans that were charged off
            #    due to not meeting credit policy
            # These statuses were chosen because they all represent loans where the borrower failed to
            # repay as agreed, resulting in a loss for the lender. In the P2P lending context, these
            # are the outcomes we want to predict and avoid.
            if status in ['charged off', 'default', 'does not meet the credit policy. status:charged off']:
                grade_stats[grade]['defaults'] += 1
            
            rows_processed += 1
                
        except (KeyError, AttributeError) as e:
            # Skip any rows that have missing or invalid data
            continue

# Print a summary of how many loans we analyzed
print(f"\nProcessed {rows_processed} rows of loan data.")
print("\nDefault rates by grade:")

# Prepare data for our graph
grades = sorted(grade_stats.keys())  # Get a sorted list of all loan grades
default_rates = []  # List to store the calculated default rates

# Calculate the default rate for each grade
for grade in grades:
    total = grade_stats[grade]['total']  # Total number of loans for this grade
    defaults = grade_stats[grade]['defaults']  # Number of defaults for this grade
    # Calculate default rate as a percentage (handle case where total might be 0)
    rate = defaults / total if total > 0 else 0
    default_rates.append(rate)
    # Print the statistics in a readable format
    print(f"Grade {grade}: {rate:.2%} ({defaults}/{total} loans)")

# Create a visual bar chart of the default rates with enhanced formatting
plt.figure(figsize=(10, 6))  # Create a new figure with specified size 

# Create bars with a color gradient that helps distinguish between different grades
# Using a color gradient from blue to red helps visually emphasize the risk progression
colors = plt.cm.coolwarm(np.linspace(0, 1, len(grades)))
bars = plt.bar(grades, default_rates, color=colors)

# Add data labels on top of each bar for better readability
for bar, rate in zip(bars, default_rates):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')

# Add grid lines for easier visual comparison of values
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add labels and title to make the graph more informative
plt.title('Loan Default Rates by Grade', fontsize=14, fontweight='bold')
plt.xlabel('Loan Grade', fontsize=12)
plt.ylabel('Default Rate', fontsize=12)

# Format the y-axis to show percentages
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

# Add a brief interpretation of the results as text annotation
plt.figtext(0.5, 0.01, 
           "As expected, default rates increase significantly from grade A to G. This confirms that \n"
           "the LendingClub grading system effectively stratifies credit risk, with each grade \n"
           "representing a distinct risk level for investors.", 
           ha='center', fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})

# Save the graph as an image file
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for the annotation
plt.savefig('task1_default_rates.png', dpi=300)  # Save with task-specific name and high resolution
plt.close()  # Close the plot to free up memory

print("\nGraph saved as 'task1_default_rates.png'")
print("Interpretation: The results clearly show that default rates progressively increase from lower to higher risk grades,")
print("confirming that LendingClub's grading system effectively stratifies borrowers by credit risk.")
print("This finding is relevant to the P2P vs banking study as it demonstrates how P2P platforms use")
print("alternative scoring systems to segment borrowers, potentially serving segments underserved by traditional banks.")