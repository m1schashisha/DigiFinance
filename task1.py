# Import required libraries that we'll use in this program
import matplotlib.pyplot as plt  # matplotlib helps us create visual graphs and charts
import csv  # csv module helps us read data from CSV (spreadsheet) files
from collections import defaultdict  # defaultdict is a special dictionary that can create default values automatically

# defaultdict is like a smart dictionary that automatically creates a new entry with default values
# when we try to access a key that doesn't exist yet. This helps us avoid writing extra code to check
# if a grade exists in our dictionary. The lambda function below creates a new dictionary with
# {total: 0, defaults: 0} whenever we access a new grade for the first time.
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
            # We consider a loan as defaulted if it's either 'charged off' or explicitly marked as 'default'
            # or if it was charged off due to not meeting credit policy
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

# Create a visual bar chart of the default rates
plt.figure(figsize=(10, 6))  # Create a new figure with specified size (width=10, height=6)
plt.bar(grades, default_rates, color='skyblue')  # Create bars for each grade

# Add labels and title to make the graph more informative
plt.title('Loan Default Rates by Grade')  # Add a title to the graph
plt.xlabel('Loan Grade')  # Label for the x-axis (horizontal)
plt.ylabel('Default Rate')  # Label for the y-axis (vertical)
# Format the y-axis to show percentages
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

# Save the graph as an image file
plt.savefig('task1_default_rates.png')  # Save with task-specific name
plt.close()  # Close the plot to free up memory

