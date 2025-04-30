# 2. Use the "sample" command to randomly select certain index numbers and then use the selected index 
# numbers to divide the dataset into a training dataset and testing dataset. 
# Please use 30% of the data for testing and the rest of the 70% for training.

# Import pandas - a powerful data manipulation library that offers advanced functionality
# for working with tabular data like CSV files
import pandas as pd

# We chose pandas over scikit-learn's train_test_split for several reasons:
# 1. The assignment specifically mentions using a "sample" command - pandas provides this directly
# 2. Pandas is more versatile for initial data exploration and manipulation
# 3. Pandas' sampling maintains the original index, making it easier to track which rows went where
# 4. It's more memory-efficient for large datasets as we can drop indices without copying all data
print("\n--- Task 2: Train-Test Split using Pandas ---")

# Load the dataset with low_memory=False to handle mixed data types
print("Loading dataset...")
df = pd.read_csv('data/loan_data_2017.csv', low_memory=False)
print(f"Dataset loaded with {len(df)} loans and {len(df.columns)} features.")

# Use pandas sample to create test set (30%) and use the remaining as train set (70%)
# We set random_state=42 for reproducibility - this ensures that:
# 1. The exact same split will be obtained each time the code runs
# 2. Others can replicate our results exactly when using the same random_state
# 3. It's a convention in data science (42 is arbitrary but commonly used)
print("Performing 70/30 train-test split...")
test_data = df.sample(frac=0.3, random_state=42)  # 30% for testing
train_data = df.drop(test_data.index)             # remaining 70% for training

# Print the split sizes to verify
print(f"Total rows: {len(df)}")
print(f"Training set size: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
print(f"Testing set size: {len(test_data)} ({len(test_data)/len(df)*100:.1f}%)")

# Save the training and testing datasets to CSV files for future use
# This is good practice as it ensures reproducibility of subsequent analyses
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
print("Split datasets saved as 'train_data.csv' and 'test_data.csv'")

print("\nThe train-test split is crucial for evaluating predictive models because:")
print("- It prevents overfitting by testing on data the model hasn't seen during training")
print("- The 70/30 ratio balances having enough data for training while ensuring")
print("  robust evaluation on the test set")
print("- In the context of the P2P lending study, this split enables us to measure how well")
print("  we can predict defaults based on loan characteristics, which is essential for")
print("  understanding risk assessment differences between P2P platforms and traditional banks.")