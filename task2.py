# 2. Use the "sample"command to randomly select certain index numbers and then use the selected index 
# numbers to divide the dataset into a training dataset and testing dataset. 
# Please use 30% of the data for testing and the rest of the 70% for training.

# Import pandas - we only need pandas since it has sampling functionality built-in
import pandas as pd

# Load the dataset with low_memory=False to handle mixed data types
df = pd.read_csv('data/loan_data_2017.csv', low_memory=False)

# Use pandas sample to create test set (30%) and use the remaining as train set (70%)
test_data = df.sample(frac=0.3, random_state=42)  # 30% for testing
train_data = df.drop(test_data.index)             # remaining 70% for training

# Print the split sizes to verify
print(f"Total rows: {len(df)}")
print(f"Training set size: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
print(f"Testing set size: {len(test_data)} ({len(test_data)/len(df)*100:.1f}%)")