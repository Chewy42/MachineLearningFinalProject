import pandas as pd

# Read the CSV file
df = pd.read_csv('dataset/train/_annotations.csv')

# Get unique classes and sort them
unique_classes = sorted(df['class'].unique())

# Print the classes and their indices
print("Unique classes found:")
for i, class_name in enumerate(unique_classes):
    print(f"{i}: {class_name}")

# Create the YAML format string
yaml_format = "names:\n"
for i, class_name in enumerate(unique_classes):
    yaml_format += f"  {i}: {class_name}\n"

print("\nYAML format:")
print(yaml_format)