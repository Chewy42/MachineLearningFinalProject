import os
import pandas as pd
import shutil
import yaml

def load_yaml_classes():
    """Load class names and their IDs from dataset.yaml"""
    with open('dataset.yaml', 'r') as f:
        yaml_data = yaml.safe_load(f)
    # Convert from {id: name} to {name: id} format
    return {str(name).lower(): id for id, name in yaml_data.get('names', {}).items()}

def get_unique_classes():
    """Get all unique classes from the CSV files"""
    all_classes = set()
    for split in ['train', 'valid', 'test']:
        csv_file = f"dataset/{split}/_annotations.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            all_classes.update(df['class'].unique())
    return sorted(list(all_classes))

def clean_class_name(class_name):
    """Clean and standardize class names"""
    if class_name in ['0', '1', '2', '3']:  # Add any other numeric classes you find
        return 'bear'  # Default to 'bear' for numeric classes based on dataset.yaml
    elif class_name.lower() == 'dogs-cats':
        return 'dog'  # Default to 'dog' for combined classes
    elif class_name.lower() == 'rat':
        return 'otherentities'  # Map unknown animals to otherentities
    return class_name

# Load class mapping from YAML
yaml_classes = load_yaml_classes()
print("YAML classes:", yaml_classes)

# Get classes from CSV files
csv_classes = get_unique_classes()
print("Found classes in CSV:", csv_classes)

# Create class mapping using YAML IDs
class_map = {}
for class_name in csv_classes:
    cleaned_name = clean_class_name(class_name)
    class_lower = str(cleaned_name).lower()
    
    # Check if the class exists in yaml_classes
    if class_lower in yaml_classes:
        class_id = yaml_classes[class_lower]
        class_map[class_name] = class_id  # Map original name
        class_map[class_lower] = class_id  # Map lowercase name
        
        # Add common variations
        if class_lower == 'human':
            class_map['person'] = class_id
        elif class_lower == 'dog':
            class_map['gaurd dog'] = class_id
            class_map['guard dog'] = class_id
            class_map['dogs-cats'] = class_id
        elif class_lower == 'goat':
            class_map['goa'] = class_id
        elif class_lower == 'bicycle':
            class_map['bycicle'] = class_id
    else:
        print(f"Warning: Class '{class_name}' not found in dataset.yaml")

print("Final class mapping:", class_map)

def convert_bbox(row):
    """Convert bounding box from [xmin, ymin, xmax, ymax] to YOLO format [x_center, y_center, width, height]"""
    width = float(row['width'])
    height = float(row['height'])
    
    x_center = (float(row['xmin']) + float(row['xmax'])) / 2 / width
    y_center = (float(row['ymin']) + float(row['ymax'])) / 2 / height
    bbox_width = (float(row['xmax']) - float(row['xmin'])) / width
    bbox_height = (float(row['ymax']) - float(row['ymin'])) / height
    
    # Clean the class name before looking up in the mapping
    class_name = clean_class_name(row['class'])
    try:
        class_id = class_map[class_name]
    except KeyError:
        print(f"Warning: Unknown class '{row['class']}' (cleaned to '{class_name}'). Using 'otherentities' class.")
        class_id = class_map['otherentities']
    
    return f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}"

def process_split(split):
    """Process train/valid/test split"""
    # Create directories
    img_dir = f"dataset/{split}"
    label_dir = f"dataset/{split}/labels"
    new_img_dir = f"dataset/{split}/images"
    
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(new_img_dir, exist_ok=True)
    
    # Read annotations
    df = pd.read_csv(f"dataset/{split}/_annotations.csv")
    
    # Group by filename to handle multiple objects per image
    for filename, group in df.groupby('filename'):
        # Create label file
        basename = os.path.splitext(filename)[0]
        label_file = os.path.join(label_dir, f"{basename}.txt")
        
        # Convert and save annotations
        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                yolo_bbox = convert_bbox(row)
                f.write(f"{yolo_bbox}\n")
        
        # Move image to images directory
        src_img = os.path.join(img_dir, filename)
        dst_img = os.path.join(new_img_dir, filename)
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)

def main():
    # Process each split
    for split in ['train', 'valid', 'test']:
        print(f"Processing {split} split...")
        process_split(split)
        print(f"Finished processing {split} split")

if __name__ == "__main__":
    main()