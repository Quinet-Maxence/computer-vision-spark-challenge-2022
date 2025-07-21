import os
import pandas as pd
import shutil
from PIL import Image

# Paths to your dataset and output directories
dataset_path = '/mnt/aiongpfs/users/nsharma/yolo_project/data'  
yolo_output_path = '/mnt/aiongpfs/users/nsharma/yolo_project/yolo_dataset'  

# Paths to CSV files
labels_path = os.path.join(dataset_path, "labels")
train_csv_path = os.path.join(labels_path, "train.csv")
val_csv_path = os.path.join(labels_path, "val.csv")


test_images_path = os.path.join(dataset_path, "test/test")
train_images_path = os.path.join(dataset_path, "train/train")
val_images_path = os.path.join(dataset_path, "val/val")


# Classes and class-to-ID mapping
classes = ['smart_1', 'cheops', 'lisa_pathfinder', 'debris', 'proba_3_csc', 
           'proba_3_ocs', 'soho', 'earth_observation_sat_1', 'proba_2', 'xmm_newton', 'double_star']
class_to_id = {cls: idx for idx, cls in enumerate(classes)}

# Convert bounding box to YOLO format
def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    R_min, C_min, R_max, C_max = bbox
    x_center = (C_min + C_max) / 2 / img_width
    y_center = (R_min + R_max) / 2 / img_height
    width = (C_max - C_min) / img_width
    height = (R_max - R_min) / img_height
    return [x_center, y_center, width, height]

# The function convert_bbox_to_yolo_format takes bounding box coordinates in the format [R_min, C_min, R_max, C_max] 
# and converts them to the YOLO format [x_center, y_center, width, height].

# Explanation of Parameters:
# Input Format:

# R_min, C_min, R_max, C_max:
# R_min = Top boundary (y-coordinate or ymin).
# C_min = Left boundary (x-coordinate or xmin).
# R_max = Bottom boundary (y-coordinate or ymax).
# C_max = Right boundary (x-coordinate or xmax).
# Output Format (YOLO):

# [x_center, y_center, width, height]:
# x_center: Horizontal center of the box as a fraction of image width.
# y_center: Vertical center of the box as a fraction of image height.
# width: Box width as a fraction of image width.
# height: Box height as a fraction of image height.
# Conversion Steps:
# Calculate Center Coordinates:

# x_center = (xmin + xmax) / 2 / img_width:

# Finds the horizontal midpoint (xmin + xmax) / 2.
# Normalizes it by dividing by img_width.
# y_center = (ymin + ymax) / 2 / img_height:

# Finds the vertical midpoint (ymin + ymax) / 2.
# Normalizes it by dividing by img_height.
# Calculate Width and Height:

# width = (xmax - xmin) / img_width:

# Finds the horizontal size of the bounding box (xmax - xmin).
# Normalizes it by dividing by img_width.
# height = (ymax - ymin) / img_height:

# Finds the vertical size of the bounding box (ymax - ymin).
# Normalizes it by dividing by img_height.


# Process DataFrame and save YOLO annotations
def save_yolo_annotations_and_images(df, split, images_path):
    labels_dir = os.path.join(yolo_output_path, split, "labels")
    images_dir = os.path.join(yolo_output_path, split, "images")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Initialize class statistics
    class_stats = {cls: 0 for cls in classes}

    for _, row in df.iterrows():
        img_name = row['filename'].replace(".png", ".jpg")  # Convert .png to .jpg
        class_name = row['class']
        class_id = class_to_id.get(class_name)
        if class_id is None:
            print(f"Invalid class name: {class_name} in {img_name}")
            continue

        try:
            bbox = eval(row['bbox'])
        except Exception as e:
            print(f"Invalid bbox for {img_name}: {e}")
            continue

        image_path = os.path.join(images_path, img_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Get image dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # Convert bbox to YOLO format
        yolo_bbox = convert_bbox_to_yolo_format(bbox, img_width, img_height)

        # Save label
        label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            f.write(f"{class_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")

        # Copy image
        dest_img_path = os.path.join(images_dir, img_name)
        shutil.copy(image_path, dest_img_path)

        # Update class statistics
        class_stats[class_name] += 1

    # Print class statistics
    print(f"Class statistics for {split}:")
    for cls, count in class_stats.items():
        print(f"Class '{cls}': {count} images")

# Process train and validation datasets
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)
save_yolo_annotations_and_images(train_df, "train", train_images_path)
save_yolo_annotations_and_images(val_df, "val", val_images_path)

# Copy test images
test_images_dir = os.path.join(yolo_output_path, "test", "images")
os.makedirs(test_images_dir, exist_ok=True)
for img_file in os.listdir(test_images_path):
    if img_file.endswith(".jpg"):
        shutil.copy(os.path.join(test_images_path, img_file), os.path.join(test_images_dir, img_file))

# Create dataset.yaml file
yaml_content = f"""
train: {os.path.join(yolo_output_path, 'train/images')}
val: {os.path.join(yolo_output_path, 'val/images')}
test: {os.path.join(yolo_output_path, 'test/images')}

nc: {len(classes)}
names: {classes}
"""
yaml_path = os.path.join(yolo_output_path, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"YOLO dataset created at: {yolo_output_path}")












