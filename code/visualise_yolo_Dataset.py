import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Paths
yolo_dataset_path = "/mnt/aiongpfs/users/nsharma/yolo_project/yolo_dataset"
output_visualizations_path = "/mnt/aiongpfs/users/nsharma/yolo_project/visualized_yolo_dataset"
os.makedirs(output_visualizations_path, exist_ok=True)

# Classes
classes = ['smart_1', 'cheops', 'lisa_pathfinder', 'debris', 'proba_3_csc', 
           'proba_3_ocs', 'soho', 'earth_observation_sat_1', 'proba_2', 'xmm_newton', 'double_star']

# Function to visualize YOLO dataset and calculate statistics
def visualize_yolo_dataset(split):
    class_stats = {cls: {"count": 0, "images": []} for cls in classes}

    images_path = os.path.join(yolo_dataset_path, split, "images")
    labels_path = os.path.join(yolo_dataset_path, split, "labels")
    output_images_dir = os.path.join(output_visualizations_path, split, "images")
    output_stats_dir = os.path.join(output_visualizations_path, "class_statistics")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_stats_dir, exist_ok=True)

    for img_file in os.listdir(images_path):
        label_file = os.path.join(labels_path, img_file.replace(".jpg", ".txt"))
        img_path = os.path.join(images_path, img_file)
        if not os.path.exists(label_file):
            print(f"Label file not found for {img_file}")
            continue

        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        with open(label_file) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:]))
                x_center, y_center, width, height = bbox
                C_min = int((x_center - width / 2) * img_width)
                C_max = int((x_center + width / 2) * img_width)
                R_min = int((y_center - height / 2) * img_height)
                R_max = int((y_center + height / 2) * img_height)
                class_name = classes[class_id]

                # Draw bounding box
                cv2.rectangle(img, (C_min, R_min), (C_max, R_max), (255, 0, 0), 2)
                cv2.putText(img, class_name, (C_min, R_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Update class statistics
                class_stats[class_name]["count"] += 1
                if img_file not in class_stats[class_name]["images"]:
                    class_stats[class_name]["images"].append(img_file)

        output_img_path = os.path.join(output_images_dir, img_file)
        plt.imsave(output_img_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Save class statistics
    csv_data = [
        {"class": cls, "count": stats["count"], "images": ", ".join(stats["images"])}
        for cls, stats in class_stats.items()
    ]
    stats_df = pd.DataFrame(csv_data)
    stats_df.to_csv(os.path.join(output_stats_dir, f"{split}_class_stats.csv"), index=False)

# Visualize train and val splits
for split in ["train", "val"]:
    visualize_yolo_dataset(split)

print(f"Visualization and class statistics saved in: {output_visualizations_path}")
