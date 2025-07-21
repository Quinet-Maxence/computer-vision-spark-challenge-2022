import os
import pandas as pd

# Paths
predictions_dir = "/mnt/aiongpfs/users/nsharma/yolo_project/runs/detect/predict6"
outpredictions_dir = "/mnt/aiongpfs/users/nsharma/yolo_project/"
images_dir = os.path.join(predictions_dir)  # Folder with prediction images
labels_dir = os.path.join(predictions_dir, "labels")  # Folder with prediction labels
output_csv_path = os.path.join(outpredictions_dir, "submission_50rpoch11l_0_85conf.csv")
class_stats_csv_path = os.path.join(outpredictions_dir, "class_stats_50epoch11l_0_85conf.csv")

# Classes mapping (with 'background' as default for empty files or missing labels)
classes = ['smart_1', 'cheops', 'lisa_pathfinder', 'debris', 'proba_3_csc',
           'proba_3_ocs', 'soho', 'earth_observation_sat_1', 'proba_2', 'xmm_newton', 'double_star']
background_class = "background"

# Add background class to the list for statistics
class_stats = {cls: 0 for cls in classes}
class_stats[background_class] = 0

# Function to convert YOLO format to [R_min, C_min, R_max, C_max]
def yolo_to_bbox(yolo_bbox, img_width, img_height):
    """
    Convert YOLO bounding box format to [R_min, C_min, R_max, C_max].

    Args:
    yolo_bbox (list): YOLO format [x_center, y_center, width, height]
    img_width (int): Image width
    img_height (int): Image height

    Returns:
    list: Bounding box in [R_min, C_min, R_max, C_max] format
    """
    x_center, y_center, width, height = yolo_bbox
    C_min = int((x_center - width / 2) * img_width)
    C_max = int((x_center + width / 2) * img_width)
    R_min = int((y_center - height / 2) * img_height)
    R_max = int((y_center + height / 2) * img_height)
    return [R_min, C_min, R_max, C_max]

# Process YOLO predictions
def convert_predictions_to_csv(images_dir, labels_dir, output_csv_path, classes, background_class):
    global class_stats
    rows = []
    for img_file in os.listdir(images_dir):
        if not img_file.endswith(".jpg"):
            continue

        label_file = os.path.join(labels_dir, img_file.replace(".jpg", ".txt"))
        img_name = img_file

        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                lines = f.readlines()
                if lines:  # If label file is not empty
                    for line in lines:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        yolo_bbox = list(map(float, parts[1:5]))

                        # Assume image dimensions are known (replace with actual dimensions if needed)
                        img_width, img_height = 1024, 1024  # Placeholder dimensions
                        bbox = yolo_to_bbox(yolo_bbox, img_width, img_height)

                        rows.append({
                            "filename": img_name.replace(".jpg",".png"),
                            "class": classes[class_id],
                            "bbox": bbox
                        })

                        # Update class statistics
                        class_stats[classes[class_id]] += 1
                else:  # Handle empty label file
                    rows.append({
                        "filename": img_name.replace(".jpg",".png"),
                        "class": " ",
                        "bbox": "[]"
                    })
                    class_stats[background_class] += 1
        else:  # Handle missing label file
            rows.append({
                "filename": img_name.replace(".jpg",".png"),
                "class": " ",
                "bbox": "[]"
            })
            class_stats[background_class] += 1

    # Create DataFrame and save as CSV
    df = pd.DataFrame(rows, columns=["filename", "class", "bbox"])
    df.to_csv(output_csv_path, index=False)
    print(f"Submission CSV saved at: {output_csv_path}")

# Save class statistics as CSV
def save_class_stats_csv(class_stats, output_path):
    stats_data = [{"class": cls, "count": count} for cls, count in class_stats.items()]
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(output_path, index=False)
    print(f"Class statistics CSV saved at: {output_path}")

# Run conversion and save statistics
convert_predictions_to_csv(images_dir, labels_dir, output_csv_path, classes, background_class)
save_class_stats_csv(class_stats, class_stats_csv_path)
