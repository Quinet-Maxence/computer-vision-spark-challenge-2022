<h1 align="center">
# 🚀 Computer Vision — SPARK Challenge 2022 (3rd Place) 🏆
</h1>

![GitHub Repo Stars](https://img.shields.io/github/stars/Quinet-Maxence/computer-vision-spark-challenge-2022?style=social)  
![GitHub Repo Forks](https://img.shields.io/github/forks/Quinet-Maxence/computer-vision-spark-challenge-2022?style=social)  
![GitHub Last Commit](https://img.shields.io/github/last-commit/Quinet-Maxence/computer-vision-spark-challenge-2022)  

📌 **This repository contains my 3rd place solution for the [SPARK 2022 Challenge](https://cvi2.uni.lu/spark2022/).**  
The challenge focuses on **spacecraft detection and pose estimation** using advanced **Computer Vision and Deep Learning** techniques for **Space Situational Awareness (SSA)**.

---

## 📚 Table of Contents  
- [🚀 Overview](#-overview)  
- [🛠 Dataset](#-dataset)  
- [⚙ Approach](#-approach)  
- [📂 Project Structure](#-project-structure)
- [🏆 Results](#-results)  
- [📜 Credits](#-credits)  

---

## 🚀 Overview  
The **SPARK Challenge 2022** is a large-scale Computer Vision competition focusing on **Spacecraft Recognition Leveraging Knowledge of the Space Environment**.  
This project demonstrates:  
- **Stream-1**: Spacecraft Detection (bounding box + classification for 11 categories, including debris).  

---

## 🛠 Dataset  
- **Stream-1**:  
  - 11 categories (10 satellites + debris).  
  - **10,000 images per class** (110,000 images total).  

Dataset details: [SPARK 2022 Dataset](https://cvi2.uni.lu/spark-2022-dataset/).

---

## ⚙ Approach  
The solution leverages:  
- **YOLOv11 & CNN-based models** for object detection.  
- **Pose estimation pipelines** based on temporal data and 3D geometry.  
- Training performed on the **HPC (High-Performance Computing)**.

---

## 📂 Project Structure  

This repository contains the following main Python files:  

1. **`dataset_conv.py`** – Converts the raw dataset into **YOLO format**.  
2. **`main11l.py`** – Code for **training the detection model**.  
3. **`predict.py`** – Script to **run inference using the trained model**.  
4. **`visualise_yolo_dataset.py`** – Visualizes training and validation data (bounding boxes & class labels) to verify dataset conversion.  
5. **`yolo_to_csv_new.py`** – Converts YOLO detection outputs into a **submission.csv** file for leaderboard evaluation.  

---

## 🏆 Results  
- **Final Rank:** 3rd place 🥉 among all submissions.  
- Achieved **high detection accuracy** across spacecraft categories.  
- Bonus points awarded for **pose estimation pipeline**.

## 🏆 Results  

<h3 align="center">1. Confusion Matrix</h3>  
<p align="center">
  <img src="results_imgs/confusion_matrix.png" width="25%">
  <img src="results_imgs/confusion_matrix_normalized.png" width="25%">
</p>

<h3 align="center">2. F1-Confidence Curve</h3>  
<p align="center">
  <img src="results_imgs/F1_curve.png" width="25%">
</p>

<h3 align="center">3. Labels & Data Distribution</h3>  
<p align="center">
  <img src="results_imgs/labels.jpg" width="25%">
  <img src="results_imgs/labels_correlogram.jpg" width="25%">
</p>

<h3 align="center">4. Precision, Recall & PR Curve</h3>  
<p align="center">
  <img src="results_imgs/P_curve.png" width="25%">
  <img src="results_imgs/R_curve.png" width="25%">
  <img src="results_imgs/PR_curve.png" width="25%">
</p>

<h3 align="center">5. Training & Validation Samples</h3>  
<p align="center">
  <img src="results_imgs/train_batch0.jpg" width="25%">
  <img src="results_imgs/train_batch1.jpg" width="25%">
  <img src="results_imgs/train_batch2.jpg" width="25%">
  <img src="results_imgs/train_batch165000.jpg" width="25%">
  <img src="results_imgs/train_batch165001.jpg" width="25%">
  <img src="results_imgs/train_batch165002.jpg" width="25%">
</p>

<h3 align="center">6. Validation Predictions</h3>  
<p align="center">
  <img src="results_imgs/val_batch0_labels.jpg" width="25%">
  <img src="results_imgs/val_batch0_pred.jpg" width="25%">
  <img src="results_imgs/val_batch1_labels.jpg" width="25%">
  <img src="results_imgs/val_batch1_pred.jpg" width="25%">
  <img src="results_imgs/val_batch2_labels.jpg" width="25%">
  <img src="results_imgs/val_batch2_pred.jpg" width="25%">
</p>

<h3 align="center">Bonus : Loss Results Values</h3>  
<p align="center">
  <img src="results_imgs/results.png" width="45%">
</p>



## 📜 Credits  
- **Author**: Maxence QUINET  
- **Challenge Organizers**: [Interdisciplinary Centre for Security, Reliability and Trust (SnT)](https://www.uni.lu/snt/)  
- **Official Challenge Page**: [SPARK 2022](https://cvi2.uni.lu/spark2022/)  
