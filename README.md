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

## 🏆 Results  
- **Final Rank:** 3rd place 🥉 among all submissions.  
- Achieved **high detection accuracy** across spacecraft categories.  
- Bonus points awarded for **pose estimation pipeline**.

---

## 📜 Credits  
- **Author**: Maxence QUINET  
- **Challenge Organizers**: [Interdisciplinary Centre for Security, Reliability and Trust (SnT)](https://www.uni.lu/snt/)  
- **Official Challenge Page**: [SPARK 2022](https://cvi2.uni.lu/spark2022/)  
