# 🛡️ Data-Driven Predictive Modeling: Gun Detection Using CNN Classification

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.13-red.svg)](https://keras.io/)
[![Colab](https://img.shields.io/badge/Google%20Colab-TPU%20Runtime-yellow.svg)](https://colab.research.google.com/)

---

## 📌 1. Project Overview

This project implements a Convolutional Neural Network (CNN) pipeline inside a [Google Colab Notebook](https://colab.research.google.com/drive/1wCoCiBpeDUUMGDDk18ACDgUFeNU1vy-8#scrollTo=GrVXRri_nXaB) to detect firearms from image assets fetched dynamically via `kagglehub` from the [Gun Detection Dataset on Kaggle](https://www.kaggle.com/datasets/atulyakumar98/gundetection).

---

## ⚠️ 2. Problem Statement & Instructor Feedback on Dataset Structure

### The Core Problem Reported by the Instructor
The instructor evaluated the initial data setup and pointed out that **the dataset was not structured in the proper way** for standard categorical classification:
* **Raw Annotation Mismatch:** The raw files combined `.jpg` imagery with parallel `.txt` annotation files containing YOLO-style bounding box coordinates (e.g., `1 0.3887 0.4867 0.2347 0.1408`), rather than cleanly separated subfolders for each target class.
* **Positive-Only Annotation Constraint:** The dataset inherently provides annotation text files exclusively for positive instances (images containing guns). Background images lacking firearms often lack a corresponding text file or contain empty annotations, creating an unstructured, asymmetric format.

### The Fix & Solution Implemented
To satisfy the instructor's structural requirements and process the dataset accurately:
1. **Custom Parsing Layer (`get_label_from_txt`):** Built a dedicated helper function that checks if a matching `.txt` file exists and contains the gun class token (`1` or `gun`).
2. **Asymmetric Handling:** If the corresponding text file exists with valid coordinate data, it assigns a positive label (`1.0`). If the text file is missing or empty (representing negative instances), it safely defaults to a negative class label (`0.0`).
3. **High-Performance `tf.data.Dataset` Pipeline:** Bypassed traditional folder restrictions by zipping image file paths and parsed text labels into an optimized `tf.data.Dataset` pipeline with batching (`BATCH_SIZE = 32`) and `.prefetch(tf.data.AUTOTUNE)`.

---

## 📊 3. Dataset Partitioning & Distribution

The dataset of 3,000 total images was split into reproducible stratified partitions using a fixed random seed (`random.seed(123)`):
* 🟢 **Train Data:** 2,100 images (70%)
* 🔵 **Validation Data:** 450 images (15%)
* 🔴 **Test Data:** 450 images (15%)

---

## 🛠️ 4. Technical Stack & Dependencies

* **Language:** Python 3.13
* **Deep Learning Framework:** TensorFlow / Keras 3
* **Data Source:** [Kaggle Gun Detection Dataset](https://www.kaggle.com/datasets/atulyakumar98/gundetection)
* **Environment:** Google Colab (Google Compute Engine TPU/GPU Backend)

---

## 🚀 5. Getting Started & Execution

1. Open the [Google Colab Notebook](https://colab.research.google.com/drive/1wCoCiBpeDUUMGDDk18ACDgUFeNU1vy-8#scrollTo=GrVXRri_nXaB).
2. Run the automated data downloading and mapping blocks sequentially to parse the text annotations and build the training/validation tensors.
