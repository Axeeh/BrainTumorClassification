# Brain Tumor Classification & Segmentation

### Deep Learning Pipeline – README

## 📌 Overview

This project implements a complete pipeline for **brain tumor image analysis**, including:

* **Image classification**
* **Semantic segmentation**
* **COCO annotation processing**
* **Custom visualization and preprocessing tools**

The notebook leverages a labeled dataset from Kaggle and applies modern deep-learning frameworks such as **TensorFlow/Keras** and **scikit-learn**, along with classical image-processing techniques.

---

## 📂 Dataset

* **Source:** [Kaggle](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation/data) – *Brain Tumor Image Dataset (Semantic Segmentation)*
* **Annotation type:** COCO JSON (`_annotations.coco.json`)
* Includes bounding boxes and category labels for tumor regions.

Example categories as extracted from the annotation file:

* `0` – Non tumor
* `1`, `2` – Tumor Categories (as defined by the author)

---

## 📦 Project Structure

The notebook uses the following components:

### **Imports**

* **General-purpose:** `numpy`, `pandas`, `json`, `random`, `matplotlib`, `seaborn`
* **Image processing:** `skimage`, `io`
* **DL frameworks:** `tensorflow`, `keras`
* **ML evaluation:** `accuracy_score`, `precision_score`, `f1_score`, `confusion_matrix`, etc.
* **Custom utilities** from `functions.py`:

  * `display_images_by_category`
  * `display_images_with_coco_annotations`
  * `visualize_annotation_mask`
  * `create_mask`
  * `extract_patches`
  * `segment_full_image`

### **Reproducibility**

All randomness is controlled via:

```python
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
```

---

## 🛠️ Notebook Workflow

### **1. Dataset Retrieval**

The dataset is automatically downloaded using `kagglehub`.
The script then loads COCO annotations:

```python
with open('Dataset/train/_annotations.coco.json', 'r') as file:
    annotations = json.load(file)
```

### **2. Annotation Exploration**

The notebook prints and inspects:

* Dataset metadata (version, licenses)
* Full image list
* Category definitions
* Bounding box structures

This is essential for understanding the segmentation task.

### **3. Custom Image Processing Tools**

The custom functions allow:

* Visualizing images grouped by tumor category
* Displaying COCO bounding boxes
* Creating segmentation masks
* Extracting image patches for model training
* Segmenting full-size MRI images

### **4. Model Training & Evaluation**

The notebook uses both ML and DL models:

* Logistic Regression (baseline classifier)
* CNN or UNet-like architectures (segmentation models)
* Evaluation through:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion matrix
  * Classification reports

---

## 📊 Visualizations

The notebook includes plots for:

* Samples from each tumor class
* COCO annotation overlays
* Generated segmentation masks
* Training curves (loss/accuracy)
* Model performance metrics

---

## 🚀 Goals of the Project

This notebook aims to explore and compare approaches for:

* **Detecting** brain tumor regions
* **Classifying** tumor types
* **Segmenting** tumor shapes and boundaries
* Understanding the dataset’s structure and annotations
* Creating a reusable training/visualization pipeline

---

## 👥 Authors

* [**Alessio Carnevale**](https://github.com/Axeeh)
* [**Manuel Cattoni**](https://github.com/ManuCa93)
* [**Carlo Schillaci**](https://github.com/CarloSchillaci)

---

## 📎 Notes

* This project requires a GPU-enabled environment for efficient training.
* Ensure the `functions.py` file is included in the working directory.
* COCO annotations must be kept in the correct relative folder structure (`Dataset/`).