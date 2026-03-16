
# PhD Research Code Repository

This repository contains the **complete collection of code, experiments, dataset references, and research artifacts** developed during my PhD.
The work focuses on **few-shot learning**, **medical image analysis**, and **adaptive sampling strategies** to improve generalization, robustness, and interpretability in low-data regimes.

This repository is structured to ensure **reproducibility**, **clarity**, and **ease of navigation** for reviewers, collaborators, and future researchers.

## 📂 Repository Structure

```
Phd_Work_Codes-main/
│
└── Phd_Codes-main/
    │
    ├── Affinity Sampling/
    │   ├── codes/
    │   │   └── CE_Affinity(CONV4).ipynb
    │   └── datasets/
    │       └── dataset.txt
    │
    ├── Class-Wise PNet/
    │   ├── codes/
    │   │   └── Class_Wise FS PNet.ipynb
    │   ├── datasets/
    │   │   └── dataset.txt
    │   └── papers/
    │       └── ISBI_Satellite_Workshop_2024.pdf
    │
    ├── DLAS/
    │   ├── codes/
    │   │   ├── CE_95_ACS+AIS_CONV4.ipynb
    │   │   ├── CE_95_Class_Confidence(CONV4).ipynb
    │   │   └── CE_95_Instance(CONV4).ipynb
    │   ├── datasets/
    │   │   └── dataset.txt
    │   └── papers/
    │       └── Dual-Level_Adaptive_Sampling_for_Enhanced_Few-Shot_Medical_Image_Classification.pdf
    │
    ├── IPNet/
    │   ├── codes/
    │   │   ├── IPNet.py
    │   │   ├── PNet.ipynb
    │   │   └── RRPNet.py
    │   ├── datasets/
    │   │   └── Datasets.txt
    │   └── papers/
    │
    ├── PANet/
    │   ├── codes/
    │   │   └── PANet.py
    │   ├── datasets/
    │   │   └── dataset.txt
    │   └── papers/
    │
    ├── MCS/
    │   ├── codes/
    │   │   └── CE_MetaData_Class_CONV4_2_2.ipynb
    │   └── datasets/
    │       └── dataset.txt
    │
    └── Refined Feature Selection PNet/
        ├── codes/
        │   └── FS(GAP+Others)_CE_PNET(CONV4_2_2).ipynb
        ├── datasets/
        │   └── dataset.txt
        └── papers/
```
        
## 🎯 Primary Objective of the Research

The primary objective of this research is to develop **efficient few-shot learning frameworks for medical image classification** by integrating **adaptive sampling strategies and prototype-based deep learning models**.

Medical imaging datasets often suffer from **limited labeled data**, making it challenging for traditional deep learning models to achieve high performance. This research addresses this challenge by designing methods that can learn effectively from **a small number of labeled samples per class**.

The main goals of the research include:

- Developing **prototype-based learning models** to represent each class using discriminative feature prototypes.
- Designing **adaptive sampling strategies** to select informative training samples in few-shot learning scenarios.
- Improving **feature representation and generalization** of deep learning models for medical image classification.
- Enhancing learning efficiency through **Dual-Level Adaptive Sampling (DLAS)** and affinity-based sampling methods.
- Evaluating the proposed methods on **publicly available medical imaging datasets** such as ChestMNIST, PathMNIST, BloodMNIST, and ODIR.

Overall, the research aims to **improve the robustness, efficiency, and accuracy of few-shot medical image classification models**, enabling reliable performance even when training data is limited.

---

## 📦 Module-wise Description

### 🔹 Module 1: Data Acquisition and Preprocessing

This module focuses on collecting and preparing medical imaging datasets for experimentation. Publicly available datasets such as **ChestMNIST, PathMNIST, BloodMNIST, and ODIR** are used.

Key steps include:

- Image normalization and resizing  
- Noise removal and preprocessing  
- Dataset organization for few-shot learning tasks  
- Splitting datasets into **training, validation, and testing sets**

These preprocessing steps ensure that the data is standardized and suitable for training deep learning models.

---

### 🔹 Module 2: Affinity-Based Sampling

This module implements **affinity-based sampling strategies** to select informative samples during training.

Main objectives:

- Compute similarity relationships between samples  
- Identify representative instances within each class  
- Select informative samples to improve training efficiency

This approach reduces redundant samples and improves learning performance in **few-shot learning environments**.

---

### 🔹 Module 3: Class-Wise Prototype Network (PNet)

This module focuses on **prototype-based learning** using Class-Wise Prototype Networks.

Key concepts include:

- Learning a **prototype representation** for each class  
- Using **distance-based similarity** for classification  
- Improving generalization when only a **few labeled samples** are available per class

Prototype networks help create effective decision boundaries in the feature space.

---

### 🔹 Module 4: Dual-Level Adaptive Sampling (DLAS)

The **DLAS module** introduces adaptive sampling strategies to improve training efficiency.

The framework includes:

- **Adaptive Class Sampling (ACS)** – prioritizes important classes during training  
- **Adaptive Instance Sampling (AIS)** – selects informative samples within each class  

Combining these techniques improves **sample diversity, learning efficiency, and model generalization**.

---

### 🔹 Module 5: Prototype-Based Learning Architectures

This module includes implementations of advanced prototype learning models.

**IPNet (Influential Prototype Network)**

- Learns robust class prototypes using instance-level information  
- Reduces the influence of noisy samples

**PANet (Prototype Alignment Network)**

- Improves alignment between **support and query samples**  
- Enhances feature representation for classification tasks

These models improve **classification performance in few-shot learning scenarios**.

---

### 🔹 Module 6: Metadata-Guided Class Sampling (MCS)

This module explores the use of **metadata information** to guide sample selection during training.

Key ideas:

- Use dataset-level and class-level statistics  
- Improve the sampling strategy using metadata insights  
- Enhance learning stability in limited-data scenarios

This approach helps models focus on **informative and challenging samples**.

## 📂 Datasets Used

The experiments in this research are conducted on publicly available **medical imaging datasets** designed for benchmarking machine learning models in healthcare.

- **ChestMNIST**
  - A large-scale chest X-ray dataset derived from the NIH ChestX-ray dataset.
  - Contains multiple thoracic disease labels for multi-label classification tasks.

- **PathMNIST**
  - A pathology image dataset consisting of histopathological images.
  - Used for multi-class classification of different tissue types.

- **BloodMNIST**
  - A microscopic blood cell image dataset.
  - Designed for classification of different blood cell types.

- **ODIR (Ocular Disease Intelligent Recognition)**
  - A retinal fundus image dataset for ocular disease diagnosis.
  - Contains images representing various eye diseases.
- **DermaMNIST**
  - A dermatoscopic skin lesion dataset from the MedMNIST collection.
  - Contains images of different skin lesion categories for multi-class classification tasks.

- **OrganAMNIST**
  - A medical imaging dataset consisting of abdominal CT images.
  - Used for multi-class classification of different abdominal organ regions.

- **Derm7pt**
  - A dermoscopic image dataset developed for skin lesion analysis.
  - Includes seven diagnostic criteria used for melanoma detection.

- **SD-198**
  - A large-scale skin disease dataset containing clinical images of various dermatological conditions.
  - Designed for classification across a wide range of skin diseases.

- **ISIC-2018**
  - A dermoscopic skin lesion dataset from the International Skin Imaging Collaboration challenge.
  - Used for lesion classification and melanoma detection tasks.

- **ISIC-2019**
  - An extended dermoscopic image dataset from the ISIC challenge.
  - Contains multiple skin lesion categories for large-scale skin cancer classification.

These datasets are used to evaluate the effectiveness of **few-shot learning, adaptive sampling, and prototype-based learning approaches** in medical image classification.

---

## 🧰 Packages Used

The implementation is developed using **Python-based deep learning and data science libraries**.

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **OpenCV**
- **tqdm**

These packages support **deep learning model development, data preprocessing, visualization, and evaluation**.

---

## 📦 Packages Required

Install the required packages using **pip** before running the project.

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn opencv-python tqdm
```

Optional (for visualization and analysis):
```bash
seaborn
tensorboard
```

---

## ▶️ How to Run

1. Clone or extract the repository
2. Navigate to the desired module folder
3. Open the notebook inside `codes/`
4. Update dataset paths
5. Run cells sequentially

---

## 📄 Publications and Outputs

- Conference and workshop papers are included in the respective `papers/` folders
- Experimental notebooks support full reproducibility

---

## ⚠️ Notes

- Code is intended for **academic and research use only**
- Results may vary depending on dataset splits and random seeds
- Please cite appropriately if reusing the code

---

## 👩‍🎓 Author

**Ranjana**
PhD Researcher
Research Areas:
Medical Imaging · Few-Shot Learning · Adaptive Sampling · Explainable AI

---

## 📜 License

This repository is released for **non-commercial academic use only**.
Please contact the author for extended usage permissions.
