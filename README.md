
# PhD Research Code Repository

This repository contains the **complete collection of code, experiments, dataset references, and research artifacts** developed during my PhD.
The work focuses on **few-shot learning**, **medical image analysis**, and **adaptive sampling strategies** to improve generalization, robustness, and interpretability in low-data regimes.

This repository is structured to ensure **reproducibility**, **clarity**, and **ease of navigation** for reviewers, collaborators, and future researchers.

---

📌 Research Scope and Objectives

-The primary objectives of this PhD research are:

-Designing adaptive and affinity-based sampling strategies (e.g., Adaptive Class Sampling and Adaptive Instance Sampling) for few-shot learning

-Improving class-confidence and instance-level generalization under limited medical training data

-Developing prototype-based learning frameworks, including class-wise prototype networks (PNet) for few-shot classification

-Applying the proposed methods to medical imaging datasets such as ChestMNIST, PathMNIST, BloodMNIST, and ODIR

-Enhancing interpretability, robustness, and reliability in AI models for medical image classification

---

## 📂 Repository Structure (Detailed)

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

---

## 📦 Module-wise Description

### 🔹 Module 1: Data Acquisition and Preprocessing
This module focuses on collecting and preparing medical imaging datasets for experimentation. Publicly available datasets such as **ChestMNIST, PathMNIST, BloodMNIST, and ODIR** are used. The preprocessing stage includes image normalization, resizing, noise removal, and dataset partitioning into training, validation, and testing sets suitable for few-shot learning scenarios.

---

### 🔹 Module 2: Adaptive Sampling Strategy Development
This module develops **adaptive and affinity-based sampling strategies** to efficiently select informative samples during training. Techniques such as **Adaptive Class Sampling (ACS)** and **Adaptive Instance Sampling (AIS)** are implemented to dynamically prioritize important classes and instances, improving learning performance in few-shot environments.

---

### 🔹 Module 3: Class-Confidence and Instance-Level Learning
This module focuses on improving **class-level and instance-level generalization** by introducing **class-confidence–based sampling mechanisms**. The system evaluates prediction confidence to guide sample selection and balance class representation, helping the model learn more effectively from limited training data.

---

### 🔹 Module 4: Prototype-Based Learning Framework
This module implements **prototype-based learning models**, particularly **Class-Wise Prototype Networks (PNet)**. The framework learns representative prototypes for each class in the feature space and uses **distance-based classification** to improve decision boundaries and support few-shot classification tasks.

---

### 🔹 Module 5: DLAS-Based Feature Learning and Optimization
This module integrates **Distance Learning with Adaptive Sampling (DLAS)** methods to enhance feature representation and improve sample selection during training. The combination of affinity-based sampling and prototype learning helps achieve better feature discrimination and model stability.

---

### 🔹 Module 6: Evaluation and Performance Analysis
This module evaluates the proposed methods using metrics such as **Accuracy, Precision, Recall, F1-score, and Confusion Matrix**. Comparative analysis with baseline models is conducted to assess improvements in **generalization, robustness, and interpretability** for medical image classification tasks.

## ⚙️ Implementation

The implementation of this research is organized into multiple stages to develop and evaluate adaptive sampling and prototype-based few-shot learning methods for medical image classification.

1. **Dataset Preparation**
   - Medical imaging datasets including **ChestMNIST, PathMNIST, BloodMNIST, and ODIR** are collected.
   - Images are preprocessed through normalization, resizing, and dataset splitting into **training, validation, and testing sets**.
   - Few-shot settings are created by limiting the number of labeled samples per class.

2. **Feature Extraction**
   - Deep learning models are used to extract discriminative features from medical images.
   - Feature embeddings are generated to represent each sample in a latent feature space.

3. **Adaptive Sampling Mechanisms**
   - **Adaptive Class Sampling (ACS)** dynamically selects informative classes during training.
   - **Adaptive Instance Sampling (AIS)** identifies representative instances within each class.
   - These strategies improve the efficiency of learning under limited data conditions.

4. **Prototype-Based Learning**
   - A **Class-Wise Prototype Network (PNet)** is implemented to learn representative prototypes for each class.
   - Distance-based classification is used to assign labels based on similarity to class prototypes.

5. **DLAS Integration**
   - **Distance Learning with Adaptive Sampling (DLAS)** combines affinity-based sampling and prototype learning.
   - This integration enhances feature discrimination and improves training stability.

6. **Model Evaluation**
   - Performance is evaluated using **Accuracy, Precision, Recall, F1-score, and Confusion Matrix**.
   - Comparative experiments are conducted against baseline models to demonstrate improvements in generalization and robustness.

---

## ⭐ Research Contributions

The major contributions of this research include:

- Development of **adaptive and affinity-based sampling strategies** to improve training efficiency in few-shot learning environments.

- Introduction of **class-confidence and instance-level sampling mechanisms** to enhance model generalization under limited training data.

- Design of a **prototype-based learning framework using Class-Wise Prototype Networks (PNet)** for improved representation learning and classification.

- Integration of **Distance Learning with Adaptive Sampling (DLAS)** to optimize sample selection and feature learning.

- Comprehensive evaluation of the proposed methods on **medical imaging datasets such as ChestMNIST, PathMNIST, BloodMNIST, and ODIR**.

- Demonstration of improved **robustness, interpretability, and performance** in AI models for medical image classification.

---

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
