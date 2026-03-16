
# PhD Research Code Repository

This repository contains the **complete collection of code, experiments, dataset references, and research artifacts** developed during my PhD.
The work focuses on **few-shot learning**, **medical image analysis**, and **adaptive sampling strategies** to improve generalization, robustness, and interpretability in low-data regimes.

This repository is structured to ensure **reproducibility**, **clarity**, and **ease of navigation** for reviewers, collaborators, and future researchers.

## 🎯 Research Overview

My research focuses on improving **few-shot learning for medical image classification** by enhancing **Prototypical Networks within a meta-learning framework**. The work addresses the critical challenge of **data scarcity in clinical AI**, where medical datasets often contain very limited labeled samples.

Specifically, the research aims to design more **robust and discriminative prototype-based learning mechanisms** by improving how prototypes are constructed, how features are represented and refined, and how episodic training tasks are sampled.

---

## ⭐ Key Research Contributions

### 1️⃣ Prototype Construction Improvement
Developing **influence-based reweighting strategies** so that more informative support samples contribute more strongly to class prototypes, improving the robustness of prototype representations.

### 2️⃣ Feature Refinement
Selecting and emphasizing the most **discriminative feature maps and channels** to improve prototype quality and enhance classification performance.

### 3️⃣ Domain-Specific Feature Integration
Incorporating **texture-aware representations** such as **wavelet-based features** to capture subtle morphological patterns commonly found in medical images.

### 4️⃣ Optimized Episodic Sampling
Designing **adaptive and metadata-guided sampling strategies** to construct more informative and balanced few-shot learning tasks during meta-training.

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
        


---

## 🚀 Research Goal

Overall, this research aims to develop **efficient, robust, and interpretable few-shot learning frameworks** capable of accurately classifying medical images even when **only a few labeled samples are available**.  

By addressing the data scarcity problem in medical imaging, the proposed approaches aim to improve the **reliability and applicability of AI systems in real-world healthcare environments**.

Overall, the research aims to **improve the robustness, efficiency, and accuracy of few-shot medical image classification models**, enabling reliable performance even when training data is limited.

---
## 📦 Module-wise Description

### 🔹 Module 1: Influential Prototypical Networks (IPNet)

This module implements **Influential Prototypical Networks (IPNet)**, which improve prototype construction in few-shot learning.

Traditional Prototypical Networks compute class prototypes using a simple average of support samples, which may be sensitive to noisy or non-representative instances.

IPNet addresses this limitation by:

- Assigning **importance weights to support samples**
- Identifying **influential samples based on statistical alignment**
- Reducing the impact of **outliers and noisy data**

This results in **more reliable class prototypes and improved classification performance in few-shot medical image datasets**.

---

### 🔹 Module 2: Class-Wise Feature Map Selection Prototypical Network

This module introduces **Class-Wise Feature Map Selection** to improve prototype quality.

Standard PNet models treat all feature channels equally when constructing prototypes. However, many channels may contain redundant or less informative information.

This module improves representation by:

- Ranking feature maps using **Global Average Pooling (GAP)**
- Selecting **top-K discriminative feature channels**
- Masking less relevant channels before prototype aggregation

This selective feature strategy enhances **prototype discriminability and classification accuracy**.

---

### 🔹 Module 3: Refined Feature Selection Prototypical Network

This module extends the previous feature selection approach by introducing **multiple feature selection strategies**.

Instead of relying on a single selection criterion, it combines several statistical measures to identify informative feature maps.

The approach includes:

- **Global Average Pooling (GAP)**
- **Mixed Pooling**
- **Variance-based feature selection**

By combining these strategies, the model captures feature importance from different perspectives, resulting in **more robust prototype representations**.

---

### 🔹 Module 4: Prototypical Aggregate Network (PANet)

This module introduces **PANet**, a domain-aware framework designed for medical image classification.

PANet integrates **spatial and textural information** to improve feature representation.

Key components include:

- **Discrete Wavelet Transform (DWT)** to extract texture information  
- Integration of **frequency-domain features with spatial features**
- **Mean Feature Aggregation Module (MFAM)** to combine multi-level features

This architecture preserves **fine-grained morphological details** such as edges and textures that are critical for medical image analysis.

---

### 🔹 Module 5: Dual-Level Adaptive Sampling (DLAS)

This module proposes **Dual-Level Adaptive Sampling (DLAS)** to improve episodic training in few-shot learning.

Standard meta-learning frameworks rely on random sampling of classes and instances, which may produce uninformative training tasks.

DLAS introduces two adaptive strategies:

**Adaptive Class Sampling (ACS)**  
- Prioritizes **difficult or underperforming classes**

**Adaptive Instance Sampling (AIS)**  
- Selects **informative or uncertain samples** within each class

By combining these two mechanisms, DLAS creates **more informative training episodes and improves model generalization**.

---

### 🔹 Module 6: Affinity-Guided Adaptive Sampling

This module introduces an **Affinity-Based Sampling Strategy** for constructing better training tasks.

The method first analyzes **relationships between samples and classes** using similarity measures.

Key ideas include:

- Construction of a **Sample Affinity Matrix (SAM)**
- Estimation of **Class Affinity Scores (CAS)**
- Sampling classes based on **similarity relationships**

Unlike conventional sampling approaches, this method **first analyzes instance relationships and then derives class selection**, leading to more structured episodic tasks.

---

### 🔹 Module 7: Metadata-Guided Class Sampling (MCS)

This module explores **metadata-driven task sampling** to further improve few-shot learning.

Instead of relying only on learned embeddings, this approach uses **dataset-level meta-information**.

The framework:

- Extracts **class-level statistics from dataset attributes**
- Uses a **Random Forest classifier** to estimate class difficulty
- Prioritizes **challenging or underperforming classes** when constructing episodes

This metadata-guided strategy helps the model focus on **informative and difficult classes**, improving training efficiency and classification performance.
.

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
