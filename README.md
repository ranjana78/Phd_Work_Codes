
# PhD Research Code Repository

This repository contains the **complete collection of code, experiments, dataset references, and research artifacts** developed during my PhD.
The work focuses on **few-shot learning**, **medical image analysis**, and **adaptive sampling strategies** to improve generalization, robustness, and interpretability in low-data regimes.

This repository is structured to ensure **reproducibility**, **clarity**, and **ease of navigation** for reviewers, collaborators, and future researchers.

---

📌 Research Scope and Objectives

The primary objectives of this PhD research are:

Designing adaptive and affinity-based sampling strategies (e.g., Adaptive Class Sampling and Adaptive Instance Sampling) for few-shot learning

Improving class-confidence and instance-level generalization under limited medical training data

Developing prototype-based learning frameworks, including class-wise prototype networks (PNet) for few-shot classification

Applying the proposed methods to medical imaging datasets such as ChestMNIST, PathMNIST, BloodMNIST, and ODIR

Enhancing interpretability, robustness, and reliability in AI models for medical image classification

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

## 📁 Module-wise Description

### 1️⃣ Affinity Sampling

**Objective:**
Improve few-shot learning performance by selecting **semantically similar and informative samples** based on affinity measures.

**Key Ideas:**
- Affinity-based sample selection
- Improved class separability
- Reduced intra-class variance

**Implementation:**
- Backbone: Conv4
- Loss: Cross-Entropy
- Notebook: `CE\_Affinity(CONV4).ipynb`

**Use Case:**
Few-shot classification under limited labeled data, especially in medical imaging.

---

### 2️⃣ Class-Wise Prototypical Network (Class-Wise PNet)

**Objective:**
Enhance standard Prototypical Networks by **learning class-wise discriminative prototypes**.

**Key Ideas:**
- Class-level representation refinement
- Better handling of inter-class similarity
- Stable prototype estimation

**Implementation:**
- Few-shot Prototypical Network
- Episodic training
- Notebook: `Class\_Wise FS PNet.ipynb`

**Associated Publication:**
- ISBI Satellite Workshop 2024

---

### 3️⃣ DLAS – Dual-Level Adaptive Sampling

**Objective:**
Jointly adapt sampling at **class level (ACS)** and **instance level (AIS)** to maximize learning efficiency.

**Key Components:**

#### 🔹 Adaptive Class Sampling (ACS)
- Dynamically prioritizes harder or underrepresented classes

#### 🔹 Adaptive Instance Sampling (AIS)
- Focuses training on informative/hard instances

**Implementation:**
- Backbone: Conv4
- Loss: Cross-Entropy
- Notebooks:
  - `CE\_95\_ACS+AIS\_CONV4.ipynb`
  - `CE\_95\_Class\_Confidence(CONV4).ipynb`
  - `CE\_95\_Instance(CONV4).ipynb`

**Contribution:**
Forms the **core contribution** of the PhD by integrating ACS and AIS into a unified framework.

---

## 📊 Datasets Used

Experiments are conducted on multiple **medical imaging datasets**, including:

- Derm7pt (Explainable dermatology)
- ISIC 2018 / ISIC 2019 (Skin lesion analysis)
- HAM10000
- BreakHis (BC-100X)
- ChestMNIST
- PathMNIST
- BloodMNIST
- Ocular Disease (ODIR-5K)

> Dataset download links and licenses are documented separately.

---

## 🧪 Experimental Setup

- Framework: **PyTorch**
- Training paradigm: Episodic few-shot learning
- Backbones: Conv4 (extendable to ResNet variants)
- Evaluation: Accuracy, confidence-based metrics

---

## 📦 Required Packages

The following Python packages are required to run the experiments:

```bash
python >= 3.8
torch
torchvision
numpy
scipy
scikit-learn
matplotlib
tqdm
opencv-python
pandas
jupyter
medmnist
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
