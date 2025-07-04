# Heart Disease Risk Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)

This repository contains the code and resources for the project **"Heart Disease Risk Prediction Using Machine Learning"**, developed as part of the INFO 531: Data Warehousing and Analytics in the Cloud course.

---

## 📖 Project Overview

Cardiovascular diseases are among the leading causes of death globally. Early identification of at-risk individuals can significantly reduce morbidity and mortality through timely intervention. This project develops a machine learning model to predict the risk of heart disease using clinical attributes and an advanced **QuantumPatternFeature**.

The model predicts whether a person is likely to have heart disease (binary classification).

---

## 📊 Dataset

* **Source**: Kaggle - [Heart Prediction Quantum Dataset](https://www.kaggle.com/datasets/shantanugarg274/heart-prediction-dataset-quantum/data)
* **Records**: 500 patient records
* **Features**:

  * `Age`: Patient age (years)
  * `Gender`: 0 = Female, 1 = Male
  * `BloodPressure`: Resting blood pressure (mm Hg)
  * `Cholesterol`: Serum cholesterol (mg/dl)
  * `HeartRate`: Heart rate (bpm)
  * `QuantumPatternFeature`: Synthetic feature inspired by quantum pattern representation
  * `HeartDisease`: Target variable (0 = No, 1 = Yes)

---

## 🛠 Tools and Libraries

* Google Colab/ Python Jupyter Notebook
* Python (Pandas, NumPy)
* Visualization: Matplotlib, Seaborn
* Machine Learning: Scikit-learn

---

## 🧑‍💻 Project Workflow

1. **Data Preparation**

   * Cleaned column names and handled missing values.
   * Detected and removed outliers using IQR.
   * Encoded categorical variables (Gender).
   * Normalized numerical features using `StandardScaler`.
2. **Exploratory Data Analysis (EDA)**

   * Visualized class distribution, feature distributions, and correlations.
3. **Model Development**

   * Trained **Logistic Regression** and **Random Forest** classifiers.
   * Hyperparameter tuning with `GridSearchCV` for Random Forest.
4. **Evaluation**

   * Metrics: Accuracy, Precision, Recall, F1-Score, ROC Curve, and AUC.
   * Best model: Random Forest with \~91% accuracy and AUC = 0.97.

---

## 📈 Results

* **Best Model**: Random Forest Classifier
* **Performance**:

  * Accuracy: **91%**
  * AUC: **0.97**
  * Balanced precision and recall across classes

This model demonstrates robust performance for binary classification in predicting heart disease risk.

---

## 📂 Repository Structure

```
heart-disease-prediction-ml/
│
├── data/                     # Contains the dataset
├── notebook/                 # Jupyter notebook for analysis and modeling
├── project report/           # Project report and related documents
├── README.md                 # Project overview and instructions
└── requirements.txt          # Python dependencies for reproducibility
```

---

## 🚀 Getting Started

1. Clone this repository:

```bash
git clone https://github.com/panditpooja/heart-disease-prediction-ml.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
cd notebook
jupyter notebook heart_disease_prediction.ipynb
```

---

## ✍️ Author
Pooja Pandit  
Master’s Student in Information Science (Machine Learning)  
The University of Arizona
