# 💳 Credit Card Fraud Detection using Machine Learning and GANs

## 📌 Project Overview

Credit card fraud detection is a challenging real-world problem due to the **extreme imbalance of transaction data** and the continuously evolving nature of fraudulent behavior. Traditional rule-based systems lack adaptability and fail to generalize to unseen fraud patterns.

This project presents a **machine learning–based fraud detection pipeline enhanced with Generative Adversarial Networks (GANs)** to address data scarcity in the minority (fraud) class. In addition to training supervised classifiers, a GAN is used to **generate realistic synthetic fraudulent transactions**, improving model generalization and recall.

The project integrates concepts from:
- Imbalanced learning
- Synthetic data generation using GANs
- Supervised classification
- Metric-driven evaluation for risk-sensitive systems

---

## 🎯 Research Objective

The primary objectives of this project are:

- To design a fraud detection system robust to **severe class imbalance**
- To evaluate the effectiveness of **GAN-generated synthetic fraud samples**
- To maximize **fraud detection recall** while controlling false positives

> The core hypothesis is that **GAN-based data augmentation improves minority-class representation and enhances fraud detection performance**.

---

## 🧠 Methodology

### 1️⃣ Dataset Description

- Anonymized credit card transaction dataset
- Numerical features obtained via **PCA transformation** for privacy preservation
- Binary target variable:
  - `0` → Legitimate transaction
  - `1` → Fraudulent transaction
- Fraudulent transactions constitute **less than 1%** of the dataset

---

### 2️⃣ Data Preprocessing

- Feature standardization to normalize transaction attributes
- Stratified train-test splitting
- Baseline imbalance handling using resampling techniques for comparison

---

### 3️⃣ GAN-Based Synthetic Data Generation

To address minority-class scarcity, a **Generative Adversarial Network (GAN)** was trained exclusively on fraudulent transaction samples.

- The **Generator** learns to create realistic synthetic fraud records
- The **Discriminator** distinguishes between real and synthetic fraud samples
- Training continues until the generator produces statistically similar fraud transactions

The generated synthetic fraud samples were combined with the original dataset to create a **balanced training set**, improving classifier learning capacity.

This approach is inspired by recent research demonstrating the effectiveness of GANs in financial fraud detection.

---

### 4️⃣ Supervised Model Training

The following classifiers were trained on both original and GAN-augmented datasets:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Performance comparisons were conducted to measure the impact of GAN-based augmentation.

---

### 5️⃣ Evaluation Strategy

Given the imbalanced nature of the problem, accuracy was not used as a primary metric. Instead, evaluation focused on:

- **Recall (Sensitivity)** – priority metric
- Precision
- F1-score
- ROC–AUC
- Confusion Matrix analysis

Recall maximization was emphasized due to the high cost of undetected fraud.

---

## 📊 Results & Key Findings

- GAN-augmented datasets significantly improved minority-class recall.
- Ensemble models (Random Forest) demonstrated the best robustness.
- Synthetic fraud generation reduced model bias toward the majority class.
- Results validate the effectiveness of GANs for imbalanced financial datasets.

---

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Libraries & Tools:**
  - NumPy
  - Pandas
  - Scikit-learn
  - TensorFlow / PyTorch (GAN implementation)
  - Imbalanced-learn
  - Matplotlib / Seaborn

---

## 📈 Future Scope

- Conditional GANs for controlled fraud pattern generation
- Online learning for concept drift handling
- Model explainability using SHAP / LIME
- Deployment using FastAPI for real-time detection

---

## 📚 References

1. **Fiore, U., De Santis, A., Perla, F., Zanetti, P., & Palmieri, F.**  
   *Using Generative Adversarial Networks for Improving Classification Effectiveness in Credit Card Fraud Detection*,  
   Information Sciences, Elsevier, 2019.

2. Dal Pozzolo, A. et al., *Adversarial Drift Detection in Credit Card Fraud*, IEEE TNNLS.

3. Kaggle – Credit Card Fraud Detection Dataset

4. Scikit-learn Documentation

5. Goodfellow, I. et al., *Generative Adversarial Networks*, NeurIPS 2014.

---

## 👨‍💻 Author

**Asutosh Ranjan**  
B.Tech CSE (Data Science)  
SRM Institute of Science and Technology

---

## ⭐ Acknowledgments

This project was developed to explore the intersection of **machine learning, generative models, and financial security systems**.
