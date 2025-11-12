# 💰 EMIPredict AI - Intelligent Financial Risk Assessment Platform

[![Streamlit App](Streamlit_Dashboard_Image.png)]
[![Project Domain](https://img.shields.io/badge/Domain-FinTech%20%26%20Banking-blue?style=flat-square)](https://github.com/EMIPredict-AI-Platform)
[![Best Model](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)](https://github.com/EMIPredict-AI-Platform)

## 💡 Project Goal

This project delivers a comprehensive, data-driven platform designed to solve the critical issue of inadequate financial risk assessment and poor EMI planning. By leveraging Machine Learning and MLOps principles, the platform provides automated, real-time insights into a borrower's financial capacity.

### Problem Solved
To build an intelligent system that predicts loan risk and determines the maximum safe Equated Monthly Installment (EMI) amount a customer can afford, utilizing a dataset of **400,000 financial records**.

---

## 🏗️ Architecture and Dual ML Solution

The EMIPredict AI platform employs a dual Machine Learning approach to provide holistic risk assessment:

| Component | Target Variable | ML Task | Key Metric Goal |
| :--- | :--- | :--- | :--- |
| **Classification Model** | `emi_eligibility` (Not\_Eligible, High\_Risk, Eligible) | Multi-Class Prediction | **Accuracy > 90%** |
| **Regression Model** | `max_monthly_emi` (Continuous INR Value) | Value Prediction | **RMSE < 2000 INR** |

### MLOps and Deployment Pipeline

* **MLflow:** Used for organized **Experiment Tracking**, logging all model parameters, metrics (Accuracy, RMSE), and maintaining a **Model Registry** for the production-ready models.
* **Streamlit:** The interactive, multi-page web application is built with Streamlit for an intuitive user interface.
* **Streamlit Cloud:** Provides automated **Cloud Deployment** for production access via a GitHub pipeline.

---

## 🛠️ Technology Stack

* **Language:** Python 3.x
* **Core Libraries:**
    * **Machine Learning:** `scikit-learn`, `XGBoost`
    * **MLOps:** `MLflow`
    * **Data Analysis:** `pandas`, `numpy`
    * **Web Application:** `streamlit`

## 📊 Key Results (Validation Set Performance)

The final selected models (XGBoost for both tasks) successfully met the project's performance criteria:

| Task | Model | Key Metric | Value Achieved | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Classification** (Eligibility) | XGBoost Classifier | Accuracy | **~93.8%** | ✅ Success |
| **Regression** (Max EMI) | XGBoost Regressor | RMSE | **~1780.2 INR** | ✅ Success |

---

## 🚀 Getting Started (Local Setup)

To run the application locally or contribute to the project, follow these steps.

### Prerequisites

1.  Python 3.8+
2.  A running MLflow tracking server (optional, for full MLOps tracking)

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/EMIPredict-AI-Platform.git](https://github.com/YOUR_GITHUB_USERNAME/EMIPredict-AI-Platform.git)
    cd EMIPredict-AI-Platform
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Ensure Data/Models:** Make sure the `emi_prediction_dataset.csv` file and all required model artifacts (or the MLflow configuration to load them) are correctly set up.
2.  **Run Streamlit:**
    ```bash
    streamlit run app.py
    ```

The application will open in your browser at `http://localhost:8501`.

---

