# customer-risk-intelligence
# Customer Risk Intelligence System

An end-to-end AI decision system that predicts customer risk, optimizes operational actions using cost-based strategies, and generates explainable insights through machine learning, SHAP, and LLM-powered explanations.

Live Application  
https://customer-risk-intelligence.streamlit.app/

---

# Project Overview

Modern digital platforms process large volumes of customer interactions every day, including product reviews, purchases, refunds, and support requests. Among these interactions, some may signal operational risk such as fraudulent behavior, refund abuse, suspicious activity, or product dissatisfaction that may lead to financial loss.

Traditional rule-based systems struggle to capture complex behavioral patterns, while fully automated decision systems can introduce unacceptable error rates without transparency.

This project builds a **Customer Risk Intelligence System** that integrates machine learning, decision optimization, explainable AI, and natural language explanations to support intelligent and transparent operational decision-making.

The system predicts the **probability of risk associated with customer interactions**, evaluates the **expected cost of different operational responses**, and generates **human-readable explanations of predictions and decisions**. The final solution is delivered through an interactive Streamlit application.

---

# Problem Statement

Organizations must balance two competing priorities:

• Detect potentially risky customer interactions  
• Maintain efficient and cost-effective operations  

Manual review processes are slow and expensive, while fully automated decisions may introduce errors that result in financial losses or customer dissatisfaction.

The key challenge addressed by this project is:

**How can machine learning be used to identify risky interactions while determining the most cost-effective operational response and maintaining transparency in decision making?**

---

# Project Objectives

The primary objectives of this system are to:

• Predict the probability that a customer interaction is risky  
• Ensure predicted probabilities are reliable through probability calibration  
• Translate model predictions into actionable business decisions  
• Optimize operational actions using cost-based decision strategies  
• Provide interpretable explanations of model predictions  
• Generate clear natural language explanations using a large language model  
• Deliver an interactive interface for exploring predictions and decisions  

---

# System Architecture

The project follows an end-to-end machine learning pipeline:

Data Collection  
↓  
Data Cleaning & Feature Engineering  
↓  
Machine Learning Risk Prediction Models  
↓  
Probability Calibration  
↓  
Decision Strategy Optimization  
↓  
Explainable AI (SHAP)  
↓  
LLM Explanation Layer  
↓  
Streamlit Interactive Application

This architecture simulates how modern organizations deploy AI-driven decision support systems.

---

# Machine Learning Models

Three machine learning models were trained and evaluated:

• Logistic Regression  
• Random Forest  
• XGBoost  

These models predict the **probability that a customer interaction represents potential risk**.

To ensure predictions can be reliably used for operational decision-making, **probability calibration (Platt Scaling)** is applied to align predicted probabilities with real-world event frequencies.

---

# Decision Strategy Engine

Instead of producing predictions alone, the system converts risk probabilities into operational actions using a **cost-based decision framework**.

Three possible actions are considered:

**Approve Automatically**  
Low-risk interactions are processed automatically to maximize operational efficiency.

**Human Review**  
Moderate-risk interactions are escalated to human analysts for verification.

**Reject / Flag**  
High-risk interactions are rejected or flagged to prevent potential financial loss.

The system calculates the **expected cost of each strategy** based on prediction probabilities and operational error rates, then recommends the most economically efficient action.

---

# Explainable AI

Model transparency is provided through **SHAP (SHapley Additive Explanations)**.

SHAP analysis identifies the features that most strongly influence model predictions, allowing users to understand:

• Why a customer interaction was flagged as risky  
• Which features contributed most to the prediction  
• How different signals influence the model

This improves trust and interpretability in AI-assisted decisions.

---

# LLM Explanation Layer

To make insights accessible to non-technical stakeholders, the system integrates a **large language model (Google Gemini)**.

The LLM converts technical outputs such as:

Risk probability  
Feature contributions  
Recommended actions  

into **clear natural language explanations** that business teams can easily understand.

---

# Streamlit Application

The final system is deployed as an interactive **Streamlit dashboard**.

The application allows users to:

• Explore predicted customer risk scores  
• Visualize model explanations using SHAP  
• Understand which features influence predictions  
• View recommended operational actions  
• Read natural language explanations of decisions  

Live app:  
https://customer-risk-intelligence.streamlit.app/

---

# Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
XGBoost  
SHAP  
Streamlit  
Google Gemini API  
Matplotlib  

---

The notebook contains the full machine learning pipeline, including data preprocessing, feature engineering, model training, calibration, explainability, and decision strategy design.

---

# Running the Project Locally

Clone the repository:
git clone https://github.com/MaryaD97/customer-risk-intelligence.git

Navigate to the project directory:
cd customer-risk-intelligence

Install dependencies:
pip install -r requirements.txt


Run the Streamlit application
streamlit run app.py


The application will open in your browser.

---

# Example Use Cases

This type of system can support several operational scenarios:

• Fraud detection in e-commerce platforms  
• Customer refund abuse detection  
• Risk monitoring in digital marketplaces  
• Customer support decision automation  
• Product feedback risk analysis  

Organizations can use systems like this to combine **machine learning predictions with operational cost optimization and explainable AI**.

---

# Future Improvements

Potential extensions for production systems include:

• Real-time streaming data integration  
• Behavioral anomaly detection models  
• Automated monitoring and model retraining pipelines  
• Advanced decision optimization frameworks  
• Containerized deployment for scalable production environments  

---

# Author

Marya D  
Customer Risk Intelligence System  
Data Science Bootcamp Final Project

