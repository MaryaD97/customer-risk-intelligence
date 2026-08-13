# Customer Risk Intelligence System

An end-to-end machine learning decision support system that predicts customer interaction risk and uses those predictions to recommend cost-aware operational actions through an interactive Streamlit application.

---

## Live Demo

**Live Application:**  
https://customer-risk-intelligence.streamlit.app/

**GitHub Repository:**  
https://github.com/MaryaD97/customer-risk-intelligence

The project started as a machine learning notebook and was then developed into an interactive application that allows users to upload transaction data, configure business assumptions, run the risk model, and explore the resulting decisions and business impact.

---

## Why I Built This

I wanted to build a project that went beyond training a classification model and reporting an accuracy score.

In a real business setting, a risk prediction is usually only the starting point. A model may identify an interaction as risky, but the business still has to decide what to do with it:

- Should it be handled automatically?
- Should it be sent for manual review?
- How much does each option cost?
- When is automation worth the additional risk?

This project explores that problem by connecting **machine learning predictions with a cost-based decision layer**.

The original model was developed using customer review data, where signals such as ratings, sentiment, review activity, helpfulness, and verified purchases were transformed into features for risk prediction. I then extended the machine learning work into an application that considers transaction value and operational costs when recommending an action.

Another important part of the project was making the output understandable. A risk score on its own is difficult for a non-technical user to act on, so the system also provides transaction-level risk signals and decision rationale alongside the predicted risk and expected cost.

The final result is a small decision-support system rather than a standalone prediction model:

**Data → Risk Prediction → Expected Cost → Recommended Action**

---

## Problem Statement

Organizations need to identify potentially risky customer interactions while balancing two competing priorities:

- **Reduce financial and operational risk**
- **Keep the cost of investigation and manual review under control**

A purely manual process can become slow and expensive as the number of transactions increases. On the other hand, automating every decision can reduce operational costs while increasing the consequences of incorrectly handling a risky transaction.

The key challenge addressed by this project is:

> **How can machine learning be used to estimate customer interaction risk and then translate those predictions into cost-aware operational decisions?**

The project approaches this as a **decision problem** rather than only a classification problem. A predicted risk probability is combined with transaction value and configurable business assumptions to estimate the expected cost of different handling strategies.

This creates a workflow of:

**Customer Data → Risk Prediction → Expected Cost → Recommended Action**

The goal is not to replace human decision-making completely, but to show how a machine learning model can be connected to a practical operational decision process.

## 5. Project Goals

The goal of the project is to move from a risk prediction model to a system that can support practical operational decisions.

The application enables users to:

- Upload transaction datasets in **CSV or Excel format**
- Preview and validate uploaded data before analysis
- Map their dataset columns to the fields required by the model
- Automatically suggest column mappings based on common field names
- Clean and standardize the mapped data
- Generate a **risk probability** for each transaction
- Configure assumptions around fraud loss and manual review costs
- Estimate the expected cost of different decision strategies
- Identify the **lowest-cost strategy for each transaction**
- Review recommended actions at both dataset and transaction level
- Explore projected savings and the operational impact of the decisions

The project therefore focuses on both sides of the problem:

**Predict risk accurately enough to support decisions, and make those predictions useful in a business context.**

---

## 6. End-to-End Workflow

The project follows an end-to-end workflow that connects the original machine learning pipeline to the final decision-support application.

```text
Customer Review Data
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Engineering
        ↓
Risk Label Construction
        ↓
Model Training & Evaluation
        ↓
Model Selection
        ↓
Probability Calibration
        ↓
Risk Probability Prediction
        ↓
Transaction Value + Business Assumptions
        ↓
Expected Cost Calculation
        ↓
Decision Strategy Selection
        ↓
Business Impact Analysis
        ↓
Interactive Streamlit Application

```

The deployed application adds an additional data-handling layer for user-provided datasets:

**Upload Dataset**  
↓  
**Data Validation**  
↓  
**Column Mapping**  
↓  
**Data Cleaning**  
↓  
**Risk Prediction**  
↓  
**Cost-Based Decision Engine**  
↓  
**Recommended Actions**  
↓  
**Business Impact Summary**

This structure was important to the project because the final goal was not simply to produce a model prediction. The prediction needed to become an input to a decision process that a user could actually interact with.

## 7. Project Development Journey

This project developed in two main stages.

### Stage 1 — Machine Learning Notebook

The project began as a machine learning pipeline built around customer review data.

The notebook was used to:

- Explore and validate the dataset
- Clean review and behavioral data
- Engineer customer-level and review-level features
- Construct a risk label from observable customer signals
- Train and evaluate multiple classification models
- Compare model performance
- Calibrate predicted probabilities
- Explore model behavior using SHAP
- Develop and test the cost-based decision strategy

This stage established the predictive and decision-making logic behind the system.

### Stage 2 — Interactive Application

The next step was to turn the notebook work into something a user could interact with.

The selected model was saved and loaded into a Streamlit application, where the machine learning output became part of a multi-step decision workflow.

Additional application logic was then developed for:

- Dataset upload and validation
- Flexible column mapping
- Data cleaning
- Business-cost configuration
- Decision simulation
- Transaction-level recommendations
- Risk and cost summaries
- Business impact reporting

The final application therefore extends beyond the original notebook. The notebook demonstrates how the model and decision logic were developed, while the Streamlit application packages that work into an interactive decision-support tool.

## 8. Machine Learning Pipeline

The machine learning component was developed in the Jupyter notebook using customer review data from the Amazon Electronics reviews dataset.

### Data Preparation

The dataset contains information such as:

- Customer ratings
- Review text
- Helpful votes
- Verified purchase information
- User identifiers
- Product information

Because the original dataset is large, a working subset was used for experimentation.

The preprocessing pipeline included:

- Handling missing values
- Converting and validating data types
- Cleaning review text
- Removing unnecessary text noise
- Extracting review length
- Processing review timestamps
- Creating customer-level behavioral features

### Feature Engineering

The project combined direct customer feedback with behavioral signals.

The final model features included:

- `rating`
- `sentiment_score`
- `verified_purchase`
- `review_length`
- `helpfulness_ratio`

These features were designed to capture different aspects of customer behavior, including satisfaction, sentiment, engagement, trust, and review credibility.

The notebook also explored additional behavioral features such as rating-sentiment mismatch, rating deviation, helpfulness signals, and customer segmentation before constructing the final risk label.

### Risk Label Construction

The original review dataset does not provide a direct fraud label.

Instead, the project constructed a risk label from observable signals associated with potentially problematic customer interactions.

The label combines conditions such as:

- Low customer rating
- Negative sentiment
- Large differences between rating and sentiment
- Low helpfulness
- High behavioral risk score

This makes the task a constructed risk classification problem, rather than supervised learning from a ground-truth fraud dataset.

### Train/Test Split

The data was divided into training and test sets using an 80/20 stratified split with a fixed random state for reproducibility.

Feature scaling was applied where appropriate, particularly for Logistic Regression.

### Models Evaluated

Three classification models were trained and compared:

#### Logistic Regression

Used as an interpretable baseline and reference point.

#### Random Forest

Used to capture non-linear relationships and interactions between features.

#### XGBoost

Used as the final candidate because it achieved the strongest overall performance among the evaluated models.

The models were evaluated using metrics including:

- Precision
- Recall
- F1-score
- ROC-AUC

The comparison in the notebook showed XGBoost achieving the highest ROC-AUC among the three evaluated models.

### Probability Calibration

Because the model's probabilities are later used by the decision engine, the project did not treat probability estimates as interchangeable with ordinary classification outputs.

The notebook applies sigmoid calibration (Platt Scaling) using CalibratedClassifierCV.

Calibration was evaluated using:

- ROC-AUC
- Calibration curves
- Brier score

The calibrated probabilities are then used in the decision-engine stage.

### Model Explainability

The notebook also uses SHAP to examine why the selected XGBoost model produces particular risk predictions.

Both global and individual-level explanations were explored, including:

- Overall feature importance
- Feature contribution to individual predictions
- Waterfall explanations

This provides an interpretable layer between the model's predictions and the decisions made using those predictions.

## 9. Decision Engine

The main extension beyond the machine learning model is the cost-based decision engine.

A risk probability by itself does not determine what an organization should do. The same risk probability can lead to different actions depending on transaction value, fraud impact, and the cost of human intervention.

The decision engine therefore combines:

**Risk Probability + Transaction Value + Business Assumptions**

to estimate the expected cost of different operational strategies.

### Strategies Considered

The underlying decision logic evaluates three approaches:

#### AI Automation

The transaction is handled automatically.

This has a lower operational cost but assumes that the automated system will occasionally miss risky transactions.

#### Human Review

The transaction is sent to a human reviewer.

This adds a fixed review cost but assumes a lower error rate than full automation.

#### Hybrid

Lower-risk cases are handled automatically while higher-risk cases are routed for review.

The hybrid approach is intended to balance automation with additional human oversight.

### Business Assumptions

The application allows users to change two key assumptions:

- **Fraud Loss Multiplier** — represents the financial impact of a missed risky transaction relative to its transaction value
- **Cost per Manual Review** — represents the operational cost of investigating one transaction

The application also uses assumed effectiveness levels for the two handling approaches:

- Manual review catches approximately 90% of fraud
- Automation catches approximately 60% of fraud

These are simulation assumptions rather than measured production performance, so the resulting financial figures should be interpreted as projected outcomes under the selected assumptions.

### Expected Cost

For each transaction, the system calculates the expected cost of the available strategies.

The decision engine then selects the strategy with the lowest expected cost.

This allows the system to move from:

> "How risky is this transaction?"

to:

> "Given its risk and financial exposure, what is the most cost-effective way to handle it?"

### Business Impact

The application then aggregates the decisions to estimate:

- Baseline cost
- Optimized cost
- Projected savings
- Automated decisions
- Manual reviews
- Strategy distribution

Users can also adjust the cost assumptions after the initial analysis and see how the recommended decisions and projected financial impact change.

This is the part of the project that connects the machine learning model to a practical business problem: the model provides the risk estimate, while the decision engine determines how that estimate can be used under different operational constraints.


The details above are grounded in the notebook's actual data, target construction, model comparison, calibration, and SHAP work, and in the current application's upload, mapping, cost, decision, and impact workflow. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14} :contentReference[oaicite:15]{index=15}

## 10. Streamlit Application

The machine learning pipeline was turned into an interactive **Streamlit application** so that the model could be used through a guided workflow rather than only through notebook code.

The application is organized into five steps:

**Upload Data → Set Costs → Generate Decisions → Decisions → Insights**

### Step 1 — Upload Data

Users can either upload their own dataset or start with the provided sample data.

The application accepts:

- **CSV files**
- **Excel files (`.xlsx` and `.xls`)**

Before the analysis can continue, the application validates the uploaded data and provides a preview.

Users can then map their own column names to the fields expected by the model. The application includes common aliases for fields such as:

- Customer rating
- Sentiment score
- Review length
- Helpfulness ratio
- Verified purchase
- Transaction value

The mapping process also checks for issues such as:

- Duplicate column selections
- Missing columns
- Empty columns
- Invalid numeric fields
- Invalid verified-purchase values

This makes the application less dependent on users having a dataset with exactly the same column names as the original training data.

### Step 2 — Set Business Assumptions

Users can configure the assumptions used by the decision engine through the application.

The two adjustable inputs are:

- **Fraud Loss Multiplier**
- **Cost per Manual Review**

This allows the decision process to be tested under different business conditions instead of relying on one fixed cost configuration.

### Step 3 — Generate Decisions

Once the data and business assumptions are ready, the application:

1. Loads the trained model
2. Generates a risk probability for each transaction
3. Calculates the expected cost of the available strategies
4. Selects the strategy with the lowest expected cost
5. Assigns a risk tier to each transaction

The model and its feature configuration are loaded from saved deployment artifacts rather than retraining the model inside the application.

### Step 4 — Explore Decisions

The application provides a transaction-level view of the recommendations.

Users can:

- View the recommended action for each transaction
- See the predicted fraud risk score
- View expected cost
- Sort transactions by risk or cost
- Filter for high-risk transactions
- Review the main risk signals associated with an individual transaction

The application also summarizes the overall decision distribution, including the proportion of transactions handled automatically versus those requiring review.

### Step 5 — Business Impact

The final step translates the individual decisions into a higher-level operational summary.

The application reports:

- **Transactions analyzed**
- **Automated decisions**
- **Manual reviews**
- **Baseline cost**
- **Optimized cost**
- **Projected savings**

Users can also change the cost assumptions during the decision analysis and see how those assumptions affect the recommended strategies and projected business impact.

### From Model Output to User Interface

One of the main goals of building the Streamlit application was to make the machine learning output easier to use.

Instead of requiring a user to interpret model probabilities directly, the application presents the information as:

**Risk Score → Expected Cost → Recommended Action → Business Impact**

This creates a clearer connection between the technical model and the operational decision a user needs to make.

### Explainability in the Final System

The notebook includes SHAP-based model analysis, including global feature importance and individual prediction explanations.

The current deployed Streamlit application takes a simpler approach for transaction-level explanations. It presents human-readable risk signals such as high transaction value, low customer rating, low engagement activity, and unverified purchase status.

This distinction is intentional in the README: **SHAP was part of the model-development and explainability work in the notebook, while the current deployed interface uses a lightweight rule-based explanation layer rather than running SHAP calculations for every uploaded transaction.**

---

## 11. Technologies Used

### Data Processing

- **Python**
- **Pandas**
- **NumPy**

Used for data cleaning, transformation, feature preparation, and decision calculations.

### Machine Learning

- **Scikit-learn**
- **XGBoost**

Used for model training, evaluation, probability calibration, and risk prediction.

### Natural Language Processing

- **TextBlob**
- **Scikit-learn text-processing utilities**

Used during the notebook stage to derive sentiment-related features from customer review text.

### Explainable AI

- **SHAP**

Used in the notebook to investigate global feature importance and individual model predictions.

### Application and Deployment

- **Streamlit**
- **Joblib**

Streamlit was used to build the interactive application, while Joblib is used to save and load the trained model and feature configuration.

### Visualization

- **Matplotlib**
- **Seaborn**

Used during the analysis and model-development stage.

### LLM Experimentation

- **Google Gemini API**

Gemini was explored in the notebook as an explanation layer for converting structured model and decision outputs into natural-language explanations.

The current deployed application does not make live Gemini API calls. The final application instead uses deterministic transaction-level risk signals and decision rationale.

---

## 12. Repository Structure

The repository separates the model-development work from the deployed application and its supporting artifacts.

```text
customer-risk-intelligence/
│
├── customer_risk_intelligence_pipeline.ipynb
│   └── End-to-end machine learning pipeline,
│       model evaluation, calibration,
│       explainability and decision-engine development
│
├── streamlit_app.py
│   └── Interactive Streamlit application
│
├── models/
│   ├── risk_model.pkl
│   └── feature_columns.pkl
│
├── sample_data.csv
│   └── Sample dataset for testing the application
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation
```

The notebook contains the development and experimentation work, while `streamlit_app.py` contains the application logic used to run the deployed decision-support workflow.

## 13. Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/MaryaD97/customer-risk-intelligence.git

Navigate to the project directory:

cd customer-risk-intelligence

Install the required dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run streamlit_app.py

```

The application will open in your browser.

### Using the Application

You can either:

- Select **Use Sample Data** to try the application with the provided sample dataset
- Upload your own CSV or Excel dataset and map its columns to the fields required by the model

The uploaded data must contain information that can be mapped to the model's required features and a transaction value.

## 14. Future Improvements

The current project is designed as a portfolio and proof-of-concept system rather than a production fraud platform.

Some areas I would explore next are:

- Real fraud-labelled data to replace the constructed risk label with a ground-truth target
- More robust model validation, including cross-validation and additional calibration evaluation
- Model monitoring to detect changes in data and prediction behaviour over time
- Automated retraining pipelines for maintaining model performance
- Real-time or API-based inference instead of file-based analysis
- More advanced decision optimization, including sensitivity analysis across different business assumptions
- Production-grade explainability, including integrating SHAP explanations directly into the deployed application
- Stronger deployment infrastructure, such as containerization and more scalable cloud deployment

These improvements would move the project from an interactive decision-support prototype toward a more complete production ML system.

## 15. Lessons Learned

Building this project from a notebook into an application changed the way I approached the problem.

One of the main lessons was that a good machine learning model is only one part of a useful data science system. The model produces a probability, but the surrounding system has to determine how that probability can actually be used.

I also learned the importance of making assumptions explicit. The cost of manual review, the financial impact of a missed risky transaction, and the assumed effectiveness of different strategies all affect the final recommendation. Making these inputs configurable made it possible to explore how business decisions change under different conditions.

Turning the notebook into a Streamlit application also introduced problems that do not appear in a typical modeling workflow. Uploaded datasets need to be validated, column names may differ from the training data, invalid inputs need to be handled, and technical model outputs need to be presented in a form that a non-technical user can understand.

Overall, the project helped me understand the difference between building a model and building something around a model that a user can actually work with.

## 16. Author

Marya D

Data Science Bootcamp Final Project

GitHub
