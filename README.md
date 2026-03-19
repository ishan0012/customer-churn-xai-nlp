# 📊 Customer Churn Intelligence & Explainable AI (XAI)

## 🚀 Project Overview
This project addresses the "Black Box" problem in Customer Churn prediction. By combining **Structured Behavioral Data** (tenure, billing) with **Unstructured Text Feedback** (NLP), this system predicts customer attrition with high precision. It further implements **SHAP** (Shapley Additive Explanations) to provide business stakeholders with transparent reasons for every prediction.

## 🛠️ Tech Stack
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** XGBoost (Gradient Boosting)
- **NLP:** TextBlob (Sentiment Analysis)
- **Explainable AI:** SHAP (Feature Attribution)
- **Persistence:** Joblib

## 📈 Key Features & Methodology
1. **Multi-Modal Feature Engineering:** Extracted polarity scores from customer feedback using NLP to quantify "Customer Frustration" as a predictive feature.
2. **Advanced Preprocessing:** Handled class imbalance in Churn using `scale_pos_weight` and converted `TotalCharges` from string to numeric.
3. **Interpretability:** Used SHAP summary plots to identify that **Contract Type** and **Feedback Sentiment** are the top 2 indicators of churn.
4. **Model Deployment Readiness:** Exported the trained pipeline as a `.pkl` file for integration into production APIs.

## 📁 Repository Structure
- `notebooks/`: Contains the end-to-end Jupyter Notebook.
- `exports/`: Contains the serialized `.pkl` model and feature list.
- `data/`: The Telco Churn dataset with feedback.
- `requirements.txt`: List of necessary Python libraries.

## 📋 How to Reproduce
1. Clone the repo: `git clone https://github.com/your-username/customer-churn-xai-nlp.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook notebooks/churn_analysis.ipynb`

## 📊 Results & Visualization
The model achieved an **ROC-AUC of 0.85+**. Below is the SHAP summary plot showing how features like 'MonthlyCharges' and 'Sentiment' impact the churn probability:
<img width="963" height="826" alt="Screenshot (29)" src="https://github.com/user-attachments/assets/2bb7b5d9-d665-4ef2-8326-92a0dcdbef1d" />

