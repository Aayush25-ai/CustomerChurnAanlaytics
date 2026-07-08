#  Customer Churn Predictor — End-to-End ML Project

A complete, production-style machine learning pipeline that predicts whether a
telecom customer will **churn** (leave the service) or stay, based on their
account, service, and billing information.

This project is built to be a **learning resource** as well as a working
pipeline — every script is heavily commented to explain *why* each step is
done, not just *what* it does.

---

## 📁 Project Structure

```
customer-churn-predictor/
│
├── data/
│   └── README.md              # instructions to download the dataset
│
├── src/
│   ├── data_preprocessing.py  # cleaning, encoding, feature engineering
│   ├── train_model.py         # trains + tunes + saves the model
│   ├── evaluate.py            # generates metrics & plots on test data
│   └── predict.py             # loads saved model, predicts on new data
│
├── models/
│   └── (saved .pkl model + preprocessor go here after training)
│
├── app/
│   └── app.py                 # Streamlit web app for live predictions
│
├── notebooks/
│   └── EDA.ipynb              # (optional) exploratory data analysis
│
├── requirements.txt
├── .gitignore
└── README.md                  # you are here
```

---

##  Dataset

**Recommended dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(available on Kaggle, originally published by IBM).

- ~7,043 customer rows, 21 columns
- Target column: `Churn` (Yes/No)
- Features include: tenure, contract type, monthly charges, total charges,
  internet service, payment method, tech support, etc.
- Free, small, well-documented, and realistic — perfect for a churn project.

### How to get it
1. Go to the Kaggle link above (you'll need a free Kaggle account).
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. Place it inside the `data/` folder of this project as `data/telco_churn.csv`.

Alternatively, download via the Kaggle CLI:
```bash
pip install kaggle
kaggle datasets download -d blastchar/telco-customer-churn -p data/ --unzip
mv data/WA_Fn-UseC_-Telco-Customer-Churn.csv data/telco_churn.csv
```

---

##  Pipeline Overview

1. **Data Preprocessing** (`src/data_preprocessing.py`)
   - Load raw CSV
   - Handle missing/invalid values (e.g. blank `TotalCharges`)
   - Encode categorical variables
   - Scale numeric features
   - Split into train/test sets
   - Save the fitted preprocessing pipeline (so it can be reused on new data)

2. **Model Training** (`src/train_model.py`)
   - Train multiple candidate models (Logistic Regression, Random Forest, XGBoost)
   - Handle class imbalance with `class_weight` / SMOTE
   - Cross-validate and tune hyperparameters (GridSearchCV)
   - Pick the best model by ROC-AUC
   - Save the final model as a `.pkl` file

3. **Evaluation** (`src/evaluate.py`)
   - Confusion matrix, precision/recall/F1, ROC-AUC
   - Feature importance plot
   - Saves plots to `models/plots/`

4. **Prediction** (`src/predict.py`)
   - Loads the saved model + preprocessor
   - Accepts new customer data (CSV or dict)
   - Outputs churn probability + label

5. **App** (`app/app.py`)
   - A Streamlit app so anyone can enter customer details in a form
   - Instantly returns churn prediction + probability

---

##  Setup

```bash
# 1. Clone the repo (after you've pushed it, see below)
git clone https://github.com/<your-username>/customer-churn-predictor.git
cd customer-churn-predictor

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset (see Dataset section above)

# 5. Run the pipeline
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate.py

# 6. Try a prediction
python src/predict.py

# 7. (Optional) Launch the interactive app
streamlit run app/app.py
```

---

##  Pushing This Project to GitHub

Run these commands from inside the `customer-churn-predictor/` folder:

```bash
# 1. Initialize git (only once)
git init

# 2. Add a .gitignore (already included) so venv/data/models aren't pushed by mistake

# 3. Stage all files
git add .

# 4. Commit
git commit -m "Initial commit: customer churn prediction pipeline"

# 5. Create a new empty repo on GitHub (via github.com/new), then link it
git remote add origin https://github.com/<your-username>/customer-churn-predictor.git

# 6. Rename branch to main (if needed)
git branch -M main

# 7. Push
git push -u origin main
```

**Tip:** Don't push the raw dataset if it's large or under a license
restriction — the `.gitignore` already excludes `data/*.csv`. Instead, keep
the download instructions in `data/README.md` (included) so others can fetch
it themselves.

Later, whenever you make changes:
```bash
git add .
git commit -m "Describe what you changed"
git push
```

---

##  Possible Extensions (great for learning further)
- Add SHAP values for model explainability
- Deploy the Streamlit app on Streamlit Community Cloud or Render
- Add a FastAPI endpoint for production-style serving
- Add MLflow for experiment tracking
- Add a GitHub Actions workflow to auto-run tests on push

---

## License
This project is for educational purposes. The Telco Customer Churn dataset
is publicly available on Kaggle under its own license — check the dataset
page for details.
