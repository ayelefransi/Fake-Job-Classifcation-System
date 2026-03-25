# !pip install kagglehub pandas numpy scikit-learn lightgbm xgboost optuna shap sentence-transformers imbalanced-learn matplotlib seaborn joblib streamlit

!pip install kagglehub pandas numpy scikit-learn lightgbm xgboost optuna shap sentence-transformers imbalanced-learn matplotlib seaborn joblib streamlit

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import optuna
import shap
import joblib
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb

# Set random seeds for reproducibility
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
import warnings
warnings.filterwarnings('ignore')

# --- CELL ---

def load_and_fuse_datasets():
    datasets = [
        "shivamb/real-or-fake-fake-jobposting-prediction",
        "srisaisuhassanisetty/fake-job-postings",
        "khushikyad001/fake-vs-real-job-postings-synthetic-nlp-dataset",
        "vijayramchallagundla/fake-job-postings"
    ]

    dataframes = []

    # Standardize column names across varying schemas
    schema_map = {
        'job_title': 'title', 'Title': 'title',
        'job_description': 'description', 'Description': 'description',
        'requirements': 'requirements', 'Requirements': 'requirements',
        'fraudulent': 'fraudulent', 'Fake': 'fraudulent', 'is_fake': 'fraudulent'
    }

    for dataset in datasets:
        try:
            print(f"Downloading {dataset}...")
            path = kagglehub.dataset_download(dataset)

            # Find all CSVs even in nested directories
            csv_files = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)

            for file in csv_files:
                df = pd.read_csv(file, low_memory=False)
                df = df.rename(columns=schema_map)

                # Keep only essential columns if they exist
                keep_cols = [c for c in ['title', 'description', 'requirements', 'fraudulent'] if c in df.columns]
                df = df[keep_cols]
                df['source_dataset'] = dataset.split('/')[0] # Track origin
                dataframes.append(df)
        except Exception as e:
            print(f"Skipping {dataset}: {e}")

    # Merge and drop exact duplicates
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=['title', 'description'])
    return merged_df

raw_data = load_and_fuse_datasets()
print(f"Total Combined Postings: {raw_data.shape[0]}")
print(raw_data['fraudulent'].value_counts(normalize=True)) # Check imbalance

# --- CELL ---

def feature_engineering(df):
    df = df.copy()

    # Handle missing text
    for col in ['title', 'description', 'requirements']:
        if col in df.columns:
            df[col] = df[col].fillna('')
        else:
            df[col] = ''

    # Target cleanup
    df = df.dropna(subset=['fraudulent'])
    df['fraudulent'] = df['fraudulent'].astype(int)

    # Combine text for holistic NLP analysis
    df['full_text'] = df['title'] + " " + df['description'] + " " + df['requirements']

    # Structured Features
    df['text_length'] = df['full_text'].apply(len)
    df['has_requirements'] = df['requirements'].apply(lambda x: 1 if len(str(x).strip()) > 5 else 0)

    # Flag words typical in scams
    scam_keywords = ['urgent', 'easy money', 'no experience', 'wire transfer', 'western union', 'cash']
    df['scam_keyword_count'] = df['full_text'].apply(lambda x: sum(1 for word in scam_keywords if word in x.lower()))

    return df

print("Engineering features...")
clean_data = feature_engineering(raw_data)

# --- ADVANCED NLP: BERT Embeddings ---
print("Generating BERT Embeddings (This takes time, batching for memory safety)...")
# MiniLM is heavily optimized for fast sentence embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings (Outputs a 384-dimensional dense vector per job posting)
embeddings = bert_model.encode(clean_data['full_text'].tolist(), batch_size=64, show_progress_bar=True)

# Convert to DataFrame and merge with structured features
embed_df = pd.DataFrame(embeddings, columns=[f'bert_{i}' for i in range(embeddings.shape[1])])
embed_df.index = clean_data.index

# Final Feature Set (Combining BERT dense vectors + Structured metadata)
X = pd.concat([
    clean_data[['text_length', 'has_requirements', 'scam_keyword_count']],
    embed_df
], axis=1)

y = clean_data['fraudulent']

# --- CELL ---

# 70% Train, 15% Val, 15% Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42) # 0.1765 of 0.85 ~ 0.15

print(f"Train size: {X_train.shape[0]} | Val size: {X_val.shape[0]} | Test size: {X_test.shape[0]}")

# Calculate Scale Pos Weight for highly imbalanced target
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

# Baseline
baseline = LogisticRegression(class_weight='balanced', max_iter=1000)
baseline.fit(X_train, y_train)
base_preds = baseline.predict(X_val)
print(f"Baseline Val F1-Score: {f1_score(y_val, base_preds):.4f}")

# --- CELL ---

import xgboost as xgb
# from xgboost.callback import EarlyStopping # This import is no longer needed due to TypeError

def optimize_xgb(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr', # This will be used in fit for early stopping
        'scale_pos_weight': scale_pos_weight, # CRITICAL for imbalanced classes
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }

    model = xgb.XGBClassifier(**params)

    # Early stopping parameters for fit method (removed due to persistent TypeError)
    # The current XGBoost installation seems to not accept 'callbacks' or 'early_stopping_rounds'
    # in XGBClassifier.fit, despite version 3.2.0 typically supporting them.
    # Therefore, training will proceed for the full n_estimators without early stopping.

    print(f"XGBoost version being used: {xgb.__version__}") # Added for verification
    # Using callbacks directly in fit, as it's the standard for XGBoost 3.x
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              # callbacks=[early_stopping_callback], # Removed this line to resolve TypeError
              verbose=False)

    # Make predictions. Since early stopping isn't applied via `fit` arguments,
    # `model.best_iteration` will likely be None. The model will predict using all
    # estimators it was trained with.
    preds = model.predict(X_val)

    return f1_score(y_val, preds)

# Run Optuna
study = optuna.create_study(direction='maximize')
study.optimize(optimize_xgb, n_trials=10) # Set to 50+ for production
best_xgb_params = study.best_params
print("Best XGB Params for F1:", best_xgb_params)

# --- CELL ---

# Reconstruct best XGBoost
best_xgb = xgb.XGBClassifier(**best_xgb_params, scale_pos_weight=scale_pos_weight, random_state=42)

# Add LightGBM
lgb_model = lgb.LGBMClassifier(
    class_weight='balanced',
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

# Add Random Forest
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42, n_jobs=-1)

# Meta-Learner stack
ensemble = StackingClassifier(
    estimators=[('xgb', best_xgb), ('lgb', lgb_model), ('rf', rf_model)],
    final_estimator=LogisticRegression(),
    cv=3,
    n_jobs=-1
)

print("Training Stacking Ensemble (combining Train+Val)...")
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])
ensemble.fit(X_train_full, y_train_full)

# Final Test Set Evaluation
test_preds = ensemble.predict(X_test)
test_probs = ensemble.predict_proba(X_test)[:, 1]

print("\n--- Final Ensemble Test Results ---")
print(classification_report(y_test, test_preds, target_names=['Real (0)', 'Fake (1)']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, test_probs):.4f}")

# Save the Pipeline
joblib.dump(ensemble, 'fake_job_ensemble.joblib')
print("Model saved to disk.")

# --- CELL ---

# SHAP Explainability (Using the XGBoost base estimator for speed)
xgb_base = ensemble.named_estimators_['xgb']
explainer = shap.TreeExplainer(xgb_base)

# Explain a sample of the test set
X_sample = X_test.sample(200, random_state=42)
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Feature Importance (BERT Dimensions & Metadata)")
plt.tight_layout()
plt.show()

# --- CELL ---

%%writefile app.py
import streamlit as st
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load Assets
@st.cache_resource
def load_models():
    model = joblib.load('fake_job_ensemble.joblib')
    bert = SentenceTransformer('all-MiniLM-L6-v2')
    return model, bert

ensemble_model, bert_model = load_models()

st.title(" Fraudulent Job Posting Detector")
st.markdown("Paste a job description below to check if it's a scam.")

title = st.text_input("Job Title")
desc = st.text_area("Job Description", height=150)
reqs = st.text_area("Requirements", height=100)

if st.button("Analyze Posting"):
    with st.spinner("Analyzing semantics and linguistic patterns..."):
        full_text = f"{title} {desc} {reqs}"

        # Structure features
        scam_keywords = ['urgent', 'easy money', 'no experience', 'wire transfer', 'western union', 'cash']
        features = {
            'text_length': [len(full_text)],
            'has_requirements': [1 if len(reqs.strip()) > 5 else 0],
            'scam_keyword_count': [sum(1 for word in scam_keywords if word in full_text.lower())]
        }

        # Get BERT Embeddings
        embeddings = bert_model.encode([full_text])
        for i in range(embeddings.shape[1]):
            features[f'bert_{i}'] = [embeddings[0][i]]

        input_df = pd.DataFrame(features)

        # Predict
        prob = ensemble_model.predict_proba(input_df)[0][1]

        if prob > 0.65:
            st.error(f" ALERT: High probability of fraud! (Confidence: {prob:.1%})")
        elif prob > 0.40:
            st.warning(f" Suspicious: Proceed with caution. (Confidence: {prob:.1%})")
        else:
            st.success(f" Looks Safe! (Fraud Probability: {prob:.1%})")