# train_save.py
import os
import json
import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, auc
from xgboost import XGBClassifier

# -------------------------
# 1) Load data (adapt paths)
# -------------------------

contract = pd.read_csv('contract.csv')
internet = pd.read_csv('internet.csv')
personal = pd.read_csv('personal.csv')
phone = pd.read_csv('phone.csv')

# 1.1. 'contract' dataset preprocessing:
contract.rename(columns = {'customerID': 'customer_id', 'BeginDate': 'begin_date', 'EndDate': 'end_date', 'Type': 'type', 'PaperlessBilling': 'paperless_billing', 'PaymentMethod': 'payment_method', 'MonthlyCharges': 'monthly_charges', 'TotalCharges': 'total_charges'}, inplace=True)

empty_values = (contract['total_charges'] == '').sum() + (contract['total_charges'].str.isspace().sum()) 
print(f'empty_values: {empty_values}')
# The empty 'whitespace' values correspond to the customers who just made first month (2020-02-01) payment. 
empty_space_rows = contract[contract['total_charges'].str.isspace()]
print(f'\n {empty_space_rows}')

'''Because these customers made the first (current) month payment, their total payment is equal to their first month payment. 
Thus, we fill ' ' values of the 'total_charges' with the corresponding values of the 'monthly_charges':'''

contract.loc[contract['total_charges'].str.strip() == '', 'total_charges'] = contract['monthly_charges']

# 'monthly_charges' column values are 'float' datatype, while 'total_carges' column values are 'object'. We convert the latter into float datatype: 
# Now, we convert 'total_charges' values into float datatype:
contract['total_charges']= contract['total_charges'].astype(float)
contract['churn'] = np.where(contract['end_date'] == 'No', 0, 1)   # 0 - no_churned, 1 - churned



# here we convert 'begin_date' and 'end_date' columns from 'object' datatype to datetime datatype
contract['begin_date'] = pd.to_datetime(contract['begin_date'])
# for 'end_date' convertion, we use errors = 'coerce' in order to keep 'No' values as a 'NaT' values of the column
contract['end_date'] = pd.to_datetime(contract['end_date'], errors = 'coerce')
#  Here, we find the cutoff date
cutoff_date = contract['end_date'].max()
# now we compute tenure
contract['tenure_days'] = (contract['end_date'].fillna(cutoff_date) - contract['begin_date']).dt.days
# now we can drop both 'begin_date' and 'end_date' columns. they have already been used to create 'churn' and 'tenure_days' columns. 
contract = contract.drop(columns=['begin_date', 'end_date'])


# 1.2. 'internet" dataset preprocessing
internet.rename(columns={'customerID': 'customer_id', 'InternetService': 'internet_service', 'OnlineSecurity': 'online_security', 'OnlineBackup': 'online_backup', 'DeviceProtection': 'device_protection', 'TechSupport': 'tech_support', 'StreamingTV': 'streaming_tv', 'StreamingMovies': 'streaming_movies'}, inplace=True)

# 1.3.  Preprocessing of 'personal' dataset
personal.rename(columns={'customerID': 'customer_id', 'SeniorCitizen': 'senior_citizen', 'Partner':'partner', 'Dependents': 'dependents'}, inplace=True)

# 1.4. Preprocessing of 'phone' dataset
phone.rename(columns={'customerID': 'customer_id', 'MultipleLines': 'multiple_lines'}, inplace=True)


# 2. Merging datasets and EDA
# 2.1. Merging datasets

df = contract.merge(personal, on='customer_id', how='left')
df = df.merge(internet, on='customer_id', how='left')
df = df.merge(phone, on='customer_id', how='left')

# missing values appeared after left merging the datasets. we fill the missing values 

categorical_cols = [
    'internet_service', 'online_security', 'online_backup',
    'device_protection', 'tech_support', 'streaming_tv',
    'streaming_movies', 'multiple_lines'
]

# Fill NaN with 'No'
df[categorical_cols] = df[categorical_cols].fillna('No')


# EDA shows that 'churn' behaves as independendt from 'male' and 'female' values.
#Therefore we remove 'gender' column from the dataset.
# Also, now we can remove 'customer_id' column beacuse it does not have a value for churn prediction. 
# drop columns
df = df.drop(columns={'customer_id', 'gender'})



# 3. ML training

X = df.drop('churn', axis=1)
y = df['churn']

# train/val/test splits
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=12345, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, random_state=12345, stratify=y_temp)

# -------------------------
# 2) Preprocessor (FIXED)
# -------------------------
categorical_features = ['type', 'paperless_billing', 'payment_method', 'partner', 'dependents', 
                        'internet_service', 'online_security', 'online_backup', 'device_protection', 
                        'tech_support', 'streaming_tv', 'streaming_movies', 'multiple_lines']

numerical_features = ['monthly_charges', 'total_charges', 'tenure_days', 'senior_citizen']   # removed senior_citizen here

# keep senior_citizen as passthrough if you prefer not to scale it:
passthrough_features = ['senior_citizen']

categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # sparse=False -> easier downstream
numerical_transformer = StandardScaler()
passthrough_transformer = FunctionTransformer()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features),
        ('pass', passthrough_transformer, passthrough_features)
    ],


    remainder='drop'
)

# -------------------------
# 3) Pipeline + Grid search
# -------------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=42))
])

param_grid = {
    'model__n_estimators': [20, 50],
    'model__max_depth': [3, 5, 10],
    'model__learning_rate': [0.1, 0.02],
    'model__subsample': [0.5, 0.8],
    'model__colsample_bytree': [0.5, 0.8],
}

grid = GridSearchCV(pipeline, param_grid, scoring=['roc_auc','accuracy','f1'], refit='roc_auc', cv=5, n_jobs=-1, verbose=2, return_train_score=True)
grid.fit(X_train, y_train)

best_pipeline = grid.best_estimator_
best_params = grid.best_params_

# -------------------------
# 4) Evaluate on test
# -------------------------
y_test_pred = best_pipeline.predict(X_test)
y_test_pred_proba = best_pipeline.predict_proba(X_test)[:,1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
test_f1 = f1_score(y_test, y_test_pred)

print("BEST PARAMS:", best_params)
print(f"TEST ACC: {test_accuracy:.4f}, ROC AUC: {test_roc_auc:.4f}, F1: {test_f1:.4f}")

# -------------------------
# 5) Save model + metadata
# -------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(best_pipeline, "artifacts/model_pipeline.joblib", compress=3)

metadata = {
    "model_version": "v1.0",
    "saved_at": datetime.datetime.now().isoformat(),
    "model_class": str(type(best_pipeline.named_steps['model'])),
    "features": list(X_train.columns),   # original input columns your app must supply
    "best_params": best_params,
    "metrics": {
        "test_accuracy": float(test_accuracy),
        "test_roc_auc": float(test_roc_auc),
        "test_f1": float(test_f1)
    },
    "notes": "Pipeline contains ColumnTransformer (OneHotEncoder/StandardScaler) + XGBClassifier"
}

with open("artifacts/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)

print("Saved artifacts/model_pipeline.joblib and artifacts/model_metadata.json")
