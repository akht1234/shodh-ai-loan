import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('/kaggle/input/lending-club/accepted_2007_to_2018Q4.csv.gz')

SAMPLE_FRAC = 0.10
RANDOM_STATE = 42
if SAMPLE_FRAC is not None and 0 < SAMPLE_FRAC < 1:
    df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)

if 'term' in df.columns:
    df['term'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float)
if 'int_rate' in df.columns:
    df['int_rate'] = df['int_rate'].astype(str).str.replace('%','').replace('nan','',regex=False)
    df.loc[df['int_rate']=='', 'int_rate'] = np.nan
    df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')
if 'emp_length' in df.columns:
    df['emp_length'] = df['emp_length'].astype(str).replace(['n/a','nan','None'],'0', regex=False)
    df['emp_length'] = df['emp_length'].str.replace(r'\+','',regex=True)
    df['emp_length'] = df['emp_length'].str.replace(r'([<> ]+years?)','',regex=True).str.replace('year','',regex=False).str.strip()
    df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce').fillna(0)
if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
    df['fico_mean'] = df[['fico_range_low','fico_range_high']].mean(axis=1)
if 'revol_util' in df.columns:
    df['revol_util'] = df['revol_util'].astype(str).str.replace('%','').replace('nan','',regex=False)
    df.loc[df['revol_util']=='', 'revol_util'] = np.nan
    df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce')

def map_status(s):
    s = str(s).lower()
    if 'fully paid' in s:
        return 0
    if 'charged off' in s or 'default' in s:
        return 1
    return np.nan

df['target'] = df['loan_status'].apply(map_status)
df = df.dropna(subset=['target']).reset_index(drop=True)
df['target'] = df['target'].astype(int)

features = [
    'loan_amnt','term','int_rate','installment',
    'grade','sub_grade','emp_length','home_ownership',
    'annual_inc','verification_status','purpose',
    'dti','delinq_2yrs','fico_mean','inq_last_6mths',
    'open_acc','pub_rec','revol_bal','revol_util','total_acc'
]
use_feats = [c for c in features if c in df.columns]
data = df[use_feats + ['target']].copy()

num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in use_feats if c not in num_cols]

numeric_features = [c for c in num_cols if c != 'target']
categorical_features = cat_cols

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X = preprocessor.fit_transform(data.drop(columns=['target']))
y = data['target'].values

processed_df = pd.DataFrame(X)
processed_df['target'] = y
processed_df.to_csv("/kaggle/working/lc_processed_sample.csv", index=False)
print("Saved processed dataset with shape:", processed_df.shape)
