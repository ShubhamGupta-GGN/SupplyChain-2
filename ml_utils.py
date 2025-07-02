
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
                             mean_squared_error, r2_score)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------------- Helper Functions ----------------
def get_counts(df: pd.DataFrame, col: str):
    exploded = df[col].astype(str).str.split(', ').explode()
    return exploded.value_counts()

# ---------------- Classification ----------------
def prepare_classification_data(df: pd.DataFrame):
    work_df = df.copy()
    work_df['PilotProgramInterest'] = work_df['PilotProgramInterest'].map({'Yes':1, 'No':0, 'Maybe':0})
    categorical = work_df.select_dtypes(include=['object']).columns
    categorical = [c for c in categorical if c != 'PilotProgramInterest']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoded = encoder.fit_transform(work_df[categorical])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical))
    final_df = pd.concat([work_df[['PilotProgramInterest']].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return final_df, encoder

def train_test_split_scaled(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test, scaler

def evaluate_classifiers(models, X_train, X_test, y_train, y_test):
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        rows.append({
            'Model': name,
            'Train Acc': accuracy_score(y_train, pred_train),
            'Test Acc': accuracy_score(y_test, pred_test),
            'Precision': precision_score(y_test, pred_test, zero_division=0),
            'Recall': recall_score(y_test, pred_test, zero_division=0),
            'F1': f1_score(y_test, pred_test, zero_division=0)
        })
    return pd.DataFrame(rows).round(3)

def plot_confusion_matrix(model, X_test, y_test):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap='Blues')
    ax.set_title('Confusion Matrix')
    return fig

def plot_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:,1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    return fig

def predict_new(model, new_df, encoder, scaler, train_feature_cols):
    # Encode using the SAME encoder categories as training
    cat_cols = new_df.select_dtypes(include=['object']).columns
    encoded = encoder.transform(new_df[cat_cols])
    X_new = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    # Align columns to training feature order
    X_new_aligned = X_new.reindex(columns=train_feature_cols, fill_value=0)
    # Scale
    X_scaled = scaler.transform(X_new_aligned)
    return model.predict(X_scaled)

# ---------------- Clustering ----------------
def cluster_data(num_df, k_range):
    inertias = {}
    labeled_df = num_df.copy()
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=0, n_init='auto')
        km.fit(num_df)
        inertias[k] = km.inertia_
        if k == max(k_range):
            labeled_df['Cluster'] = km.labels_
    return labeled_df, inertias

def plot_elbow(inertias):
    fig, ax = plt.subplots()
    ks = list(inertias.keys())
    ax.plot(ks, [inertias[k] for k in ks], marker='o')
    ax.set_xlabel('k'); ax.set_ylabel('Inertia'); ax.set_title('Elbow Method')
    return fig

def describe_clusters(cluster_df):
    return cluster_df.groupby('Cluster').mean().round(2)

# ---------------- Association Rules ----------------
def run_association_rules(df, cols, min_support, min_conf):
    from mlxtend.frequent_patterns import apriori, association_rules
    transactions = df[cols].astype(str).str.get_dummies(sep=', ')
    frequent = apriori(transactions, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric='confidence', min_threshold=min_conf)
    return rules

# ---------------- Regression ----------------
def prepare_regression_data(df):
    mapping = {
        'Less than 500': 250,
        '500 – 1,000': 750,
        '1,001 – 5,000': 3000,
        '5,001 – 10,000': 7500,
        'More than 10,000': 12000
    }
    temp = df.copy()
    temp['SpendBin'] = temp['WillingnessToSpendPerMonth'].map(mapping)
    temp = temp.dropna(subset=['SpendBin'])
    cat_cols = temp.select_dtypes(include=['object']).columns
    cat_cols = [c for c in cat_cols if c not in ['WillingnessToSpendPerMonth']]
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc_df = pd.DataFrame(enc.fit_transform(temp[cat_cols]), columns=enc.get_feature_names_out(cat_cols))
    final = pd.concat([temp[['SpendBin']].reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)
    return final

def simple_train_test_split(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=0)

def evaluate_regressors(models, X_train, X_test, y_train, y_test):
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            'Model': name,
            'RMSE': mean_squared_error(y_test, pred, squared=False),
            'R2': r2_score(y_test, pred)
        })
    return pd.DataFrame(rows).round(3)
