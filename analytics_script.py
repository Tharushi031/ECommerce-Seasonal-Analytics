"""
The Impact of Seasonal Sales Trends on Consumer Purchasing Behavior and E-commerce Business Performance
An Empirical Study Using the UCI Online Shoppers Intention Dataset

Author: S.D.T.S.Gimhami
Analytical Tools: Python (pandas, scikit-learn, matplotlib, seaborn, scipy)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix

# ─── 1. LOAD DATASET ─────────────────────────────────────────────────────────
# Download the dataset from the UCI Machine Learning Repository or place 'online_shoppers_intention.csv' in your directory.
try:
    df = pd.read_csv('online_shoppers_intention.csv')
    print("Dataset loaded successfully. Shape:", df.shape)
except FileNotFoundError:
    print("Error: 'online_shoppers_intention.csv' file not found. Please place it in the working directory.")
    # Exiting execution if file isn't present
    import sys; sys.exit()

# ─── 2. DATA PRE-PROCESSING ──────────────────────────────────────────────────
print("\n--- Starting Data Preprocessing ---")

# Encode categorical variables using Label Encoding
le_month = LabelEncoder()
df['Month_encoded'] = le_month.fit_transform(df['Month'])

le_visitor = LabelEncoder()
df['VisitorType_encoded'] = le_visitor.fit_transform(df['VisitorType'])

# Convert Boolean variables to binary integers (0 and 1)
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)

# Define feature columns and target variable for Logistic Regression
X = df.drop(columns=['Revenue', 'Month', 'VisitorType'])
y = df['Revenue']

# Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data preprocessing completed.")

# ─── 3. LOGISTIC REGRESSION MODELING ─────────────────────────────────────────
print("\n--- Training Logistic Regression Model ---")

# Handle the 84:16 class imbalance using class_weight="balanced"
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = lr_model.predict(X_test_scaled)
y_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

# Model Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_prob)

print(f"Overall Accuracy: {accuracy:.4f}")
print(f"AUC-ROC Score: {auc_roc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Extract Standardized Coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient (Beta)': lr_model.coef_[0]
}).sort_values(by='Coefficient (Beta)', ascending=False)

print("\nTop Predictor Coefficients:")
print(coefficients)

# ─── 4. K-MEANS CLUSTERING (MARKET SEGMENTATION) ─────────────────────────────
print("\n--- Executing K-Means Market Segmentation ---")

# Behavioral features selected for clustering
cluster_features = [
    "ProductRelated", "ProductRelated_Duration", 
    "BounceRates", "ExitRates", "PageValues", 
    "Administrative", "Administrative_Duration"
]

X_clustering = df[cluster_features]

# Scale features specifically for clustering
scaler_cl = StandardScaler()
X_cl_scaled = scaler_cl.fit_transform(X_clustering)

# Determine Elbow Curve values (k=2 to k=9)
wcss = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cl_scaled)
    wcss.append(km.inertia_)

# Fit optimal K-Means model (k=3 as identified in the paper)
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = km3.fit_predict(X_cl_scaled)

print("\nSummary Profile of the 3 Behavioral Clusters:")
cluster_profiles = df.groupby("Cluster").agg({
    'Revenue': 'mean',
    'PageValues': 'mean',
    'ProductRelated': 'mean',
    'BounceRates': 'mean'
})
cluster_profiles.columns = ['Conversion Rate', 'Avg Page Value', 'Avg Product Pages', 'Avg Bounce Rate']
print(cluster_profiles)

# ─── 5. INFERENTIAL STATISTICAL TESTS ────────────────────────────────────────
print("\n--- Conducting Statistical Inference Tests ---")

# Test 1: Chi-square Test of Independence (Weekend vs Revenue)
chi2, p_val_chi2, dof, expected = stats.chi2_contingency(pd.crosstab(df["Weekend"], df["Revenue"]))
print(f"Chi-square Test (Weekend vs Revenue): χ² = {chi2:.2f}, p-value = {p_val_chi2:.4f}")

# Test 2: Mann-Whitney U Test (PageValues distribution for converting vs non-converting)
g0 = df[df["Revenue"] == 0]["PageValues"]
g1 = df[df["Revenue"] == 1]["PageValues"]
u_stat, p_val_mw = stats.mannwhitneyu(g0, g1, alternative="two-sided")
print(f"Mann-Whitney U Test (PageValues vs Revenue): U = {u_stat:,}, p-value = {p_val_mw}")

# Test 3: Point-biserial Correlation Coefficient (ProductRelated_Duration vs Revenue)
r_pb, p_val_pb = stats.pointbiserialr(df["ProductRelated_Duration"], df["Revenue"])
print(f"Point-biserial Correlation (Product Duration vs Revenue): r = {r_pb:.3f}, p-value = {p_val_pb}")

print("\nScript run execution finished successfully.")