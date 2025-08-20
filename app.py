import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load dataset and preprocess
df = pd.read_csv('Telco-Customer-Churn.csv')
df.drop(['customerID'], axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

target = 'Churn'

# Selected features list
features = [
    'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

df = df[features + [target]]

# Convert SeniorCitizen to string for categorical processing
categorical = df.select_dtypes(include='object').columns.tolist()
if 'SeniorCitizen' in features and 'SeniorCitizen' not in categorical:
    categorical.append('SeniorCitizen')

numeric = list(set(features) - set(categorical))

df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

# Save features list for Streamlit use
joblib.dump(features, 'model_features.pkl')

# Prepare data for training
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', SimpleImputer(strategy='median'), numeric)
])

# Random Forest pipeline and expanded hyperparameter grid
rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])

rf_params = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2],
    'clf__max_features': ['sqrt', 'log2']
}

rf_random = RandomizedSearchCV(
    rf_pipe, rf_params, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1
)
rf_random.fit(X_train, y_train)

best_rf_pipe = rf_random.best_estimator_

print("📊 Tuned Random Forest Results:")
rf_preds = best_rf_pipe.predict(X_test)

print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("F1 Score:", f1_score(y_test, rf_preds))
print("ROC-AUC:", roc_auc_score(y_test, rf_preds))

# Save model pipeline
joblib.dump(best_rf_pipe, 'best_rf_model.pkl')

print("\n✅ Model pipeline and features saved successfully.")