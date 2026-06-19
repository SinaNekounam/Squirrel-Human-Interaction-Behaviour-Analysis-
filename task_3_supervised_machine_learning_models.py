import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# FEATURES AND TARGET

target = 'human_reaction'

features = [
    # Behaviour scores
    'active_score',
    'foraging_score',
    'vocal_score',
    'alert_score',

    # Time features
    'shift_enc',
    'weekday',

    # Area features
    'zone',
    'density_level',
    'conditions_enc',
    'has_threat',
    'outlier_variable',

    # Spatial features
    'X',
    'Y',

    # Squirrel characteristics
    'Age',
    'Primary Fur Color',
    'Location'
]

merged = pd.read_csv("task_1_preprocessed_squirrel_data.csv")

X = merged[features]
y = merged[target]


print("\n---------------------------------------------------------------------------")
print("Dataset shape:", merged.shape)

print("\n---------------------------------------------------------------------------")
print("Target distribution:")
print(y.value_counts())


# FEATURE TYPES

categorical_features = [
    'zone',
    'density_level',
    'Age',
    'Primary Fur Color',
    'Location'
]

numerical_features = [
    'active_score',
    'foraging_score',
    'vocal_score',
    'alert_score',
    'shift_enc',
    'weekday',
    'conditions_enc',
    'has_threat',
    'outlier_variable',
    'X',
    'Y'
]



# PREPROCESSING

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ]
)



# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# MODEL 1: LOGISTIC REGRESSION

logistic_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=2000,
        class_weight='balanced'
    ))
])

# Train
logistic_model.fit(X_train, y_train)

# Predict
y_pred_log = logistic_model.predict(X_test)



# LOGISTIC REGRESSION EVALUATION

print("\n---------------------------------------------------------------------------")
print("LOGISTIC REGRESSION RESULTS")

print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred_log, average='weighted', zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_log, average='weighted', zero_division=0))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))



# CROSS VALIDATION

cv_scores_log = cross_val_score(
    logistic_model,
    X,
    y,
    cv=5,
    scoring='f1_weighted'
)

print("\n---------------------------------------------------------------------------")
print("\nLogistic Regression CV Scores:")
print(cv_scores_log)

print("Mean CV F1:", cv_scores_log.mean())



# MODEL 2: RANDOM FOREST

rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42
    ))
])

# Train
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)



# RANDOM FOREST EVALUATION

print("\n---------------------------------------------------------------------------")
print("RANDOM FOREST RESULTS")

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_rf, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_rf, average='weighted'))

print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))



# RANDOM FOREST CROSS VALIDATION

cv_scores_rf = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring='f1_weighted'
)

print("\n---------------------------------------------------------------------------")
print("Random Forest CV Scores:")
print(cv_scores_rf)

print("Mean CV F1:", cv_scores_rf.mean())



# FEATURE IMPORTANCE (RANDOM FOREST)

# Get encoded feature names
ohe_features = (
    rf_model.named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(categorical_features)
)

all_features = (list(ohe_features) + numerical_features)

# Get importance values
importances = (
    rf_model.named_steps['classifier']
    .feature_importances_
)

importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n---------------------------------------------------------------------------")
print("Top 10 Random Forest Features:")
print(importance_df.head(10))


# LOGISTIC REGRESSION COEFFICIENTS

coefficients = (logistic_model.named_steps['classifier'].coef_)

coef_df = pd.DataFrame(coefficients, columns=all_features)

coef_df.index = [
    'Runs Away',
    'Indifferent',
    'Approaches'
]

# SAVE OUTPUTS

# Save Random Forest feature importance
importance_df.to_csv(
    "task_3_random_forest_feature_importance.csv",
    index=False
)

# Save Logistic Regression coefficients
coef_df.to_csv(
    "task_3_logistic_regression_coefficients.csv"
)

print("\n---------------------------------------------------------------------------")
print("task_3_random_forest_feature_importance.csv was saved")
print("task_3_logistic_regression_coefficients.csv was saved")
