import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('data.csv')

# Drop rows with missing values in specified columns
df.dropna(subset=['PEFR Status', 'BP_S (50-250)', 'BP_D (30-200)', 'BloodSugarReportStatus', 'Abdominal Status'], inplace=True)

# Fill missing values in 'BSR (50-550mg/dL)' column with the median
df['BSR (50-550mg/dL)'].fillna(df['BSR (50-550mg/dL)'].median(), inplace=True)

# Drop additional specified columns after encoding
df.drop(['PERSON NAME', 'so/wo', 'Date of Birth', 'REGISTRATION_QULIFICATION', 'BloodPressureStatus', 'Weight (20-200Kg)', 'HeightInInch (31-78)', 'Body Mass Index', 'Hip (9.8-59Inches)', 'BloodSugarReportStatus', 'Hip Waist Status'], axis=1, inplace=True)

encoder = LabelEncoder()    
df['Abdominal Status'] = encoder.fit_transform(df['Abdominal Status'])

# Define the target column and feature columns
target_column = 'Abdominal Status'
X = df.drop(columns=[target_column])
y = df[target_column]

# Preprocessing pipeline for numerical features
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Impute missing values
    ('scaler', StandardScaler())  # Standardize features
])

# Preprocessing pipeline for categorical features (already label encoded)
categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode
])

# Combine numerical and categorical pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, X.select_dtypes(include=['float64', 'int64']).columns.tolist()),
        ('cat', categorical_pipeline, X.select_dtypes(include=['object', 'category']).columns.tolist())
    ])

# Define the models to evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'AdaBoost': AdaBoostClassifier(),
    'Support Vector Classifier': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Initialize an empty dictionary to store accuracy results
accuracy_results = {}

from sklearn.model_selection import cross_val_score

# Initialize an empty dictionary to store results
accuracy_results = {}

# Loop through the models and evaluate each
for model_name, model in models.items():
    # Create the final pipeline with the preprocessor and the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model_pipeline.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results[model_name] = accuracy
    
    print(f'{model_name} Accuracy: {accuracy:.2f}')


# Save the pipeline
from joblib import dump

knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(
        n_neighbors=15,
        weights='distance',
        metric='manhattan'
    ))
])

# Fit the model on the entire dataset
knn_pipeline.fit(X, y)

# Save the pipeline
dump(knn_pipeline, 'knn_pipeline.joblib')

from joblib import load

# Load the trained K-Nearest Neighbors pipeline
knn_pipeline = load('knn_pipeline.joblib')

# Example of new unseen data (ensure this data has the same columns as the training data, except the target column)
new_data = pd.DataFrame({
    'Age': [55],
    'REGISTRATION_GENDER': ['Male'],
    'REGISTRATION_MARITAL_STS': ['Married'],
    'BSR (50-550mg/dL)': [145],
    'BP_S (50-250)': [140],
    'BP_D (30-200)': [90],
    'Waist (9.8-59Inches)': [34],
    'Waist-Hip Ratio': [0.84],
    'Body Mass Index Status': ['Normal'],
    'PEFR Status': ['Low']
})

# Make predictions on the new unseen data
predictions = knn_pipeline.predict(new_data)
probabilities = knn_pipeline.predict_proba(new_data)

# Print predictions with confidence scores
for i, prediction in enumerate(predictions):
    print(f'Prediction: {prediction}, Confidence Score: {probabilities[i][prediction]:.2f}')

