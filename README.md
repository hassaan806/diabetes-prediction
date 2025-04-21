# 🩺 Diabetes Prediction Web Application

This project presents a machine learning-based web application designed to predict the likelihood of an individual being diabetic. By inputting key health metrics—**glucose level**, **BMI**, and **age**—users receive an assessment indicating their risk of diabetes. The application leverages a **Random Forest Classifier** trained on a curated dataset to provide accurate predictions.

---

## 📊 Overview

- **Objective**: Predict diabetes risk based on user-provided health parameters.
- **Model**: Random Forest Classifier.
- **Framework**: Flask for web application development.
- **Dataset**: Sourced from Kaggle, focusing on relevant features.

---

## 🛠️ Features

- **User-Friendly Interface**: Input health metrics easily through a web form.
- **Real-Time Prediction**: Immediate feedback on diabetes risk.
- **Data Preprocessing**:
  - Removal of outliers and irrelevant features.
  - Feature selection focusing on glucose, BMI, and age.
  - Data balancing using SMOTE to address class imbalance.
  - Normalization with StandardScaler for consistent feature scaling.
- **Model Training**: Utilizes RandomForestClassifier for robust prediction capabilities.

---

## 🧪 How It Works

1. **Data Collection**: The dataset is obtained from Kaggle, containing various health-related features.
2. **Data Preprocessing**:
   - Outliers and non-contributing features (e.g., Pregnancies, BloodPressure) are removed.
   - The dataset is balanced using SMOTE to ensure equal representation.
   - Features are normalized to improve model performance.
3. **Model Training**:
   - A Random Forest Classifier is trained on the processed data.
   - The model learns patterns associated with diabetic and non-diabetic individuals.
4. **Web Application**:
   - Built with Flask, the app provides a simple interface for users to input their health metrics.
   - Upon submission, the app processes the input and displays the prediction result.

---

## 📁 Project Structure

diabetes-prediction/ ├── app.py # Flask web application ├── diabetes-prediction.ipynb # Jupyter notebook for data analysis and model training ├── data.csv # Dataset used for training ├── requirements.txt # Python dependencies ├── .gitignore # Files and directories to be ignored by Git


---

## 📌 Notes

- The application focuses on three primary features: glucose level, BMI, and age.
- The model is trained to provide a binary prediction: diabetic or not diabetic.
- For a more comprehensive analysis, additional features and more complex models can be incorporated.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- The open-source community for continuous support and resources.
