# 🎓 Student Performance Predictor
A complete end-to-end Data Science project in Google Colab.

from google.colab import files
uploaded = files.upload()

import pandas as pd

# Load the CSV file
df = pd.read_csv('card_transdata.csv')
df.head()

# Data Exploration
df.info()
df.describe()
df.shape
df.columns

# Check for Missing Values and Duplicates
print(df.isnull().sum())
print(f'Duplicates: {df.duplicated().sum()}')

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Identify Target and Features
target = 'your_target_column_name'  # Change this to the actual target
features = [col for col in df.columns if col != target]

# Convert Categorical Columns to Numerical
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].apply(lambda x: x.astype('category').cat.codes)

# One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# Step 1: Set your actual target column
target = 'fraud'  # Replace 'fraud' with your actual target column name

# Step 2: Perform feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=target))  # X_scaled is your scaled feature matrix

# Optionally, set up the labels (y)
y = df[target]


# Train-Test Split
from sklearn.model_selection import train_test_split

X = X_scaled
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Make Predictions from New Input
sample = [0.5] * X.shape[1]  # Replace with actual feature values
prediction = model.predict([sample])
print("Prediction:", prediction)

# Convert to DataFrame and Encode
new_data = pd.DataFrame([sample], columns=df.drop(columns=target).columns)

# Predict the Final Grade (or Output)
predicted_value = model.predict(new_data)
print("Predicted:", predicted_value[0])

# Install Gradio
!pip install gradio

# Create a Prediction Function
def predict_student_performance(*inputs):
    import numpy as np
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Create the Gradio Interface
import gradio as gr

input_fields = [gr.Number(label=col) for col in df.drop(columns=target).columns]

interface = gr.Interface(fn=predict_student_performance,
                         inputs=input_fields,
                         outputs="label",
                         title="🎓 Student Performance Predictor")
interface.launch()
