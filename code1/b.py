import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('college.csv')

# Strip spaces from column names (important!)
df.columns = df.columns.str.strip()

# Debug: print actual column names
print("Actual column names:", df.columns.tolist())

# Encode categorical columns
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])         # Male=1, Female=0
df['Stream'] = le.fit_transform(df['Stream'])         # Stream to numbers

# Check if 'placed' column exists
if 'placed' not in df.columns:
    raise ValueError("The column 'placed' is not found in the dataset. Please check the CSV headers.")

# Separate features and target
x = df.drop(['placed'], axis=1)
y = df['placed']

# Normalize feature values
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define models to train
models = {
    'SVC': SVC(),
    'DecisionTree': DecisionTreeClassifier(),
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(n_estimators=50),
    'KNeighbors': KNeighborsClassifier()
}

# Train and evaluate models
for name, model in models.items():
    print(f'\nTraining model: {name}')
    scores = cross_val_score(model, x, y, cv=3)
    print(f'{name} Cross-Validation Accuracy: {scores.mean():.4f}')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{name} - Confusion Matrix')
    plt.show()

    print(f"{name} Training Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"{name} Testing Accuracy: {model.score(X_test, y_test):.4f}")
