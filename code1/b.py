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

df = pd.read_csv('college.csv')

print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.duplicated().sum())
print(df.isnull().sum())
print(df.info())
print(df.describe())
print(df.nunique())

#plt.figure(figsize=(15, 6))
#sns.countplot(x='Age', data=df, palette='hls')
#plt.show()

#plt.figure(figsize=(15, 6))
#sns.histplot(df['Age'], kde=True, bins=10)
#plt.show()

#plt.figure(figsize=(15, 6))
#sns.countplot(x='Gender', data=df, palette='hls')
#plt.show()

#plt.figure(figsize=(10, 6))
#label_data = df['Gender'].value_counts()
#plt.pie(label_data, labels=label_data.index, colors=['blue', 'red'],pctdistance=0.65, shadow=True, startangle=0,explode=(0.0, 0.1), autopct='%1.1f%%')
#plt.title('Gender Distribution', size=20)
#plt.show()

#plt.figure(figsize=(15, 6))
#sns.countplot(x='Stream', data=df, palette='hls')
#plt.show()

#plt.figure(figsize=(15, 6))
#sns.barplot(x='Stream', y='PlacedOrNot', data=df)
#plt.show()

#plt.figure(figsize=(15, 6))
#sns.countplot(x='Internships', data=df, palette='hls')
#plt.show()

#plt.figure(figsize=(10, 6))
#label_data = df['Internships'].value_counts()
#plt.pie(label_data, labels=label_data.index, colors=['blue', 'red', 'green', 'orange'],pctdistance=0.65, shadow=True, startangle=0,explode=(0.0, 0.1, 0.1, 0.1), autopct='%1.1f%%')
#plt.title('Internships Distribution', size=20)
#plt.show()

#plt.figure(figsize=(15, 6))
#sns.countplot(x='CGPA', data=df, palette='hls')
#plt.show()

#plt.figure(figsize=(10, 6))
#label_data = df['CGPA'].value_counts()
#plt.pie(label_data, labels=label_data.index, colors=['blue', 'red', 'green', 'orange', 'violet'],pctdistance=0.65, shadow=True, startangle=0,explode=(0.0, 0.1, 0.1, 0.1, 0.1), autopct='%1.1f%%')
#plt.title('CGPA Distribution', size=20)
#plt.show()

#plt.figure(figsize=(15, 6))
#sns.countplot(x='Hostel', data=df, palette='hls')
#plt.show()

#plt.figure(figsize=(10, 6))
#label_data = df['Hostel'].value_counts()
#plt.pie(label_data, labels=label_data.index, colors=['blue', 'red'],pctdistance=0.65, shadow=True, startangle=0,explode=(0.0, 0.1), autopct='%1.1f%%')
#plt.title('Hostel Distribution', size=20)
#plt.show()

#plt.figure(figsize=(15, 6))
#sns.countplot(x='HistoryOfBacklogs', data=df, palette='hls')
#plt.show()

#plt.figure(figsize=(10, 6))
#label_data = df['HistoryOfBacklogs'].value_counts()
#plt.pie(label_data, labels=label_data.index, colors=['blue', 'red'],pctdistance=0.65, shadow=True, startangle=0,explode=(0.0, 0.1), autopct='%1.1f%%')
#plt.title('History of Backlogs', size=20)
#plt.show()

#plt.figure(figsize=(15, 6))
#sns.countplot(x='PlacedOrNot', data=df, palette='hls')
#plt.show()

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Stream'] = le.fit_transform(df['Stream'])

x = df.drop(['placed or not'], axis=1)
y = df['placed or not']

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

models = {
    'SVC': SVC(),
    'DecisionTree': DecisionTreeClassifier(),
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(n_estimators=50),
    'KNeighbors': KNeighborsClassifier()
}

for name, model in models.items():
    scores = cross_val_score(model, x, y, cv=3)
    print(f'{name}: {scores.mean()}')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    

    print("Training Accuracy:", model.score(X_train, y_train))
    print("Testing Accuracy:", model.score(X_test, y_test))
