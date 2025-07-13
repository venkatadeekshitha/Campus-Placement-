import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


# Load data
data = pd.read_csv("college.csv")
print(data)

# Assuming "Gender" and "Stream" are categorical variables
# Encode categorical variables
label_encoder = LabelEncoder()
data["Gender"] = label_encoder.fit_transform(data["Gender"])
data["Stream"] = label_encoder.fit_transform(data["Stream"])

# Features and target
x = data[["Age", "Gender", "Stream", "Internships", "CGPA", "Hostel", "History of backlogs"]]
y = data["placed "]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Logistic Regression model
logreg_model = LogisticRegression()
logreg_model.fit(x_train, y_train)
y_pred_logreg = logreg_model.predict(x_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("Logistic Regression Accuracy:", accuracy_logreg*100)
classification_report_logreg = classification_report(y_test, y_pred_logreg)
print("Logistic Regression Classification Report:\n", classification_report_logreg)

# Random Forest Classifier model
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Classifier Accuracy:", accuracy_rf*100)
classification_report_rf = classification_report(y_test, y_pred_rf)
print("Random Forest Classifier Classification Report:\n", classification_report_rf)
model=RandomForestClassifier()
model.fit(x_train,y_train)

# Naive Bayes model
NB_model = GaussianNB()
NB_model.fit(x_train, y_train)
y_pred_NB = NB_model.predict(x_test)
accuracy_NB = accuracy_score(y_test, y_pred_NB)
print("GaussianNB Accuracy:", accuracy_NB*100)
classification_report_NB = classification_report(y_test, y_pred_NB)
print("GaussianNB Classification Report:\n", classification_report_NB)

# SVM (Support Vector Machine) model
svm_model = SVC()
svm_model.fit(x_train, y_train)
y_pred_svm = svm_model.predict(x_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm*100)
classification_report_svm = classification_report(y_test, y_pred_svm)
print("SVM Classification Report:\n", classification_report_svm)
joblib.dump(model,"mymodel.h5")




