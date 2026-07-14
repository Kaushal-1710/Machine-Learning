from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd


# Load the Breast Cancer dataset

pima_data = pd.read_csv('pima-indians-diabetes.csv')

X = pima_data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = pima_data['Outcome']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize 
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train)
X_test1 = scaler.transform(X_test)
svc_model=SVC(kernel='linear',C=1,gamma=10)
dtc = tree.DecisionTreeClassifier(random_state=42)
naive_bayes = GaussianNB()
for k in range(6,17):
    # Instantiate learning model (k = 6 to 16)
    classifier = KNeighborsClassifier(n_neighbors=k)


# fit the model
svc_model.fit(X_train, y_train)
log_reg_model = LogisticRegression()
rf_classifier.fit(X_train, y_train)
log_reg_model.fit(X_train, y_train)
dtc.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)
classifier.fit(X_train, y_train)



# Predict the labels for test set
y_predRF = rf_classifier.predict(X_test)
y_predLo = log_reg_model.predict(X_test)
y_predSV = svc_model.predict(X_test)
y_predictDTC=dtc.predict(X_test)
y_prednb=naive_bayes.predict(X_test)
y_predKNN = classifier.predict(X_test)

# Calculate the accuracy
accuracyRF = accuracy_score(y_test, y_predRF)
accuracyLO = accuracy_score(y_test, y_predLo)
accuracySV = accuracy_score(y_test, y_predSV)
accuracyNB = accuracy_score(y_test, y_prednb)
accuracyKNN = accuracy_score(y_test, y_predKNN)
print("Accuracy Of naive byres :", accuracyKNN)
print("Accuracy Of naive byres :", accuracyNB)
print('Accuracy of Decision Tree-Test: ', accuracy_score(y_predictDTC, y_test))
print("Accuracy Of randomforest :", accuracyRF)
print("Accuracy of Logistic Regression :", accuracyLO)
print("Accuracy Of SVM :", accuracySV)


# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_predRF))
