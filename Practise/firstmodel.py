import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


#loaded the dataset 
sheet2 = pd.read_excel("datasets111.xlsx", sheet_name="Sheet2")

sheet3 = pd.read_excel("datasets111.xlsx", sheet_name="Sheet3")


#print(sheet2)
#print(sheet3)

#convert into numerical 
encoder = LabelEncoder()

print(encoder)

for column in sheet2.columns:
    sheet2[column] = encoder.fit_transform(sheet2[column])


X = sheet2.iloc[:, :-1]
y = sheet2.iloc[:, -1]


print(X)
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Initialize Gaussian Naive Bayes classifier
naive_bayes = GaussianNB()
naive_bayesM= MultinomialNB()
naive_bayesB = BernoulliNB()

# Train the classifier
naive_bayes.fit(X_train, y_train)
naive_bayesM.fit(X_train, y_train)
naive_bayesB.fit(X_train, y_train)


# Predict the labels for test data
y_pred = naive_bayes.predict(X_test)
y_predM =naive_bayesM.predict(X_test)
y_predB =naive_bayesB.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracyM = accuracy_score(y_test, y_predM)
accuracyB = accuracy_score(y_test, y_predB)
print("Accuracy:", accuracy)
print("Accuracy:", accuracyM)
print("Accuracy:", accuracyB)
