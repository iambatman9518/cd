import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"NaiveBdata.csv")

# Encode labels (Spam -> 1, Ham -> 0)
le = LabelEncoder()
df['class'] = le.fit_transform(df['label'])

# Vectorize text data using TfidfVectorizer
ve = TfidfVectorizer(stop_words='english')
X = ve.fit_transform(df['text'])
y = df['class']

# Multinomial Naive Bayes
mn = MultinomialNB()
mn.fit(X, y)

# Cross-validation predictions
y_pred = cross_val_predict(mn, X, y)

# Metrics for Multinomial Naive Bayes
print("Multinomial Naive Bayes:\n")
print("accuracy: ", accuracy_score(y, y_pred))
print("recall: ", recall_score(y, y_pred))
print("f1_score: ", f1_score(y, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y, y_pred, target_names=["Ham", "Spam"], zero_division=0))

# Scatter plot of message length vs label
plt.figure(figsize=(8, 5))
plt.scatter(df['class'], df['text'].apply(len), alpha=0.5, c=df['class'], cmap='coolwarm')
plt.xlabel("Label (0 = Ham, 1 = Spam)")
plt.ylabel("Message Length")
plt.title("Scatter Plot of Label vs Message Length")
plt.grid(True)
plt.show()

# Sample message prediction using Multinomial Naive Bayes
sample_message = input("Input Text: ")
sm = ve.transform([sample_message])
re = mn.predict(sm)

if re[0] == 0:
    print("Not Spam")
else:
    print("Spam!")
