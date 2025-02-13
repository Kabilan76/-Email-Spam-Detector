import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset (Replace with your dataset if needed)
url = "https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/spam.csv"
data = pd.read_csv(url, encoding='latin-1')

# Keep only necessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels to binary (spam=1, ham=0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naïve Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# Print accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Function to predict spam or not
def predict_spam(email_text):
    email_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(email_tfidf)[0]
    return "Spam 🚨" if prediction == 1 else "Not Spam ✅"

# Test the model
email = input("Enter an email message: ")
print("Prediction:", predict_spam(email))
