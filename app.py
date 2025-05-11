import pandas as pd
import random
import uuid
import re
import string
import nltk 
import sys
sys.stdout.reconfigure(encoding='utf-8')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

# Load and clean the dataset
file_path = "enron_spam_data.csv"
df = pd.read_csv(file_path)
df["Spam/Ham"] = df["Spam/Ham"].str.strip().str.capitalize()
df["spam"] = df["Spam/Ham"].map({"Ham": 0, "Spam": 1})
df = df.dropna(subset=["spam"])
df["subject"] = df["Subject"].fillna("")
df["body"] = df["Message"].fillna("")

# Prepare stopwords once
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Apply cleaning
df["clean_subject"] = df["subject"].apply(clean_text)
df["clean_body"] = df["body"].apply(clean_text)
df["text"] = df["clean_subject"] + " " + df["clean_body"]

# Vectorize and train
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"])
y = df["spam"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy =100*accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Model training complete!")

# Predict function
def predict_email(subject, body):
    email_text = clean_text(subject) + " " + clean_text(body)
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    print("\nEmail:")
    print("Subject:", subject)
    print("Body:", body)
    print("Prediction:", result)
    return result

# Test example
sample_subject = "Meeting request for project discussion"
sample_body = "Dear team, I would like to schedule a meeting to discuss the upcoming project. Please let me _know your availability. Best regards, John Doe."
prediction_result = predict_email(sample_subject, sample_body)

#Not Spam
#Meeting request for project discussion
#Dear team, I would like to schedule a meeting to discuss the upcoming project. Please let me know your availability. Best regards, John Doe.

#Spam
#Congratulations! You've won a $1000 gift card 🎁
#Click here to claim your reward now: http://scamlink.com. This is a limited-time offer! Act fast and don't miss out on your free prize.


