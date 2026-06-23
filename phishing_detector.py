import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load dataset
try:
    data = pd.read_csv("emails.csv")
except FileNotFoundError:
    print("Error: emails.csv file not found!")
    exit()

# Check columns
if "text" not in data.columns or "label" not in data.columns:
    print("CSV must contain 'text' and 'label' columns")
    exit()

# Text preprocessing
def preprocess(text):
    text = str(text).lower()

    # Replace URLs
    text = re.sub(r"http\S+|www\S+", " URL ", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text

data["text"] = data["text"].apply(preprocess)

# Features and Labels
X = data["text"]
y = data["label"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("PHISHING EMAIL DETECTION MODEL")
print("==============================")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, str(cm[i][j]),
                 ha='center', va='center')

plt.tight_layout()
plt.show()

# Save Report
with open("report.txt", "w") as file:
    file.write("PHISHING EMAIL DETECTION REPORT\n")
    file.write("=" * 40 + "\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(classification_report(y_test, y_pred))

print("\nReport saved as report.txt")

# Live Testing
while True:

    user_email = input(
        "\nEnter Email Text (type 'exit' to quit): "
    )

    if user_email.lower() == "exit":
        break

    processed_email = preprocess(user_email)

    email_vector = vectorizer.transform(
        [processed_email]
    )

    prediction = model.predict(email_vector)[0]

    probability = max(
        model.predict_proba(email_vector)[0]
    )

    print("\nPrediction:", prediction.upper())
    print(
        f"Confidence: {probability * 100:.2f}%"
    )

    if prediction == "phishing":
        print("⚠️ WARNING: Possible Phishing Email")
    else:
        print("✅ Safe Email")

print("\nProgram Ended")