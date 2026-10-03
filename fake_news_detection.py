import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Synthetic Dataset Generation
np.random.seed(42)

sample_fake_news = [
    "Breaking: Aliens landed in New York City yesterday!",
    "Secret cure for aging found in backyard weeds!",
    "Government announces free gold coins for every citizen.",
    "Scientists discover drinking water turns hair green.",
    "Celebrity claims eating glass improves immunity."
] * 100

sample_real_news = [
    "Central Bank raises benchmark interest rate by 25 basis points.",
    "NASA successfully launches new weather monitoring satellite.",
    "Global renewable energy capacity increased by 10 percent this year.",
    "Tech companies report quarterly earnings amid market shift.",
    "New highway project completed to ease urban traffic flow."
] * 100

articles = sample_fake_news + sample_real_news
labels = ['FAKE'] * len(sample_fake_news) + ['REAL'] * len(sample_real_news)

df = pd.DataFrame({'text': articles, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 3. Text Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# 4. Model Training (Passive Aggressive Classifier)
pac = PassiveAggressiveClassifier(max_iter=50, random_state=42)
pac.fit(tfidf_train, y_train)

# 5. Model Evaluation
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print(f"--- Model Accuracy: {score*100:.2f}% ---")
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 6. Sample Prediction Test
sample_input = ["Breaking: Government gives free cars to everyone!"]
sample_tfidf = tfidf_vectorizer.transform(sample_input)
prediction = pac.predict(sample_tfidf)
print(f"\nSample Input: '{sample_input[0]}'")
print(f"Prediction: {prediction[0]}")
