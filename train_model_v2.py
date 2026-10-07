import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")


# ==========================================
# 2. ADD LABELS
# ==========================================

fake["label"] = 0
true["label"] = 1


# ==========================================
# 3. COMBINE DATA
# ==========================================

data = pd.concat(
    [fake, true],
    ignore_index=True
)

data = data.drop_duplicates()


# ==========================================
# 4. REMOVE MISSING VALUES
# ==========================================

data = data.dropna(
    subset=["title", "text"]
)


# ==========================================
# 5. COMBINE TITLE + ARTICLE
# ==========================================

data["content"] = (
    data["title"].astype(str)
    + " "
    + data["text"].astype(str)
)


X = data["content"]
y = data["label"]


# ==========================================
# 6. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 7. WORD + CHARACTER TF-IDF
# ==========================================

word_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1, 2),
    sublinear_tf=True
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    sublinear_tf=True
)


vectorizer = FeatureUnion([
    ("word", word_vectorizer),
    ("char", char_vectorizer)
])


X_train_vectorized = vectorizer.fit_transform(X_train)

X_test_vectorized = vectorizer.transform(X_test)


# ==========================================
# 8. TRAIN MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_vectorized,
    y_train
)


# ==========================================
# 9. TEST MODEL
# ==========================================

y_pred = model.predict(
    X_test_vectorized
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n===================================")
print("IMPROVED MODEL TRAINING COMPLETED")
print("===================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 10. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "model/fake_news_model_v2.pkl"
)

joblib.dump(
    vectorizer,
    "model/vectorizer_v2.pkl"
)


print("\n===================================")
print("MODEL SAVED SUCCESSFULLY")
print("===================================")

print("model/fake_news_model_v2.pkl")
print("model/vectorizer_v2.pkl")