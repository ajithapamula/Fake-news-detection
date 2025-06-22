# fake_news_detection/main.py

import pandas as pd
import numpy as np
import re
import string
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# -----------------------------
# Load and preprocess data
# -----------------------------
df_fake = pd.read_csv("fake.csv")
df_true = pd.read_csv("true.csv")

df_fake["class"] = 0
df_true["class"] = 1

# Remove last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
df_fake = df_fake.iloc[:-10]

df_true_manual_testing = df_true.tail(10)
df_true = df_true.iloc[:-10]

df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
df_manual_testing.to_csv("manual_testing.csv")

df_merge = pd.concat([df_fake, df_true], axis=0)
df = df_merge.drop(["title", "subject", "date"], axis=1)
df = df.sample(frac=1).reset_index(drop=True)

# -----------------------------
# Text Preprocessing Function
# -----------------------------
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df["text"] = df["text"].apply(wordopt)

# -----------------------------
# Train/Test Split
# -----------------------------
x = df["text"]
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# TF-IDF Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# -----------------------------
# Train Models
# -----------------------------
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
print("\n[Logistic Regression Results]")
print(classification_report(y_test, pred_lr))

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
print("\n[Decision Tree Results]")
print(classification_report(y_test, pred_dt))

GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
pred_gbc = GBC.predict(xv_test)
print("\n[Gradient Boosting Results]")
print(classification_report(y_test, pred_gbc))

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
print("\n[Random Forest Results]")
print(classification_report(y_test, pred_rfc))

# -----------------------------
# Manual Testing Function
# -----------------------------
def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_df = pd.DataFrame(testing_news)
    new_df["text"] = new_df["text"].apply(wordopt)
    new_xv = vectorization.transform(new_df["text"])

    print("\n[Manual Testing Results]")
    print("LR Prediction:", output_label(LR.predict(new_xv)[0]))
    print("DT Prediction:", output_label(DT.predict(new_xv)[0]))
    print("GBC Prediction:", output_label(GBC.predict(new_xv)[0]))
    print("RFC Prediction:", output_label(RFC.predict(new_xv)[0]))

# -----------------------------
# Input Prompt (Uncomment to Test)
# -----------------------------
# while True:
#     news = input("\nEnter news text: ")
#     manual_testing(news)
