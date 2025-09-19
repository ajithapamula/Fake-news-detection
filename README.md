```
## 📰 Fake News Detection

This project uses machine learning and NLP to detect whether a news article is fake or real.
```
## 🔍 Features
```
- Preprocessing of raw text using regular expressions
- TF-IDF vectorization for feature extraction
- Multiple model training and evaluation:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Manual testing interface to test user-input news in real-time
- Evaluation using classification report and accuracy score
```
## 🧠 Technologies Used
```
- Python
- Pandas, NumPy
- Scikit-learn
- TfidfVectorizer
- Regex
- Matplotlib
```
## 🧪 Model Performance
```
Each model prints a classification report showing:
- Precision
- Recall
- F1-score
- Accuracy
```
## 🖥️ Manual Testing
You can uncomment the input loop to test news manually:
```python
while True:
    news = input("Enter news text: ")
    manual_testing(news)
