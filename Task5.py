import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import make_pipeline

nltk.download('stopwords')
nltk.download('punkt')

# Step 1: Explanatory Data Analysis and Feature Engineering
url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
df = pd.read_csv(url, compression='zip', dtype=str)

# Filter relevant columns
df = df[['Product', 'Consumer complaint narrative']].dropna()

# Explore the distribution of categories
df['Product'].value_counts().plot(kind='bar', title='Distribution of Categories')

# Step 2: Text Pre-Processing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(str(text))
    words = [ps.stem(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['Consumer complaint narrative'].apply(preprocess_text)

# Step 3: Selection of Multi Classification Model
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['Product'], test_size=0.2, random_state=42)

# Using a simple Naive Bayes classifier for illustration
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Step 4: Comparison of Model Performance
y_pred = model.predict(X_test)

# Step 5: Model Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 6: Prediction
new_complaints = ["I found an error in my credit report.",
                  "A debt collector is harassing me.",
                  "I need a consumer loan for my business.",
                  "I want to inquire about mortgage options."]

new_complaints_cleaned = [preprocess_text(complaint) for complaint in new_complaints]

predictions = model.predict(new_complaints_cleaned)

for complaint, prediction in zip(new_complaints, predictions):
    print(f"Complaint: {complaint}\nPredicted Category: {prediction}\n{'='*30}")
