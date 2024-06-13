import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from pypdf import PdfReader
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


df = pd.read_csv('UpdatedResumeDataSet.csv')

def cleaning(text):
    text = re.sub(r"(https?://[^\s]+)", " ", text)
    text = re.sub(r"(?:RT|cc|#\S+|@\S+)", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['cleaned'] = df['Resume'].apply(lambda x: cleaning(x))
df['cleaned'] = df['cleaned'].str.lower()

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['cleaned'] = df['cleaned'].apply(preprocess)


tfidf = TfidfVectorizer()
resume = tfidf.fit_transform(df['cleaned'])


le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])


X_train, X_test, y_train, y_test = train_test_split(resume, df['Category'], test_size=0.2, random_state=45)


clf=SVC()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
acc=accuracy_score(y_test,y_pred)


with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)


with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

def prediction(pdf_path, vectorizer):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            reader = PdfReader(pdf_file)
            text=""
            for page in reader.pages:
                text+=page.extract_text()
            text = preprocess(text)
            text_vectorized = vectorizer.transform([text])
            pred = clf.predict(text_vectorized)
            category = le.inverse_transform(pred)
            print(f"Predicted Category: {category[0]}")
    except FileNotFoundError:
        print(f"Error: Could not find file '{pdf_path}'.")


pdf_path = r"C:\Users\neash\Desktop\eashwar\N-Eashwar-Resume-1.pdf"
prediction(pdf_path, vectorizer)
