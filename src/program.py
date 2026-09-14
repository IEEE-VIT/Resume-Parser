import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np
import re
import nltk
import xgboost as xgb
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from pdfminer.high_level import extract_text
import pickle
import warnings

import argparse
import sys

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

df = pd.read_csv('UpdatedResumeDataSet.csv')

def cleaning(text):
    text = re.sub(r"(https?://[^\s]+)", "", text)
    text = re.sub(r"(?:RT|cc|#\S+|@\S+)", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"[^\x00-\x7F]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['cleaned'] = df['Resume'].apply(lambda x: cleaning(str(x)))
df['cleaned'] = df['cleaned'].str.lower()

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

skilllist = [
    "Java", "Python", "C++", "JavaScript", "C#", "PHP", "Ruby", "Go", "Swift", "Kotlin",
    "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Apache", "Nginx",
    "MySQL", "Oracle", "PostgreSQL", "MongoDB", "Cassandra", "Redis",
    "AWS", "Azure", "GCP", "Bitcoin", "Ethereum", "Solidity", "C",
    "Git", "version control systems", "Jenkins", "GitLab CI/CD", "AUTO CAD", "Fanuc Series",
    "TCP/IP", "network protocols", "routing", "firewalls",
    "Network security principles", "encryption", "IDS",
    "Windows (Server and Desktop)", "Linux (Ubuntu, Red Hat, CentOS)", "macOS",
    "Visual Studio", "PyCharm", "IntelliJ IDEA", "VS Code", "Sublime Text", "debuggers",
    "unit testing frameworks", "Requirements gathering", "process modeling",
    "stakeholder management", "data analysis", "Agile", "Waterfall", "MS Project", "Jira",
    "Lead generation", "market research", "sales strategy", "negotiation", "communication",
    "Supply chain management", "logistics", "Lean Six Sigma", "data-driven decision making",
    "Digital marketing", "social media marketing", "content marketing", "SEO", "SEM",
    "customer relationship management (CRM)", "SPIN Selling", "prospecting", "closing deals",
    "financial statements", "budgeting", "financial modeling", "risk management",
    "Recruitment", "onboarding", "performance management", "compensation and benefits",
    "employee relations", "Crisis communication", "media relations", "reputation management",
    "brand storytelling", "CRM systems (Salesforce, HubSpot)", "Asana", "Trello", "Tableau",
    "Power BI", "Communication", "problem-solving", "critical thinking", "leadership",
    "teamwork", "time management", "organization", "Design principles", "color theory",
    "typography", "layout", "Adobe Photoshop", "Illustrator", "InDesign", "UI design",
    "UX design", "Figma", "Sketch", "Adobe XD", "Adobe Premiere Pro", "motion graphics",
    "Canva", "Piktochart", "Linear algebra", "calculus", "probability", "statistics",
    "hypothesis testing", "regression analysis", "Supervised learning", "unsupervised learning",
    "deep learning", "Data cleaning", "EDA", "Spark", "Hadoop", "R", "Jupyter Notebook",
    "RStudio", "AWS SageMaker", "Azure Machine Learning", "Microsoft Office Suite",
    "Word", "Excel", "PowerPoint", "Cybersecurity Awareness"
]

def extract_skills(text):
    extracted_skills = []
    for skill in skilllist:
        pattern = r'\b{}\b'.format(re.escape(skill.lower()))
        if re.search(pattern, text, re.IGNORECASE):
            extracted_skills.append(skill)
    return extracted_skills

if 'Skills' not in df.columns:
    df['Skills'] = df['cleaned'].apply(extract_skills)

df['Skills'] = df['Skills'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
df_filtered = df[df['Skills'].notnull() & df['Category'].notnull()]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df_filtered['Skills'])

df['Category'] = df['Category'].fillna('Unknown')

le = LabelEncoder()
y = le.fit_transform(df_filtered['Category'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifiers
clf_svm = SVC()
clf_svm.fit(X_train, y_train)

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)

def extract_text_from_file(file_path):
    file_path = str(file_path)
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return extract_text(file_path)
    if suffix in {".txt", ".md", ".csv"}:
        for encoding in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                return Path(file_path).read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported resume format: {suffix}")

def prediction(file_path, vectorizer):
    text = extract_text_from_file(file_path)
    cleaned_text = cleaning(text.strip())
    text_skills = extract_skills(cleaned_text)
    text_skills_str = ' '.join(text_skills)
    text_vectorized = vectorizer.transform([text_skills_str])
    prediction = rf_clf.predict(text_vectorized)
    predicted_category = le.inverse_transform(prediction)
    return predicted_category[0]

# --- Command-line interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a resume PDF into categories.")
    parser.add_argument("resume_path", nargs="?", help="Path to the resume PDF file to classify")
    args = parser.parse_args()

    if not args.resume_path:
        print("Usage: python program.py <path_to_resume.pdf>")
        sys.exit(1)

    resume_file = Path(args.resume_path)
    if not resume_file.exists():
        print(f"Error: File not found at {resume_file}")
        sys.exit(1)

    category = prediction(str(resume_file), tfidf)
    print("The category is:", category)
