# watchdog_core.py
"""
Smart Ingredient Watchdog - Core Engine
This file contains the IngredientWatchdog class with:
- Dataset loading
- Database integration (SQLite)
- OCR & NLP preprocessing
- Machine Learning model training & inference
- Ingredient risk analysis
"""

import os
import re
import pickle
import sqlite3
import numpy as np
import pandas as pd
import cv2
import pytesseract
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Download NLTK data (first time)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass


class IngredientWatchdog:
    def __init__(self):
        self.db_path = "ingredient_watchdog.db"
        self.dataset_path = r"C:\Users\lavan\OneDrive\Desktop\project\AIML intern\data\ingredients_dataset.csv"

        self.model_path = os.path.join("models", "trained_model.pkl")

        self.conn = None
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()

        try:
            self.stop_words = set(stopwords.words("english"))
        except:
            self.stop_words = {"a", "the", "and", "is", "in", "of"}

    # -------------------- DAY 2: Load Dataset --------------------
    def load_dataset(self):
        try:
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Dataset loaded: {len(df)} ingredients")
            return df
        except Exception as e:
            print(f"❌ Dataset load error: {e}")
            return None

    # -------------------- DAY 3: OCR --------------------
    def extract_text_from_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return ""
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(rgb, config="--psm 6")
            return text.lower()
        except Exception as e:
            print(f"❌ OCR Error: {e}")
            return ""

    def parse_ingredients_from_text(self, text):
        text = text.lower().strip()
        match = re.search(r"ingredients?\s*:?\s*([^\.]+)", text)
        ingredients_text = match.group(1) if match else text
        ingredients = re.split(r"[,;]", ingredients_text)
        return [i.strip() for i in ingredients if len(i.strip()) > 1]

    # -------------------- DAY 4: NLP --------------------
    def preprocess_ingredient(self, ingredient_text):
        text = ingredient_text.lower().strip()
        e_nums = re.findall(r"e\d+", text)
        text = re.sub(r"[^\w\s-]", " ", text)
        tokens = word_tokenize(text)
        processed = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 1
        ]
        processed.extend(e_nums)
        return " ".join(processed)

    # -------------------- DAY 5: Database --------------------
    def create_database(self, df):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingredients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                label INTEGER,
                side_effects TEXT,
                risk_level TEXT
            )
        """)
        for _, row in df.iterrows():
            cur.execute("""
                INSERT OR IGNORE INTO ingredients (name, label, side_effects, risk_level)
                VALUES (?, ?, ?, ?)
            """, (row["ingredient"].lower(), int(row["label"]), row["side_effects"], row["risk_level"]))
        self.conn.commit()
        print("✅ Database ready")

    def query_ingredient(self, name):
        cur = self.conn.cursor()
        cur.execute("SELECT name,label,side_effects,risk_level FROM ingredients WHERE name=?", (name.lower(),))
        return cur.fetchone()

    # -------------------- DAY 6: Train Model --------------------
    def build_and_train_model(self, df):
        X = df["ingredient"].apply(self.preprocess_ingredient)
        y = df["label"]

        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
        X_vec = self.vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, stratify=y, random_state=42)

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000)
        }

        best_model, best_score = None, 0
        for name, model in models.items():
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            print(f"{name} Accuracy: {acc:.2f}")
            if acc > best_score:
                best_score, best_model = acc, model

        self.model = best_model
        os.makedirs("models", exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "vectorizer": self.vectorizer}, f)

        print(f"✅ Model trained and saved: {self.model_path}")

    def load_model(self):
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self.model, self.vectorizer = data["model"], data["vectorizer"]
            print("✅ Model loaded")
            return True
        except:
            return False

    # -------------------- DAY 7: Analysis --------------------
    def analyze_ingredients(self, ingredients):
        results = {"safe": [], "harmful": [], "unknown": []}
        for ing in ingredients:
            db = self.query_ingredient(ing)
            if db:
                name, label, se, risk = db
                if label == 1:
                    results["harmful"].append({"name": name, "side_effects": se, "risk": risk})
                else:
                    results["safe"].append({"name": name, "info": se})
            elif self.model:
                vec = self.vectorizer.transform([self.preprocess_ingredient(ing)])
                pred = self.model.predict(vec)[0]
                if pred == 1:
                    results["harmful"].append({"name": ing, "side_effects": "AI predicted harmful"})
                else:
                    results["unknown"].append({"name": ing})
            else:
                results["unknown"].append({"name": ing})
        return results

    # -------------------- DAY 10: Run System --------------------
    def run_complete_system(self):
        df = self.load_dataset()
        if df is None:
            return False
        self.create_database(df)
        if not self.load_model():
            self.build_and_train_model(df)
        return True
if __name__ == "__main__":
    print("=" * 70)
    print("🔍 SMART INGREDIENT WATCHDOG - AI DETECTION SYSTEM")
    print("=" * 70)

    watchdog = IngredientWatchdog()
    success = watchdog.run_complete_system()

    if success:
        print("\n🎉 System initialized successfully!")
        # quick test: analyze a sample ingredient list
        sample_text = "Water, Sugar, Sodium Benzoate, MSG, Natural Flavors"
        ings = watchdog.parse_ingredients_from_text(sample_text)
        results = watchdog.analyze_ingredients(ings)
        print("\nSample Analysis:", results)
    else:
        print("❌ Failed to initialize system")

