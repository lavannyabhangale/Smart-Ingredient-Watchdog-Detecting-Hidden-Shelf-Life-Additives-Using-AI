# 🕵️ Smart Ingredient Watchdog: Detecting Hidden Shelf-Life Additives Using AI
*(Alternate Title: TruthScanner AI: Exposing Unsafe Additives with Machine Learning)*

## 📌 Overview
Smart Ingredient Watchdog is an AI-powered application that scans food product labels (image or text) to detect hidden and potentially harmful additives.  
Using **OCR, NLP, Machine Learning, and Database integration**, it flags risky ingredients, highlights possible side effects, and helps consumers make safer food choices.  
This project demonstrates how **AI can drive consumer safety, ingredient transparency, and real-world problem-solving** in food tech and health.

---

## 🛠 Tools & Technologies
- **Python** – Core language for ML & automation  
- **Pandas, NumPy** – Data manipulation  
- **Scikit-learn, ** – Classification models    
- **OpenCV + Tesseract OCR** – Extract text from scanned food labels  
- **NLTK / spaCy** – Natural Language Processing  
- **SQLite ** – Store ingredient data, aliases, side effects, and risk levels  
- **Streamlit ** – Frontend for uploading and analyzing labels  
-Jupyter Notebook** – Development environment  


---

## 🎯 Objective
The project aims to build an application that:  
- 📸 Scans ingredient labels (via OCR or text input)  
- 🔎 Identifies suspicious / hidden additives  
- 🗂 Cross-checks with a database of harmful ingredients  
- 🚨 Raises alerts with **side effects** and **risk levels**


**Output Analysis:**  
- ✅ Safe:  
  - *Water → Essential for life*  
  - *Sugar → Safe in moderation*  
- ❌ Harmful:  
  - *Sodium Benzoate → DNA damage with Vitamin C (Risk: Moderate)*  
  - *MSG → MSG syndrome, headaches (Risk: Moderate)*  
- ⚠ Unknown:  
  - *Natural Flavors*  

---


