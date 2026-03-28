# 📩 SMS Spam Detector — Streamlit App

## Files needed
Put ALL of these in the same folder:
```
app.py
requirements.txt
spam_model.pkl
vectroizer.pkl
spam.csv          ← your original dataset (needed to refit the vectorizer)
```

> **Note:** Your `vectroizer.pkl` was saved as the class (`TfidfVectorizer`) instead of the
> fitted instance (`tfidf`). The app automatically handles this by refitting on `spam.csv`
> if it's present. **Include spam.csv for best accuracy.**

---

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy on Streamlit Community Cloud (free)

1. Push all files to a **GitHub repo**
2. Go to → https://share.streamlit.io
3. Click **"New app"** → connect your GitHub repo
4. Set **Main file path** = `app.py`
5. Click **Deploy** 🚀

---

## How the model works
| Step | What happens |
|------|-------------|
| Input | User types any SMS |
| Preprocess | Lowercase → remove punctuation/numbers/HTML/emojis → remove stopwords → stem |
| Vectorize | TF-IDF (max 3000 features) |
| Predict | Multinomial Naive Bayes → Spam (1) or Ham (0) |
| Output | Result + confidence % shown |
