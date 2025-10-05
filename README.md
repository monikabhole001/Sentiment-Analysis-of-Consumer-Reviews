# Sentiment-Analysis-of-Consumer-Reviews

## 📖 Project Overview  

This project performs **Sentiment Analysis** on **Yelp consumer reviews** using **VADER (Valence Aware Dictionary and sEntiment Reasoner)** and **SpaCy** for text preprocessing.  
The goal is to automatically classify customer feedback as **positive** or **negative** and improve interpretability by filtering out neutral and non-English reviews.

It demonstrates key **Natural Language Processing (NLP)** and **Data Science** steps:  
- Text preprocessing (tokenization, lemmatization)  
- Sentiment polarity extraction (VADER)  
- Binary sentiment labeling  
- Accuracy evaluation and visualization  

---

##  Tech Stack  

| Category | Tools Used |
|-----------|------------|
| Programming | Python 3.10 |
| NLP | SpaCy, NLTK (VADER) |
| ML & Metrics | Scikit-learn |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

##  Folder Structure  

```
Sentiment-Analysis-of-Consumer-Reviews/
│
├── yelp_sentiment_analysis.ipynb     # Main Jupyter notebook
├── README.md                         # Project documentation
├── requirements.txt                  # Dependencies list

```

---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/monikabhole001/Sentiment-Analysis-of-Consumer-Reviews.git
cd Sentiment-Analysis-of-Consumer-Reviews
```

### 2️⃣ Create Virtual Environment  
```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux  
venv\Scripts\activate        # Windows
```

### 3️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Download SpaCy English Model  
```bash
python -m spacy download en_core_web_lg
```

---

## 🧹 Data Preprocessing  

Preprocessing is done using **SpaCy** to clean and lemmatize reviews before sentiment analysis.

```python
import spacy
nlp = spacy.load("en_core_web_lg")

lemma_text_list = []
for doc in nlp.pipe(df["text"], n_process=-1):
    lemma_text_list.append(" ".join(token.lemma_ for token in doc))

df["cleaned_review"] = lemma_text_list
```

**Explanation:**  
- Uses `nlp.pipe()` for batch processing (parallelized).  
- Converts each word to its lemma (root form).  
- Maintains original order of words.  
- Adds cleaned, lemmatized reviews as a new column.  

Example:
| Original Text | Cleaned Review |
|----------------|----------------|
| "The burgers were amazing!" | "the burger be amazing" |

---

## 💬 Sentiment Analysis using VADER  

VADER is a rule-based sentiment analyzer tuned for social media and reviews.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon")

analyzer = SentimentIntensityAnalyzer()
df["compound"] = df["cleaned_review"].apply(lambda text: analyzer.polarity_scores(text)["compound"])
```

Each review receives a **compound score** between -1 and +1:
- **> 0** → Positive  
- **< 0** → Negative  
- **= 0** → Neutral  

---

## 🧾 Labeling Sentiments  

Convert compound scores into binary sentiment labels.

```python
import numpy as np

df["predicted_sentiment"] = np.where(df["compound"] > 0, 1, 0)
```

| Compound | Predicted Sentiment |
|-----------|---------------------|
| 0.82 | 1 (Positive) |
| -0.41 | 0 (Negative) |
| 0.00 | Neutral (filtered out) |

---

## 🧹 Filtering Neutral Reviews  

Neutral reviews (compound = 0) often occur due to **non-English text** or **lack of emotional polarity**.  
To improve interpretability and evaluation reliability, they are removed.

```python
df = df.loc[df["compound"] != 0]
```

---

## 📊 Evaluation  

We measure how well predicted sentiments match the true labels using scikit-learn.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(df["sentiment"], df["predicted_sentiment"])
print(f"Accuracy: {accuracy * 100:.2f}%")
```

**Example Output:**  
```
Accuracy: 86.87%
```

You can also print a classification report:

```python
from sklearn.metrics import classification_report
print(classification_report(df["sentiment"], df["predicted_sentiment"]))
```

---


##  Key Learnings  

- Lemmatization reduces noise and vocabulary size.  
- VADER provides interpretable sentiment polarity without model training.  
- Neutral scores (compound = 0) often result from non-English or ambiguous reviews.  
- Filtering such cases enhances model clarity and performance.  

---

##  Future Improvements  

-  Integrate **language detection** (`langdetect`) to remove non-English reviews earlier.  
-  Extend to a **multi-class model** (positive, neutral, negative).  
-  Compare rule-based vs ML-based sentiment classifiers (e.g., Logistic Regression, BERT).  
-  Build a **FastAPI app** for real-time sentiment prediction.  
-  Log experiments using **MLflow** for tracking model performance.  

---

## 📚 Example Results  

| Review | Cleaned Review | Compound | Sentiment |
|--------|----------------|-----------|------------|
| "Loved the food, great service!" | love the food great service | 0.84 | Positive |
| "Terrible experience, never again." | terrible experience never again | -0.65 | Negative |
| "It was okay, nothing special." | it be okay nothing special | 0.00 | Neutral (removed) |

---

## 📦 Requirements  


To install:
```bash
pip install -r requirements.txt
```

---

## 👤 Author  

**Monika Bhole**  

🔗 [LinkedIn](https://www.linkedin.com/in/monikabhole001)  
---


## 🧩 Acknowledgements  

- **VADER Sentiment Analyzer** — Hutto & Gilbert (2014)  
- **SpaCy** — Explosion AI  
- **Scikit-learn** — Pedregosa et al. (2011)  

---

