import numpy as np
import pandas as pd
import re
import gensim
import gensim.corpora as corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --- Category Mapping ---
category_mapping = {
    "Home_and_Kitchen_5": "Home and Kitchen",
    "Sports_and_Outdoors_5": "Sports and Outdoors",
    "Electronics_5": "Electronics",
    "Movies_and_TV_5": "Movies and TV",
    "Tools_and_Home_Improvement_5": "Tools and Home Improvement",
    "Pet_Supplies_5": "Pet Supplies",
    "Kindle_Store_5": "Kindle Store",
    "Books_5": "Books",
    "Toys_and_Games_5": "Toys and Games",
    "Clothing_Shoes_and_Jewelry_5": "Clothing Shoes and Jewelry"
}

# --- Text Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens

# --- Load Dataset ---
print("Loading dataset...")
df = pd.read_csv("train.csv")  # Ensure it has 'category' and 'text_' columns
print("Dataset loaded successfully!")

df["review"] = df["text_"]
df["category"] = df["category"].map(category_mapping)
df["tokens"] = df["review"].apply(preprocess_text)

# --- Create Dictionary & Corpus ---
print("Creating dictionary and corpus...")
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]

# --- Train LDA Model ---
print("Training LDA model...")
lda_model = gensim.models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=10)

# --- Get Topic Vectors ---
def get_topic_vector(text):
    bow = dictionary.doc2bow(preprocess_text(text))
    topic_dist = lda_model[bow]
    vec = np.zeros(10)
    for topic_id, weight in topic_dist:
        vec[topic_id] = weight
    return vec

df["topic_vector"] = df["review"].apply(get_topic_vector)

# --- TF-IDF on Categories ---
categories = list(category_mapping.values())
print("Training TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=10)
category_vectors = vectorizer.fit_transform(categories).toarray()

# --- Compute Relevance Scores ---
def compute_relevance(category, review_vector):
    category_vector = vectorizer.transform([category]).toarray()
    return cosine_similarity(category_vector, [review_vector])[0][0]

df["relevance"] = df.apply(lambda row: compute_relevance(row["category"], row["topic_vector"]), axis=1)

# --- Save Models and Results ---
print("Saving models...")
os.makedirs("models", exist_ok=True)
lda_model.save("models/lda_model.gensim")
dictionary.save("models/dictionary.gensim")
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Saving training relevance output...")
df[["category", "review", "label", "relevance"]].to_csv("category_review_relevance.csv", index=False)
print("Training complete and models saved!")
