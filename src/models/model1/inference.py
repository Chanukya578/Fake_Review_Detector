import pandas as pd
import numpy as np
import re
import pickle
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

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

# --- Load Models ---
print("Loading models...")
lda_model = gensim.models.LdaModel.load("models/lda_model.gensim")
dictionary = corpora.Dictionary.load("models/dictionary.gensim")
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
print("Models loaded!")

# --- Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens

# --- Get Topic Vector ---
def get_topic_vector(text):
    bow = dictionary.doc2bow(preprocess_text(text))
    topic_dist = lda_model[bow]
    vec = np.zeros(10)
    for topic_id, weight in topic_dist:
        vec[topic_id] = weight
    return vec

# --- Relevance Computation ---
def compute_relevance(category, review_vector):
    category_vector = vectorizer.transform([category]).toarray()
    return cosine_similarity(category_vector, [review_vector])[0][0]

# --- Main Inference Function ---
def compute_relevance_for_file(file_path, output_path="val_category_relevance.csv"):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    df["review"] = df["text_"]
    df["category"] = df["category"].map(category_mapping)

    print("Computing topic vectors and relevance scores...")
    df["topic_vector"] = df["review"].apply(get_topic_vector)
    df["relevance"] = df.apply(lambda row: compute_relevance(row["category"], row["topic_vector"]), axis=1)

    print(f"Saving results to {output_path}...")
    df[["category", "review", "label", "relevance"]].to_csv(output_path, index=False)
    print("Inference complete!")

# Example usage:
compute_relevance_for_file("dataset/val.csv")
