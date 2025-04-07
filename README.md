# Fake_Review_Detector

## Project File Structure:
```
Fake_Review_Detector/
├── data/
│   ├── processed/       # Processed data files
│   │   ├── preprocessed_test.csv
│   │   ├── preprocessed_train.csv
│   │   ├── preprocessed_val.csv
│   │   └── val_combined.csv
│   └── raw/             # Raw input data files
│       ├── fake_reviews.csv
│       ├── test.csv
│       ├── train.csv
│       └── val.csv
│
├── notebooks/           # Jupyter notebooks for exploration and visualization
│   └── eda.ipynb
│
├── src/                 # Source code
│   ├── config.py        # Configuration settings
│   ├── embeddings/      # Text vectorization methods
│   │   └── word2vec.py  # Bag of Words vectorization
│   │
│   ├── ensemble/        # Ensemble learning implementation
│   │   ├── __init__.py
│   │   ├── test_probs.csv
│   │   ├── val_probs.csv
│   │   └── weighted_ensemble.py
│   │
│   ├── models/          # ML model implementations
│   │   ├── decision_tree.py
│   │   ├── model1/
│   │   │   ├── category_review_relevance.csv
│   │   │   ├── dataset/
│   │   │   │   ├── test.csv
│   │   │   │   ├── train.csv
│   │   │   │   └── val.csv
│   │   │   ├── inference.py
│   │   │   ├── models/
│   │   │   │   ├── dictionary.gensim
│   │   │   │   ├── label_encoder.pkl
│   │   │   │   ├── lda_model.gensim
│   │   │   │   ├── lda_model.gensim.expElogbeta.npy
│   │   │   │   ├── lda_model.gensim.state
│   │   │   │   └── tfidf_vectorizer.pkl
│   │   │   ├── threshold.py
│   │   │   ├── train.py
│   │   │   ├── val_category_relevance.csv
│   │   │   ├── val_results.csv
│   │   │   └── val_set_metrics.csv
│   │   └── model2/
│   │       ├── dataset.csv
│   │       ├── metrics_with_threshold_per_rating.csv
│   │       ├── model_bert.py
│   │       ├── test.csv
│   │       ├── thresold.py
│   │       ├── train.csv
│   │       └── val.csv
│   │
│   ├── preprocessing/   # Text preprocessing pipeline
│   │   ├── clean_text.py       # Text cleaning functions
│   │   ├── lemmatization.py    # Word lemmatization
│   │   ├── preprocessing_pipeline.py  # Complete pipeline
│   │   └── stemming.py         # Word stemming
│   │
│   ├── main.py          # Main script to run the project
│   └── utils.py         # Utility functions
│
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── Report.pdf           # Project report
```
