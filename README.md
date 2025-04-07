# Fake_Review_Detector

## Project Overview: 
This project aims to detect fake reviews (Computer-Generated or Fake - CG) vs original reviews (Human-Written - OR) using **Natural Language Processing (NLP) techniques** and **Machine Learning models**. The system processes textual data, extracts meaningful features, trains multiple classifiers, and combines their outputs using an **ensemble learning approach** to enhance detection accuracy.

## Pre-trained Models
Download the pre-trained models from: [Google Drive](https://drive.google.com/file/d/1n14eSuAM8nX_6pQe2HVd_CF7azJQmWxi/view?usp=sharing)

Extract the downloaded zip file into the `models/` directory of the project.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Fake_Review_Detector.git
cd Fake_Review_Detector
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Download the pre-trained models (if you want to skip training):
```bash
# Download from Google Drive link above and extract to the models/ directory
```

## Usage

### Data Preprocessing

To preprocess the dataset (split and clean the data):

```bash
python src/main.py --mode preprocess
```

### Training Models

Train a specific model:

```bash
python src/main.py --mode train --embedding tfidf --classifier svm
```

Train all model combinations:

```bash
python src/main.py --mode train --all-models
```

### Evaluating Models

Evaluate a specific model:

```bash
python src/main.py --mode evaluate --embedding tfidf --classifier svm
```

Evaluate all trained models:

```bash
python src/main.py --mode evaluate --all-models
```

### Train and Evaluate in One Step

```bash
python src/main.py --mode train_and_evaluate --embedding tfidf --classifier svm
```

or for all models:

```bash
python src/main.py --mode train_and_evaluate --all-models
```
### Training with BERT

To train the model using BERT:

```bash
python src/models/model_bert.py
```

### Inference and Threshold Calculation

To infer and calculate thresholds using the trained model:

1. Download the model from: [Google Drive](https://drive.google.com/file/d/1f3-kes7OcHLdDDf89ipK87fDqyG9uC6h/view?usp=sharing)
2. Run the following command:

```bash
python src/models/threshold.py
```

## Project File Structure: 
Fake_Review_Detector/
├── data/
│   ├── raw/             # Raw input data files
│   │   └── fake_reviews.csv  # Original dataset
│   └── preprocessed/    # Preprocessed data files
│
├── models/              # Saved trained models
│
├── notebooks/           # Jupyter notebooks for exploration and visualization
│
├── src/                 # Source code
│   ├── embeddings/      # Text vectorization methods
│   │   ├── tfidf_vectorizer.py  # TF-IDF vectorization
│   │   └── word2vec.py  # Bag of Words vectorization
│   │
│   ├── models/          # ML model implementations
│   │   ├── decision_tree.py
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── svm.py
│   │   ├── model_bert.py
│   │   └── thresold.py
│   │
│   ├── preprocessing/   # Text preprocessing pipeline
│   │   ├── clean_text.py       # Text cleaning functions
│   │   ├── lemmatization.py    # Word lemmatization
│   │   ├── preprocessing_pipeline.py  # Complete pipeline
│   │   └── stemming.py         # Word stemming
│   │
│   ├── config.py        # Configuration settings
│   ├── main.py          # Main script to run the project
│   └── utils.py         # Utility functions
│
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation