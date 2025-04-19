# Biometric Text Analysis on Twitter Data

A scalable and intelligent machine learning pipeline for user authentication using stylometric features derived from tweets. Built with Apache Spark for preprocessing and H2O for modeling, this project addresses the challenges of authorship verification, style variability, and high-dimensional textual data at scale.

## Authors

- Youssef Nakhla  
- Zeina Kishk  
- Jana Al Morsy  
- Hanya Zamir  

## Dataset

We used the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140), containing 1.6 million tweets with sentiment labels.

### Fields Used:

- `user`: Anonymized Twitter username  
- `text`: Tweet content  
- `target`: Sentiment polarity (used optionally as a feature)

---

## Preprocessing Pipeline (Apache Spark)

- **Feature Cleansing**: Dropped unnecessary columns like `flag`, `date`, `ids`
- **User Filtering**: Removed users with ≤ 5 tweets
- **Bot Detection**: Flagged accounts with >90% identical tweets
- **Text Normalization**: Removed mentions, URLs, non-ASCII characters
- **Tokenization + Stopword Removal**: Using Spark ML

---

## Feature Engineering

Features were categorized into:

### Lexical
- Average word length  
- Type-token ratio  
- Capital word ratio  

### Structural
- Word, character, sentence counts  
- Average sentence length  
- Capitalization ratio  

### Syntactic
- POS tagging distribution: `NOUN`, `VERB`, `ADJ`, `ADV`, `PRON` (via SpaCy)

### Stylistic
- Stopword ratio  
- Emoji / hashtag / slang count  
- Symbol and punctuation usage  
- Repeated characters, emphasis markers (e.g., `"!!!"`, `"???"`)

### Semantic
- TF-IDF vectors from 300-dimension hashed term frequency features  

---

## Format Transformation

- Flattened all arrays and maps into individual feature columns  
- Removed intermediate Spark vectors and NA-heavy columns  
- Final feature matrix used for classification  

---

## Balancing Classes (SMOTE)

- **Target**: Classify tweets from “User X” vs. all others  
- **Original imbalance**: 281 vs. 567,539 samples  

### Applied:
- Undersampling of majority class → 250,000  
- SMOTE on minority class → 250,000  

Final dataset: **500,000 rows**, perfectly balanced  
Hold-out test set (20%) stratified by class  

---

## Models (H2O.ai Framework)

Trained on the balanced dataset using the following models:

| Model                | Accuracy | F1 Score |
|---------------------|----------|----------|
| Logistic Regression | 0.93     | 0.90     |
| SVM (CNN-like)      | 0.87     | 0.80     |
| Random Forest       | 0.93     | 0.91     |
| XGBoost             | 0.73     | 0.33     |

### Observations
- Logistic Regression and Random Forest performed best  
- SVM lagged slightly due to sparsity  
- XGBoost struggled significantly with class 1 identification  

---

## Evaluation Metrics

- Confusion Matrix  
- Accuracy  
- F1-Score  
- *(Optional: AUC, ROC Curves in future expansion)*

---

## Limitations

While the proposed pipeline demonstrated strong performance, several limitations must be acknowledged:

- The synthetic samples generated through SMOTE, though useful for addressing class imbalance, may not fully capture the linguistic variability and contextual richness of authentic tweets.  
- The limited number of genuine tweets from the target user (User X) may have constrained the classifier's capacity to learn a robust and distinctive authorial profile.  
- Due to computational resource constraints, we were unable to scale the pipeline to support dynamic training and evaluation across all users in the dataset.  

Future work should explore more sophisticated oversampling strategies, investigate meta-learning approaches for user-specific generalization, and implement distributed training solutions to extend scalability.

---

## Future Work

- Implement Siamese Networks for author similarity learning  
- Expand to multi-user authorship classification  
- Deploy as a real-time Spark streaming service  
- Explore cross-lingual authorship analysis  

---

## Acknowledgments

- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)  
- H2O.ai for model interpretability tools  
- Spark and PySpark for distributed NLP processing  
- SpaCy and NLTK for linguistic feature extraction


