import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import compute_class_weight

if __name__ == '__main__':
    # Constants and file names
    DATA_DIR = '../quora_insincere_data'
    DATA_FILE = os.path.join(DATA_DIR, 'train.csv')

    # Load data
    data = pd.read_csv(DATA_FILE, sep=',')
    data_text = data['question_text']
    data_labels = data['target']

    # Define feature extractor
    vectorizer = TfidfVectorizer(min_df=0.00001)  # Ignore words where document freq is lt 1% or gt 70%.

    # Train vectorizer on the text data
    vectorizer.fit(data_text)
    all_features = vectorizer.transform(data_text)
    print ("Length of vocabulary of vectorizer", len(vectorizer.vocabulary_))

    # Do train test split
    X_train, X_val, y_train, y_val = train_test_split(
        all_features, data_labels, random_state=3, test_size=0.3
    )

    # Define Ridge regression object
    lr = LogisticRegression(penalty='l2', class_weight='balanced', random_state=5)

    # Define GridSearch object
    param_grid = {
        'C': [1e-2, 1e-1, 1, 10]
    }
    lr_gs = GridSearchCV(lr, param_grid=param_grid, n_jobs=-1, scoring='f1')

    # Fit the grid search object and print cv results
    lr_gs.fit(X_train, y_train)
    print (lr_gs.cv_results_)

    # Predict and find precision and recall
    y_preds = lr_gs.predict(X_val)
    print ("Precision", precision_score(y_val, y_preds))
    print ("Recall", recall_score(y_val, y_preds))
