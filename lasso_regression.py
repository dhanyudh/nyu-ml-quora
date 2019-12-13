import os
import re
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
    del data

    # Do train test split
    train_texts, test_texts, y_train, y_test = train_test_split(
        data_text, data_labels, random_state=3, test_size=0.3, shuffle=True
    )

    # Define feature extractor
    vectorizer = TfidfVectorizer(min_df=0.00001)  # Ignore words where document freq is lt 1% or gt 70%.

    # Train vectorizer on the text data
    vectorizer.fit(train_texts)
    X_train = vectorizer.transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    print ("Length of vocabulary of vectorizer", len(vectorizer.vocabulary_))
    del train_texts, test_texts

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
    y_preds = lr_gs.predict(X_test)

    count_ones = np.count_nonzero(y_preds)
    count_zeros = len(y_preds) - count_ones
    print ("Count of ones", count_ones)
    print ("Count of zeros", count_zeros)

    print ("Precision", precision_score(y_test, y_preds))
    print ("Recall", recall_score(y_test, y_preds))
