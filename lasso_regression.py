from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Constants and file names
    DATA_DIR = '../quora_insincere_data'
    DATA_FILE = os.path.join(DATA_DIR, 'train.csv')

    # Load data
    data = pd.read_csv(DATA_FILE, sep=',')
    data_text = data['question_text']
    data_labels = data['target']

    # Define feature extractor
    vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.7)  # Ignore words where document freq is lt 1% or gt 70%.

    # Train vectorizer on the text data
    vectorizer.fit(data_text)
    print ("Length of vocabulary of vectorizer", len(vectorizer.vocabulary_))

    # Do train test split
    train_test_split()