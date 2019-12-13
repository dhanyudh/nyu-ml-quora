from collections import Counter
import itertools

import os
import json
import numpy as np
import pandas as pd
import re

import sys
from gensim.models import KeyedVectors
from keras import backend as K
from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.engine import Model
from keras.layers import Dense, Dropout, MaxPooling2D, merge, Flatten, Embedding, Reshape, Conv2D
from keras.optimizers import SGD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, compute_class_weight


def recall_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.clip(y_true, 0, 1))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_metric(y_true, y_pred):
    prec = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    f1_score = (2 * prec * recall) / (prec + recall + K.epsilon())

    return f1_score


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", ' ', string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r',', ' , ', string)
    string = re.sub(r'!', ' ! ', string)
    string = re.sub(r'\(', ' \( ', string)
    string = re.sub(r'\)', ' \) ', string)
    string = re.sub(r'\?', ' \? ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip().lower()


def train_custom_cnn(seq_length, embedding_dim, vocab_size, emb_init_matrix):

    input = Input(shape=(seq_length,), name='text_input')

    embedding = Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=seq_length,
                          weights=[emb_init_matrix])(input)

    reshape = Reshape((seq_length, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(NUM_FILTERS, FILTER_SIZES[0], embedding_dim, border_mode='valid', init='normal',
                    activation='relu', dim_ordering='tf')(reshape)
    conv_1 = Conv2D(NUM_FILTERS, FILTER_SIZES[1], embedding_dim, border_mode='valid', init='normal',
                    activation='relu', dim_ordering='tf')(reshape)
    conv_2 = Conv2D(NUM_FILTERS, FILTER_SIZES[2], embedding_dim, border_mode='valid', init='normal',
                    activation='relu', dim_ordering='tf')(reshape)

    maxpool_0 = MaxPooling2D(pool_size=(seq_length-FILTER_SIZES[0]+1, 1), strides=(1, 1), border_mode='valid',
                             dim_ordering='tf')(conv_0)
    maxpool_1 = MaxPooling2D(pool_size=(seq_length-FILTER_SIZES[1]+1, 1), strides=(1, 1), border_mode='valid',
                             dim_ordering='tf')(conv_1)
    maxpool_2 = MaxPooling2D(pool_size=(seq_length-FILTER_SIZES[2]+1, 1), strides=(1, 1), border_mode='valid',
                             dim_ordering='tf')(conv_2)

    merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)

    flatten = Flatten()(merged_tensor)

    drop = Dropout(rate=0.5)(flatten)

    final_output = Dense(1, activation='sigmoid')(drop)

    final_model = Model(inputs=[input], outputs=[final_output])

    return final_model


if __name__ == '__main__':

    # # Just checking model summary
    # NUM_FILTERS = 512
    # FILTER_SIZES = [3, 4, 5]
    # emb_init_matrix = np.zeros((100, 300))
    # txt_cnn = train_custom_cnn(50, 300, 100, emb_init_matrix)
    #
    # print (txt_cnn.summary())
    # exit()

    # Constants and file names
    use_google_colab = int(sys.argv[1])
    if use_google_colab:
        DATA_DIR = os.path.join('/content/drive/My Drive/iml_project/quora_insincere_data')
    else:
        DATA_DIR = '../quora_insincere_data'
    DATA_FILE = os.path.join(DATA_DIR, 'train.csv')
    DNN_BEST_MODEL = 'embed_own_dnn_v2.hdf5'
    VOCAB_FILE = 'vocab.json'
    VOCAB_INV_FILE = 'vocab_inv.npy'
    embeddings_dir = os.path.join(DATA_DIR, 'embeddings/GoogleNews-vectors-negative300/')
    GOOGLE_WORD_2_VEC = os.path.join(embeddings_dir, 'GoogleNews-vectors-negative300.bin')

    # Model constants
    NUM_FILTERS = 512
    FILTER_SIZES = [3, 4, 5]
    seq_length = 40
    embed_size = 300
    EPOCHS_PATIENCE_BEFORE_STOPPING = 5
    EPOCHS_PATIENCE_BEFORE_DECAY = 3

    # Extract text and labels from the data
    data = pd.read_csv(DATA_FILE, sep=',')
    data_text = data['question_text']
    data_labels = data['target']
    del data

    # Clean data_text
    for i in range(len(data_text)):
        data_text[i] = clean_str(data_text[i])

    # Shuffle and make train and test text and label arrays
    data_text, data_labels = shuffle(data_text, data_labels, random_state=3)
    train_text, test_text, train_labels, test_labels = train_test_split(
        data_text, data_labels, test_size=0.2, shuffle=True
    )
    del data_text, data_labels

    # Convert train_text and test_text to lists
    train_text = train_text.tolist()
    test_text = test_text.tolist()

    # Define feature extractor. This is just used to filter out words in vocab which are rarely used.
    vectorizer = TfidfVectorizer(min_df=0.00001)  # Ignore words where document freq is lt 0.001%

    # Train vectorizer on the text data
    vectorizer.fit(train_text)
    print ("Length of vocabulary of vectorizer", len(vectorizer.vocabulary_))

    # Obtain the relevant words in vocabulary and their  indices
    vocabulary = vectorizer.vocabulary_
    vocabulary['UNQ'] = len(vocabulary)  # Add new word UNQ to vocab, so that words out of vocab in train and text can
    # be replaced by this
    vocabulary_inv = ['UNQ'] * (len(vocabulary) + 1)
    for word in vocabulary:
        index = vocabulary[word]
        vocabulary_inv[index] = word

    # Create initial embedding matrix which will be fine tuned
    w2v_google = KeyedVectors.load_word2vec_format(GOOGLE_WORD_2_VEC, binary=True)
    emb_init_matrix = np.zeros((len(vocabulary.keys()), embed_size), dtype=float)
    for ind, word in enumerate(vocabulary.keys()):
        try:
            emb_init_matrix[ind] = w2v_google.wv[word]
        except KeyError:
            continue
    del w2v_google

    # # Compute class weights
    class_weight = compute_class_weight('balanced', np.unique(train_labels), train_labels)

    # # For sanity experiments
    # train_text = train_text[:1000]
    # test_text = test_text[:1000]
    # train_labels = train_labels[:1000]
    # test_labels = test_labels[:1000]

    # Convert train text to a list of list of words
    for i in range(len(train_text)):
        train_text[i] = train_text[i].split()
        no_words = len(train_text[i])
        if no_words < seq_length:
            train_text[i] = train_text[i] + ['UNQ'] * (seq_length - no_words)

    # Convert test text to a list of list of words
    for i in range(len(test_text)):
        test_text[i] = test_text[i].split()
        no_words = len(test_text[i])
        if no_words < seq_length:
            test_text[i] = test_text[i] + ['UNQ'] * (seq_length - no_words)

    # Modify given train_text and test_text with vocabulary dictionary
    for i in range(len(train_text)):
        train_text[i] = train_text[i][:seq_length]
        for j in range(seq_length):
            word = train_text[i][j]
            try:
                train_text[i][j] = vocabulary[word]
            except KeyError:
                train_text[i][j] = vocabulary['UNQ']

    for i in range(len(test_text)):
        test_text[i] = test_text[i][:seq_length]
        for j in range(seq_length):
            word = test_text[i][j]
            try:
                test_text[i][j] = vocabulary[word]
            except KeyError:
                test_text[i][j] = vocabulary['UNQ']

    train_feats = np.array(train_text)
    test_feats = np.array(test_text)

    # Reshape train and test features
    train_embed = train_feats.reshape(train_feats.shape[0], seq_length)
    test_embed = test_feats.reshape(test_feats.shape[0], seq_length)

    print(train_embed.shape)
    print(test_embed.shape)

    cnn = train_custom_cnn(seq_length=seq_length, embedding_dim=embed_size, vocab_size=len(vocabulary),
                           emb_init_matrix=emb_init_matrix)
    # cnn.layers[1].trainable = False
    sgd = SGD(lr=1e-2, momentum=0.9)
    cnn.compile(optimizer=sgd, loss='binary_crossentropy', metrics=[precision_metric, recall_metric, f1_metric])

    check_pointer = ModelCheckpoint(filepath=DNN_BEST_MODEL, verbose=1, save_best_only=True)
    early_stopper = EarlyStopping(monitor='val_loss', patience=EPOCHS_PATIENCE_BEFORE_STOPPING)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=EPOCHS_PATIENCE_BEFORE_DECAY,
                                             verbose=1, min_lr=1e-7)

    cnn.fit(train_embed, train_labels,
            validation_data=[test_embed, test_labels],
            class_weight=class_weight,
            batch_size=128, epochs=100,
            callbacks=[check_pointer, early_stopper],
            verbose=2)



