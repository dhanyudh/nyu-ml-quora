from collections import Counter
import itertools

import os
import json
import numpy as np
import pandas as pd
import re

from keras import backend as K
from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.engine import Model
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, merge, Flatten, Embedding, Reshape
from keras.optimizers import SGD
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


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))

    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Above will be collection of all unique words ['Aniket', 'He', 'I', 'am', 'intelligent', 'is', 'smart']
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    # Dictionary of the form as below
    # {'Aniket': 0, 'He': 1, 'I': 2, 'am': 3, 'inteligennt': 4, 'is': 5, 'smart': 6}
    return vocabulary, vocabulary_inv


def train_custom_cnn(seq_length, embedding_dim, vocab_size):

    input = Input(shape=(seq_length,), name='text_input')

    embedding = Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=seq_length)(input)

    reshape = Reshape((seq_length, embedding_dim, 1))(embedding)

    conv_0 = Convolution2D(NUM_FILTERS, FILTER_SIZES[0], embedding_dim, border_mode='valid', init='normal',
                           activation='relu', dim_ordering='tf')(reshape)
    conv_1 = Convolution2D(NUM_FILTERS, FILTER_SIZES[1], embedding_dim, border_mode='valid', init='normal',
                           activation='relu', dim_ordering='tf')(reshape)
    conv_2 = Convolution2D(NUM_FILTERS, FILTER_SIZES[2], embedding_dim, border_mode='valid', init='normal',
                           activation='relu', dim_ordering='tf')(reshape)

    maxpool_0 = MaxPooling2D(pool_size=(seq_length-FILTER_SIZES[0]+1, 1), strides=(1, 1), border_mode='valid',
                             dim_ordering='tf')(conv_0)
    maxpool_1 = MaxPooling2D(pool_size=(seq_length-FILTER_SIZES[1]+1, 1), strides=(1, 1), border_mode='valid',
                             dim_ordering='tf')(conv_1)
    maxpool_2 = MaxPooling2D(pool_size=(seq_length-FILTER_SIZES[2]+1, 1), strides=(1, 1), border_mode='valid',
                             dim_ordering='tf')(conv_2)

    merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)

    flatten = Flatten()(merged_tensor)

    drop = Dropout(0.5)(flatten)

    final_output = Dense(1, activation='sigmoid')(drop)

    final_model = Model(inputs=[input], outputs=[final_output])

    return final_model


# Constants and file names
DATA_DIR = '../quora_insincere_data'
DATA_FILE = os.path.join(DATA_DIR, 'train.csv')
DNN_BEST_MODEL = 'embed_own_dnn.hdf5'
VOCAB_FILE = 'vocab.json'
VOCAB_INV_FILE = 'vocab_inv.npy'

# Model constants
NUM_FILTERS = 512
FILTER_SIZES = [3, 4, 5]
seq_length = 50
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

# # Compute class weights
class_weight = compute_class_weight('balanced', np.unique(train_labels), train_labels)

# For sanity experiments
train_text = train_text[:1000]
test_text = test_text[:1000]
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

# Convert train_text and test_text to lists
train_text = train_text.tolist()
test_text = test_text.tolist()

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

# Generate vocabulary and vocabulary_inv using train_text
vocabulary, vocabulary_inv = build_vocab(train_text)
with open(VOCAB_FILE, 'w') as VOCAB_FILE:
    json.dump(vocabulary, VOCAB_FILE)
np.save(VOCAB_INV_FILE, vocabulary_inv)

print('Length of vocabulary = :', len(vocabulary_inv))

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

train_embed = np.array(train_text)
test_embed = np.array(test_text)

# Reshape train and test embeddings
train_embed = train_embed.reshape(train_embed.shape[0], seq_length)
test_embed = test_embed.reshape(test_embed.shape[0], seq_length)

print(train_embed.shape)
print(test_embed.shape)

cnn = train_custom_cnn(seq_length=seq_length, embedding_dim=embed_size, vocab_size=len(vocabulary))
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
        callbacks=[check_pointer, early_stopper])

