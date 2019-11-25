import argparse
import os
import numpy as np
import pandas as pd


if __name__ == '__main__':

    # Arguments parser
    parser = argparse.ArgumentParser(description='Data exploration')
    parser.add_argument('--only-insincere', type=bool, default=False,
                        help='Whether to consider only insincere questions for no of words analysis')
    args = parser.parse_args()

    # Constants and file names
    DATA_DIR = '../quora_insincere_data'
    DATA_FILE = os.path.join(DATA_DIR, 'train.csv')

    # Extract text and labels from the data
    data = pd.read_csv(DATA_FILE, sep=',')
    if args.only_insincere:
        data_text = data[data['target'] == 1]['question_text']
    else:
        data_text = data['question_text']
    del data

    # Build array of length of texts
    lengths_of_quests = []
    for text in data_text:
        lengths_of_quests.append(len(text.split()))
    lengths_arr = np.array(lengths_of_quests)
    del data_text

    # Sort the lengths arr
    lengths_arr = np.sort(lengths_arr)

    # Get 50, 80, 90, 95, 99, 100th percentile of length of questions
    index_50th = int(len(lengths_arr) * 0.5)
    index_80th = int(len(lengths_arr) * 0.8)
    index_90th = int(len(lengths_arr) * 0.9)
    index_99th = int(len(lengths_arr) * 0.99)
    index_100th = len(lengths_arr) - 1

    val_50th = lengths_arr[index_50th]
    val_80th = lengths_arr[index_80th]
    val_90th = lengths_arr[index_90th]
    val_99th = lengths_arr[index_99th]
    val_100th = lengths_arr[index_100th]

    print ("No of words in sentence with length at 50th percentile", val_50th)
    print ("No of words in sentence with length at 80th percentile", val_80th)
    print ("No of words in sentence with length at 90th percentile", val_90th)
    print ("No of words in sentence with length at 99th percentile", val_99th)
    print ("No of words in sentence with length at 100th percentile", val_100th)
