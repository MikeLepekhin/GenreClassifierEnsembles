import argparse
import glob
import os.path
import pandas as pd
import random
import shutil
import torch
import time

from classic_models import *
from interpretation import *
from os import makedirs, remove
from sklearn.svm import SVC


def train_svm_model(train_data_filename, test_data_filename, model_dir, random_seed=42,
                   max_word_features=5000, max_char_features=10000, rewrite_dir=False):
    '''
    Trains a transformer-like model with usage of the AllenNLP framework.
    
    Parameters.
    1) train_data_filename - name of the train data file in csv format,
    2) test_data_filename - name of the test data file in csv format,
    3) model_dir - directory where to save the model after training,
    4) random_state - random seed for the trainer,
    5) max_word_features - the number of features for word-based tf-idf vectorizer,
    6) max_char_features - the number of features for char-based tf-idf vectorizer,
    7) rewrite_dir - indicates whether should the model_dir be rewrite.if it already exists.
    '''
    
    if os.path.exists(model_dir):
        print("directory or file " + model_dir + " already exists")
        if rewrite_dir:
            shutil.rmtree(model_dir)
            print("the directory is removed")
        else:
            print("can't start training the model")
            return
    
    raw_train = pd.read_csv(train_data_filename)
    train_clean_text = raw_train.text.values
    train_clean_label = raw_train.target.values
    
    raw_test = pd.read_csv(test_data_filename)
    test_clean_text = raw_test.text.values
    test_clean_label = raw_test.target.values
    
    vectorizer = BigVectorizer(max_word_features, max_char_features)
    
    print("# vectorization started")
    cur_time = time.time()
    train_vect = vectorizer.fit_transform(train_clean_text)
    test_vect = vectorizer.transform(test_clean_text)
    print('vectorization time:', time.time() - cur_time)
    print("# vectorization finished")
    
    svm_estimator = SVC(random_state=random_seed, max_iter=1000, probability=True)
    print("# training started")
    cur_time = time.time()
    svm_estimator.fit(train_vect, train_clean_label)
    print('training time:', time.time() - cur_time)
    print("# training finished")
    
    makedirs(model_dir, exist_ok=True)
    save_model(svm_estimator, vectorizer, model_dir)
    evaluate(svm_estimator, train_vect, test_vect, train_clean_label, test_clean_label)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a SVM classifier.')
    
    parser.add_argument('--train-data-filename', type=str, help='name of the train data file in csv format')
    parser.add_argument('--test-data-filename', type=str, help='name of the test data file in csv format')
    parser.add_argument('--model-dir', type=str, help='directory where to save the model after training')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--max-word-features', type=int, default=5000)
    parser.add_argument('--max-char-features', type=int, default=10000)
    parser.add_argument(
        '--rewrite-dir', type=bool, default=False, 
         help='indicates whether should the model_dir be rewrited if it already exists'
    )
    
    args = parser.parse_args()
    
    train_svm_model(
        args.train_data_filename, args.test_data_filename, args.model_dir,
        args.random_seed, args.max_word_features, args.max_char_features,
        args.rewrite_dir
    )