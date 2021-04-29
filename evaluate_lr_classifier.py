import argparse
import glob
import logging
import os.path
import pandas as pd
import shutil
import random
import torch

from classic_models import *
from data_processing import *
from models import *
from os import makedirs, remove
from sklearn.metrics import accuracy_score
from training import *
from tqdm import tqdm


def predict_probs_for_texts(text_list, model, vectorizer):
    test_vect = vectorizer.transform(text_list)
    return model.predict_proba(text_vect)

def predict_labels_for_texts(text_list, model, vectorizer):
    test_vect = vectorizer.transform(text_list)
    return model.predict(text_vect)

def predict_probs_file(filename, model, vectorizer):
    return predict_probs_for_texts(
        pd.read_csv(filename).text.values, model, vectorizer
    )

def predict_file(filename, model, vectorizer):
    return predict_labels_for_texts(
        pd.read_csv(filename).text.values, model, vectorizer
    )

def get_lr_predictions(test_data_filename, model_dir):
    '''
    Parameters.
    1) test_data_filename - name of the test data file in csv format,
    2) model_dir - directory where to save the model after training.
    '''
    
    model, vectorizer = load_model(model_dir)
    return predict_file(test_data_filename, model, vectorizer)


def get_lr_prob_predictions(test_data_filename, model_dir, probs_filename=None):
    '''
    Parameters.
    1) test_data_filename - name of the test data file in csv format,
    2) model_dir - directory where to save the model after training,
    3) probs_filename - the name of the file where to save the predicted probabilities.
    '''
    
    model, vectorizer = load_model(model_dir)
    model_probs = predict_probs_file(test_data_filename, model, vectorizer)
    
    if probs_filename is not None:
        vocab = Vocabulary().from_files(os.path.join(model_dir, 'vocab'))
        label_to_id = {label: label_id for label_id, label in enumerate(model.classes_)}
        all_labels = ['A1', 'A11', 'A12', 'A14', 'A16', 'A17', 'A4', 'A7', 'A8', 'A9']
        
        normalized_probs = np.zeros_like(model_probs).astype(float)
        
        for label_id, label in enumerate(all_labels):
            normalized_probs[:, label_id] = model_probs[:, label_to_id[label]]
        np.save(probs_filename, normalized_probs)
    
    return model_probs

def evaluate_lr_classifier(test_data_filename, model_dir):
    '''
    Parameters.
    1) test_data_filename - name of the test data file in csv format,
    2) model_dir - directory where to save the model after training.
    '''
    
    model_predictions = get_lr_model_predictions(test_data_filename, model_dir)
    calc_classifier_metrics(model_predictions, list(pd.read_csv(test_data_filename).target.values))

    
def get_lr_classifier_accuracy(test_data_filename, model_dir, probs_filename=None):
    '''
    Parameters.
    1) test_data_filename - name of the test data file in csv format,
    2) model_dir - directory where to save the model after training,
    3) probs_filename - the file where the classifier probs will be solved.
    '''
    
    if probs_filename is not None:
        all_labels = ['A1', 'A11', 'A12', 'A14', 'A16', 'A17', 'A4', 'A7', 'A8', 'A9']
        
        model_predictions = np.argmax(
            get_lr_prob_predictions(test_data_filename, model_dir, probs_filename=probs_filename),
            axis=-1
        )
        model_predictions = [all_labels[label_id] for label_id in model_predictions]
    else:
        model_predictions = get_lr_predictions(test_data_filename, model_dir)
   
    return accuracy_score(
        list(pd.read_csv(test_data_filename).target.values),
        model_predictions
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a transformer model.')
    
    parser.add_argument('--test-data-filename', type=str, help='name of the test data file in csv format')
    parser.add_argument('--model-dir', type=str, help='directory where to save the model after training')
   
    args = parser.parse_args()
    
    evaluate_lr_classifier(
        args.test_data_filename, args.model_dir, args.use_bert_pooler
    )