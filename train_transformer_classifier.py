import argparse
import glob
import os.path
import shutil
import random
import torch

from data_processing import *
from interpretation import *
from models import *
from os import makedirs, remove
from training import *


def train_transformer_model(transformer_model, train_data_filename,
                            test_data_filename, model_dir, epochs_num=10, 
                            cuda_device=-1, rewrite_dir=False,
                            use_bert_pooler=False):
    '''
    Trains a transformer-like model with usage of the AllenNLP framework.
    
    Parameters.
    1) transformer_model - model type (example: bert-base-cased),
    2) train_data_filename - name of the train data file in csv format,
    3) test_data_filename - name of the test data file in csv format,
    4) model_dir - directory where to save the model after training,
    5) epochs_num - the number of the epochs,
    6) cuda_device - cuda device id on which the model trains 
    (if set to -1, the training performs on CPU),
    7) rewrite_dir - indicates whether should the model_dir be rewrited if it already exists,
    8) use_bert_pooler - indicates whether should we take the embedding
    of the first token as the text embedding.
    
    '''
    
    if os.path.exists(model_dir):
        print("directory or file " + model_dir + " already exists")
        if rewrite_dir:
            shutil.rmtree(model_dir)
            print("the directory is removed")
        else:
            print("can't start training the model")
            return
    
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    best_model = os.path.join(checkpoints_dir, 'best.th')
    
    tokenizer = PretrainedTransformerTokenizer(transformer_model)
    train_dataset_reader = build_transformer_dataset_reader(transformer_model, lower=True)
    val_dataset_reader = build_transformer_dataset_reader(transformer_model, lower=True)

    train_data, dev_data = read_data(
        train_data_filename, 
        test_data_filename,
        train_dataset_reader, 
        val_dataset_reader
    )
    vocab = build_vocab(train_data + dev_data)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    
    # initialize the model and place it to gpu is needed
    if use_bert_pooler:
        model = build_transformer_model(vocab, transformer_model)
    else:
        model = build_pool_transformer_model(vocab, transformer_model)
    if cuda_device != -1:
        model = model.cuda(cuda_device)
    
    # make data loader
    makedirs(checkpoints_dir)
    train_loader, dev_loader = build_data_loaders(train_data, dev_data)

    # model training
    trainer = build_classifier_trainer(
        model,
        checkpoints_dir,
        train_loader,
        dev_loader,
        epochs_num,
        cuda_device=cuda_device
    )
    print("Starting training")
    trainer.train()
    print("Finished training")
    
    # remove extra files
    extra_files = glob.glob(checkpoints_dir + '/*.json') +\
                  glob.glob(checkpoints_dir + '/model*') +\
                  glob.glob(checkpoints_dir + '/training*')
    for f in extra_files:
        remove(f)

    # save the model vocabulary to file
    model.vocab.save_to_files(os.path.join(model_dir, 'vocab'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a transformer model.')
    
    parser.add_argument('--transformer-model', type=str, help='model type (example: bert-base-cased)')
    parser.add_argument('--train-data-filename', type=str, help='name of the train data file in csv format')
    parser.add_argument('--test-data-filename', type=str, help='name of the test data file in csv format')
    parser.add_argument('--model-dir', type=str, help='directory where to save the model after training')
    parser.add_argument('--epochs-num', type=int, default=10, help='the number of the epochs')
    parser.add_argument(
        '--rewrite-dir', type=bool, default=False, 
         help='indicates whether should the model_dir be rewrited if it already exists'
    )
    parser.add_argument(
        '--use-bert-pooler', type=bool, default=False, 
         help='indicates whether should we take the embedding of the first token as the text embedding'
    )
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--cuda-device', type=int, default=-1)
   
    args = parser.parse_args()
    
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    
    train_transformer_model(args.transformer_model, args.train_data_filename,
                            args.test_data_filename, args.model_dir, args.epochs_num,
                            args.cuda_device, args.rewrite_dir, args.use_bert_pooler)