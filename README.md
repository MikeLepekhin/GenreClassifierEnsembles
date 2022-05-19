# GenreClassifierEnsembles

You can find all the datasets we use in `data` directory.

## Train models

### Transformer-based models

Train XLM-RoBERTa

```python train_transformer_classifier.py \
        --transformer-model xlm-roberta-base \
        --train-data-filename data/new_valid_livejournal/new_pure_livejournal_ftd_train1.csv \
        --test-data-filename data/new_valid_livejournal/new_pure_livejournal_ftd_valid1.csv \
        --model-dir models/xlm_roberta_half_livejournal_ru_seed_42 \
        --random-seed 42 \
        --cuda-device 0
```

Train RuBERT

```python train_transformer_classifier.py \
        --transformer-model DeepPavlov/rubert-base-cased \
        --train-data-filename data/new_valid_livejournal/new_pure_livejournal_ftd_train1.csv \
        --test-data-filename data/new_valid_livejournal/new_pure_livejournal_ftd_valid1.csv \
        --model-dir models/xlm_roberta_half_livejournal_ru_seed_42 \
        --random-seed 42 \
	--use-bert-pooler True \
        --cuda-device 0
```


### Classic ML

Train Logistic Regression:

```python train_lr_classifier.py \
        --train-data-filename data/new_valid_livejournal/new_pure_livejournal_ftd_train1.csv \
        --test-data-filename data/new_valid_livejournal/new_pure_livejournal_ftd_valid1.csv \
        --model-dir models/svm_genre/new_valid_lr_livejournal_ftd_100_seed_42_1
```

Train SVM Classifier:

```python train_svm_classifier.py \
        --train-data-filename new_valid_livejournal/new_pure_livejournal_ftd_train1.csv \
        --test-data-filename data/new_valid_livejournal/new_pure_livejournal_ftd_valid1.csv \
        --model-dir models/svm_genre/new_valid_svm_livejournal_ftd_100_seed_42_1
```
