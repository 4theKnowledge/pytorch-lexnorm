# Pytorch-lexnorm

A Pytorch-based model for lexical normalisation, as detailed in the paper [Word-level Lexical Normalisation using Context-Dependent Embeddings](https://arxiv.org/abs/1911.06172).

## Setting up

Place your data under `data/datasets/<dataset_name>`, e.g. `data/datasets/us_accidents_self`, using the names `train.txt` and `test.txt`. The format for the training and test datasets must be a list of (word, correct_form), one per line, e.g:

    word <SELF>
    eror error

Whenever a word does not require normalisation, the second column should be `<SELF>`. If it does require normalisation, the second column should be the correct form of that word. Each document is separated by an additional newline character.

A sample train and test set has been provided (under `data/datasets/us_accidents_self`. The full version of the provided sample can be found at [https://github.com/Michael-Stewart-Webdev/us-accidents-dataset](https://github.com/Michael-Stewart-Webdev/us-accidents-dataset).

### Modifying the config file

The config file, `config.py`, can be modified according to your desired parameters. Notable options are:

    CF_DATASET = "US Acc"           # The name of the dataset.
    CF_PRETRAINED = False           # Whether to use pretrained word embeddings, which must be saved under `data/<dataset_embeddings>`.
    CF_EMBEDDING_MODEL = "Uniform"  # The embedding model to use. We found "Uniform" (i.e. randomly generated embeddings following a uniform distribution) generally works best, as detailed in the paper.

At this stage modifying the `CF_DATASET` requires adjusting the dictionaries defined later in the `config.py` code, so the easiest way to run the code is to simply replace the datasets under `data/datasets/us_accidents_self` with your own datasets and not modify the `CF_DATASET` (i.e. keep it as "US Acc").

Hyperparameters for the neural network are listed in the `__init__` function. The default hyperparameters were found to be the best performing hyperparameters for the US Accidents dataset.

## Running the code

First, run the `build_data.py` script to build the input data for the neural model:

    $ python build_data.py


Then, run the training script:

    $ python train.py

The script will evaluate the model's performance on the test set every epoch. The predictions at each epoch will be saved under `asset/<model folder>`, where `<model folder>` is an automatically generated name based on the parameters specified in `config.py`.



