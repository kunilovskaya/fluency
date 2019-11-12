#! python3
# coding: utf-8

## heavily based on the tutorial https://github.com/akutuzov/nlp_lm
## this is a module to be used with the training script (my_train_lm.py) and testing script (LM_oracles_void.py)

import re
import sys
import pickle
import numpy as np
import time
from collections import Counter
## Change your import to tensorflow.keras
from keras.utils import to_categorical
from keras import backend
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
## tensorflow is required (for tf==1.14.0, but not for tf==1.5! or for biLM where tf==1.13.1, tf-gpu==1.12.0) for loading the model to avoid ValueError: Unknown initializer: GlorotUniform
from tensorflow.keras.models import load_model
from keras.callbacks import TensorBoard, EarlyStopping

## gensim==3.8.1 smart-open==1.8.4
from smart_open import open
import logging
import gensim


def tokenize(string, ext_emb_UD=True):
    token_pattern = re.compile('(?u)[\w,."-:?)(–;!]+') ## add the list of empirically observed punctuation to avoid the UKN lempos of '_PUNCT'
    if ext_emb_UD:
        tokens = [t for t in token_pattern.findall(string)]
    else:
        ### if training HMM/RNN on lempos
        tokens = [t for t in token_pattern.findall(string)]
        ### if using raw corpus
        # tokens = [t.lower() for t in token_pattern.findall(string)]
    return tokens

class MarkovLanguageModel:
    """
    This model predicts the next word based on tri-gram statistics from the corpus
    (so it finally uses the context).
    """

    def __init__(self, k=2):
        self.vocab = Counter()
        self.trigrams = {}
        self.k = k
        self.corpus_size = 0
        self.probs = None

    def train(self, strings):
        for string in strings:
            self.vocab.update(string)
            for nr, token in enumerate(string):
                self.corpus_size += 1
                if nr < self.k:
                    continue
                prev_context = (string[nr - 2], string[nr - 1])
                if prev_context not in self.trigrams:
                    self.trigrams[prev_context] = Counter()
                self.trigrams[prev_context].update([token])
        print('Vocabulary built:', len(self.vocab), file=sys.stderr)
        # Word probabilities:
        self.probs = {word: self.vocab[word] / self.corpus_size for word in self.vocab}
        print('Trigram model built:', len(self.trigrams), 'trigrams', file=sys.stderr)
        return self.vocab, self.trigrams, self.probs

    def score(self, entity, context=None):
        # print(entity, context)
        oov = 0
        in_train_voc = 0
        if context in self.trigrams:
            ## keys in trigram dict: ('поэтому_ADV', 'порядок_NOUN')
            ## variant is class 'collections.Counter': 'действительно_ADV': 1, 'дмитрий_PROPN': 1,
            variants = self.trigrams[context]
            ### if token in the 3d member of the recorded trigrams
            if entity in variants:
                # print('this is the freq of the token with this context', variants[entity])
                # print('this should be the accumulated freqs of other possibilities observed in the train', sum(variants.values()), len(variants))
                probability = variants[entity] / sum(variants.values())  # Relative to context
                ### what happens if not???
                return probability, oov, in_train_voc
        if entity in self.probs:
            in_train_voc += 1
            probability = self.probs[entity]  # Proportional to frequency
        # Decreased probability for out-of-vocabulary words:
        else:
            oov += 1
            probability = 0.99 / self.corpus_size
        return probability, oov, in_train_voc

    def generate(self, context=None):
        if context in self.trigrams:
            variants = self.trigrams[context]
            bigram_freq = sum(variants.values())
            words = list(variants)
            probabilities = [variants[word] / bigram_freq for word in words]
            prediction = np.random.choice(words, p=probabilities)
        else:
            words = list(self.probs)
            probabilities = [self.probs[w] for w in words]
            prediction = np.random.choice(words, p=probabilities)
        return prediction

    def save(self, filename):
        print('Saving the model to', filename, file=sys.stderr)
        out_dump = [self.probs, self.trigrams, self.corpus_size]
        with open(filename, 'wb') as out:
            pickle.dump(out_dump, out)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.probs, self.trigrams, self.corpus_size = pickle.load(f)
        print('Model loaded from', filename, file=sys.stderr)

class RNNLanguageModel:
    """
    This model trains a simple LSTM on the training corpus and casts the next word prediction
    as a classification task (choose from all the words in the vocabulary).
    """

    def __init__(self, k=2, lstm=128, emb_dim=5, batch_size=6, ext_emb=None, mincount=None):
        backend.clear_session()
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.k = k
        self.vocab = Counter()
        self.embed = emb_dim
        self.rnn_size = lstm
        self.word_index = None
        self.inv_index = {}
        self.model = None
        self.corpus_size = 0
        self.batch_size = batch_size
        self.mincount = mincount
        self.ext_emb = ext_emb
        self.ext_vectors = None
        self.ext_vocab = None
        if self.ext_emb:
            external_embeddings = None
            if ext_emb.endswith('.model'):
                external_embeddings = gensim.models.KeyedVectors.load(ext_emb)
            elif ext_emb.endswith('bin') or ext_emb.endswith('bin.gz'):
                external_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                    ext_emb, binary=True)
            elif ext_emb.endswith('.txt.gz') or ext_emb.endswith('.txt') \
                    or ext_emb.endswith('.vec.gz') or ext_emb.endswith('.vec'):  # Текстовый формат
                external_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                    ext_emb, binary=False, unicode_errors='replace')
            else:
                print('Wrong file format for the external embedding file!', file=sys.stderr)
                print('Please use either Gensim models or binary word2vec models', file=sys.stderr)
                exit()
            self.ext_vectors = external_embeddings.vectors
            self.ext_vocab = external_embeddings.index2entity
            self.ext_word_index = {}
            for nr, word in enumerate(self.ext_vocab):
                self.ext_word_index[word] = nr

    def train(self, strings, val_split=None):
        for string in strings:
            ## building vocabulary from train/test corpus
            self.vocab.update(string)
        if self.mincount:
            self.vocab = {word: self.vocab[word] for word in self.vocab
                          if self.vocab[word] >= self.mincount}
        print('Vocabulary built:', len(self.vocab), file=sys.stderr)
        ## this is the train data vocabulary
        vocab_size = len(self.vocab)
        
        self.word_index = list(self.vocab)
        for nr, word in enumerate(self.word_index):
            ## indexing all words in train wrt to the voc of train
            self.inv_index[word] = nr

        sequences = list()
        for string in strings:
            for nr, token in enumerate(string):
                self.corpus_size += 1
                ## skip the first few words to have k in the left window
                if nr < self.k:
                    continue
                if self.ext_emb:
                    ## this generates chunks of text = the proverbial sliding window, if k=2, it returns trigrams
                    data = [string[nr - 2], string[nr - 1], token]
                    ## if the first (two) words in the trigram are found in embeddings voc and the token is found in the keys of the train voc
                    if all([word in self.ext_word_index for word in data[:2]]) \
                            and token in self.inv_index:
                        ## replace word-strings with indices
                        encoded_context = [self.ext_word_index[w] for w in data[:2]]
                        ## get the index of the word in train
                        encoded_token = [self.inv_index[token]]
                        ## I get two indices from the embeddings voc and one from the train voc???
                        encoded = encoded_context + encoded_token
                        sequences.append(encoded)
                else:
                    data = [string[nr - 2], string[nr - 1], token]
                    if all([word in self.inv_index for word in data]):
                        encoded = [self.inv_index[w] for w in data]
                        ## or from the train voc only
                        sequences.append(encoded)
        print('Total sequences to train on:', len(sequences), file=sys.stderr)
        sequences = np.array(sequences)

        # Describe the model architecture
        self.model = Sequential()
        if self.ext_emb:
            weights = self.ext_vectors
            
            # Take the weights from the Gensim model, freeze the layer
            self.model.add(Embedding(weights.shape[0], weights.shape[1], weights=[weights],
                                     input_length=self.k, trainable=False, name='embeddings'))
        else:
            self.model.add(Embedding(vocab_size, self.embed, input_length=self.k,
                                     name='embeddings'))
        self.model.add(LSTM(self.rnn_size, name='LSTM'))
        self.model.add(Dense(vocab_size, activation='softmax', name='output'))
        print(self.model.summary(), file=sys.stderr)

        # Model compilation:
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        loss_plot = TensorBoard(log_dir='logs/LSTM')
        ## to look at the visualizations: import tensorflow as tf, %load_ext tensorboard, tensorboard --logdir logs/LSTM
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=2, verbose=2)

        # We are using the last lines of the corpus as a validation set:
        ### for a 16.7 K texts the last 0.005 of all texts make around 80 texts (16k x 0.005=80), they have around 60K tokens that we will use as validation and test
        ## for lempos_rncP_5papers.gz будем тестировать на 95,804 словах в 6,037 предложениях, что составляет 0.0005th долю от 12,074,050 (весь объем корпуса в предложениях)
        val_split = val_split
        train_data_len = int(len(sequences) * (1 - val_split))
        val_data_len = int(len(sequences) * val_split)

        train_data = sequences[0:train_data_len, :]
        val_data = sequences[-val_data_len:, :]

        print('Training on:', train_data_len, file=sys.stderr)
        print('Validating on:', val_data_len, file=sys.stderr)

        val_contexts, val_words = val_data[:, :-1].astype(int), val_data[:, -1].astype(int)

        val_words = to_categorical(val_words, num_classes=vocab_size) ## ohoho!
        val_data = val_contexts, val_words ## this .astype(int) does not help

        # How many times per epoch we will ask the batch generator to yield a batch?
        steps = train_data_len / self.batch_size ### double slash fixed the problem of TypeError: 'float' object cannot be interpreted as an integer
        print('Steps:', int(steps), file=sys.stderr)

        # Training:
        start = time.time()

        ## Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
        history = self.model.fit_generator(self.batch_generator(
            train_data, vocab_size, self.batch_size), steps_per_epoch=steps, epochs=10,
            verbose=2, callbacks=[earlystopping,loss_plot], validation_data=val_data) #,callbacks=[earlystopping,loss_plot],
        end = time.time()
        training_time = int(end - start)
        print('LSTM training with %s embeddings took %.2f minites' % (self.ext_emb, training_time/60), file=sys.stderr)

        return self.vocab

    def batch_generator(self, data, vocab_size, batch_size):
        print('Voc size: %s\nbatch_size: %s\n' % (vocab_size, batch_size), file=sys.stderr)
        """
        Generates training batches
        """

        while True:
            # Separating input from output:
            contexts = np.empty((batch_size, self.k), dtype=int)
            words = np.empty((batch_size, vocab_size), dtype=int)
            inst_counter = 0
            for row in data:
                context, word = row[:-1], row[-1]
                word = to_categorical(word, num_classes=vocab_size)
                contexts[inst_counter] = context
                words[inst_counter] = word
                inst_counter += 1
                if inst_counter == batch_size:
                    yield (contexts, words)
                    contexts = np.empty((batch_size, self.k))
                    words = np.empty((batch_size, vocab_size))
                    inst_counter = 0

    def score(self, entity, context=None):
        # print(entity, context)
        oov = 0
        in_train_voc = 0
        if self.ext_vocab:
            context_vocab = self.ext_word_index  # if we use external word embeddings
        else:
            context_vocab = self.inv_index  # if we use only training corpus information
        ## filter out trigrams where the context words don't have vectors
        if not all([word in context_vocab for word in context]):
            pass

        ## if entity is the train (!! loaded with the model at test time) corpus voc
        if entity in self.inv_index and all([word in context_vocab for word in context]):
            in_train_voc = 1
            entity_id = self.inv_index[entity]
            context_ids = np.array([[context_vocab[w] for w in context]])
            # Probability distribution: probabilities of all words in the voc, given the context, produced by the model called to predict
            prediction = self.model.predict(context_ids).ravel()
            probability = prediction[entity_id]  # Probability of the correct word
        # Decreased probability for out-of-the-model-vocabulary-(i.-e-train-corpus) words
        else:
            oov = 1
            probability = (1 / self.corpus_size)
        return probability, oov, in_train_voc  ### the division is really tricky!!! the double slash ruined the output by returning zero!


    def generate(self, context=None):
        if self.ext_vocab:
            context_vocab = self.ext_word_index  # if we use external word embeddings
        else:
            context_vocab = self.inv_index  # if we use only ming corpus information

        if all([word in context_vocab for word in context]):
            context_ids = np.array([[context_vocab[w] for w in context]])
            prediction = self.model.predict(context_ids).ravel()  # Probability distribution
            word_id = prediction.argmax()  # Entry with the highest probability
            word = self.word_index[word_id]  # Word corresponding to this entry
        else:
            word = np.random.choice(self.word_index)
        return word

    def save(self, filename):
        self.model.save(filename)
        out_dump = [self.inv_index, self.corpus_size, self.ext_vocab]
        with open(filename.split('.')[0] + '.pickle.gz', 'wb') as out:
            pickle.dump(out_dump, out)
        print('Model saved to {} and {} (vocabulary)'.format(filename, filename.split('.')[0] +
                                                             '.pickle.gz'), file=sys.stderr)

    def load(self, filename):
        self.model = load_model(filename)
        voc_file = filename.split('.')[0] + '.pickle.gz'
        with open(voc_file, 'rb') as f:
            self.inv_index, self.corpus_size, self.ext_vocab = pickle.load(f)
        self.word_index = sorted(self.inv_index, key=self.inv_index.get)
        if self.ext_vocab:
            self.ext_word_index = {}
            for nr, word in enumerate(self.ext_vocab):
                self.ext_word_index[word] = nr
        print('Model loaded from {} and {}'.format(filename, voc_file), file=sys.stderr)
        print(self.model.summary(), file=sys.stderr)
