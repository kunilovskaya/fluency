#! python3
# coding: utf-8

## fluency module component:
# this calculates perplexity of a model trained on the genre-compareble non-translations when it is applied to good and bad translations
# the assumption is that bad translations are more dysfluent and perplexing for models trained on the normative language

## requires my_models module and tf==1.14.0 for cuda=10.1 and CPU with AVX (or tf==1.5 for no-AVX CPU), keras, smart-open, etc

## USAGE:
## training a RNN with embeddings pre-trained on the same comparable corpus:
# (gpuTF) masha@MAK:~/HTQE$ python3 my_LMs/my_train_lm.py --train resources/lempos_rnc5papers.gz --embeddings rncP5papers_lempos_w2v_skipgram_5_300_hs.model.bin --model rnn --save oracles/rnn_rnc5papers_lempos.h5

## training a HMM
## (base) lpvoid@rig1:~/masha$ python3.6 -W ignore my_LMs/my_train_lm.py --train corpora/lempos_rnc5papers.gz --model hmm --save my_LMs/oracles/hmm_rnc5papers_lempos.h5 --device gpu

### I don't have enough RAM to fit all the 12 mln sentences of the real corpus into memory (lempos_rnc5papers.gz), so all results pertain to a 100k sentences sample thereof, deemed most genre-comparable to our test-data


import argparse
from my_models import *
import sys, os
from tensorflow.python.client import device_lib
## 0 = (default, debug) all messages are logged (default behavior); 1 = INFO messages are not printed; 2 = INFO and WARNING messages are not printed;
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', help="Path to training file", required=True)
    parser.add_argument('--embeddings', '-e', help="Path to external embeddings file")
    parser.add_argument('--ngrams', '-k', help="Number of context words to consider",
                        type=int, default=2)
    parser.add_argument('--model', '-m', default='rnn', required=True,
                        choices=['hmm', 'rnn'])
    parser.add_argument('--save', '-s', help='Save model to (filename)...')
    parser.add_argument('--device', type=str, default='gpu', help='Choose the device. Options: cpu, gpu')

    args = parser.parse_args()
    
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
    if args.device == 'gpu':
        print('== NUMBER of devices??', device_lib.list_local_devices())
    

    EOL = 'endofline'  # Special token for line breaks
    k = args.ngrams

    lines = []  # Training corpus
    for line in open(args.train, 'r'):
        res = line.strip() + ' ' + EOL
        lines.append(tokenize(res, ext_emb_UD=True))
    print("Our first training line:\n", lines[0])

    if args.model == 'hmm':
        model = MarkovLanguageModel(k=k)
        model.train(lines)
    elif args.model == 'rnn':
        if args.embeddings:
            model = RNNLanguageModel(k=k, mincount=10, lstm=128, ext_emb=args.embeddings)
        else:
            model = RNNLanguageModel(k=k, mincount=10)
            ## choose your val_split wisely: for lempos_rncP_5papers with 12M sentences, the 0.0005 share is 6037 sents (with 95804 lempos) out of 12074050
            model.train(lines, val_split=0.0005)
    else:
        raise ValueError

    print('Training...', file=sys.stderr)
    

    
    if args.save:
        model.save(args.save)
