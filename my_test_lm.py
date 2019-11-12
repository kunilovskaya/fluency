#! python3
# coding: utf-8

## this script expects the test corpus in one-sentence-per-line format

##real data
## == rnn
## from HTQE: python3 -W ignore my_LMs/my_test_lm.py --test resources/lempos_small_rncP_1697.gz --model rnn --modelfile small_rncP_lempos_rnnLMe.h5 --device gpu
## == hmm
## (gpuTF) masha@MAK:~/HTQE$ python3 -W ignore my_LMs/my_test_lm.py --test data/LMs_predict_quality/bad_lempos.ol --model hmm --modelfile my_LMs/oracles/hmm_rnc1697_lempos.h5 --device gpu

import time
import argparse
from my_models import *
import os
from tensorflow.python.client import device_lib
## 0 = (default, debug) all messages are logged (default behavior); 1 = INFO messages are not printed; 2 = INFO and WARNING messages are not printed;
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

## A language model is a probability distribution over sentences.


if __name__ == "__main__":
    count = 0
    parser = argparse.ArgumentParser()
    # smart-open should handle .gz okey??
    parser.add_argument('--test', '-t', help="Path to testing file", required=True)
    parser.add_argument('--model', '-m', default='random', required=True,
                        choices=['hmm', 'rnn'])
    parser.add_argument('--modelfile', '-mf', required=True, help='Language Model name')
    parser.add_argument('--device', type=str, default='gpu', help='Choose the device. Options: cpu, gpu')
    
    args = parser.parse_args()
    start = time.time()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
    if args.device == 'gpu':
        print('== NUMBER of devices??', len(device_lib.list_local_devices()))
    
    EOL = 'endofline'  # Special token for line breaks

    print('Loading test corpus...', file=sys.stderr)
    lines = []
    ## smart-open should handle .gz okey??
    ## this is a sentence
    for line in open(args.test, 'r'):
        res = line.strip() + ' ' + EOL
        lines.append(tokenize(res))
        
    k = 2

    if args.model == 'hmm':
        model = MarkovLanguageModel(k=k)
    elif args.model == 'rnn':
        model = RNNLanguageModel(k=k, mincount=2)
    else:
        raise ValueError
    
    model.load(args.modelfile)  # Loading the model from file

    print('Testing...', file=sys.stderr)
    
    perplexities0 = []
    perplexities1 = []
    perplexities2 = []
    
    count_oov = 0
    count_in = 0
    for l in lines:
        
        count += 1
        probabilities = []
        sent_entropies = []
        new_entropies = []
        
        for nr, token in enumerate(l):
            if nr < k:
                continue
            context = (l[nr - 2], l[nr - 1])
            probability, boo, boo1 = model.score(token, context=context)
            count_oov += boo
            count_in += boo1
            
            probabilities.append(probability)

            ## entropy per unit is the inverse probability of the test set normalized for the number of words
            ### higher probability, smaller entropy and vice versa
            word_entropy = - 1 * np.log2(probability)
            if np.isfinite(word_entropy):
                sent_entropies.append(word_entropy)
            else:
                ## there are three offensive lines that return infs 59739, 57148, 49384 in lempos_small_rncP_1697.gz
                print('EARLier GOTCHA\n', word_entropy, type(word_entropy), count, '\n========\n', l, '\n========\n')
            
        ## default
        sent_perplexity0 = [2 ** ent for ent in sent_entropies]
        ## collect the sentence-level (averaged over the sentence words) perplexities into a list for the corpus
        if np.isfinite(np.mean(sent_perplexity0)):
            ## this is normalisation at the texy level
            perplexities0.append(np.mean(sent_perplexity0))
        else:
            ## there are three offensive lines that return infs 59739, 57148, 49384 in lempos_small_rncP_1697.gz
            print('GOTCHA\n', sent_perplexity0, type(sent_perplexity0), count, '\n========\n', l, '\n========\n')
            
       ##  Per-word cross-entropy of the trigram model for every sentence: sum of negative logs of the words probabilities / number of words in the sent
        # (see lascarides: https://www.inf.ed.ac.uk/teaching/courses/fnlp/lectures/04_slides-2x2.pdf)
        mean_ent = np.mean(sent_entropies)
        
        ## perplexity is the exponentiation of the entropy!
        ### exponentiate only once
        perplexity = 2 ** mean_ent
    
        if np.isfinite(perplexity):
            perplexities1.append(perplexity)
        else:
            ## there are three offensive lines that return infs 59739, 57148, 49384 in lempos_small_rncP_1697.gz
            print('GOTCHA for perplexities1\n', perplexity, type(perplexity), count, '\n========\n', l, '\n========\n')

        ## H(p) is the entropy of the distribution p(x): H(p) =-Σ p(x) log p(x). from https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3
        ## based on proper formular which gets the entropy of a sent as a sum of products of (probs X log2of probs) of all word in the sent

        ## CROSS-entropy: the sum of products of probability of a word observed in the train and log2 of the one predicted for the test (a LM can be tested on the train)
        
        for prob in probabilities:
            lst_new_entropies = [-prob * np.log2(prob) for prob in probabilities] ## you might be able to save time by using -np.dot(a, b) in place of -np.sum(a * b)
            new_entropies.append(lst_new_entropies)
            
        av_entropy = np.mean(new_entropies) ## this is where the summing of all per-word entropy happens and normalisation to the number of words in the sent occurs
        new_perplexity = 2 ** av_entropy
        if np.isfinite(new_perplexity):
            perplexities2.append(new_perplexity)
        else:
            ## there are three offensive lines that return infs 59739, 57148, 49384 in lempos_small_rncP_1697.gz
            print('GOTCHA for perplexities2\n', new_perplexity, type(new_perplexity), count, '\n========\n', l, '\n========\n')

    print('\nDefault Perplexity0: {0:.5f}, {1} running trigrams'.format(np.mean(perplexities0),len(perplexities0)))
    ## this is Per-word cross-entropy of the trigram model for every sentence, suggested in Lascarides
    print('Perplexity1  (from exponentiated average per-sent entropies): %s' % np.mean(perplexities1))
    print('Perplexity2 derived from Shanons (cross)entropy: negative sum of products of prob and log of prob\t%s' % np.mean(perplexities2))

    print('Number of OOV?', count_oov)
    print('Number of words from test seen in the models voc:', count_in)

    end = time.time()
    processing_time = int(end - start)
    print('Time spent on getting models probabilities and calculating perplexities for each sent %s took %.2f minites' % (args.test, processing_time / 60), file=sys.stderr)
