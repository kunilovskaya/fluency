#! python3
# coding: utf-8

# fluency module component: this calculates perplexity of a model trained on the genre-compareble
# non-translations on good and bad translations on the assumption that bad translations are distinctively dysfluent
# ## it is my_test_lm.py adapted to good-bad classification problem
# this script (unlike my_test_lm.py) expects separate sentence-per-line text files as input rather than onebig corpus the models were trained on

"""
A language model is a probability distribution over sentences.
We are testing three models (oracles), the first two trained on the genre-comparable corpus, ELMO is pre-trained on wikipedia+ruscorpora
(1) statistical trigram model (aka HMM) trained on lempos
(2) RNN for k=2 (i.e. for trigramms) and
(3) elmo
The script expects a folder with the classification labels as names of subfolders containing text instances as separate documents (ex. lempos_rnn/good/*.lempos and lempos_rnn/bad/*.lempos)

In the first scenario run with
(the real model: lempos_rncP_5papers_rnn_lm.h5)
python3 -W ignore my_LMs/LM_oracles.py --test data/LMs_predict_quality/lempos_rnn --model rnn --modelfile small_rncP_lempos_rnnLMe.h5

In the second scenario:
### IMPORTANT! don't forget to wrap each file in a folder for ELMo!
python3 -W ignore my_LMs/LM_oracles.py --test data/LMs_predict_quality/test_elmo --model elmo --modelfile /home/masha/HTQE/resources/vectors/out_ruwiki --device gpu
"""

## masha@MAK:~/HTQE$

from __future__ import absolute_import, division, print_function, unicode_literals

from keras import backend as K
import tensorflow as tf
import pandas as pd
import os, sys
import argparse
## for trigram rnn-based LM
from my_models import *
from oracle_support import get_textlines, get_perplexities, cv_classify, visual
## for ELMo
from bilm.training import test, load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset


parser = argparse.ArgumentParser(description='Compute test perplexity with rnn-based LM or ELMo last softmax layer')
parser.add_argument('--test', required=True, help="Path to the folder with one-sentence per line texts and a class label for a folder name")
parser.add_argument('--model', required=True, choices=['hmm', 'elmo', 'rnn'])
parser.add_argument('--modelfile', required=True, help='Language Model file or ELMo folder')
parser.add_argument('--algo', type=str, default='XGBoost', help='Which classifier to run on the perplexities: SVM, dummy, RF, XGBoost')
## additional arguments required for ELMo
# parser.add_argument('--vocab_file', help='Vocabulary file for ELMo') ## hard-coded now as a file from modelfile folder like options
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for ELMo; keep 1 for per-word entropies and providing for the stateful nature of ELMo')
parser.add_argument('--device', type=str, default='gpu', help='Choose the device. Options: cpu, gpu')

args = parser.parse_args()
start = time.time()

if args.device == 'cpu':
    ## select the number of CPU workers=threads to be used
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4,
    inter_op_parallelism_threads=4, allow_soft_placement=True) #,device_count = {'CPU' : 2, 'GPU' : 1}
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if args.device == 'gpu':
    config = tf.compat.v1.ConfigProto()
    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    print('=== DEFAULT availability',K.tensorflow_backend._get_available_gpus())
    # sess = tf.Session(config=config)
    ## select what you want to run it on
    # config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 4})
    # config.gpu_options.visible_device_list = '1'  # only see the gpu 1
    config.gpu_options.visible_device_list = '0,1'  # see the gpu 0, 1, 2
    
    ## replace: tf.ConfigProto by tf.compat.v1.ConfigProto
    print("====== Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    tf.compat.v1.logging.set_verbosity(tf.logging.INFO)
    
print('=== Loading pre-trained %s language model ...' % args.model.upper())
## size of context for rnn-based LM
k = 2
res_perplexities2 = []

if args.model == 'elmo':
    ## loading the ELMo model just once
    options, ckpt_file = load_options_latest_checkpoint(args.modelfile)
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = load_vocab(args.modelfile + '/vocab.txt', max_word_length)
    
    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }
    
    model = None

## a simple LSTM that casts the next word prediction as a classification task (choose from all the words in the vocabulary)
elif args.model == 'rnn':
    model = RNNLanguageModel(k=k, mincount=2)
    model.load(args.modelfile)  # Loading the model from file
    res_perplexities0 = []
    res_perplexities1 = []
    
    count_in = 0
    count_oov = 0
    options = None
    kwargs = None
    vocab = None
    ckpt_file = None
    
## usues trigram statistics
elif args.model == 'hmm':
    model = MarkovLanguageModel(k=k)
    model.load(args.modelfile)  # Loading the model from file
    res_perplexities0 = []
    res_perplexities1 = []
    
    count_in = 0
    count_oov = 0
    options = None
    kwargs = None
    vocab = None
    ckpt_file = None
else:
    raise ValueError


print('=== Loading test corpus...', file=sys.stderr)

labels = []
fns = []

outfile = open('/home/masha/HTQE/%s_perplexities.tsv' % args.model, 'w')
count = 0
for subdir, dirs, files in os.walk(args.test):
    for i, file in enumerate(files):
        count += 1
        filepath = subdir + os.sep + file
        print('== CURRENT text:', filepath)
        if args.model == 'rnn':
            label = (subdir + os.sep).split('/')[-2]
        if args.model == 'hmm':
            label = (subdir + os.sep).split('/')[-2]
        if args.model == 'elmo':
            label = (subdir + os.sep).split('/')[-3]
        
        labels.append(label)
        
        fns.append(file)
        ## collect textlines
        sents = get_textlines(filepath)
        
        # calculate and collect perplexities, i.e. our X feature
        if args.model == 'rnn':
            ## res are mean perplexities over all sentences in each text
            res0, res1, res2, OOV, IN = get_perplexities(sents, model, k=k)
            res_perplexities2.append(res2)
            count_in += OOV
            count_oov += IN
            res_perplexities0.append(res0)
            res_perplexities1.append(res1)
        
        if args.model == 'hmm':
            res0, res1, res2, OOV, IN = get_perplexities(sents, model, k=k)
            res_perplexities2.append(res2)
            count_in += OOV
            count_oov += IN
            res_perplexities0.append(res0)
            res_perplexities1.append(res1)
            
        if args.model == 'elmo':

            filepath = subdir + os.sep
            if options.get('bidirectional'):
                data = BidirectionalLMDataset(filepath, vocab, **kwargs)
                # print(data)
            else:
                data = LMDataset(filepath, vocab, **kwargs)

            res2 = test(options, ckpt_file, data, batch_size=args.batch_size)

            res_perplexities2.append(res2)

        outfile.write(file + '\t' + label + '\t' + str(res2) + '\n')

        if count % 5 == 0:
            print('I have calculated perplexities for %s files' % count, file=sys.stderr)



print('=== Just a sanity check on the perplexity calculations: ')
print(labels[:5], fns[:5], res_perplexities2[:5])

print('Texts with the most extreme text-level perplexities:')
df = pd.DataFrame(list(zip(fns, labels, res_perplexities2)),
               columns =['files', 'label', 'perpl'])
## saving before sorting
df.to_csv('/home/masha/HTQE/%s_perplexities_backup.tsv' % args.model, sep='\t', index=False)

df = df.sort_values(by=['perpl'], ascending=False)
print('Top 5 most perplexing:')
print(df.iloc[:5])
print('Bottom 5 least perplexing:')
print(df.iloc[-5:])

print('=== Here go the classification results on the average text-level perplexities for good-bad from %s' % args.model.upper())

res_perplexities2 = np.array([res_perplexities2]).transpose()
print(res_perplexities2.shape)

print('\nOn the perplexities from cross-entropies:')

cv_classify(res_perplexities2, labels, algo=args.algo, class_weight='balanced', n_splits=10, minor=None, random=42)

if args.model == 'rnn' or args.model == 'hmm':
    print('\n=== Additionally, if you are running rnn-based LM, check out results on the alternative ways to calculate text-level perplexities:')

    print('Number of OOV (wds from test not in train)?', count_oov)
    print('Number of words from test seen in the models voc:', count_in)

    ### transform into a 2-D array to be feed to the classifier
    res_perplexities0 = np.array([res_perplexities0]).transpose()
    print(res_perplexities0.shape)

    res_perplexities1 = np.array([res_perplexities1]).transpose()
    print(res_perplexities1.shape)

    print('\nOn default huge numbers:')
    cv_classify(res_perplexities0, labels, algo=args.algo, class_weight='balanced', n_splits=5, minor=None, random=42) ## SVM, dummy, RF, XGBoost

    print('\nOn the perplexities from sentence-average entropies (%s):' % args.algo)
    cv_classify(res_perplexities1, labels, algo=args.algo, class_weight='balanced', n_splits=5, minor=None, random=42)

end = time.time()
processing_time = int(end - start)
print('Time spent on getting models probabilities, calculating perplexities and running a one-feature classification on %s for %s texts in %s took %.2f minites' % (args.device.upper(), len(labels), args.test, processing_time / 60), file=sys.stderr)

outfile.close()