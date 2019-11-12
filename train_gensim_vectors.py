#!python

## python3 venv/resources/train_gensim_vectors.py -t test/ontable.onebig.gz -fr w2v -m skipgram -w 5 -minc 10 -iter 5 -s binary
# python3 train_gensim_vectors.py -t /home/masha/HTQE/resources/input_corpora/rncP5papers_lempos.gz -fr w2v -m skipgram -w 5 -minc 10 -iter 5 -s binary
# python3 train_gensim_vectors.py -t "/media/masha/Seagate Expansion Drive/corpora_July2019/enwiki20191001.lempos.gz" -fr w2v -m skipgram -w 5 -minc 10 -iter 5 -s binary

### learning mock en and ru vectors to test MUSE ability to accept w2v text
## python3 /home/masha/HTQE/resources/train_gensim_vectors.py -t /home/masha/bnc6cat_650UD_lempos.gz -fr w2v -m skipgram -w 5 -minc 10 -iter 5 -s text
## python3 /home/masha/HTQE/resources/train_gensim_vectors.py -t /home/masha/HTQE/resources/input_corpora/lempos_small_rncP_1697.gz -fr w2v -m skipgram -w 5 -minc 10 -iter 1 -s text

## these need to be fed to
## (MUSE) masha@MAK:~/masha/HTQE$ ../MUSE/biling_emb/emb_code/python supervised.py --src_lang en --tgt_lang ru --src_emb /home/masha/bnc6cat_650UD_lempos_w2v_skipgram_5_300_hs.model.txt.gz --tgt_emb /home/masha/HTQE/resources/input_corpora/lempos_small_rncP_1697_w2v_skipgram_5_300_hs.model.txt.gz --dico_train resources/en-ru.0-5000_lempos.tsv --dico_eval resources/en-ru.5000-6500_lempos.tsv
## (MUSE) masha@MAK:~/masha/HTQE$ python ../MUSE/biling_emb/emb_code/unsupervised.py --src_lang en --tgt_lang ru --src_emb temp/ --tgt_emb temp/


import gensim
import logging, sys
### I am learning throu wrappers
from gensim.models import FastText
from gensim.models import Word2Vec
import multiprocessing
import argparse

import time

start = time.time()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# How many workers (CPU cores) to use during the training?
cores = multiprocessing.cpu_count()  # Use all cores we have access to
logger.info('Number of cores to use: %d' % cores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', help="Path to training file", required=True)
    parser.add_argument('--framework', '-fr', default='w2v', type=str, help="Choose framework", choices=['w2v', 'ft'])
    parser.add_argument('--model', '-m', default='skipgram', help="Name of the model", choices=['skipgram', 'cbow'])
    parser.add_argument('--window', '-w', help="Number of context words to consider", type=int, default=5)
    parser.add_argument('--mincount', '-minc', default=10, type=int, help="Min freq for vocabulary items: for raw input 10, for lemmatized 50")
    parser.add_argument("-iter", help="Run more training iterations (default 5)", type=int, default=5)
    parser.add_argument('--save', '-s', default='binary', help="Name of the model", choices=['binary', 'gensim', 'text'])
    args = parser.parse_args()


corpus = args.train
if args.model == 'skipgram':
        sg = 1
elif args.model == 'cbow':
        sg = 0
framework = args.framework
window = args.window
mincount = args.mincount
save = args.save
iterations = args.iter

logger.info(corpus)


data = gensim.models.word2vec.LineSentence(corpus)

if framework == 'w2v':
        outfile = corpus.replace('.gz', '_w2v') + '_' + str(args.model) + '_' + str(window)+ '_300_hs'
        ## Initialize and train a :class:`~gensim.models.word2vec.Word2Vec` model
        model = Word2Vec(data, size=300, sg=sg, min_count=mincount, window=window,
                hs=0, negative=5, workers=cores, iter=iterations, seed=42)
        
        if args.save == 'gensim':
                model.save(outfile + '.model')
        elif args.save == 'binary':
                model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
        elif args.save == 'text':
                model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)
        
if framework == 'ft':
        ### I don't understand abt these formats
        outfile = corpus.replace('.gz','_ft')+'_'+str(args.model)+'_'+str(window)+ '_3-6_hs' + '.model'
        ## Train, use and evaluate word representations learned using FastText with subword info
        model = FastText(data, size=300, sg=sg, min_count=mincount, window=window, min_n=3, max_n=6,
                hs=0, negative=5, workers=cores, iter=iterations, seed=42)
        ## The model can be stored / loaded via its: meth:`~gensim.models.fasttext.FastText.save`
        # and :meth: `~gensim.models.fasttext.FastText.load` methods
        ## or
        # model = FastText.load(fname)
        model.save(outfile)

end = time.time()
training_time = int(end - start)
print('Training embeddings on %s took %.2f minites' % (corpus.split('/')[-1], training_time/60))

