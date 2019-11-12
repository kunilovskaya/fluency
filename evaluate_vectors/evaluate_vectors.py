#!/usr/bin/python2
# coding: utf-8

import gensim
import logging
import sys
from scipy import stats
import os
# Question 1 How do I load fasttext model with Gensim?
## The FastText binary format isnt compatible with Gensim's word2vec format; the former contains additional information about subword units, which word2vec doesn't make use of.
from gensim.models.wrappers import FastText


'''
this script
(1) collects the similarities for the word pairs (that have human scores) produced by the embeddings that are being evaluated
(2) measure Spearman correlation between the similarities produced by the models and the scores provided in the datasets
(3) compare to previously reported results for RNC 200mln: 42.55 and Aranea 10bln: 41.46 (see Kutuzov, A., & Kunilovskaya, M. (2018).
Size vs. structure in training corpora for word embedding models: Araneum Russicum maximum and Russian national corpus.
In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)
(Vol. 10716 LNCS).
'''

lemma = False
lang = 'en'

if lang == 'ru':
    if lemma:
        ## lemmatized_postagged ruwikiruscorpora no functional words #182
        m = '/home/masha/HTQE/resources/vectors/ruwikiruscorpora_upos_skipgram_300_2_2019/model.txt'
        
        ### genre-comparable RNC newspaper
        # m = '/home/masha/HTQE/resources/vectors/rncP5papers_lempos_w2v_skipgram_5_300_hs.model.bin'
        
        ### lempos with functionals #183 from rusvectores (99M texts, almost 1bln tokens)
        # m = '/home/masha/HTQE/resources/vectors/ruwikiruscorpora183_func_lempos_skipgram_5_300_hs.txt.gz'

        ## lempos with functionals in shared space
        # m = '/home/masha/HTQE/resources/vectors/en2ru_space/func_lempos/vectors-ru.txt.gz'

        # RU word similarity testset (a list of lines lemma_POS \t lemma_POS \t similarity score)
        simfile = '/home/masha/HTQE/resources/evaluate_vec/ruSimLex965_mystem2upos.tsv'

    elif not lemma:
        ## default CommonCrawl tokens vectors
        # m = '/home/masha/HTQE/resources/vectors/ru_cc300_func_nolem.vec.gz'

        ## the same default CommonCrawl tokens transposed into the shared vector space
        m = '/home/masha/HTQE/resources/vectors/en2ru_space/tokens/vectors-ru.txt.gz'
        
        ## genre-comparable RNC newspaper
        # m = '/home/masha/HTQE/resources/vectors/rncP_tokens_w2v_skipgram_5_300_hs.model.bin'
        
        ### if evaluating the model learnt on raw text
        simfile = '/home/masha/HTQE/resources/evaluate_vec/ruSimLex965.tsv'
        
if lang == 'en':
    if lemma:
        ## lempos, no functionals
        # m = "/home/masha/HTQE/resources/vectors/enwiki_upos_skipgram_300_5_2017/model.txt"
        ## lempos with functionals
        # m = '/home/masha/HTQE/resources/vectors/enwiki2019_func_lempos_skipgram_5_300.txt.gz'
        
        ## lempos with functionals in shared space
        m = '/home/masha/HTQE/resources/vectors/in_shared_space/vectors-en.txt'
        simfile = "/home/masha/HTQE/resources/evaluate_vec/en_lempos_SimLex.csv"
        
    elif not lemma:
        ## default CommonCrawl tokens vectors
        # m = "/home/masha/HTQE/resources/vectors/en_cc300_func_nolem.vec.gz"
        ## the same default CommonCrawl tokens transposed into the shared vector space
        m = '/home/masha/HTQE/resources/vectors/en2ru_space/tokens/vectors-en.txt.gz'
        
        simfile = "/home/masha/HTQE/resources/evaluate_vec/en_SimLex999.csv"
        
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Detecting the model format
if m.endswith('.txt.gz') or m.endswith('.vec.gz') or m.endswith('.txt'):
    model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=False)
elif m.endswith('.bin'):
    model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=True)
elif m.endswith('.model'): ## native gensim format
    model = gensim.models.Word2Vec.load(m)
else:
    print('I am not sure what format is this')
    
# get the vector of the word
if lang == 'ru':
    if lemma:
        # print(model['вдова_NOUN'])
        print(model.most_similar('вдова_NOUN'))
        print(model.similarity('жена_NOUN', 'вдова_NOUN'))
## ft vectors most similar for 'вдова_NOUN' = [('комова_NOUN', 0.8125011324882507), ('вова_NOUN', 0.796205461025238), ('баранова_NOUN', 0.7814720273017883), ('обнова_NOUN', 0.780508279800415), ('босанова_NOUN', 0.774837076663971), ('коснова_NOUN', 0.7725213170051575), ('вдова_VERB', 0.7646628618240356), ('ова_NOUN', 0.759023904800415), ('вдова_ADJ', 0.7581200003623962), ('бажова_NOUN', 0.7523174285888672)]
## ft vectors most similar for 'лошадь_NOUN' [('лошадь_PROPN', 0.758854329586029), ('налошадь_NOUN', 0.7587173581123352), ('эрдь_NOUN', 0.7378658652305603), ('рудь_NOUN', 0.7206859588623047), ('ложь_NOUN', 0.7165038585662842), ('.ть_NOUN', 0.7150841951370239), ('-и-ь_NOUN', 0.7124443650245667), ('лофть_NOUN', 0.7118619084358215), (',ть_NOUN', 0.7116912007331848), ('ть_NOUN', 0.7093919515609741)]
## №183 from RusVectores:
# [('вдова_PROPN', 0.7162735462188721), ('вдова_ADJ', 0.7052091956138611), ('вдовый_ADJ', 0.7033309936523438), ('вдовец_NOUN', 0.7006356716156006), ('дочь_NOUN', 0.6808878183364868), ('жена_NOUN', 0.6720510721206665), ('племянница_NOUN', 0.6712895631790161), ('дочь_VERB', 0.6685841083526611), ('родственница_NOUN', 0.6529833674430847), ('зять_NOUN', 0.631388783454895)]
### my rncP_4papers_lempos
## [('вдова_PROPN', 0.678353488445282), ('вдовец_NOUN', 0.6399790048599243), ('вдова_ADJ', 0.6314741969108582), ('жена_NOUN', 0.6173982620239258), ('мать_NOUN', 0.5711338520050049), ('вдовый_ADJ', 0.5702030658721924), ('супруга_NOUN', 0.5696138143539429), ('вдова_VERB', 0.5617521405220032), ('мария::владимировна_PROPN', 0.5614156723022461), ('покойная_ADJ', 0.557407557964325)]
    
    if not lemma:
        print(model.most_similar('вдова'))
        print(model.similarity('жена', 'вдова'))
# on №183 lemmas 0.6720511
### my rncP_4papers_lempos: 0.6173982
## on rncP_tokens: 0.7025673

# Pre-calculating vector norms
model.init_sims(replace=True)

### this does the overall word similarity!
a = model.evaluate_word_pairs(simfile, dummy4unknown=True, case_insensitive=False, restrict_vocab=500000)

## but if you want to reproduce manually
hu_scores = []
vec_scores = []

pairs = open(simfile,'r').readlines()

for i in pairs:
    # i = i.strip().split('\t')
    # print(len(i), i)
    if i.startswith('#'):
        continue
        
    lemma1, lemma2, sim = i.strip().split('\t')
    
    try:
        vec_score = model.similarity(lemma1, lemma2)
        # print(lemma1, '\t', lemma2, '\t', vec_score)
        # print(type(sim), type(vec_score))
        hu_scores.append(float(sim))
        vec_scores.append(vec_score)
    except KeyError:
        print(lemma1, lemma2, file=sys.stderr)
        continue
    #print model.similarity(u'крепостной_ADJ',u'раб_NOUN')

#tuples = model.most_similar(positive=u'алкоголь_NOUN', topn=5)
#for t in tuples:
    #print t[0].encode('utf-8'), '\t', t[1]
    
## Pearson correlation assumes that the data we are comparing is normally distributed.
print('Pearson corr: ', stats.pearsonr(hu_scores, vec_scores))
## on №183 lemmas: 0.32074; pvalue=1.962944303374411e-24
## on rncP_tokens 0.29614; pvalue=3.512508944308719e-20
### my rncP_4papers_lempos: 0.33202060839054137, 5.206948447375931e-26

## Spearman rank correlation is a non-parametric correlation measure
print('Spearman corr: ', stats.spearmanr(hu_scores, vec_scores))
## on №183 lemmas: 0.31050, pvalue=6.348178240006709e-23 Pairs with unknown words ratio: 0.4%
### my rncP_4papers_lempos: 0.3260818388307458, pvalue=4.278427536812984e-25 Pairs with unknown words ratio: 1.0%

## on rncP_tokens 0.26866720058523086, pvalue=9.294657332927851e-17

### remember the error of learning FT on postagged data!

### №183 OOV
# 2019-10-22 01:33:34,759 : INFO : Pairs with unknown words ratio: 0.4%
# врач_NOUN ортодонт_NOUN
# дождь_NOUN изморось_NOUN
# кажущийся_ADJ очевидный_ADJ
# ортодонт_NOUN дантист_NOUN

### my rncP_4papers_lempos
# 2019-10-23 16:19:13,294 : INFO : Pairs with unknown words ratio: 1.0%
# врач_NOUN ортодонт_NOUN
# дождь_NOUN изморось_NOUN
# износоустойчивость_NOUN повязка_NOUN
# кажущийся_ADJ очевидный_ADJ
# обручаться_VERB жениться_VERB
# ортодонт_NOUN дантист_NOUN
# ребячливый_ADJ безрассудный_ADJ
# счастие_NOUN удача_NOUN
# счастие_NOUN умиротворенность_NOUN
# ущелие_NOUN долина_NOUN




