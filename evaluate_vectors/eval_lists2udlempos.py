#!/usr/bin/python2
# coding: utf-8

# TASK: lemmatize+postag
# EN word similarity list from https://fh295.github.io/simlex.html (see SOTA there)
# EN-RU bilingual glossary from https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries

from __future__ import print_function
from __future__ import division
from future import standard_library
import sys, os
import pandas as pd
from collections import Counter
from ufal.udpipe import Model, Pipeline
from alphabet_detector import AlphabetDetector
ad = AlphabetDetector()

simlex = '/home/masha/venv/resources/SimLex-999/SimLex-999.txt'
# biling = '/home/masha/HTQE/resources/en-ru.0-5000.txt'
biling = '/home/masha/HTQE/resources/en-ru.5000-6500.txt'

# df = pd.read_csv(simlex, delimiter="\t")
# print(df.head())
# d = Counter()
# all_pos = df['POS'].tolist()
#
# for pos in all_pos:
#     d[pos] += 1
# # print(d)
#
# df['POS'] = df['POS'].replace(to_replace='A',value='ADJ', regex=True).replace(to_replace='V',value='VERB', regex=True).replace(to_replace='N',value='NOUN', regex=True)
#
# # print(df.head())
#
# ## creat a list with lempos: glue the pos tag from one column to the lemma in the other
# lempos1 = df.apply(lambda x:'%s_%s' % (x['word1'],x['POS']),axis=1)
# lempos2 = df.apply(lambda x:'%s_%s' % (x['word2'],x['POS']),axis=1)
#
# # new_col = 'labels'
# df.insert(0, 'lempos1', lempos1)
# df.insert(1, 'lempos2', lempos2)
#
# # print(df.head())
# # print(df.columns.tolist())
#
# ## drop the old columns by retaining only the three necessary ones
# df1 = df[['lempos1', 'lempos2', 'SimLex999']]
# # print(df1.head())

# export_csv = df1.to_csv('/home/masha/venv/resources/en_lempos_SimLex.csv', index = None, header=True, sep="\t")


def process(pipeline, text='word'):

    processed = pipeline.process(text)
    # пропускаем строки со служебной информацией:
    content = [l for l in processed.split('\n') if not l.startswith('#')]

    # извлекаем из обработанного текста леммы, тэги и морфологические характеристики
    tagged = [w.split('\t') for w in content if w]

    for t in tagged:
        if len(t) != 10:
            continue
        (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
        lemma = lemma
        pos = pos

    this_lempos = '%s_%s' % (lemma, pos)

    return lemma, pos

def tag_lists(en, ru, langs=['ru', 'en']):
    count_all = 0
    for lang in langs:
        if lang == 'en':
            en_tagged = []
            udpipe_filename = '/home/masha/tools/udpipe/english-ewt-ud-2.3-181115.udpipe'
            model = Model.load(udpipe_filename)
            process_pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
            for w in en:
                count_all += 1
                w = w.strip()
                en_lem, en_pos = process(process_pipeline, text=w)
                en_tagged.append((en_lem, en_pos))
                
        elif lang == 'ru':
            ru_tagged = []
            udpipe_filename = '/home/masha/tools/udpipe/udpipe_syntagrus.model'
            model = Model.load(udpipe_filename)
            process_pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
            for w in ru:
                w = w.strip()
                ru_lem, ru_pos = process(process_pipeline, text=w)
                ru_tagged.append((ru_lem, ru_pos))
    filtered = []
    both = zip(en_tagged, ru_tagged)
    for i in both:
        if i[0][1] == i[1][1]:
            if all(ad.is_latin(uchr) for uchr in i[1][0] if uchr.isalpha()):
                continue
            else:
                en_w = '_'.join([i[0][0],i[0][1]])
                ru_w = '_'.join([i[1][0], i[1][1]])
                filtered.append((en_w, ru_w))
    print(count_all, len(filtered))
    
    return filtered

bi = open(biling, "r")
eng = []
rus = []

for i in bi:
    i = i.strip()
    bits = i.split()
    eng.append(bits[0])
    rus.append(bits[1])
    
if len(eng) == len(rus):
    
    pairs = tag_lists(eng,rus)
    print(len(pairs))
    set_pairs = set(pairs)
    print(len(set_pairs))

    with open('/home/masha/HTQE/resources/en-ru.5000-6500_lempos.tsv', 'w') as out:
        for i in set_pairs:
            # print(i)
            # print(i[0] + '\t' + i[1])
            out.write(i[0] + '\t' + i[1] + '\n')
        
        

        

