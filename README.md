# fluency
This is an attempt to capture fluency as an aspect of translation quality along with accuracy
(for `accuracy` module results see [this project](https://github.com/kunilovskaya/accuracy))

*`fluency` module*
* using language model perplexities as a measure of fluency for classifying good/bad translations

## Fluency module
Learn language models on untranslated (genre-comparable) Russian corpus and measure the models perplexity on good/bad translations on the assumption that bad translations should return higher values for the models' perplexity. Use these values as a single feature in a (LR) classification task

## Models, methods and training resorces
### Models attempted:
*A language model is a probability distribution (of/for the words in a given vocabulary) over sentences.*
1. MarkovModel (HMM) (trigram statistics: what is the probability of a word to occur after the 2-word context, given the earlier observed frequency of all trigrams)
1. an LSTM (RNN) which returns the probability of a word to occur next against all other words in the vocabulary; the probability is retrieved from solving a (very multi-class) classification task
1. Language model layer of a pre-trained ELMo which returnes perplexities from cross-entropy loss function for each word

### Methods to calculate LM result (i.e. text-level perplexity) and the average perplexity values on train, good, bad 
1. *word-level perplexity averaged twice*

LM probability for each word, given the context of k=2, turned into entropy understood as inverse probability (entropy per unit is the inverse probability of the test set (normalized for the number of words)) (word\_entropy = - 1 * np.log2(probability), and then exponentiated to get perplexities: lst\_perplexity0 = [2 ** ent for ent in sent_entropies], and text-level normalisation perplexities0.append(np.mean(lst_perplexity0))

| model |                train                   |                    good         |         bad         |
|-------|----------------------------------------|---------------------------------|---------------------|
|  HMM  |                   74.07                |       117,639.9                 |     120,406.2       |
|-------|----------------------------------------|---------------------------------|---------------------|
|  RNN  |95,783,640,050,446,191,521,813,233,664.0|299,911,314,419,825,181,196,288.0|503,315,467,063,556.4|
|-------|----------------------------------------|---------------------------------|---------------------|

(NB! Insane values for RNN on the same training corpus, same test data, same script: LM\_oracles\_void.py)

1. *exponentiated average per-sentence entropy, normalised per number of sentences*: 

mean\_ent = np.mean(sent\_entropies), then one exponentiation for each sentence: perplexity = 2 ** mean\_ent. The method is described [here](https://www.inf.ed.ac.uk/teaching/courses/fnlp/lectures/04_slides-2x2.pdf) as per-word cross-entropy of the trigram model for every sentence: sum of negative logs of the words probabilities / number of words in the sent see lascarides) ## I don't understand why it is CROSS-entropy here??

| model	|   train   |   good       |    bad      |
|-------|-----------|--------------|-------------|
|  HMM  |    5.40   |    1104.89   |     1035.71 |
|-------|-----------|--------------|-------------|
|  RNN  | 989,353.9 | 13,207,078.2 | 2,197,852.9 |
|-------|-----------|--------------|-------------|

1. *Shanon's (cross)entropy: negative sum of products of probability and log of probability*: 

the negative sum of products of probability of a word observed in the train and log2 of the one predicted for the test (mind that a LM can be tested on the train), H(p) =-Σ p(x) log p(x); and perplexity is the exponentiation of the entropy.

| model	|   train  |    good  |    bad   |
|-------|----------|----------|----------|
|  HMM  |    1.14  | 1.065991 | 1.065730 |
|-------|----------|----------|----------|
|  RNN  | 1.033206 | 1.031687 | 1.031241 |
|-------|----------|----------|----------|
| ELMo  | --       |  104.28  |  118.55  |
|-------|----------|----------|----------|

**The comparison above yields controvertional, counter-intuitive results that raise doubts about the sanity of the approach in general**

### Resources used
* HMM/RNN is trained on toy genre-comparable corpus of 1697 texts, 100k sents, 17mln tokens (a sample from the newspaper subcorpus of the RNC, lempos with functional wds)
* ELMo is trained on wiki+ruscorpora (989M tokens) \cite{KutuzovKuzmenko2017}

### Our HTQ-labeled data
542 hand-annotated targets in Russian to 105 English sources (all in the general domain of mass-media texts)
The data was comes from [RusLTC](https://www.rus-ltc.org/static/html/about.html), which sources quality-labeled translations from a number of translation competitions and university exam settings.
The translations were graded or ranked by either university translation teachers or by a panel of professional jurors in the contest settings. 

| labels |  words  | texts |
|--------|---------|-------|
| good   | 127,192 |  329  |
|--------|---------|-------|
|  bad   |  87,367 |  213  |
|--------|---------|-------|
| source |  46,218 |  105  |

## Results
### TQ classification
binary classification on cross-entropies, XGBoost
* HMM trigram LM             .50
* RNN-based trigram LM       .56
* embeddings from LM (ELMo)  .54

FYI: stratified dummy F1=.46

### Materials
* the seven pairs of vectors (en, ru, 16 GiB when unpacked) that we are using for accuracy and fluency modules can be downloaded [here](https://dev.rus-ltc.org/static/misc/vectors.tar.gz)
* the quality labels and preprocessed data for each experimental set up is [here](https://dev.rus-ltc.org/static/misc/LMs_predict_quality.tar.gz)
* learnt models are stored [here](https://dev.rus-ltc.org/static/misc/oracles.tar.gz)





