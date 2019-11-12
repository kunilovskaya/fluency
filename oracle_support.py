### this is a module with the support functions for LM_oracles


import pandas as pd
import numpy as np
import sys
from collections import OrderedDict

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# from sklearn.metrics import precision_recall_curve, make_scorer, recall_score, accuracy_score, precision_score
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
# from sklearn.metrics import roc_curve, auc, roc_auc_score
# from sklearn.metrics import recall_score
# # from sklearn.metrics import balanced_accuracy_score # aka "macro-averaged recall"
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from my_models import *




def get_textlines(f):
    EOL = 'endofline'  # Special token for line break
    lines = []
    ## this is a sentence
    for line in open(f, 'r'):
        res = line.strip() + ' ' + EOL
        # print(res)
        lines.append(tokenize(res))
    
    return lines

def get_perplexities(lines, model, k=2):
    perplexities0 = []
    perplexities1 = []
    perplexities2 = []
    text_count_in = 0
    text_count_oov = 0
    for l in lines:
        probabilities = []
        entropies = []
        entropies2 = []
    
        for nr, token in enumerate(l):
            if nr < k:
                continue
            # Model prediction:
            context = (l[nr - 2], l[nr - 1])
            ## this gets the probability of the correct word, given the k words in the left context, returned by the model
            probability, boo, boo1 = model.score(token, context=context)
            text_count_oov += boo
            text_count_in += boo1
        
            probabilities.append(probability)
        
            ## H(p) is the entropy of the distribution p(x): H(p) =-Σ p(x) log p(x). from https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3
            ## entropy per unit is the inverse probability of the test set normalized for the number of words
            ### higher probability, smaller entropy and vice versa
            entropy = - 1 * np.log2(probability)
            entropies.append(entropy)
    
        ## default method
        lst_perplexity0 = [2 ** ent for ent in entropies]
        ## collect the sentence-level (averaged over the sentence words) perplexities into a list for the corpus
        perplexities0.append(np.mean(lst_perplexity0))
        ##  Per-word cross-entropy of the trigram model for every sentence: sum of negative logs of the words probabilities / number of words in the sent
        # (see lascarides: https://www.inf.ed.ac.uk/teaching/courses/fnlp/lectures/04_slides-2x2.pdf)
        mean_ent = np.mean(entropies)
        ## perplexity is the exponentiation of the entropy!
        perplexity = 2 ** mean_ent
    
        ### exponentiate only once
        perplexities1.append(perplexity)
    
        ## based on proper formular which gets the entropy of a sent as a sum of products of (probs X log2of probs) of all word in the sent
    
        ## the formular of entropy to be seen in Wikipedia is that of CROSS-entropy where value is found as the sum of
        # products of probability of a word observed in the train and log2 of the one predicted for the test (a LM can be tested on the train)
    
        for prob in probabilities:
            lst_new_entropies = [-prob * np.log2(prob) for prob in
                                 probabilities]  ## you might be able to save time by using -np.dot(a, b) in place of -np.sum(a * b)
            entropies2.append(lst_new_entropies)
    
        av_entropy = np.mean(
            entropies2)  ## this is where the summing of all per-word entropy happens and normalisation to the number of words in the sent occurs
        ## perplexity is exponentiated entropy
        new_perplexity = 2 ** av_entropy
        perplexities2.append(new_perplexity)

    # print('\n=====Results for the last sentence:')
    # print(l)
    # print('Number of words:', len(probabilities))
    # print('Models per-word probabilities:', probabilities)
    #
    # print('\n==Method 1===\n')
    # print('Entropies0:', entropies)
    # print('Entropies0 averaged:', mean_ent)
    # print('Perplexities0', lst_perplexity0)
    # print('Averaged perplexity for sent:', np.mean(lst_perplexity0))
    #
    # print('\n==Method 2===\n')
    # print('Entropies1 averaged:', mean_ent)
    # print('Perplexity1 for sent:', perplexity)
    #
    # print('\n==Method 3===\n')
    # print('New Entropies2 averaged:', lst_new_entropies)
    # print('Averaged entropies2:', av_entropy)
    # print('Perplexity2 for sent:', new_perplexity)
    #
    # print('\n===OVERALL RESULTS (averaged over sentences of the whole corpus) ===\n')
    #
    # print('\nDefault Perplexity0: {0:.5f}, {1} running trigrams'.format(np.mean(perplexities0), len(perplexities0)))
    # ## this is Per-word cross-entropy of the trigram model for every sentence, suggested in Lascarides
    # print('Perplexity1  (from exponentiated average per-sent entropies): %s' % np.mean(perplexities1))
    # print('Perplexity2 derived from Shanons (cross)entropy: negative sum of products of prob and log of prob\t%s' % np.mean(
    #         perplexities2))

    return np.mean(perplexities0), np.mean(perplexities1), np.mean(perplexities2), text_count_oov, text_count_in

 ## SVM, dummy, RF, XGBoost
def cv_classify(x, y, algo=None, class_weight='balanced', n_splits=5, minor=None, random=42):
    if algo == 'LR':
        ##  solvers: "newton-cg", "sag", "lbfgs" and "liblinear"
        clf = LogisticRegression(random_state=random, solver='liblinear', class_weight=class_weight) ## multi_class='ovo', 'ovr'
    elif algo == 'SVM':
        clf = SVC(decision_function_shape='ovo', gamma='scale', random_state=random, verbose=False, probability=True, class_weight=class_weight)
    elif algo == 'dummy':
        strategy = 'stratified'
        print('\n====DummyBaseline (%s)====' % strategy)
        clf = DummyClassifier(strategy=strategy, random_state=random)  # 'stratified','uniform', 'most_frequent'
    elif algo == 'RF':
        clf = RandomForestClassifier(class_weight=class_weight, n_jobs=-1, random_state=random, min_samples_split=5, n_estimators=300, max_depth=15)
    elif algo == 'XGBoost':
        clf = XGBClassifier()
    else:
        print('I am not sure which algorythm to use for the clasification')
        clf = None
    skf = StratifiedKFold(n_splits=n_splits)
    clf.fit(x, y)
    
    # Generate cross-validated estimates for each input data point
    preds = cross_val_predict(clf, x, y, cv=skf)
    print('Cross-validated estimates for data points')
    print(classification_report(y, preds))
    
    # get measures on the minority class
    if minor:
        my_dict = classification_report(y, preds, output_dict=True)
        minorf1 = my_dict[minor]['f1-score']
        print('F1 on the minority class (this FTD):', np.average(minorf1).round(3))
    #     print("Classification report:\n", metrics.classification_report(yy, output))
    
    print('Compute confusion matrix')
    cnf_matrix = confusion_matrix(y, preds)
    print(cnf_matrix)
    
    # Use cross_validate to measure generalization error.
    # The scoring parameter: https://scikit-learn.org/stable/modules/model_evaluation.html
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    cv_scores = cross_validate(clf, x, y, cv=skf, scoring=scoring, n_jobs=2)
    
    print('Performance on cross-validation (macro-measures averaged):')
    
    # print('average accuracy %5.2f (+/- %0.2f)' % (cv_scores['test_accuracy'].mean(),
    #     #       cv_scores['test_accuracy'].std() * 2)()
    #
    # print('average macro-F1 %5.2f (+/- %0.2f)' % cv_scores['test_f1_macro'].mean(),
    #     #       cv_scores['test_f1_macro'].std() * 2))

    print('accuracy %5.2f (+/- %0.2f); macro-F1 %5.2f (+/- %0.2f)' % (cv_scores['test_accuracy'].mean(),
                                                                      cv_scores['test_accuracy'].std() * 2,
                                                                      cv_scores['test_f1_macro'].mean(),
                                                                      cv_scores['test_f1_macro'].std() * 2))
    
    
def visual(data, labels, classes):
    # Here goes the 2-D plotting of the data...
    pca = PCA(n_components=2)
    x_r = pca.fit_transform(data)
    plt.figure()
    # consistent colors
    cols = ['red', 'green', 'orange', 'blue', 'grey']
    colors = {}
    for i,name in enumerate(classes):
        colors[name] = cols[i]
#     colors = {'bad': 'red', 'good': 'green'}
    lw = 2

    for target_name in classes:
        plt.scatter(x_r[labels == target_name, 0], x_r[labels == target_name, 1], s=1, color=colors[target_name],
                    label=target_name, alpha=.8, lw=lw)
    plt.legend(loc='best', scatterpoints=1, prop={'size': 15})
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    #plt.savefig('plot.png', dpi=300)
    plt.show()
    plt.close()
    return x_r