import csv
import math
import numpy as np
import pandas as pd
import gensim
import re
import pronouncing
import nltk

import humor_regression_lstm

from flair.data import Sentence
from scipy.spatial import distance
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from nltk import ngrams
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from flair.embeddings import ELMoEmbeddings
from nltk import edit_distance
from nltk.corpus import brown

from sklearn.preprocessing import PolynomialFeatures
# import tensorflow as tf
# import tensorflow_hub as hub # https://towardsdatascience.com/elmo-contextual-language-embedding-335de2268604


# new version

def read_data_task1(filename, max=None, mode='orig', train=True, pad=True):
    """ method to read in data in csv format
        Arguments:
        filename        path to the file to read in
        max             maximum number of words per headline, needed for test set
    """
    texts = []
    labels = []
    ids = []
    indices = []
    tuples = []
    with open(filename, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(reader)
        regex = re.compile('[^a-zA-Z\d\s\']') # pattern to remove non-alphanumeric characters
        max_num_words = 0
        cnt = 0
        for line in reader:
            # if cnt > 100:
            #     break
            ids.append(line[0].strip())
            if train:
                labels.append(line[4].strip())
            text = line[1]
            start = text.find("<")
            end = text.find("/>")
            orig = text[start:end + 2]
            edit = line[2]
            if mode is 'edit':
                text = text.replace(orig, edit)
            else:
                while " " in orig:
                    white = orig.find(" ")
                    new_orig = orig[white+1:]
                    text = text.replace(orig, new_orig)
                    orig = new_orig
            text = text.strip()
            text = text.strip('\n')
            text = text.lower()
            text = re.sub(regex, "", text)
            orig = orig.strip()
            orig = orig.lower()
            orig = re.sub(regex, "", orig)
            edit = edit.strip()
            edit = edit.lower()
            tup = (orig, edit)
            tuples.append(tup)
            edit = re.sub(regex, "", edit)
            text_list = re.split('\s+', text)
            if mode is 'edit':
                edit_index = text_list.index(edit)
            else:
                edit_index = text_list.index(orig)
            if len(text_list) > max_num_words:
                max_num_words = len(text_list)
            indices.append(edit_index)
            texts.append(text)
            cnt = cnt + 1
    if pad:
        if max:
            max_num_words = max # ignore longest essay if max was passed as argument"
        padded = []
        for i in range(len(texts)):
            t = texts[i].split()
            t.insert(0, "<")
            if len(t) > max_num_words: # truncate longer essays
                t = t[0:max_num_words+1]
            while len(t) <= max_num_words: # pad shorter essays
                t.append(">")
            t = " ".join(t)
            padded.append(t)
        texts = padded
    labels = np.array(labels, dtype=float)
    if train:
        return texts, labels, indices, ids, tuples
    else:
        return texts, indices, ids, tuples


def read_data_task2(filename, max=None, mode='orig', train=True, pad=True):
    """ method to read in data in csv format
        Arguments:
        filename        path to the file to read in
        max             maximum number of words per headline, needed for test set
    """
    texts1 = []
    texts2 = []
    means1 = []
    means2 = []
    ids = []
    indices1 = []
    indices2 = []
    tuples1 = []
    tuples2 = []
    # 0: both sentences equally funny, 1: 1st sentence funnier, 2: 2nd sentence funnier
    labels = []
    with open(filename, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(reader)
        regex = re.compile('[^a-zA-Z\d\s\']') # pattern to remove non-alphanumeric characters
        max_num_words = 0
        cnt = 0
        for line in reader:
            # if cnt > 100:
            #     break
            # both sentences
            ids.append(line[0].strip())
            if train:
                means1.append(line[4].strip())
                means2.append(line[8].strip())
                labels.append(line[9].strip())
            # SENTENCE 1
            text1 = line[1]
            start1 = text1.find("<")
            end1 = text1.find("/>")
            orig1 = text1[start1:end1 + 2]
            edit1 = line[2]
            if mode is 'edit':
                text1 = text1.replace(orig1, edit1)
            else:
                while " " in orig1:
                    white = orig1.find(" ")
                    new_orig1 = orig1[white+1:]
                    text1 = text1.replace(orig1, new_orig1)
                    orig1 = new_orig1
            text1 = text1.strip()
            text1 = text1.strip('\n')
            text1 = text1.lower()
            text1 = re.sub(regex, "", text1)
            orig1 = orig1.strip()
            orig1 = orig1.lower()
            orig1 = re.sub(regex, "", orig1)
            edit1 = edit1.strip()
            edit1 = edit1.lower()
            tup1 = (orig1, edit1)
            tuples1.append(tup1)
            edit1 = re.sub(regex, "", edit1)
            text_list1 = re.split('\s+', text1)
            if mode is 'edit':
                edit_index1 = text_list1.index(edit1)
            else:
                edit_index1 = text_list1.index(orig1)
            if len(text_list1) > max_num_words:
                max_num_words = len(text_list1)
            indices1.append(edit_index1)
            texts1.append(text1)
            # SENTENCE 2
            if train:
                text2 = line[5]
                edit2 = line[6]
            else:
                text2 = line[3]
                edit2 = line[4]
            start2 = text2.find("<")
            end2 = text2.find("/>")
            orig2 = text2[start2:end2 + 2]
            if mode is 'edit':
                text2 = text2.replace(orig2, edit2)
            else:
                while " " in orig2:
                    white = orig2.find(" ")
                    new_orig2 = orig2[white + 1:]
                    text2 = text2.replace(orig2, new_orig2)
                    orig2 = new_orig2
            text2 = text2.strip()
            text2 = text2.strip('\n')
            text2 = text2.lower()
            text2 = re.sub(regex, "", text2)
            orig2 = orig2.strip()
            orig2 = orig2.lower()
            orig2 = re.sub(regex, "", orig2)
            edit2 = edit2.strip()
            edit2 = edit2.lower()
            tup2 = (orig2, edit2)
            tuples2.append(tup2)
            edit2 = re.sub(regex, "", edit2)
            text_list2 = re.split('\s+', text2)
            if mode is 'edit':
                edit_index2 = text_list2.index(edit2)
            else:
                edit_index2 = text_list2.index(orig2)
            if len(text_list2) > max_num_words:
                max_num_words = len(text_list2)
            indices2.append(edit_index2)
            texts2.append(text2)
            #count
            cnt = cnt + 1
    if pad:
        if max:
            max_num_words = max # ignore longest essay if max was passed as argument"
        padded1 = []
        padded2 = []
        # SENTENCES 1
        for i in range(len(texts1)):
            t = texts1[i].split()
            t.insert(0, "<")
            if len(t) > max_num_words: # truncate longer essays
                t = t[0:max_num_words+1]
            while len(t) <= max_num_words: # pad shorter essays
                t.append(">")
            t = " ".join(t)
            padded1.append(t)
        texts1 = padded1
        for i in range(len(texts2)):
            t = texts2[i].split()
            t.insert(0, "<")
            if len(t) > max_num_words: # truncate longer essays
                t = t[0:max_num_words+1]
            while len(t) <= max_num_words: # pad shorter essays
                t.append(">")
            t = " ".join(t)
            padded2.append(t)
        texts2 = padded2
    labels = np.array(labels, dtype=int)
    means1 = np.array(means1, dtype=float)
    means2 = np.array(means2, dtype=float)
    ids = np.array(ids)
    if train:
        return ids, texts1, texts2, means1, means2, tuples1, tuples2, indices1, indices2, labels
    else:
        return ids, texts1, texts2, tuples1, tuples2, indices1, indices2


def load_pretrained_embeddings(glove_file):
    """ helper method to read in pretrained embedding vectors from a 6b glove file
        Arguments:
        glove_file      the path to the glove file
    """
    embedding_dict = dict()
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            row = line.strip().split(' ')
            word = row[0] # get word
            vector = [float(i) for i in row[1:]] # get vector
            embedding_dict[word] = vector
    return embedding_dict

class Encoder:
    def fit(self, features):
        self.tdocvect = TfidfVectorizer(ngram_range=(2,6), max_df=0.95, min_df=0.0005)#, max_features=3000)#, stop_words='english')
        self.tdocvect.fit(features)
        #print(self.tdocvect.get_feature_names())

    def transform(self, features):
        return self.tdocvect.transform(features)

    def fit_transform(self, features):
        self.fit(features)
        return self.transform(features)


# run a linear regression
# rmse score on test set: 0.59
def linear_reg(x_train, y_train):
    cls = LinearRegression()
    cls.fit(x_train, y_train)
    return cls


# run a ridge regression
# rmse score on test set: 0.59
def ridge_reg(x_train, y_train, alpha=5000, solver='auto'):
    cls = Ridge(alpha=alpha, solver=solver)
    cls.fit(x_train, y_train)
    return cls


def trunc_SVD(x, components=700):
    svd = TruncatedSVD(n_components=components)
    svd_x_train = svd.fit_transform(x)
    return svd_x_train

def add_glove_dists(emb_dict, feats, tups):
    #print(feats.shape)
    #print(len(tups))
    new_features = []
    for i in range(len(tups)):
        #print(feats[i])
        #print(tups[i])
        feature = feats[i]
        tuple = tups[i]
        orig = tuple[0]
        edit = tuple[1]
         #print("original ", orig)
        #print("edited ", edit)
        emb_orig = emb_dict.get(orig)
        emb_edit = emb_dict.get(edit)
        if emb_edit is None or emb_orig is None:
            dist = 0.0
        else:
            dist = distance.euclidean(emb_orig, emb_edit)
        #print(dist)
        new_feat = np.append(feature, dist)
        new_features.append(new_feat)
    return np.array(new_features)

def add_glove_embeddings(emb_dict, feats, tups, dim=50):
    new_features = []
    for i in range(len(tups)):
        feature = feats[i]
        tuple = tups[i]
        orig = tuple[0]
        edit = tuple[1]
        emb_orig = emb_dict.get(orig)
        emb_edit = emb_dict.get(edit)
        if emb_edit is None:
            emb_edit = np.zeros((dim,), dtype=float)
        if emb_orig is None:
            emb_orig = np.zeros((dim,), dtype=float)
        new_feat = np.hstack((feature, emb_orig))
        new_feat = np.hstack((new_feat, emb_edit))
        new_features.append(new_feat)
    return np.array(new_features)


def add_elmo_embeddings(features, origs, edits, idx):
    new_features = []
    elmo = ELMoEmbeddings('small')
    for i in range(len(features)):
        feature = features[i]
        edit_idx = idx[i]
        orig_sent = Sentence(origs[i])
        edit_sent = Sentence(edits[i])
        print(orig_sent)
        #print(edit_sent)
        elmo.embed(orig_sent)
        elmo.embed(edit_sent)
        if edit_idx < len(orig_sent.tokens):
            orig = orig_sent.tokens[edit_idx]
        else:
            #print("WRONG LENGTH FOR SOME REASON")
            orig = orig_sent.tokens[len(orig_sent.tokens)-1]
        #print(orig)
        emb_orig = (orig.embedding).numpy()
        #print(emb_orig)
        if edit_idx < len(edit_sent.tokens):
            edit = edit_sent.tokens[edit_idx]
        else:
            #print("WRONG LENGTH EDIT")
            edit = edit_sent.tokens[len(edit_sent.tokens)-1]
        #print(edit)
        emb_edit = (edit.embedding).numpy()
        #print(emb_edit)
        #print("embedding shape: ", emb_edit.shape)
        new_feat = np.hstack((feature, emb_edit))
        new_feat = np.hstack((new_feat, emb_orig))
        #print(new_feat.shape)
        new_features.append(new_feat)
    print("elmo done")
    print(np.array(new_features).shape)
    return np.array(new_features)



def print_to_file(ids, predictions, filename='task-1-output.csv'):
    import csv

    with open(filename, mode='w') as results_file:
        writer = csv.writer(results_file, delimiter=',')
        writer.writerow(["id", "pred"])
        for index in range(len(predictions)):
            writer.writerow([ids[index], predictions[index]])

def compare_predictions(preds1, preds2):
    outcomes = []
    for i in range(len(preds1)):
        p1 = preds1[i]
        p2 = preds2[i]
        if p1 > p2:
            outcomes.append(1)
        elif p2 > p1:
            outcomes.append(2)
        else:
            outcomes.append(0)
    return np.array(outcomes, dtype=int)

def add_edit_distance(features, tupels):
    new_features = []
    for i in range(len(features)):
        feat = features[i]
        tup = tupels[i]
        dist = edit_distance(tup[0], tup[1])
        new_feat = np.hstack((feat, dist))
        new_features.append(new_feat)
    return np.array(new_features)

# nope
def add_rhyme(features, tuples):
    new_features = []
    for i in range(len(features)):
        feat = features[i]
        tup = tuples[i]
        rhyme = 0
        if tup[0] in pronouncing.rhymes(tup[1]):
            rhyme = 1
        new_feat = np.hstack((feat, rhyme))
        new_features.append(new_feat)
    return new_features


# nicht so useful
def add_encoded_pos(features, tuples):
    new_features = []
    x = dict()
    all_pos = []
    for i in range(len(features)):
        feat = features[i]
        tup = tuples[i]
        orig_tag = nltk.pos_tag(tup[0])
        edit_tag = nltk.pos_tag(tup[1])
        if orig_tag not in all_pos:
            all_pos.append(orig_tag)
        if edit_tag not in all_pos:
            all_pos.append(edit_tag)
        new_feat = np.hstack((feat, orig_tag))
        new_feat = np.hstack(new_feat, edit_tag)
        new_features.append(new_feat)
    return new_features


def scale_predictions(preds, task=1):
    new_preds = []
    for p in preds:
        # if task == 1:
        #     p = int(round(p/0.2))
        #     p = p * 0.2
        #     p = round(p, 1)
        # else:
        #     p = round(p, 3)
        if p > 3:
            p = 3.0
        elif p < 0:
            p = 0.0
        new_preds.append(p)
    return np.array(new_preds, dtype=float)


if __name__ == '__main__':
    # TASK 2
    #  training set original sentences
    ids, texts1, texts2, means1, means2, tuples1, tuples2, indices1, indices2, labels = \
        read_data_task2('data/semeval-2020-task-7-data-full/task-2/train.csv', mode='orig', pad=False)
    # edited sentences
    ed_ids, ed_t1, ed_t2, ed_m1, ed_m2, ed_tup1, ed_tup2, ed_idx1, ed_idx2, ed_labels = \
        read_data_task2('data/semeval-2020-task-7-data-full/task-2/train.csv', mode='edit', pad=False)
    # dev set
    # original sentences
    d_ids, d_texts1, d_texts2, d_means1, d_means2, d_tuples1, d_tuples2, d_indices1, d_indices2, d_labels = \
        read_data_task2('data/semeval-2020-task-7-data-full/task-2/dev.csv', mode='orig', pad=False)
    # edited sentences
    ded_ids, ded_t1, ded_t2, ded_m1, ded_m2, ded_tup1, ded_tup2, ded_idx1, ded_idx2, ded_labels = \
        read_data_task2('data/semeval-2020-task-7-data-full/task-2/dev.csv', mode='edit', pad=False)
    enc = Encoder()
    all_origs = np.append(texts1, texts2, axis=0)
    #all_origs = np.append(all_origs, featurs, axis=0)
    # all_origs = np.append(all_origs, d_texts2, axis=0)
    all_edits = np.append(ed_t1, ed_t2, axis=0)
    #all_edits = np.append(all_edits, ed_features, axis=0)
    # all_edits = np.append(all_edits, ded_t2, axis=0)
    all_tups = np.append(tuples1, tuples2, axis=0)
    #all_tups = np.append(all_tups, tups, axis=0)
    # all_tups = np.append(all_tups, ded_tup2, axis=0)
    all_idx = np.append(indices1, indices2, axis=0)
    #all_idx = np.append(all_idx, idx, axis=0)
    # all_idx = np.append(all_idx, ded_idx2, axis=0)
    all_means = np.append(means1, means2, axis=0)
    #all_means = np.append(all_means, labs, axis=0)
    # all_means = np.append(all_means, ded_m2, axis=0)
    feats1 = enc.fit_transform(all_origs)
    # truncated svd
    feats1 = trunc_SVD(feats1, components=700)
    # glove embeddings
    emb_dict = load_pretrained_embeddings('glove.6B.100d.txt')
    feats1 = add_glove_embeddings(emb_dict, feats1, all_tups, dim=100)
    # edit distance
    feats1 = add_edit_distance(feats1, all_tups)
    # elmo embeddings
    feats1 = add_elmo_embeddings(feats1, all_origs, all_edits, all_idx)
    # models
    ridge = ridge_reg(feats1, all_means, alpha=5000)

    # test data for TEST SET SUBMISSION
    test_ids, test_texts1, test_texts2, test_tuples1, test_tuples2, test_indices1, test_indices2 = \
        read_data_task2('data/semeval-2020-task-7-data-full/task-2/test.csv', mode='orig', train=False, pad=False)
    te_ids, te_texts1, te_texts2, te_tuples1, te_tuples2, te_indices1, te_indices2 = \
        read_data_task2('data/semeval-2020-task-7-data-full/task-2/test.csv', mode='edit', train=False, pad=False)
    f_dev1 = enc.transform(test_texts1) # encode
    f_dev2 = enc.transform(test_texts2)
    # # DEV SET SUBMISSION
    # f_dev1 = enc.transform(d_texts1)  # encode
    # f_dev2 = enc.transform(d_texts2)
    # BOTH
    f_dev1 = trunc_SVD(f_dev1, components=700) # svd
    f_dev2 = trunc_SVD(f_dev2, components=700)
    # TEST SET SUBMISSION
    f_dev1 = add_glove_embeddings(emb_dict, f_dev1, test_tuples1, dim=100) # glove
    f_dev2 = add_glove_embeddings(emb_dict, f_dev2, test_tuples2, dim=100)
    f_dev1 = add_edit_distance(f_dev1, test_tuples1) # edit distance
    f_dev2 = add_edit_distance(f_dev2, test_tuples2)
    f_dev1 = add_elmo_embeddings(f_dev1, test_texts1, te_texts1, te_indices1) # elmo
    f_dev2 = add_elmo_embeddings(f_dev2, test_texts2, te_texts2, te_indices2)
    # # DEV SET SUBMISSION
    # f_dev1 = add_glove_embeddings(emb_dict, f_dev1, d_tuples1, dim=100)  # glove
    # f_dev2 = add_glove_embeddings(emb_dict, f_dev2, d_tuples2, dim=100)
    # f_dev1 = add_edit_distance(f_dev1, d_tuples1)  # edit distance
    # f_dev2 = add_edit_distance(f_dev2, d_tuples2)
    # f_dev1 = add_elmo_embeddings(f_dev1, d_texts1, ded_t1, ded_idx1)  # elmo
    # f_dev2 = add_elmo_embeddings(f_dev2, d_texts2, ded_t2, ded_idx2)
    # print("training features shape ", feats1.shape)
    # print("dev1 shape ", f_dev1.shape)
    # print("dev2 shape ", f_dev2.shape)
    preds1 = ridge.predict(f_dev1) # predict
    preds2 = ridge.predict(f_dev2)
    preds1= np.array(preds1, dtype=float) # post process
    preds2 = np.array(preds2, dtype=float)
    preds1 = scale_predictions(preds1, task=2)
    preds2 = scale_predictions(preds2, task=2)
    predictions = compare_predictions(preds1, preds2)
    # #accuracy = accuracy_score(d_labels, predictions)
    # #for i in range(len(predictions)):
    #     #print("true ", d_labels[i], " predicted ", predictions[i])
    # #print("accuracy ", accuracy)
    # # print(d_ids.shape)
    # # print(predictions.shape)
    # print_to_file(d_ids, predictions, filename='task-2-output.csv') # DEV SET
    print_to_file(test_ids, predictions, filename='task-2-output.csv') #TEST SET
    # #print(rmse)
    # TASK 1
    features, labels, idx, ids, tups = read_data_task1('data/semeval-2020-task-7-data-full/task-1/train.csv', mode='orig', pad=False)
    ed_features, ed_labels, ed_idx, ed_ids, ed_tups = read_data_task1('data/semeval-2020-task-7-data-full/task-1/train.csv', mode='edit', pad=False)

    f_dev, lab_dev, ind_dev, id_dev, t_dev = read_data_task1('data/semeval-2020-task-7-data-full/task-1/dev.csv',
                                                    mode='orig', pad=False)
    ed_fdev, ed_labdev, ed_inddev, ed_iddev, ed_tdev = read_data_task1('data/semeval-2020-task-7-data-full/task-1/dev.csv',
                                                            mode='edit', pad=False)
    f_test, ind_test, id_test, t_test = read_data_task1('data/semeval-2020-task-7-data-full/task-1/test.csv', train=False,
                                                    mode='orig', pad=False)
    ed_ftest, ed_indtest, ed_idtest, ed_ttest = read_data_task1('data/semeval-2020-task-7-data-full/task-1/test.csv', train=False,
                                                            mode='edit', pad=False)
    # training features
    enc = Encoder()
    enc_feats = enc.fit_transform(features)
    # svd
    feats = trunc_SVD(enc_feats, components=700)
    emb_dict = load_pretrained_embeddings('glove.6B.100d.txt')
    # glove
    feats = add_glove_embeddings(emb_dict, feats, tups, dim=100)
    # edit distance
    feats = add_edit_distance(feats, tups)
    # elmo features
    feats = add_elmo_embeddings(feats, features, ed_features, ed_idx)
    # model
    ridge = ridge_reg(feats, labels)

    # development set
    enc_feats_dev = enc.transform(f_dev)
    enc_feats_dev = trunc_SVD(enc_feats_dev, components=700)
    enc_feats_dev = add_glove_embeddings(emb_dict, enc_feats_dev, t_dev, dim=100)
    enc_feats_dev = add_edit_distance(enc_feats_dev, t_dev)
    enc_feats_dev = add_elmo_embeddings(enc_feats_dev, f_dev, ed_fdev, ed_inddev)
    predictions_dev = ridge.predict(enc_feats_dev)
    predictions_dev = np.array(predictions_dev, dtype=float)
    predictions_dev = scale_predictions(predictions_dev)
    rmse = math.sqrt(mean_squared_error(y_true=lab_dev, y_pred=predictions_dev))
    print_to_file(id_dev, predictions_dev, filename='task1dev.csv',)
    print("ridge + glove")  # + elmo embeddings")
    print(rmse)

    # test set
    enc_feats_test = enc.transform(f_test)
    enc_feats_test = trunc_SVD(enc_feats_test, components=700)
    enc_feats_test = add_glove_embeddings(emb_dict, enc_feats_test, t_test, dim=100)
    enc_feats_test = add_edit_distance(enc_feats_test, t_test)
    enc_feats_test = add_elmo_embeddings(enc_feats_test, f_test, ed_ftest, ed_indtest)
    predictions_test = ridge.predict(enc_feats_test)
    predictions_test = np.array(predictions_test, dtype=float)
    predictions_test = scale_predictions(predictions_test)
    print_to_file(id_test, predictions_test, filename='task-1-output.csv', )







    #TUNING
    # alphas = np.array([10, 11, 12, 13, 15])
    # param_grid = {'alpha':alphas, 'solver':['sag']} #'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
    # model = Ridge()
    # grid = GridSearchCV(estimator=model, param_grid=param_grid)
    # grid.fit(feats, labels)
    # print(grid.best_score_)
    # print(grid.best_params_)

    #    for i in range(len(ids)):
    #     print("sentence 1 ", texts1[i], " tuples 1 ", tuples1[i])
    #     print("mean grade ", means1[i])
    #     print("sentence 2 ", texts2[i], " tuples 2 ", tuples2[i])
    #     print("mean grade ", means2[i])
    #     print("label ", labels[i])