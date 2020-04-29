import csv
import math
import random
import re
import sys
import numpy as np
from flair.data import Sentence
from flair.embeddings import ELMoEmbeddings
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

np.random.seed(1337)
from keras import Model, Input, Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM, merge, concatenate, Concatenate, \
    Flatten
from sklearn.metrics import f1_score, mean_squared_error


def read_data_both(filename, max=None, mode='orig', train=True, pad=True):
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
            # if cnt > 10:
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


# def read_data_old(filename, max=None, mode='orig', train=True):
#     """ method to read in data in csv format
#         Arguments:
#         filename        path to the file to read in
#         max             maximum number of words per headline, needed for test set
#     """
#     with open(filename, 'r', encoding='utf-8') as csv_file:
#         reader = csv.reader(csv_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
#         next(reader)
#         texts = []
#         labels = []
#         ids = []
#         regex = re.compile('[^a-zA-Z\d\s\']') # pattern to remove non-alphanumeric characters
#         max_num_words = 0
#         for line in reader:
#             ids.append(line[0].strip())
#             if train:
#                 labels.append(line[4].strip())
#             text = line[1]
#             start = text.find("<")
#             end = text.find("/>")
#             orig = text[start:end + 2]
#             edit = line[2]
#             if mode is 'edit':
#                 text = text.replace(orig, edit)
#             text = text.strip()
#             text = text.strip('\n')
#             text = text.lower()
#             text = re.sub(regex, "", text)
#             #print(text)
#             text_list = re.split('\s+', text)
#             texts.append(text_list)
#             if len(text_list) > max_num_words:
#                 max_num_words = len(text_list)
#     if max:
#         max_num_words = max # ignore longest essay if max was passed as argument
#     for i in range(len(texts)):
#         texts[i].insert(0, "<")
#         t = texts[i]
#         if len(t) > max_num_words: # truncate longer essays
#             texts[i] = t[0:max_num_words+1]
#         while len(texts[i]) <= max_num_words: # pad shorter essays
#             texts[i].append(">")
#     texts = np.array(texts)
#     labels = np.array(labels)
#     if train:
#         return texts, labels, ids
#     else:
#         return texts, ids


class Encoder:

    def fit(self, text):
        """ method to fit the encoder
            Arguments:
            text            training essays: array of arrays, each containing one essay
        """
        self.word_dict = sorted(set(text.ravel()))
        self.word_dict = {self.word_dict[i]: i+1 for i in range(len(self.word_dict))}
        self.word_dict['<unk>'] = 0 # set unseen words to 0

    def transform(self, text):
        """ method to encode text and labels
            Arguments:
            text        array of essays to be encoded
            labels      array of labels to be encoded
        """
        # enc_labels = []
        # for l in labels:
        #     # encode labels
        #     enc_l = np.zeros(len(self.label_dict))
        #     if l in self.label_dict:
        #         np.put(enc_l, self.label_dict[l], 1)
        #     else: # should never appear
        #         print("unknown label")
        #     enc_labels.append(enc_l)
        enc_text = []
        for t in text:
            # encode essays
            enc_essay = []
            for word in t:
                if word in self.word_dict:
                    enc_essay.append(self.word_dict[word])
                else:
                    enc_essay.append(self.word_dict["<unk>"])
            enc_text.append(enc_essay)
        enc_text = np.array(enc_text)
        #enc_labels = np.array(enc_labels)
        return(enc_text)

    def fit_transform(self, text):
        self.fit(text)
        return self.transform(text)

def load_pretrained_embeddings(glove_file):
    """ helper method to read in pretrained embedding vectors from a 6b glove file
        Arguments:
        glove_file      the path to the glove file
    """
    embedding_dict = dict()
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            row = line.strip().split(' ')
            word = row[0]   #get word
            vector = [float(i) for i in row[1:]]   #get vector
            embedding_dict[word] = vector
    return embedding_dict


def get_embedding_matrix(x, word_dict, glove_path='glove.6B/glove.6B.50d.txt'):
    """ method to train a keras cnn to predict the gender of an author given a short text
        Arguments:
        x           the input array
        y           the output array
        word_dict   the word dict needed to get embedding matrix
    """
    max_len = x.shape[1]
    embed_dict = load_pretrained_embeddings(glove_path)
    emb_dim = len(embed_dict['<']) # take some vector to get dimension
    embedding_matrix = np.zeros((len(word_dict), emb_dim)) # initialize embedding matrix
    for word, i in word_dict.items():
        embedding = embed_dict.get(word)
        if embedding is not None: # unknown words will be all zeros
            embedding_matrix[i] = embedding # fill embedding matrix
    return embedding_matrix


def train_model(x1, x2, y, dict_size, emb_matrix1, emb_matrix2):
    max_len = x1.shape[1]
    emb_dim = len(emb_matrix1[0])
    embedding_layer = Embedding(dict_size,
                                emb_dim,
                                weights=[emb_matrix1],
                                input_length=max_len,
                                trainable=True)
    inputs1 = Input(shape=(max_len,), dtype='float32')
    embedded = embedding_layer(inputs1)
    drop = Dropout(0.2)(embedded)  # prevent overfitting
    lstm1 = LSTM(4, input_shape=(1, 1))(drop)
    #m1 = Dense(1, activation='linear')(lstm)

    max_len = x2.shape[1]
    emb_dim = len(emb_matrix2[0])
    embedding_layer = Embedding(dict_size,
                                emb_dim,
                                weights=[emb_matrix2],
                                input_length=max_len,
                                trainable=True)
    inputs2 = Input(shape=(max_len,), dtype='float32')
    embedded = embedding_layer(inputs2)
    drop = Dropout(0.2)(embedded)  # prevent overfitting
    lstm2 = LSTM(4, input_shape=(1, 1))(drop)
    #m2 = Dense(1, activation='linear')(lstm)

    # model = Model(input=inputs, output=dense)
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(x, y, validation_split=0.2, epochs=50, batch_size=64)
    merged = Concatenate()([lstm1, lstm2])
    #merged = Dropout(0.2)(merged)
    merged = Dense(16)(merged)
    out = Dense(1, activation='linear')(merged)
    model = Model(input=[inputs1, inputs2], output=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # prevent overfitting
    model.fit([x1, x2], y, validation_split=0.2, epochs=200, batch_size=16, callbacks=[early_stop])
    return model


def train_model_elmo(x1, x2, y):
    inputs1 = Input(shape=(x1.shape[1:]), dtype='float32')
    drop = Dropout(0.5)(inputs1)  # prevent overfitting
    lstm1 = LSTM(4, input_shape=(drop.shape), return_sequences=True)(drop)

    inputs2 = Input(shape=(x2.shape[1:]), dtype='float32')
    drop = Dropout(0.5)(inputs2)  # prevent overfitting
    lstm2 = LSTM(4, input_shape=(drop.shape), return_sequences=True)(drop)

    merged = Concatenate()([lstm1, lstm2])
    #merged = Dropout(0.2)(merged)
    merged = Dense(16)(merged)
    flat = Flatten()(merged)
    out = Dense(1, activation='linear')(flat)
    model = Model(input=[inputs1, inputs2], output=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # prevent overfitting
    model.fit([x1, x2], y, validation_split=0.2, epochs=100, batch_size=16, callbacks=[early_stop])
    return model


def print_to_file(ids, predictions):
    with open('task-1-output.csv', mode='w') as results_file:
        writer = csv.writer(results_file, delimiter=',')
        writer.writerow(["id", "pred"])
        for index in range(len(ids)):
            writer.writerow([ids[index], predictions[index]])


def elmo_embeddings(sentences):
    embedded_sentences = []
    elmo = ELMoEmbeddings('small')
    for i in range(len(sentences)):
        sent = Sentence(sentences[i])
        print("sentence: ", sent)
        elmo.embed(sent)
        embedding = []
        for token in sent.tokens:
            emb = token.embedding.numpy()
            #print("word: ", token)
            #print("word embedding shape: ", emb.shape)
            embedding.append(emb)
        embedding = np.array(embedding)
        embedded_sentences.append(embedding)
    embedded_sentences = np.array(embedded_sentences)
    print("elmo embeddings done")
    #print("embedded sentences shape: ", embedded_sentences.shape)
    return np.array(embedded_sentences)

def scale_predictions(preds):
    new_preds = []
    for p in preds:
        #print(p)
        #print(type(p))
        if type(p) is not float:
            p = p[0]
        if p > 3:
            p = 3.0
        elif p < 0:
            p = 0.0
        new_preds.append(p)
    return np.array(new_preds, dtype=float)
    #return new_preds


if __name__ == "__main__":
    #main method that calls all methods
    text_edits, labels_edits, indices_edits, ids_edits, tuples_edits = read_data_both('data/task-1/train.csv', mode='edit')
    sent = text_edits[0].split()
    print(len(sent))
    text, labels, indices, ids, tuples = read_data_both('data/task-1/train.csv', max=len(sent)-1)

    x_orig = elmo_embeddings(text)
    x_edit = elmo_embeddings(text_edits)
    print("original elmo shape: ", x_orig.shape)
    print("edited elmo shape: ", x_edit.shape)
    model = train_model_elmo(x_orig, x_edit, labels)

    # test_text, test_labs, test_inds, test_ids, test_tuples = read_data_both('data/task-1/test_split.csv', max=len(sent)-1)
    # test_text_edits, test_labs_edits, test_inds_edits, test_ids_edits, test_tuples_edits = read_data_both('data/task-1/test_split.csv', max=len(sent)-1, mode='edit')
    #
    # x_test_orig = elmo_embeddings(test_text)
    # x_test_edit = elmo_embeddings(test_text_edits)
    #
    # y_pred = model.predict([x_test_orig, x_test_edit])
    # rmse = math.sqrt(mean_squared_error(y_true=test_labs, y_pred=y_pred))
    # print(rmse)

    dev_text, dev_inds, dev_ids, dev_tuples = read_data_both('data/semeval-2020-task-7-data-full/task-1/test.csv', max=len(sent) - 1, train=False)
    dev_text_edits, dev_inds_edits, dev_ids_edits, dev_tuples_edits = read_data_both('data/semeval-2020-task-7-data-full/task-1/test.csv', max=len(sent) - 1, mode='edit', train=False)

    x_dev_orig = elmo_embeddings(dev_text)
    x_dev_edit = elmo_embeddings(dev_text_edits)

    dev_pred = model.predict([x_dev_orig, x_dev_edit])
    dev_pred = scale_predictions(dev_pred)
    print_to_file(dev_ids, dev_pred)
    #print(rmse)


    # with open('task-1-output.csv', mode='r', encoding='utf-8') as file:
    #     writer = open('task-1-outpu1.csv', mode='w', encoding='utf-8')
    #     regex = re.compile('\[|\]')
    #     for li in file:
    #         li = re.sub(regex, "", li)
    #         writer.write(li)

    # all_text = np.append(text, text_edits)
    # glove = 'glove.6B.100d.txt'
    # enc = Encoder()
    # enc.fit(all_text)
    # # enc_x = enc.transform(text)
    # # enc_x_edits = enc.transform(text_edits)
    # # mat1 = get_embedding_matrix(enc_x, enc.word_dict, glove)
    # # mat2 = get_embedding_matrix(enc_x_edits, enc.word_dict, glove)
    # # model = train_model(enc_x, enc_x_edits, labels, len(enc.word_dict), mat1, mat2)
    # # test_text_edits, test_y = read_data('data/task-1/test_split.csv', mode='edit', max=text_edits.shape[1]-1)
    # # test_text, t_labels = read_data('data/task-1/test_split.csv', max=text_edits.shape[1]-1)
    # # print(test_text_edits.shape)
    # # print(test_text.shape)
    # # test_x = enc.transform(test_text)
    # # test_x_edits = enc.transform(test_text_edits)
    # # y_pred = model.predict([test_x, test_x_edits])
    # # y_pred = y_pred.ravel()
    # # y_pred = np.asarray(y_pred, dtype=float)
    # # test_y = np.asarray(test_y, dtype=float)
    # # rmse = math.sqrt(mean_squared_error(y_true=test_y, y_pred=y_pred))
    # # print(rmse)
    #
    # edit, labels1, ids2 = read_data('data/task-1/train.csv', mode='edit')
    # orig, labels_orig, ids_orig = read_data('data/task-1/train.csv', max=text_edits.shape[1] - 1)
    # dev_edit, dev_ids = read_data('data/task-1/dev.csv', max=text_edits.shape[1]-1, train=False, mode='edit')
    # dev_orig, dev_ids1 = read_data('data/task-1/dev.csv', max=text_edits.shape[1]-1, train=False)
    # print("edit ", edit.shape)
    # print("orig ", orig.shape)
    # print("dev edit ", dev_edit.shape)
    # print("dev orig ", dev_orig.shape)
    #
    # train_enc = enc.transform(edit)
    # train_enc_orig = enc.transform(orig)
    # dev_enc = enc.transform(dev_edit)
    # dev_enc_orig = enc.transform(dev_orig)
    # print("edit encoded ", train_enc.shape)
    # print("orig encoded ", train_enc_orig.shape)
    # print("dev encoded ", dev_enc.shape)
    # print("dev orig encoded ", dev_enc_orig.shape)
    #
    # mat_edit = get_embedding_matrix(train_enc, enc.word_dict, glove)
    # mat_orig = get_embedding_matrix(train_enc_orig, enc.word_dict, glove)
    # final_model = train_model(train_enc_orig, train_enc, labels_orig, len(enc.word_dict), mat_orig, mat_edit)
    # predictions_dev = final_model.predict([dev_enc_orig, dev_enc])
    # predictions_dev = predictions_dev.ravel()
    # predictions_dev = np.asarray(predictions_dev, dtype=float)
    # print_to_file(dev_ids, predictions_dev)




