import os
import json
from elasticsearch import Elasticsearch
import pandas as pd
import pickle
import xlsxwriter
import numpy as np

import torch
from emb_model import load_bert_model
from labels import *
from scipy.sparse import issparse


client = Elasticsearch("http://127.0.0.1:9200")


# recebe json com a estrutura dos acordaos e retorna o texto do relatório - fundamentação - decisão
def get_judgment_nucleo_text(path, file_name):
    f = open(path + file_name)
    text_list = []
    ids = []
    data = json.load(f)
    denotations = data["denotations"]
    for d in denotations:
        if d["type"] in ["relatório", "fundamentação", "decisão"]:
            ids.append((d["start"], d["end"]))
    text = data["text"]

    for p in text:
        for i in ids:
            if int(i[0]) <= int(p[1]) <= int(i[1]):
                text_list.append(p[0])

    return text_list



def create_embeddings(args):
    device = torch.device('cuda' if args['--cuda'] else 'cpu')
    print(device)
    if args['--cuda']:
        torch.cuda.set_device(int(args["--device-number"]))
    file = open(args["ARGUMENTS"][0], "rb")
    print("input", args["ARGUMENTS"][0], args["ARGUMENTS"][1])
    df = pickle.load(file)
    print(df)
    print("output", args["ARGUMENTS"][2])


    y_file = open(args["ARGUMENTS"][1], "rb")
    y = pickle.load(y_file)

    indxs = df.index.tolist()

    #assert len(indxs) == len(y)

    model_name_bert = "stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1"
    model_emb = load_bert_model(model_name_bert, device)
    X = ''

    for i, row in enumerate(df.iloc):
        print("i", i)
        text_list = row["section text"]
        X_aux = torch.zeros(1024)
        for j,p in enumerate(text_list):
            emb = torch.from_numpy(model_emb.encode(p)) #-> (1024) -> X[i,j]
            if emb.sum().data == 0:
                emb = torch.from_numpy(model_emb.encode("UNK"))
                X_aux = X_aux + emb
            else:
                X_aux = X_aux + emb

        assert len(text_list) > 0
        X_aux = torch.div(X_aux, len(text_list))
        X_aux = X_aux.to_sparse()
        X_aux = torch.unsqueeze(X_aux, 0)
        if i == 0:
            X = X_aux
        else:
            X = torch.cat((X, X_aux), 0)


    with open(args["ARGUMENTS"][2], 'wb') as handle:
        pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)


def embedding_judgment(text, device):
    model_name_bert = "stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1"
    model_emb = load_bert_model(model_name_bert, device)
    X = torch.zeros(1024)

    for p in text:
        emb = torch.from_numpy(model_emb.encode(p))
        if emb.sum().data == 0:
            emb = torch.from_numpy(model_emb.encode("UNK"))
            X = X + emb
        else:
            X = X + emb

    X = torch.div(X, len(text))
    X = X.to_sparse()

    return X


def project(V, x):
    """
    V: (embed dim, feature dim)
    x: (feature_dim, )

    return:
    vector of (embed dim, )
    """
    if issparse(x):
        x = x.toarray().T
    if len(x.shape) == 1:
        x = x[:, None]

    return (V @ x).flatten()


def precision_at_ks(true_Y, pred_Y, ks=[1, 2, 3, 4, 5, 10, 15, 20]):
    result = {}
    true_labels = [set(true_Y[i, :].nonzero()[1]) for i in range(true_Y.shape[0])]
    label_ranks = np.fliplr(np.argsort(pred_Y, axis=1))
    for k in ks:
        pred_labels = label_ranks[:, :k]
        precs = [len(t.intersection(set(p))) / min(k, len(t))
                 for t, p in zip(true_labels, pred_labels)]
        result[k] = np.mean(precs)
    return result


class X_test:

    def __init__(self, id, labels):
        self.id = id
        self.labels = labels

    def append_or_actualize_label(self, label, score):
        if label in self.labels:
            self.labels[label] += score
        else:
            self.labels[label] = score



def transform_labels(pred_y, section, x_objects):
    labels_section = return_labels(section)

    for i,y in enumerate(pred_y):
        for j,l in enumerate(y):
            label = labels_section[j]
            assert x_objects[i].id == i
            x_objects[i].append_or_actualize_label(label, l)

    return x_objects

def precision_all_models(true_Y, x_objects, section, ks=[1, 2, 3, 4, 5, 10, 15, 20]):
    result = {}
    true_labels = [set(true_Y[i, :].nonzero()[1]) for i in range(true_Y.shape[0])]
    labels_section = return_labels(section)
    true_labels_name = []
    for true in true_labels:
        x_labels = []
        for t in true:
            x_labels.append(labels_section[t])
        true_labels_name.append(x_labels)

    for k in ks:
        score = 0
        for i,x in enumerate(x_objects):
            tp = 0
            fp = 0
            pred_labels = {k: v for k, v in sorted(x.labels.items(), key=lambda item: item[1], reverse=True)}
            keys = list(pred_labels.keys())
            pred_keys = keys[0: k]

            #print(pred_keys)
            #print(true_labels_name[i])

            for key in pred_keys:
                if key in true_labels_name[i]:
                    tp += 1
                else:
                    fp += 1

            fp = min(fp, (len(true_labels_name[i]) - tp))
            precision = tp/(tp + fp)
            #print(precision)
            score += precision

        result[k] = score/len(x_objects)

    return result


def get_emb_sec():
    x_train_file = open("data/relatorio_direito/criminal/x_train_emb.pkl", "rb")
    x_train = pickle.load(x_train_file)

    x_train_file_2 = open("data/decision/criminal/x_train_emb.pkl", "rb")
    x_train_2 = pickle.load(x_train_file_2)


    x_test_file = open("data/relatorio_direito/criminal/x_test_emb.pkl", "rb")
    x_test = pickle.load(x_test_file)

    x_test_file_2 = open("data/decision/criminal/x_test_emb.pkl", "rb")
    x_test_2 = pickle.load(x_test_file_2)


    X_train = x_train + x_train_2

    X_test = x_test + x_test_2

    with open("data/relatorio_direito_decision/criminal/x_train_emb.pkl", 'wb') as handle:
        pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("data/relatorio_direito_decision/criminal/x_test_emb.pkl", 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)







