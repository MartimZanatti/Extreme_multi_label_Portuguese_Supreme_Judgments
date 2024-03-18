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

    assert len(indxs) == len(y)

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


def transform_labels(pred_y, section):
    d = {}
    labels_section = return_labels(section)





