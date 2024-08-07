"""
Usage:
    run.py embeddings-sem-seccao [ARGUMENTS ... ] [options]
    run.py embeddings-sem-seccao-test [ARGUMENTS ... ] [options]
    run.py embeddings-contencioso-seccao [ARGUMENTS ... ] [options]
    run.py embeddings-contencioso-seccao-test [ARGUMENTS ... ] [options]
    run.py embeddings-6-seccao [ARGUMENTS ... ] [options]
    run.py embeddings-6-seccao-test [ARGUMENTS ... ] [options]
    run.py embeddings-5-seccao [ARGUMENTS ... ] [options]
    run.py embeddings-5-seccao-test [ARGUMENTS ... ] [options]
    run.py embeddings-7-seccao [ARGUMENTS ... ] [options]
    run.py embeddings-7-seccao-test [ARGUMENTS ... ] [options]
    run.py embeddings-4-seccao [ARGUMENTS ... ] [options]
    run.py embeddings-4-seccao-test [ARGUMENTS ... ] [options]
    run.py embeddings-3-seccao [ARGUMENTS ... ] [options]
    run.py embeddings-3-seccao-test [ARGUMENTS ... ] [options]
    run.py embeddings-2-seccao [ARGUMENTS ... ] [options]
    run.py embeddings-2-seccao-test [ARGUMENTS ... ] [options]
    run.py embeddings-1-seccao [ARGUMENTS ... ] [options]
    run.py embeddings-1-seccao-test [ARGUMENTS ... ] [options]
    run.py train-contencioso-seccao [ARGUMENTS ... ] [options]
    run.py test-contencioso-seccao [ARGUMENTS ... ] [options]
    run.py train-sem-seccao [ARGUMENTS ... ] [options]
    run.py test-sem-seccao [ARGUMENTS ... ] [options]
    run.py train-6-seccao [ARGUMENTS ... ] [options]
    run.py test-6-seccao [ARGUMENTS ... ] [options]
    run.py train-5-seccao [ARGUMENTS ... ] [options]
    run.py test-5-seccao [ARGUMENTS ... ] [options]
    run.py train-7-seccao [ARGUMENTS ... ] [options]
    run.py test-7-seccao [ARGUMENTS ... ] [options]
    run.py train-4-seccao [ARGUMENTS ... ] [options]
    run.py test-4-seccao [ARGUMENTS ... ] [options]
    run.py train-3-seccao [ARGUMENTS ... ] [options]
    run.py test-3-seccao [ARGUMENTS ... ] [options]
    run.py train-2-seccao [ARGUMENTS ... ] [options]
    run.py test-2-seccao [ARGUMENTS ... ] [options]
    run.py train-1-seccao [ARGUMENTS ... ] [options]
    run.py test-1-seccao [ARGUMENTS ... ] [options]
    run.py transform-sem-seccao [ARGUMENTS ... ] [options]
    run.py transform-contencioso-seccao [ARGUMENTS ... ] [options]
    run.py transform-6-seccao [ARGUMENTS ... ] [options]
    run.py transform-5-seccao [ARGUMENTS ... ] [options]
    run.py transform-7-seccao [ARGUMENTS ... ] [options]
    run.py transform-4-seccao [ARGUMENTS ... ] [options]
    run.py transform-3-seccao [ARGUMENTS ... ] [options]
    run.py transform-2-seccao [ARGUMENTS ... ] [options]
    run.py transform-1-seccao [ARGUMENTS ... ] [options]
    run.py sections-1-seccao [ARGUMENTS ... ] [options]
    run.py sections-1-seccao-test [ARGUMENTS ... ] [options]
    run.py sections-2-seccao [ARGUMENTS ... ] [options]
    run.py sections-2-seccao-test [ARGUMENTS ... ] [options]
    run.py sections-3-seccao [ARGUMENTS ... ] [options]
    run.py sections-3-seccao-test [ARGUMENTS ... ] [options]
    run.py sections-4-seccao [ARGUMENTS ... ] [options]
    run.py sections-4-seccao-test [ARGUMENTS ... ] [options]
    run.py sections-5-seccao [ARGUMENTS ... ] [options]
    run.py sections-5-seccao-test [ARGUMENTS ... ] [options]
    run.py sections-6-seccao [ARGUMENTS ... ] [options]
    run.py sections-6-seccao-test [ARGUMENTS ... ] [options]
    run.py sections-7-seccao [ARGUMENTS ... ] [options]
    run.py sections-7-seccao-test [ARGUMENTS ... ] [options]
    run.py sections-contencioso-seccao [ARGUMENTS ... ] [options]
    run.py sections-contencioso-seccao-test [ARGUMENTS ... ] [options]
    run.py embeddings-civel [ARGUMENTS ... ] [options]
    run.py embeddings-civel-test [ARGUMENTS ... ] [options]
    run.py train-civel [ARGUMENTS ... ] [options]
    run.py embeddings-criminal [ARGUMENTS ... ] [options]
    run.py embeddings-criminal-test [ARGUMENTS ... ] [options]
    run.py train-criminal [ARGUMENTS ... ] [options]
    run.py sections-civel [ARGUMENTS ... ] [options]
    run.py sections-civel-test [ARGUMENTS ... ] [options]
    run.py sections-criminal [ARGUMENTS ... ] [options]
    run.py sections-criminal-test [ARGUMENTS ... ] [options]
    run.py transform-civel [ARGUMENTS ... ] [options]
    run.py transform-criminal [ARGUMENTS ... ] [options]
    run.py test-civel [ARGUMENTS ... ] [options]
    run.py test-criminal [ARGUMENTS ... ] [options]
    run.py test-civel-all [ARGUMENTS ... ] [options]
    run.py test-criminal-all [ARGUMENTS ... ] [options]

Options:
        --judgment-zone-model=<directory>   model of structuring zones [default: JudgmentModel/with_fundamentacao_separation/model.pth]
        --dropout-rate=<float>              dropout rate [default: 0.5]
        --d-max-pool=<list>                 d max pool [default: [125, 128, 128]]
        --filter-channels=<int>             filter channels [default: 128]
        --filter-sizes=<list>                filter sizes [default: [2, 4, 8]]
        --embed-size=<int>                  size of word embedding [default: 1024]
        --sequence-length=<int>             sequence length [default: 500]
        --hidden-size=<int>                 size of hidden state [default: 1024]
        --num-layers=<int>                  num of hidden layers [default: 1]
        --batch-size=<int>                  batch-size [default: 64]
        --max-epoch=<int>                   max epoch [default: 10]
        --clip-max-norm=<float>             clip max norm [default: 5.0]
        --lr=<float>                        learning rate [default: 0.001]
        --log-every=<int>                   log every [default: 10]
        --validation-every=<int>            validation every [default: 10]
        --patience-threshold=<float>        patience threshold [default: 0.98]
        --max-patience=<int>                time of continuous worse performance to decay lr [default: 4]
        --max-decay=<int>                   time of lr decay to early stop [default: 4]
        --lr-decay=<float>                  decay rate of lr [default: 0.5]
        --model-save-path=<file>            model save path [default: /home/ruimelo/martim/model/model.pth]
        --optimizer-save-path=<file>        optimizer save path [default: /home/ruimelo/martim/model/optimizer.pth]
        --cuda                              use GPU
        --device-number=<int>               device number [default: 2]
"""

from docopt import docopt
import random
import torch
from utils import *
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from collections import namedtuple
import implicit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cluster import KMeans
from scipy.sparse import vstack
import numpy as np
from tqdm import tqdm
from ensemble import Model, Ensemble
from Judgment import Judgment, get_paragraph_by_id, add_zones_to_paragraph_objects
import bilstm_crf
import ast
from bilstm_utils import id2word

params = namedtuple('args', ['num_learner', 'num_clusters',
                             'num_threads', 'SVP_neigh', 'out_dim',
                             'w_thresh', 'sp_thresh', 'cost',
                             'NNtest', 'normalize'])
params.num_learners = 1 # 1
params.num_clusters = 1 # 1
params.num_threads = 1
params.w_thresh = 0.01  # ?
params.sp_thresh = 0.01  # ?
params.normalize = 1  # ?
params.embedding_lambda = 0.1  # determined automatically in WAltMin_asymm.m
params.regressor_lambda1 = 1e-2
params.regressor_lambda2 = 1e-3


def train(args, svp, out, nn):
    x_train_file = open(args["ARGUMENTS"][0], "rb")
    x_train = pickle.load(x_train_file)

    print("input", args["ARGUMENTS"][0], args["ARGUMENTS"][1], args["ARGUMENTS"][2], args["ARGUMENTS"][3])
    print("output", args["ARGUMENTS"][4])


    if np.isnan(x_train.data).any():
        x_train.data = np.nan_to_num(x_train.data)
        x_train.eliminate_zeros()

    x_test_file = open(args["ARGUMENTS"][1], "rb")
    x_test = pickle.load(x_test_file)

    y_train_file = open(args["ARGUMENTS"][2], "rb")
    y_train = pickle.load(y_train_file)

    #print(np.shape(y_train))

    y_test_file = open(args["ARGUMENTS"][3], "rb")
    y_test = pickle.load(y_test_file)

    clusterings = []

    for i in range(params.num_learners):
        model = KMeans(n_clusters=params.num_clusters, n_init=8, max_iter=50000)
        model.fit(x_train)
        clusterings.append(model)

    learners = []



    for clus_model in tqdm(clusterings):
        models = []
        for i in range(clus_model.n_clusters):

            data_idx = np.nonzero(clus_model.labels_ == i)[0]


            X = x_train[data_idx, :]
            Y = y_train[data_idx, :]

            print('embedding learning: building kNN graph')
            # build the kNN graph
            graph = kneighbors_graph(Y, svp, mode='distance', metric='cosine',
                             include_self=True,
                             n_jobs=-1)



            graph.data = 1 - graph.data  # convert to similarity


            als_model = implicit.als.AlternatingLeastSquares(factors=out,
                                                             regularization=params.embedding_lambda)
            als_model.fit(graph)

            # the embedding
            # shape: #instances x embedding dim
            Z = als_model.item_factors

            print('linear regressor training')
            # learn the linear regressor
            if True:
                print('entrei')
                # regressor = Ridge(fit_intercept=True, alpha=params.regressor_lambda2)
                regressor = ElasticNet(alpha=0.01, l1_ratio=0.0001, max_iter=100)
                regressor.fit(X, Z)
                # shape: embedding dim x feature dim
                V = regressor.coef_
            else:
                # learn V with l2 on V and l1 on VX
                ## note that X is sparse
                V = learn_V(X.toarray(), Z,
                            lambda1=params.regressor_lambda1,
                            lambda2=params.regressor_lambda2,
                            iter_max=500,
                            print_log=True)

            # the nearest neighbour model
            fitted_Z = X.toarray() @ V.T
            Z_neighbors = NearestNeighbors(n_neighbors=nn, metric='cosine').fit(fitted_Z)

            projected_center = project(V, clus_model.cluster_centers_[i])
            learned = {
                'center_z': projected_center,
                'V': V,
                'Z_neighbors': Z_neighbors,
                'data_idx': data_idx
            }
            models.append(learned)
        learners.append(models)


    with open(args["ARGUMENTS"][4], 'wb') as handle:
        pickle.dump(learners, handle, protocol=pickle.HIGHEST_PROTOCOL)



def transform_torch_numpy(args):
    x_train_file = open(args["ARGUMENTS"][0], "rb")
    x_train = pickle.load(x_train_file)
    print("input", args["ARGUMENTS"][0], args["ARGUMENTS"][1], args["ARGUMENTS"][2], args["ARGUMENTS"][3])
    print("output", args["ARGUMENTS"][4], args["ARGUMENTS"][5], args["ARGUMENTS"][6], args["ARGUMENTS"][7])

    print(x_train.size())

    x_test_file = open(args["ARGUMENTS"][1], "rb")
    x_test = pickle.load(x_test_file)

    y_train_file = open(args["ARGUMENTS"][2], "rb")
    y_train = pickle.load(y_train_file)

    print(len(y_train))

    y_test_file = open(args["ARGUMENTS"][3], "rb")
    y_test = pickle.load(y_test_file)

    y_train = csr_matrix(y_train)
    y_test = csr_matrix(y_test)

    x_0 = x_train[0].to_dense()
    x_0 = x_0.numpy()
    x_train_nump = csr_matrix(x_0)

    for i in range(1, len(x_train)):
        x_i = x_train[i].to_dense()
        x_i = x_i.numpy()
        x_i_sparse = csr_matrix(x_i)
        x_train_nump = vstack([x_train_nump, x_i_sparse])

    x_0 = x_test[0].to_dense()
    x_0 = x_0.numpy()
    x_test_nump = csr_matrix(x_0)

    for i in range(1, len(x_test)):
        x_i = x_test[i].to_dense()
        x_i = x_i.numpy()
        x_i_sparse = csr_matrix(x_i)
        x_test_nump = vstack([x_test_nump, x_i_sparse])


    print(x_train_nump)
    print(np.shape(x_train_nump))
    with open(args["ARGUMENTS"][4], 'wb') as handle:
        pickle.dump(x_train_nump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(x_test_nump)
    print(np.shape(x_test_nump))
    with open(args["ARGUMENTS"][5], 'wb') as handle:
        pickle.dump(x_test_nump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args["ARGUMENTS"][6], 'wb') as handle:
        np.shape(y_train)
        pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args["ARGUMENTS"][7], 'wb') as handle:
        np.shape(y_test)
        pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


def test(args):
    print("input:", args["ARGUMENTS"][1], args["ARGUMENTS"][2], args["ARGUMENTS"][3], args["ARGUMENTS"][4])


    x_test_file = open(args["ARGUMENTS"][1], "rb")
    x_test = pickle.load(x_test_file)
    x_test = csr_matrix(x_test)


    y_train_file = open(args["ARGUMENTS"][2], "rb")
    y_train = pickle.load(y_train_file)

    y_test_file = open(args["ARGUMENTS"][3], "rb")
    y_test = pickle.load(y_test_file)

    with open(args["ARGUMENTS"][4], 'rb') as f:
        learners = pickle.load(f)

    models = [Model(learner, y_train)
              for learner in learners]
    ensemble = Ensemble(models)

    pred_Y = ensemble.predict_many(x_test)

    performance = precision_at_ks(y_test, pred_Y)


    return performance

#funcao que testa a area civel e a area criminal juntando os modelos das secoes e area respectivas
def test_models_joint(args):
    x_test_file = open(args["ARGUMENTS"][0], "rb")
    x_test = pickle.load(x_test_file)
    x_test = csr_matrix(x_test)

    y_test_file = open(args["ARGUMENTS"][1], "rb")
    y_test = pickle.load(y_test_file)

    x_objects = []


    for a in range(2, len(args["ARGUMENTS"]), 2):

        print(args["ARGUMENTS"][a])
        print(args["ARGUMENTS"][a + 1])

        y_train_file = open(args["ARGUMENTS"][a], "rb")

        y_train = pickle.load(y_train_file)

        with open(args["ARGUMENTS"][a + 1], 'rb') as f:
            learners = pickle.load(f)
            models = [Model(learner, y_train)
                      for learner in learners]
            ensemble = Ensemble(models)

            pred_Y = ensemble.predict_many(x_test)


            if x_objects == []:
                for i,y in enumerate(pred_Y):
                    x_objects.append(X_test(i, {}))


            sec = args["ARGUMENTS"][a + 1][18:26]

            print(sec)



            x_objects = transform_labels(pred_Y, sec, x_objects)


    performance = precision_all_models(y_test, x_objects, "3_seccao")



    return performance


def append_section_elastic_search(args):
    file = open(args["ARGUMENTS"][0], "rb")
    df = pickle.load(file)  # train data as panda file

    print(args["ARGUMENTS"][0])
    print("input", args["ARGUMENTS"][0])
    print(df)
    print("output", args["ARGUMENTS"][1])

    device = torch.device('cuda' if args['--cuda'] else 'cpu')
    if args['--cuda']:
        torch.cuda.set_device(int(args["--device-number"]))

    sections_text = []
    sections_name = []

    for j,row in enumerate(df.iloc):
        print("j", j)
        text = row["text"]
        doc = Judgment(text, "html", False)


        model = bilstm_crf.BiLSTMCRF.load(args["--judgment-zone-model"], device)
        model.eval()

        all_text, ids, text_ids = doc.get_list_text()
        output = {"wrapper": "plaintext", "text": text_ids, "denotations": []}

        # secções pela ordem mais importante
        sections_doc = {"fundamentação de direito": [], "fundamentação de facto": [], "relatório": [], "decisão": [],
                        "delimitação": [], "colectivo": [], "declaração": [], "cabeçalho": [], "foot-note": [],
                        "título": []}
        sections = model.get_sections(all_text, device)

        sections = sections[0][1:-1]
        sections_names = []
        for tag in sections:
            sections_names.append(id2word(tag))

        for i, section in enumerate(sections_names):
            if section in ["B-cabeçalho", "I-cabeçalho"]:
                sections_doc["cabeçalho"].append((section, ids[i]))
            elif section in ["B-relatório", "I-relatório"]:
                sections_doc["relatório"].append((section, ids[i]))
            elif section in ["B-delimitação", "I-delimitação"]:
                sections_doc["delimitação"].append((section, ids[i]))
            elif section in ["B-fundamentação-facto", "I-fundamentação-facto"]:
                sections_doc["fundamentação de facto"].append((section, ids[i]))
            elif section in ["B-fundamentação-direito", "I-fundamentação-direito"]:
                sections_doc["fundamentação de direito"].append((section, ids[i]))
            elif section in ["B-decisão", "I-decisão"]:
                sections_doc["decisão"].append((section, ids[i]))
            elif section in ["B-colectivo", "I-colectivo"]:
                sections_doc["colectivo"].append((section, ids[i]))
            elif section in ["B-declaração", "I-declaração"]:
                sections_doc["declaração"].append((section, ids[i]))
            elif section in ["B-foot-note", "I-foot-note"]:
                sections_doc["foot-note"].append((section, ids[i]))
            elif section == "título":
                sections_doc["título"].append((section, ids[i]))

        keys = []
        t_to_add = []

        if len(sections_doc["relatório"]) != 0:
            keys.append("relatório")
            for p in sections_doc["relatório"]:
                t = get_paragraph_by_id(p[1][0], doc)
                t_to_add.append(t)
        """
        if len(sections_doc["fundamentação de facto"]) != 0:
            keys.append("fundamentação de facto")
            for p in sections_doc["fundamentação de facto"]:
                t = get_paragraph_by_id(p[1][0], doc)
                t_to_add.append(t)
        if len(sections_doc["fundamentação de direito"]) != 0:
            keys.append("fundamentação de direito")
            for p in sections_doc["fundamentação de direito"]:
                t = get_paragraph_by_id(p[1][0], doc)
                t_to_add.append(t)
        """
        if t_to_add == []:
            key, value = max(sections_doc.items(), key=lambda x: len(set(x[1])))
            keys.append(key)
            for p in value:
                t = get_paragraph_by_id(p[1][0], doc)
                t_to_add.append(t)


        sections_text.append(t_to_add)
        sections_name.append(keys)


    df.insert(len(df.columns), "section text", sections_text)
    df.insert(len(df.columns), "sections names", sections_name)

    print(df)

    with open(args["ARGUMENTS"][1], 'wb') as f:
        pickle.dump(df, f)


def sections_to_dict(args):
    file = open(args["ARGUMENTS"][0], "rb")
    df = pickle.load(file)  # train data as panda file

    file = open(args["ARGUMENTS"][1], "rb")
    df_test = pickle.load(file)  # train data as panda file


    #print(args["ARGUMENTS"][0])
    print("input", args["ARGUMENTS"][0], args["ARGUMENTS"][1])
    print(df)
    print(df_test)
    print("output", args["ARGUMENTS"][2])

    device = torch.device('cuda' if args['--cuda'] else 'cpu')
    if args['--cuda']:
        torch.cuda.set_device(int(args["--device-number"]))

    judgment_dict = {}

    for j,row in enumerate(df.iloc):
        print("j", j)
        text = row["text"]
        doc = Judgment(text, "html_text", False)


        model = bilstm_crf.BiLSTMCRF.load(args["--judgment-zone-model"], device)
        model.eval()

        all_text, ids, text_ids = doc.get_list_text()

        # secções pela ordem mais importante
        sections_doc = {"fundamentação de direito": [], "fundamentação de facto": [], "relatório": [], "decisão": [],
                        "delimitação": [], "colectivo": [], "declaração": [], "cabeçalho": [], "foot-note": [],
                        "título": []}
        sections = model.get_sections(all_text, device)

        sections = sections[0][1:-1]
        sections_names = []
        for tag in sections:
            sections_names.append(id2word(tag))

        for i, section in enumerate(sections_names):
            if section in ["B-cabeçalho", "I-cabeçalho"]:
                sections_doc["cabeçalho"].append((section, ids[i]))
            elif section in ["B-relatório", "I-relatório"]:
                sections_doc["relatório"].append((section, ids[i]))
            elif section in ["B-delimitação", "I-delimitação"]:
                sections_doc["delimitação"].append((section, ids[i]))
            elif section in ["B-fundamentação-facto", "I-fundamentação-facto"]:
                sections_doc["fundamentação de facto"].append((section, ids[i]))
            elif section in ["B-fundamentação-direito", "I-fundamentação-direito"]:
                sections_doc["fundamentação de direito"].append((section, ids[i]))
            elif section in ["B-decisão", "I-decisão"]:
                sections_doc["decisão"].append((section, ids[i]))
            elif section in ["B-colectivo", "I-colectivo"]:
                sections_doc["colectivo"].append((section, ids[i]))
            elif section in ["B-declaração", "I-declaração"]:
                sections_doc["declaração"].append((section, ids[i]))
            elif section in ["B-foot-note", "I-foot-note"]:
                sections_doc["foot-note"].append((section, ids[i]))
            elif section == "título":
                sections_doc["título"].append((section, ids[i]))


        judgment_dict[row["id"]] = {}

        for key,values in sections_doc.items():
            if len(values) == []:
                judgment_dict[row["id"]][key] = []
            else:
                text_to_dict = []
                for p in values:
                    t = get_paragraph_by_id(p[1][0], doc)
                    text_to_dict.append(t)
                judgment_dict[row["id"]][key] = text_to_dict

    for k,row in enumerate(df_test.iloc):
        print("k", k)
        text = row["text"]
        doc = Judgment(text, "html_text", False)


        model = bilstm_crf.BiLSTMCRF.load(args["--judgment-zone-model"], device)
        model.eval()

        all_text, ids, text_ids = doc.get_list_text()

        # secções pela ordem mais importante
        sections_doc = {"fundamentação de direito": [], "fundamentação de facto": [], "relatório": [], "decisão": [],
                        "delimitação": [], "colectivo": [], "declaração": [], "cabeçalho": [], "foot-note": [],
                        "título": []}
        sections = model.get_sections(all_text, device)

        sections = sections[0][1:-1]
        sections_names = []
        for tag in sections:
            sections_names.append(id2word(tag))

        for i, section in enumerate(sections_names):
            if section in ["B-cabeçalho", "I-cabeçalho"]:
                sections_doc["cabeçalho"].append((section, ids[i]))
            elif section in ["B-relatório", "I-relatório"]:
                sections_doc["relatório"].append((section, ids[i]))
            elif section in ["B-delimitação", "I-delimitação"]:
                sections_doc["delimitação"].append((section, ids[i]))
            elif section in ["B-fundamentação-facto", "I-fundamentação-facto"]:
                sections_doc["fundamentação de facto"].append((section, ids[i]))
            elif section in ["B-fundamentação-direito", "I-fundamentação-direito"]:
                sections_doc["fundamentação de direito"].append((section, ids[i]))
            elif section in ["B-decisão", "I-decisão"]:
                sections_doc["decisão"].append((section, ids[i]))
            elif section in ["B-colectivo", "I-colectivo"]:
                sections_doc["colectivo"].append((section, ids[i]))
            elif section in ["B-declaração", "I-declaração"]:
                sections_doc["declaração"].append((section, ids[i]))
            elif section in ["B-foot-note", "I-foot-note"]:
                sections_doc["foot-note"].append((section, ids[i]))
            elif section == "título":
                sections_doc["título"].append((section, ids[i]))


        judgment_dict[row["id"]] = {}

        for key,values in sections_doc.items():
            if len(values) == []:
                judgment_dict[row["id"]][key] = []
            else:
                text_to_dict = []
                for p in values:
                    t = get_paragraph_by_id(p[1][0], doc)
                    text_to_dict.append(t)
                judgment_dict[row["id"]][key] = text_to_dict




    with open(args["ARGUMENTS"][2], 'wb') as f:
        pickle.dump(judgment_dict, f)



def create_pd_with_sections(args):
    file = open(args["ARGUMENTS"][0], "rb")
    df = pickle.load(file)  # train data as panda file

    print(df)

    df_1 = df.iloc[:3000, :]
    df_2 = df.iloc[3000:, :]


    file = open(args["ARGUMENTS"][1], "rb")
    judgment_dict = pickle.load(file)  # dict with sections

    sections = ast.literal_eval(args["ARGUMENTS"][2]) # list with sections to retain


    print("input:", args["ARGUMENTS"][0], args["ARGUMENTS"][1], args["ARGUMENTS"][2])
    print("output:", args["ARGUMENTS"][3])

    sections_text = []
    sections_name = []

    for j,row in enumerate(df_1.iloc):
        print("j", j)
        id = row["id"]

        keys = []
        t_to_add = []
        for section in sections:
            text_section = judgment_dict[id][section]
            if text_section != []:
                keys.append(section)
                for t in text_section:
                    t_to_add.append(t)

        if t_to_add == []:
            key, value = max(judgment_dict[id].items(), key=lambda x: len(set(x[1])))
            keys.append(key)
            for v in value:
                t_to_add.append(v)


        sections_text.append(t_to_add)
        sections_name.append(keys)

    df_1.insert(len(df.columns), "section text", sections_text)
    df_1.insert(len(df.columns), "sections names", sections_name)

    print(df_1)

    with open(args["ARGUMENTS"][3], 'wb') as f:
        pickle.dump(df_1, f)

    sections_text_2 = []
    sections_name_2 = []

    for k, row in enumerate(df_2.iloc):
        print("k", k)
        id = row["id"]

        keys = []
        t_to_add = []
        for section in sections:
            text_section = judgment_dict[id][section]
            if text_section != []:
                keys.append(section)
                for t in text_section:
                    t_to_add.append(t)

        if t_to_add == []:
            key, value = max(judgment_dict[id].items(), key=lambda x: len(set(x[1])))
            keys.append(key)
            for v in value:
                t_to_add.append(v)

        sections_text_2.append(t_to_add)
        sections_name_2.append(keys)

    df_2.insert(len(df.columns), "section text", sections_text_2)
    df_2.insert(len(df.columns), "sections names", sections_name_2)

    print(df_2)

    with open(args["ARGUMENTS"][4], 'wb') as f:
        pickle.dump(df_2, f)


def main():
    args = docopt(__doc__)
    random.seed(0)
    torch.manual_seed(0)
    if args['--cuda']:
        torch.cuda.manual_seed(0)
    if args["sections-1-seccao"]:
        create_pd_with_sections(args)
    elif args["sections-1-seccao-test"]:
        create_pd_with_sections(args)
    elif args["embeddings-1-seccao"]:
        create_embeddings(args)
    elif args["embeddings-1-seccao-test"]:
        create_embeddings(args)
    elif args["transform-1-seccao"]:
        transform_torch_numpy(args)
    elif args["train-1-seccao"]:
        train(args, 20, 512, 25)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-1-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["sections-2-seccao"]:
        create_pd_with_sections(args)
    elif args["sections-2-seccao-test"]:
        create_pd_with_sections(args)
    elif args["embeddings-2-seccao"]:
        create_embeddings(args)
    elif args["embeddings-2-seccao-test"]:
        create_embeddings(args)
    elif args["transform-2-seccao"]:
        transform_torch_numpy(args)
    elif args["train-2-seccao"]:
        train(args, 30, 560, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-2-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["sections-3-seccao"]:
        create_pd_with_sections(args)
    elif args["sections-3-seccao-test"]:
        create_pd_with_sections(args)
    elif args["embeddings-3-seccao"]:
        create_embeddings(args)
    elif args["embeddings-3-seccao-test"]:
        create_embeddings(args)
    elif args["transform-3-seccao"]:
        transform_torch_numpy(args)
    elif args["train-3-seccao"]:
        train(args, 30, 560, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-3-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["sections-4-seccao"]:
        create_pd_with_sections(args)
    elif args["sections-4-seccao-test"]:
        create_pd_with_sections(args)
    elif args["embeddings-4-seccao"]:
        create_embeddings(args)
    elif args["embeddings-4-seccao-test"]:
        create_embeddings(args)
    elif args["transform-4-seccao"]:
        transform_torch_numpy(args)
    elif args["train-4-seccao"]:
        train(args, 30, 512, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-4-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["sections-5-seccao"]:
        create_pd_with_sections(args)
    elif args["sections-5-seccao-test"]:
        create_pd_with_sections(args)
    elif args["embeddings-5-seccao"]:
        create_embeddings(args)
    elif args["embeddings-5-seccao-test"]:
        create_embeddings(args)
    elif args["transform-5-seccao"]:
        transform_torch_numpy(args)
    elif args["train-5-seccao"]:
        train(args, 30, 560, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-5-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["sections-6-seccao"]:
        create_pd_with_sections(args)
    elif args["sections-6-seccao-test"]:
        create_pd_with_sections(args)
    elif args["embeddings-6-seccao"]:
        create_embeddings(args)
    elif args["embeddings-6-seccao-test"]:
        create_embeddings(args)
    elif args["transform-6-seccao"]:
        transform_torch_numpy(args)
    elif args["train-6-seccao"]:
        train(args, 30, 560, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-6-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["sections-7-seccao"]:
        create_pd_with_sections(args)
    elif args["sections-7-seccao-test"]:
        create_pd_with_sections(args)
    elif args["embeddings-7-seccao"]:
        create_embeddings(args)
    elif args["embeddings-7-seccao-test"]:
        create_embeddings(args)
    elif args["transform-7-seccao"]:
        transform_torch_numpy(args)
    elif args["train-7-seccao"]:
        train(args, 30, 560, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-7-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["sections-contencioso-seccao"]:
        create_pd_with_sections(args)
    elif args["sections-contencioso-seccao-test"]:
        create_pd_with_sections(args)
    elif args["embeddings-contencioso-seccao"]:
        create_embeddings(args)
    elif args["embeddings-contencioso-seccao-test"]:
        create_embeddings(args)
    elif args["transform-contencioso-seccao"]:
        transform_torch_numpy(args)
    elif args["train-contencioso-seccao"]:
        train(args, 30, 560, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-contencioso-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["sections-civel"]:
        create_pd_with_sections(args)
    elif args["sections-civel-test"]:
        create_pd_with_sections(args)
    elif args["embeddings-civel"]:
        create_embeddings(args)
    elif args["embeddings-civel-test"]:
        create_embeddings(args)
    elif args["transform-civel"]:
        transform_torch_numpy(args)
    elif args["train-civel"]:
        train(args, 30, 560, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-civel"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["sections-criminal"]:
        create_pd_with_sections(args)
    elif args["sections-criminal-test"]:
        create_pd_with_sections(args)
    elif args["embeddings-criminal"]:
        create_embeddings(args)
    elif args["embeddings-criminal-test"]:
        create_embeddings(args)
    elif args["transform-criminal"]:
        transform_torch_numpy(args)
    elif args["train-criminal"]:
        train(args, 30, 560, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-criminal"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-civel-all"]:
        performance = test_models_joint(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-criminal-all"]:
        performance = test_models_joint(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))


if __name__ == '__main__':
    main()





