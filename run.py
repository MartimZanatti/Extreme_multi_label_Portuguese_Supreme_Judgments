"""
Usage:
    run.py embeddings-sem-seccao [ARGUMENTS ... ] [options]
    run.py embeddings-sem-seccao-test [ARGUMENTS ... ] [options]
    run.py embeddings-contencioso [ARGUMENTS ... ] [options]
    run.py embeddings-contencioso-test [ARGUMENTS ... ] [options]
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
    run.py train-contencioso [ARGUMENTS ... ] [options]
    run.py test-contencioso [ARGUMENTS ... ] [options]
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
    run.py transform-contencioso [ARGUMENTS ... ] [options]
    run.py transform-6-seccao [ARGUMENTS ... ] [options]
    run.py transform-5-seccao [ARGUMENTS ... ] [options]
    run.py transform-7-seccao [ARGUMENTS ... ] [options]
    run.py transform-4-seccao [ARGUMENTS ... ] [options]
    run.py transform-3-seccao [ARGUMENTS ... ] [options]
    run.py transform-2-seccao [ARGUMENTS ... ] [options]
    run.py transform-1-seccao [ARGUMENTS ... ] [options]


Options:
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
        --device-number=<int>               device number [default: 1]
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
        model = KMeans(n_clusters=params.num_clusters, n_init=8, max_iter=50)
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
                # regressor = Ridge(fit_intercept=True, alpha=params.regressor_lambda2)
                regressor = ElasticNet(alpha=0.1, l1_ratio=0.0001, max_iter=1000)
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
    x_train_file = open(args["ARGUMENTS"][0], "rb")
    x_train = pickle.load(x_train_file)
    x_train = csr_matrix(x_train)

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




def main():
    args = docopt(__doc__)
    random.seed(0)
    torch.manual_seed(0)
    if args['--cuda']:
        torch.cuda.manual_seed(0)
    if args["embeddings-sem-seccao"]:
        create_embeddings(args)
    elif args["embeddings-sem-seccao-test"]:
        create_embeddings(args)
    elif args["embeddings-contencioso"]:
        create_embeddings(args)
    elif args["embeddings-contencioso-test"]:
        create_embeddings(args)
    elif args["embeddings-6-seccao"]:
        create_embeddings(args)
    elif args["embeddings-6-seccao-test"]:
        create_embeddings(args)
    elif args["embeddings-5-seccao"]:
        create_embeddings(args)
    elif args["embeddings-5-seccao-test"]:
        create_embeddings(args)
    elif args["embeddings-7-seccao"]:
        create_embeddings(args)
    elif args["embeddings-7-seccao-test"]:
        create_embeddings(args)
    elif args["embeddings-4-seccao"]:
        create_embeddings(args)
    elif args["embeddings-4-seccao-test"]:
        create_embeddings(args)
    elif args["embeddings-3-seccao"]:
        create_embeddings(args)
    elif args["embeddings-3-seccao-test"]:
        create_embeddings(args)
    elif args["embeddings-2-seccao"]:
        create_embeddings(args)
    elif args["embeddings-2-seccao-test"]:
        create_embeddings(args)
    elif args["embeddings-1-seccao"]:
        create_embeddings(args)
    elif args["embeddings-1-seccao-test"]:
        create_embeddings(args)
    elif args["train-contencioso"]:
        train(args, 20, 120, 15)
        performance = test(args)
        for k,s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-contencioso"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["train-sem-seccao"]:
        train(args, 30, 120, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-sem-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["train-6-seccao"]:
        train(args, 20, 120, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-6-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["train-5-seccao"]:
        train(args, 20, 120, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-5-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["train-7-seccao"]:
        train(args, 35, 250, 20)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-7-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["train-4-seccao"]:
        train(args, 35, 560, 20)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-4-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["train-3-seccao"]:
        train(args, 35, 250, 20)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-3-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["train-2-seccao"]:
        train(args, 30, 560, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-2-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["train-1-seccao"]:
        train(args, 30, 560, 15)
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["test-1-seccao"]:
        performance = test(args)
        for k, s in performance.items():
            print('precision@{}: {:.4f}'.format(k, s))
    elif args["transform-sem-seccao"]:
        transform_torch_numpy(args)
    elif args["transform-contencioso"]:
        transform_torch_numpy(args)
    elif args["transform-6-seccao"]:
        transform_torch_numpy(args)
    elif args["transform-5-seccao"]:
        transform_torch_numpy(args)
    elif args["transform-7-seccao"]:
        transform_torch_numpy(args)
    elif args["transform-4-seccao"]:
        transform_torch_numpy(args)
    elif args["transform-3-seccao"]:
        transform_torch_numpy(args)
    elif args["transform-2-seccao"]:
        transform_torch_numpy(args)
    elif args["transform-1-seccao"]:
        transform_torch_numpy(args)


if __name__ == '__main__':
    main()





