#!/bin/sh

if [ "$1" = "embeddings-sem-seccao" ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-sem-seccao data/Sem_seccao/data_train.pkl data/Sem_seccao/y_train.pkl data/Sem_seccao/x_train_emb.pkl & disown
  #python run.py embeddings-sem-seccao data/Sem_seccao/data_train.pkl data/Sem_seccao/y_train.pkl data/Sem_seccao/x_train_emb.pkl  
elif [ "$1" = "embeddings-sem-seccao-test" ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-sem-seccao-test data/Sem_seccao/data_test.pkl data/Sem_seccao/y_test.pkl data/Sem_seccao/x_test_emb.pkl & disown
  #python run.py embeddings-sem-seccao-test data/Sem_seccao/data_test.pkl data/Sem_seccao/y_test.pkl data/Sem_seccao/x_test_emb.pkl  
elif [ "$1" = "embeddings-contencioso" ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-contencioso data/Contencioso/data_train.pkl data/Contencioso/y_train.pkl data/Contencioso/x_train_emb.pkl & disown
  #python run.py embeddings-contencioso data/Contencioso/data_train.pkl data/Contencioso/y_train.pkl data/Contencioso/x_train_emb.pkl 
elif [ "$1" = "embeddings-contencioso-test" ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-contencioso-test data/Contencioso/data_test.pkl data/Contencioso/y_test.pkl data/Contencioso/x_test_emb.pkl & disown
elif [ "$1" = "embeddings-6-seccao" ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-6-seccao data/6_seccao/data_train.pkl data/6_seccao/y_train.pkl data/6_seccao/x_train_emb.pkl & disown
  #python run.py embeddings-6-seccao data/6_seccao/data_train.pkl data/6_seccao/y_train.pkl data/6_seccao/x_train_emb.pkl  
elif [ "$1" = "train-contencioso" ]
then
  python run.py train-contencioso data/Contencioso/x_train_emb_numpy.pkl data/Contencioso/x_test_emb_numpy.pkl data/Contencioso/y_train_numpy.pkl data/Contencioso/y_test_numpy.pkl data/Contencioso/models.pkl
elif [ "$1" = "test-contencioso" ]
then
  python run.py test-contencioso data/Contencioso/x_train_emb_numpy.pkl data/Contencioso/x_test_emb_numpy.pkl data/Contencioso/y_train_numpy.pkl data/Contencioso/y_test_numpy.pkl data/Contencioso/models.pkl
elif [ "$1" = "train-sem-seccao" ]
then
  python run.py train-sem-seccao data/Sem_seccao/x_train_emb_numpy.pkl data/Sem_seccao/x_test_emb_numpy.pkl data/Sem_seccao/y_train_numpy.pkl data/Sem_seccao/y_test_numpy.pkl data/Sem_seccao/models.pkl
elif [ "$1" = "test-sem-seccao" ]
then
  python run.py test-sem-seccao data/Sem_seccao/x_train_emb_numpy.pkl data/Sem_seccao/x_test_emb_numpy.pkl data/Sem_seccao/y_train_numpy.pkl data/Sem_seccao/y_test_numpy.pkl data/Sem_seccao/models.pkl
elif [ "$1" = "transform-sem-seccao" ]
then
  python run.py transform-sem-seccao data/Sem_seccao/x_train_emb.pkl data/Sem_seccao/x_test_emb.pkl data/Sem_seccao/y_train.pkl data/Sem_seccao/y_test.pkl data/Sem_seccao/x_train_emb_numpy.pkl data/Sem_seccao/x_test_emb_numpy.pkl data/Sem_seccao/y_train_numpy.pkl data/Sem_seccao/y_test_numpy.pkl
elif [ "$1" = "transform-contencioso" ]
then
  python run.py transform-contencioso data/Contencioso/x_train_emb.pkl data/Contencioso/x_test_emb.pkl data/Contencioso/y_train.pkl data/Contencioso/y_test.pkl data/Contencioso/x_train_emb_numpy.pkl data/Contencioso/x_test_emb_numpy.pkl data/Contencioso/y_train_numpy.pkl data/Contencioso/y_test_numpy.pkl




fi





