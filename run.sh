#!/bin/sh
if [ "$1" = "sections-1-seccao" ]
then
  python run.py sections-1-seccao data/relatorio/1_seccao/data_train.pkl data/civel/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/1_seccao/data_train_with_sections.pkl  --cuda
  #python run.py sections-1-seccao data/1_seccao/data_train.pkl data/1_seccao/data_train_with_sections.pkl  --cuda
elif [ "$1" = "sections-1-seccao-test" ]
then
  python run.py sections-1-seccao-test data/relatorio/1_seccao/data_test.pkl data/civel/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/1_seccao/data_test_with_sections.pkl  --cuda
  #python run.py sections-1-seccao-test data/1_seccao/data_test.pkl data/1_seccao/data_test_with_sections.pkl  --cuda
elif [ "$1" = "embeddings-1-seccao"   ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-1-seccao data/all_sections/1_seccao/data_train_with_sections.pkl data/relatorio/1_seccao/y_train.pkl data/all_sections/1_seccao/x_train_emb.pkl --cuda & disown
  #python run.py embeddings-1-seccao data/1_seccao/data_train_with_sections.pkl data/1_seccao/y_train.pkl data/1_seccao/x_train_emb.pkl
elif [ "$1" = "embeddings-1-seccao-test"  ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-1-seccao-test data/all_sections/1_seccao/data_test_with_sections.pkl data/relatorio/1_seccao/y_test.pkl data/all_sections/1_seccao/x_test_emb.pkl > output1 --cuda & disown
  #python run.py embeddings-1-seccao-test data/1_seccao/data_test_with_sections.pkl data/1_seccao/y_test.pkl data/1_seccao/x_test_emb.pkl
elif [ "$1" = "transform-1-seccao" ]
then
  python run.py transform-1-seccao data/all_sections/1_seccao/x_train_emb.pkl data/all_sections/1_seccao/x_test_emb.pkl data/relatorio/1_seccao/y_train.pkl data/relatorio/1_seccao/y_test.pkl data/all_sections/1_seccao/x_train_emb_numpy.pkl data/all_sections/1_seccao/x_test_emb_numpy.pkl data/all_sections/1_seccao/y_train_numpy.pkl data/all_sections/1_seccao/y_test_numpy.pkl
elif [ "$1" = "train-1-seccao" ]
then
  #python run.py train-1-seccao data/1_seccao2/x_train_emb_numpy.pkl data/1_seccao2/x_test_emb_numpy.pkl data/1_seccao2/y_train_numpy.pkl data/1_seccao2/y_test_numpy.pkl data/1_seccao2/models.pkl
  PYTHONUNBUFFERED=1 nohup python run.py train-1-seccao data/all_sections/1_seccao/x_train_emb_numpy.pkl data/all_sections/1_seccao/x_test_emb_numpy.pkl data/all_sections/1_seccao/y_train_numpy.pkl data/all_sections/1_seccao/y_test_numpy.pkl data/all_sections/1_seccao/models.pkl > train1 & disown
elif [ "$1" = "test-1-seccao" ]
then
  python run.py test-1-seccao data/1_seccao2/x_train_emb_numpy.pkl data/1_seccao2/x_test_emb_numpy.pkl data/1_seccao2/y_train_numpy.pkl data/1_seccao2/y_test_numpy.pkl data/1_seccao2/models.pkl
elif [ "$1" = "sections-2-seccao" ]
then
  python run.py sections-2-seccao data/relatorio/2_seccao/data_train.pkl data/civel/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/2_seccao/data_train_with_sections.pkl  --cuda
  #python run.py sections-2-seccao data/2_seccao/data_train.pkl data/2_seccao/data_train_with_sections.pkl  --cuda
elif [ "$1" = "sections-2-seccao-test" ]
then
  python run.py sections-2-seccao-test data/relatorio/2_seccao/data_test.pkl data/civel/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/2_seccao/data_test_with_sections.pkl  --cuda
  #python run.py sections-2-seccao-test data/2_seccao/data_test.pkl data/2_seccao/data_test_with_sections.pkl  --cuda
elif [ "$1" = "embeddings-2-seccao"   ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-2-seccao data/all_sections/2_seccao/data_train_with_sections.pkl data/relatorio/2_seccao/y_train.pkl data/all_sections/2_seccao/x_train_emb.pkl --cuda & disown
  #python run.py embeddings-2-seccao data/2_seccao/data_train_with_sections.pkl data/2_seccao/y_train.pkl data/2_seccao/x_train_emb.pkl
elif [ "$1" = "embeddings-2-seccao-test"  ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-2-seccao-test data/all_sections/2_seccao/data_test_with_sections.pkl data/relatorio/2_seccao/y_test.pkl data/all_sections/2_seccao/x_test_emb.pkl --cuda > output1 & disown
  #python run.py embeddings-2-seccao-test data/2_seccao/data_test_with_sections.pkl data/2_seccao/y_test.pkl data/2_seccao/x_test_emb.pkl
elif [ "$1" = "transform-2-seccao" ]
then
  python run.py transform-2-seccao data/all_sections/2_seccao/x_train_emb.pkl data/all_sections/2_seccao/x_test_emb.pkl data/relatorio/2_seccao/y_train.pkl data/relatorio/2_seccao/y_test.pkl data/all_sections/2_seccao/x_train_emb_numpy.pkl data/all_sections/2_seccao/x_test_emb_numpy.pkl data/all_sections/2_seccao/y_train_numpy.pkl data/all_sections/2_seccao/y_test_numpy.pkl
elif [ "$1" = "train-2-seccao" ]
then
  PYTHONUNBUFFERED=1 nohup python run.py train-2-seccao data/all_sections/2_seccao/x_train_emb_numpy.pkl data/all_sections/2_seccao/x_test_emb_numpy.pkl data/all_sections/2_seccao/y_train_numpy.pkl data/all_sections/2_seccao/y_test_numpy.pkl data/all_sections/2_seccao/models.pkl > train2 & disown
  #PYTHONUNBUFFERED=1 nohup python run.py train-2-seccao data/2_seccao/x_train_emb_numpy.pkl data/2_seccao/x_test_emb_numpy.pkl data/2_seccao/y_train_numpy.pkl data/2_seccao/y_test_numpy.pkl data/2_seccao/models.pkl & disown
elif [ "$1" = "test-2-seccao" ]
then
  python run.py test-2-seccao data/2_seccao/x_train_emb_numpy.pkl data/2_seccao/x_test_emb_numpy.pkl data/2_seccao/y_train_numpy.pkl data/2_seccao/y_test_numpy.pkl data/2_seccao/models.pkl
elif [ "$1" = "sections-3-seccao" ]
then
  python run.py sections-3-seccao data/relatorio/3_seccao/data_train.pkl data/criminal/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/3_seccao/data_train_with_sections.pkl  --cuda
  #python run.py sections-3-seccao data/3_seccao/data_train.pkl data/3_seccao/data_train_with_sections.pkl  --cuda
elif [ "$1" = "sections-3-seccao-test" ]
then
  python run.py sections-3-seccao-test data/relatorio/3_seccao/data_test.pkl data/criminal/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/3_seccao/data_test_with_sections.pkl  --cuda
  #python run.py sections-3-seccao-test data/3_seccao/data_test.pkl data/3_seccao/data_test_with_sections.pkl  --cuda
elif [ "$1" = "embeddings-3-seccao"   ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-3-seccao data/all_sections/3_seccao/data_train_with_sections.pkl data/relatorio/3_seccao/y_train.pkl data/all_sections/3_seccao/x_train_emb.pkl --cuda > output1 & disown
  #python run.py embeddings-3-seccao data/3_seccao/data_train_with_sections.pkl data/3_seccao/y_train.pkl data/3_seccao/x_train_emb.pkl
elif [ "$1" = "embeddings-3-seccao-test"  ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-3-seccao-test data/all_sections/3_seccao/data_test_with_sections.pkl data/relatorio/3_seccao/y_test.pkl data/all_sections/3_seccao/x_test_emb.pkl --cuda & disown
  #python run.py embeddings-3-seccao-test data/3_seccao/data_test_with_sections.pkl data/3_seccao/y_test.pkl data/3_seccao/x_test_emb.pkl
elif [ "$1" = "transform-3-seccao" ]
then
  python run.py transform-3-seccao data/all_sections/3_seccao/x_train_emb.pkl data/all_sections/3_seccao/x_test_emb.pkl data/relatorio/3_seccao/y_train.pkl data/relatorio/3_seccao/y_test.pkl data/all_sections/3_seccao/x_train_emb_numpy.pkl data/all_sections/3_seccao/x_test_emb_numpy.pkl data/all_sections/3_seccao/y_train_numpy.pkl data/all_sections/3_seccao/y_test_numpy.pkl
elif [ "$1" = "train-3-seccao" ]
then
    PYTHONUNBUFFERED=1 nohup python run.py train-3-seccao data/all_sections/3_seccao/x_train_emb_numpy.pkl data/all_sections/3_seccao/x_test_emb_numpy.pkl data/all_sections/3_seccao/y_train_numpy.pkl data/all_sections/3_seccao/y_test_numpy.pkl data/all_sections/3_seccao/models.pkl > train3 & disown
  #PYTHONUNBUFFERED=1 nohup python run.py train-3-seccao data/3_seccao/x_train_emb_numpy.pkl data/3_seccao/x_test_emb_numpy.pkl data/3_seccao/y_train_numpy.pkl data/3_seccao/y_test_numpy.pkl data/3_seccao/models.pkl & disown
elif [ "$1" = "test-3-seccao" ]
then
  python run.py test-3-seccao data/3_seccao/x_train_emb_numpy.pkl data/3_seccao/x_test_emb_numpy.pkl data/3_seccao/y_train_numpy.pkl data/3_seccao/y_test_numpy.pkl data/3_seccao/models.pkl
elif [ "$1" = "sections-4-seccao" ]
then
  python run.py sections-4-seccao data/relatorio/4_seccao/data_train.pkl data/4_seccao/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/4_seccao/data_train_with_sections.pkl --cuda
  #python run.py sections-4-seccao data/4_seccao/data_train.pkl data/4_seccao/sections_dict.pkl --cuda
elif [ "$1" = "sections-4-seccao-test" ]
then
  python run.py sections-4-seccao-test data/relatorio/4_seccao/data_test.pkl data/4_seccao/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/4_seccao/data_test_with_sections.pkl  --cuda
  #python run.py sections-4-seccao-test data/4_seccao/data_test.pkl data/4_seccao/data_test_with_sections.pkl  --cuda
elif [ "$1" = "embeddings-4-seccao"   ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-4-seccao data/all_sections/4_seccao/data_train_with_sections.pkl data/relatorio/4_seccao/y_train.pkl data/all_sections/4_seccao/x_train_emb.pkl --cuda > output1 & disown
  #python run.py embeddings-4-seccao data/4_seccao/data_train_with_sections.pkl data/4_seccao/y_train.pkl data/4_seccao/x_train_emb.pkl
elif [ "$1" = "embeddings-4-seccao-test"  ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-4-seccao-test data/all_sections/4_seccao/data_test_with_sections.pkl data/relatorio/4_seccao/y_test.pkl data/all_sections/4_seccao/x_test_emb.pkl --cuda & disown
  #python run.py embeddings-4-seccao-test data/4_seccao/data_test_with_sections.pkl data/4_seccao/y_test.pkl data/4_seccao/x_test_emb.pkl
elif [ "$1" = "transform-4-seccao" ]
then
  python run.py transform-4-seccao data/all_sections/4_seccao/x_train_emb.pkl data/all_sections/4_seccao/x_test_emb.pkl data/relatorio/4_seccao/y_train.pkl data/relatorio/4_seccao/y_test.pkl data/all_sections/4_seccao/x_train_emb_numpy.pkl data/all_sections/4_seccao/x_test_emb_numpy.pkl data/all_sections/4_seccao/y_train_numpy.pkl data/all_sections/4_seccao/y_test_numpy.pkl
elif [ "$1" = "train-4-seccao" ]
then
    PYTHONUNBUFFERED=1 nohup python run.py train-4-seccao data/all_sections/4_seccao/x_train_emb_numpy.pkl data/all_sections/4_seccao/x_test_emb_numpy.pkl data/all_sections/4_seccao/y_train_numpy.pkl data/all_sections/4_seccao/y_test_numpy.pkl data/all_sections/4_seccao/models.pkl > train4 & disown
  #PYTHONUNBUFFERED=1 nohup python run.py train-4-seccao data/4_seccao/x_train_emb_numpy.pkl data/4_seccao/x_test_emb_numpy.pkl data/4_seccao/y_train_numpy.pkl data/4_seccao/y_test_numpy.pkl data/4_seccao/models.pkl & disown
elif [ "$1" = "test-4-seccao" ]
then
  python run.py test-contencioso-seccao data/4_seccao/x_train_emb_numpy.pkl data/4_seccao/x_test_emb_numpy.pkl data/4_seccao/y_train_numpy.pkl data/4_seccao/y_test_numpy.pkl data/4_seccao/models.pkl
elif [ "$1" = "sections-5-seccao" ]
then
  python run.py sections-5-seccao data/relatorio/5_seccao/data_train.pkl data/criminal/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/5_seccao/data_train_with_sections.pkl --cuda
  #python run.py sections-5-seccao data/5_seccao/data_train.pkl data/5_seccao/data_train_with_sections.pkl  --cuda
elif [ "$1" = "sections-5-seccao-test" ]
then
  python run.py sections-5-seccao-test data/relatorio/5_seccao/data_test.pkl data/criminal/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/5_seccao/data_test_with_sections.pkl --cuda
  #python run.py sections-5-seccao-test data/5_seccao/data_test.pkl data/5_seccao/data_test_with_sections.pkl  --cuda
elif [ "$1" = "embeddings-5-seccao"   ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-5-seccao data/all_sections/5_seccao/data_train_with_sections.pkl data/relatorio/5_seccao/y_train.pkl data/all_sections/5_seccao/x_train_emb.pkl --cuda & disown
  #python run.py embeddings-5-seccao data/5_seccao/data_train_with_sections.pkl data/5_seccao/y_train.pkl data/5_seccao/x_train_emb.pkl
elif [ "$1" = "embeddings-5-seccao-test"  ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-5-seccao-test data/all_sections/5_seccao/data_test_with_sections.pkl data/relatorio/5_seccao/y_test.pkl data/all_sections/5_seccao/x_test_emb.pkl --cuda > output1 & disown
elif [ "$1" = "transform-5-seccao" ]
then
  python run.py transform-5-seccao data/all_sections/5_seccao/x_train_emb.pkl data/all_sections/5_seccao/x_test_emb.pkl data/relatorio/5_seccao/y_train.pkl data/relatorio/5_seccao/y_test.pkl data/all_sections/5_seccao/x_train_emb_numpy.pkl data/all_sections/5_seccao/x_test_emb_numpy.pkl data/all_sections/5_seccao/y_train_numpy.pkl data/all_sections/5_seccao/y_test_numpy.pkl
elif [ "$1" = "train-5-seccao" ]
then
    PYTHONUNBUFFERED=1 nohup python run.py train-5-seccao data/all_sections/5_seccao/x_train_emb_numpy.pkl data/all_sections/5_seccao/x_test_emb_numpy.pkl data/all_sections/5_seccao/y_train_numpy.pkl data/all_sections/5_seccao/y_test_numpy.pkl data/all_sections/5_seccao/models.pkl > train5 & disown
elif [ "$1" = "test-5-seccao" ]
then
  python run.py test-5-seccao data/5_seccao/x_train_emb_numpy.pkl data/5_seccao/x_test_emb_numpy.pkl data/5_seccao/y_train_numpy.pkl data/5_seccao/y_test_numpy.pkl data/5_seccao/models.pkl
elif [ "$1" = "sections-6-seccao" ]
then
  python run.py sections-6-seccao data/relatorio/6_seccao/data_train.pkl data/civel/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/6_seccao/data_train_with_sections.pkl --cuda
elif [ "$1" = "sections-6-seccao-test" ]
then
  python run.py sections-6-seccao-test data/relatorio/6_seccao/data_test.pkl data/civel/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/6_seccao/data_test_with_sections.pkl --cuda
elif [ "$1" = "embeddings-6-seccao"   ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-6-seccao data/all_sections/6_seccao/data_train_with_sections.pkl data/relatorio/6_seccao/y_train.pkl data/all_sections/6_seccao/x_train_emb.pkl --cuda & disown
elif [ "$1" = "embeddings-6-seccao-test"  ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-6-seccao-test data/all_sections/6_seccao/data_test_with_sections.pkl data/relatorio/6_seccao/y_test.pkl data/all_sections/6_seccao/x_test_emb.pkl --cuda > output1 & disown
elif [ "$1" = "transform-6-seccao" ]
then
  python run.py transform-6-seccao data/all_sections/6_seccao/x_train_emb.pkl data/all_sections/6_seccao/x_test_emb.pkl data/relatorio/6_seccao/y_train.pkl data/relatorio/6_seccao/y_test.pkl data/all_sections/6_seccao/x_train_emb_numpy.pkl data/all_sections/6_seccao/x_test_emb_numpy.pkl data/all_sections/6_seccao/y_train_numpy.pkl data/all_sections/6_seccao/y_test_numpy.pkl
elif [ "$1" = "train-6-seccao" ]
then
    PYTHONUNBUFFERED=1 nohup python run.py train-6-seccao data/all_sections/6_seccao/x_train_emb_numpy.pkl data/all_sections/6_seccao/x_test_emb_numpy.pkl data/all_sections/6_seccao/y_train_numpy.pkl data/all_sections/6_seccao/y_test_numpy.pkl data/all_sections/6_seccao/models.pkl > train6 & disown
elif [ "$1" = "test-6-seccao" ]
then
  python run.py test-6-seccao data/relatorio/6_seccao/x_train_emb_numpy.pkl data/relatorio/6_seccao/x_test_emb_numpy.pkl data/relatorio/6_seccao/y_train_numpy.pkl data/relatorio/6_seccao/y_test_numpy.pkl data/relatorio/6_seccao/models.pkl
elif [ "$1" = "sections-7-seccao" ]
then
  python run.py sections-7-seccao data/relatorio/7_seccao/data_train.pkl data/civel/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/7_seccao/data_train_with_sections.pkl --cuda
elif [ "$1" = "sections-7-seccao-test" ]
then
  python run.py sections-7-seccao-test data/relatorio/7_seccao/data_test.pkl data/civel/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/7_seccao/data_test_with_sections.pkl --cuda
elif [ "$1" = "embeddings-7-seccao"   ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-7-seccao data/all_sections/7_seccao/data_train_with_sections.pkl data/relatorio/7_seccao/y_train.pkl data/all_sections/7_seccao/x_train_emb.pkl --cuda > output1 & disown
elif [ "$1" = "embeddings-7-seccao-test"  ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-7-seccao-test data/all_sections/7_seccao/data_test_with_sections.pkl data/relatorio/7_seccao/y_test.pkl data/all_sections/7_seccao/x_test_emb.pkl --cuda & disown
elif [ "$1" = "transform-7-seccao" ]
then
  python run.py transform-7-seccao data/all_sections/7_seccao/x_train_emb.pkl data/all_sections/7_seccao/x_test_emb.pkl data/relatorio/7_seccao/y_train.pkl data/relatorio/7_seccao/y_test.pkl data/all_sections/7_seccao/x_train_emb_numpy.pkl data/all_sections/7_seccao/x_test_emb_numpy.pkl data/all_sections/7_seccao/y_train_numpy.pkl data/all_sections/7_seccao/y_test_numpy.pkl
elif [ "$1" = "train-7-seccao" ]
then
    PYTHONUNBUFFERED=1 nohup python run.py train-7-seccao data/all_sections/7_seccao/x_train_emb_numpy.pkl data/all_sections/7_seccao/x_test_emb_numpy.pkl data/all_sections/7_seccao/y_train_numpy.pkl data/all_sections/7_seccao/y_test_numpy.pkl data/all_sections/7_seccao/models.pkl > train7 & disown
elif [ "$1" = "test-7-seccao" ]
then
  python run.py test-7-seccao data/7_seccao/x_train_emb_numpy.pkl data/7_seccao/x_test_emb_numpy.pkl data/7_seccao/y_train_numpy.pkl data/7_seccao/y_test_numpy.pkl data/7_seccao/models.pkl
elif [ "$1" = "sections-contencioso-seccao" ]
then
  python run.py sections-contencioso-seccao data/relatorio/contencioso/data_train.pkl data/contencioso/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/contencioso/data_train_with_sections.pkl  --cuda
elif [ "$1" = "sections-contencioso-seccao-test" ]
then
  python run.py sections-contencioso-seccao-test data/relatorio/contencioso/data_test.pkl data/contencioso/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/contencioso/data_test_with_sections.pkl  --cuda
elif [ "$1" = "embeddings-contencioso-seccao" ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-contencioso-seccao data/all_sections/contencioso/data_train_with_sections.pkl data/relatorio/contencioso/y_train.pkl data/all_sections/contencioso/x_train_emb.pkl --cuda > output2 & disown
  #python run.py embeddings-contencioso data/Contencioso/data_train.pkl data/Contencioso/y_train.pkl data/Contencioso/x_train_emb.pkl
elif [ "$1" = "embeddings-contencioso-seccao-test" ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-contencioso-seccao-test data/all_sections/contencioso/data_test_with_sections.pkl data/relatorio/contencioso/y_test.pkl data/all_sections/contencioso/x_test_emb.pkl --cuda > output3 & disown
elif [ "$1" = "transform-contencioso-seccao" ]
then
  python run.py transform-contencioso-seccao data/all_sections/contencioso/x_train_emb.pkl data/all_sections/contencioso/x_test_emb.pkl data/relatorio/contencioso/y_train.pkl data/relatorio/contencioso/y_test.pkl data/all_sections/contencioso/x_train_emb_numpy.pkl data/all_sections/contencioso/x_test_emb_numpy.pkl data/all_sections/contencioso/y_train_numpy.pkl data/all_sections/contencioso/y_test_numpy.pkl
elif [ "$1" = "train-contencioso-seccao" ]
then
  python run.py train-contencioso-seccao data/all_sections/contencioso/x_train_emb_numpy.pkl data/all_sections/contencioso/x_test_emb_numpy.pkl data/all_sections/contencioso/y_train_numpy.pkl data/all_sections/contencioso/y_test_numpy.pkl data/all_sections/contencioso/models.pkl
elif [ "$1" = "test-contencioso-seccao" ]
then
  python run.py test-contencioso-seccao data/contencioso/x_train_emb_numpy.pkl data/contencioso/x_test_emb_numpy.pkl data/contencioso/y_train_numpy.pkl data/contencioso/y_test_numpy.pkl data/contencioso/models.pkl
elif [ "$1" = "sections-civel" ]
then
  python run.py sections-civel data/relatorio/civel/data_train.pkl data/civel/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/civel/data_train_with_sections.pkl --cuda
elif [ "$1" = "sections-civel-test" ]
then
  python run.py sections-civel-test data/relatorio/civel/data_test.pkl data/civel/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/civel/data_test_with_sections.pkl --cuda
elif [ "$1" = "embeddings-civel"   ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-civel data/all_sections/civel/data_train_with_sections.pkl data/relatorio/civel/y_train.pkl data/all_sections/civel/x_train_emb.pkl --cuda & disown
elif [ "$1" = "embeddings-civel-test"  ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-civel-test data/all_sections/civel/data_test_with_sections.pkl data/relatorio/civel/y_test.pkl data/all_sections/civel/x_test_emb.pkl --cuda > output1 & disown
elif [ "$1" = "transform-civel" ]
then
  python run.py transform-civel data/all_sections/civel/x_train_emb.pkl data/all_sections/civel/x_test_emb.pkl data/relatorio/civel/y_train.pkl data/relatorio/civel/y_test.pkl data/all_sections/civel/x_train_emb_numpy.pkl data/all_sections/civel/x_test_emb_numpy.pkl data/all_sections/civel/y_train_numpy.pkl data/all_sections/civel/y_test_numpy.pkl
elif [ "$1" = "train-civel" ]
then
  PYTHONUNBUFFERED=1 nohup python run.py train-civel data/all_sections/civel/x_train_emb_numpy.pkl data/all_sections/civel/x_test_emb_numpy.pkl data/all_sections/civel/y_train_numpy.pkl data/all_sections/civel/y_test_numpy.pkl data/all_sections/civel/models.pkl > traincivel & disown
elif [ "$1" = "test-civel" ]
then
  python run.py test-civel data_area/civel/x_train_emb_numpy.pkl data_area/civel/x_test_emb_numpy.pkl data_area/civel/y_train_numpy.pkl data_area/civel/y_test_numpy.pkl data_area/civel/models.pkl
elif [ "$1" = "sections-criminal" ]
then
  python run.py sections-criminal data/relatorio/criminal/data_train.pkl data/criminal/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/criminal/data_train_with_sections_1.pkl  data/all_sections/criminal/data_train_with_sections_2.pkl --cuda
elif [ "$1" = "sections-criminal-test" ]
then
  python run.py sections-criminal-test data/relatorio/criminal/data_test.pkl data/criminal/sections_dict.pkl '["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "colectivo", "declaração", "foot-note", "título"]' data/all_sections/criminal/data_test_with_sections.pkl --cuda
elif [ "$1" = "embeddings-criminal"   ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-criminal data/all_sections/criminal/data_train_with_sections_2.pkl data/relatorio/criminal/y_train.pkl data/all_sections/criminal/x_train_emb_2.pkl --cuda > output1 & disown
elif [ "$1" = "embeddings-criminal-test"  ]
then
  PYTHONUNBUFFERED=1 nohup python run.py embeddings-criminal-test data/all_sections/criminal/data_test_with_sections.pkl data/relatorio/criminal/y_test.pkl data/all_sections/criminal/x_test_emb.pkl --cuda > output1 & disown
elif [ "$1" = "transform-criminal" ]
then
  python run.py transform-criminal data/all_sections/criminal/x_train_emb.pkl data/all_sections/criminal/x_test_emb.pkl data/relatorio/criminal/y_train.pkl data/relatorio/criminal/y_test.pkl data/all_sections/criminal/x_train_emb_numpy.pkl data/all_sections/criminal/x_test_emb_numpy.pkl data/all_sections/criminal/y_train_numpy.pkl data/all_sections/criminal/y_test_numpy.pkl
elif [ "$1" = "train-criminal" ]
then
    PYTHONUNBUFFERED=1 nohup python run.py train-criminal data/all_sections/criminal/x_train_emb_numpy.pkl data/all_sections/criminal/x_test_emb_numpy.pkl data/all_sections/criminal/y_train_numpy.pkl data/all_sections/criminal/y_test_numpy.pkl data/all_sections/criminal/models.pkl > traincriminal & disown
elif [ "$1" = "test-civel-all" ]
then
  python run.py test-civel-all data/all_sections/1_seccao/x_test_emb_numpy.pkl data/all_sections/1_seccao/y_test_numpy.pkl data/all_sections/1_seccao/y_train_numpy.pkl data/all_sections/1_seccao/models.pkl data/all_sections/2_seccao/y_train_numpy.pkl data/all_sections/2_seccao/models.pkl data/all_sections/6_seccao/y_train_numpy.pkl data/all_sections/6_seccao/models.pkl data/all_sections/7_seccao/y_train_numpy.pkl data/all_sections/7_seccao/models.pkl
elif [ "$1" = "test-criminal-all" ]
then
  python run.py test-criminal-all data/all_sections/3_seccao/x_test_emb_numpy.pkl data/all_sections/3_seccao/y_test_numpy.pkl data/all_sections/3_seccao/y_train_numpy.pkl data/all_sections/3_seccao/models.pkl data/all_sections/5_seccao/y_train_numpy.pkl data/all_sections/5_seccao/models.pkl
fi





