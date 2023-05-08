#comparation of graph node embedding
python 10folds_RF_C1223.py AraNetStruc2vec
python 10folds_RF_C1223.py AraNetNode2vec

#comparation of different ML
python 10folds_RF_C1223.py  EsmMean AraNetStruc2vec
python 10folds_NN_C1223.py  EsmMean AraNetStruc2vec
python 10folds_SVM_C1223.py EsmMean AraNetStruc2vec
python 10folds_Xgboost.py   EsmMean AraNetStruc2vec

#comparation of sequence-base methods
python 10folds_Xgboost_C1223.py dpc
python 10folds_Xgboost_C1223.py ct
python 10folds_Xgboost_C1223.py ac
python 10folds_Xgboost_C1223.py cksaap
python 10folds_Xgboost_C1223.py EsmMean
python 10folds_Xgboost_C1223.py doc2vec
python 10folds_Xgboost_C1223.py prottrans

#comparation of graph embedding
python 10folds_Xgboost_C1223.py AraNetStruc2vec
python 10folds_Xgboost_C1223.py AraNetNode2vec




