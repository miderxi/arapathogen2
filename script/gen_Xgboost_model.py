"""
example
python 10folds_RF_C1223.py EsmMean AraNetStruc2vec
"""
import sys
import os
import numpy as np
#from sklearn.ensemble import RandomForestClassifier
from zzd.utils.assess import multi_scores
import pickle
from sklearn.model_selection import train_test_split
from RF_features import Features_C123 as Features
import xgboost as xgb

features = Features(info=['EsmMean','AraNetStruc2vec'])
c1223 = np.genfromtxt("../data/10folds_C1223/all_pos_and_neg.txt",str)
    
print("encode file =>",end="");sys.stdout.flush();#对蛋白进行编码
X_c1223,y_c1223 = c1223[:,:2], c1223[:,2].astype(np.float32)
x_c1223 = np.array([np.hstack([features.get(j,0) for j in i]) for i in X_c1223 ]) 

model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
          colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.4,
          early_stopping_rounds=None, enable_categorical=False,
          eval_metric=None, gamma=0.2, gpu_id=-1, grow_policy='depthwise',
          importance_type=None, interaction_constraints='',
          learning_rate=0.05, max_bin=256, max_cat_to_onehot=4,
          max_delta_step=0, max_depth=12, max_leaves=0, min_child_weight=5,
          monotone_constraints='()', n_estimators=400,
          n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0,
          reg_alpha=0, reg_lambda=1)


print("training==>",end="");sys.stdout.flush();
model.fit(x_c1223, y_c1223)


model.save_model("../output/model_state/Xgboost_EsmMean_AraNetStruc2vec.json")



print("-----------------------------------")
