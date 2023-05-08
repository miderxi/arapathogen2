import tensorflow as tf
from tensorflow import keras

import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from zzd.utils.assess import multi_scores
import pickle
from sklearn.model_selection import train_test_split
from RF_features import Features_C123 as Features
from sklearn.svm import SVC


def dense_model(input_shape):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(keras.layers.Dense(512, activation="relu", name="layer1"))
    model.add(keras.layers.Dense(256, activation="relu", name="layer2"))
    #model.add(keras.layers.Dense(128, activation="relu", name="layer3"))
    #model.add(layers.Dense(64, activation="relu", name="layer4"))
    #model.add(keras.layers.Dense(32, activation="relu", name="layer5"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1,  activation="sigmoid", name="layer6"))
    model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy'])
    return model


#1. parameters
print(sys.argv)
info_list = sys.argv[1:] if len(sys.argv)>1 else None
output_dir = f"../output/preds/10folds_C1223_NN_"+"_".join(info_list)
os.makedirs(output_dir,exist_ok=True)



#2. load data
#2.1 load C1 C2 and C3 file #加载4个测试集
#c1_train_files = [f'../data/10folds_C1223/C1_train_fold{i}.txt' for i in range(10)] 
#c1_test_files = [f'../data/10folds_C1223/C1_test_fold{i}.txt' for i in range(10)] 
c1_train_files = [f'../data/10folds_C1223/C1_fold{i}_train.txt' for i in range(10)] 
c1_test_files = [f'../data/10folds_C1223/C1_fold{i}_test.txt' for i in range(10)] 
c2h_files = [f'../data/10folds_C1223/C2h_fold{i}.txt' for i in range(10)] 
c2p_files = [f"../data/10folds_C1223/C2p_fold{i}.txt" for i in range(10)]
c3_files = [f'../data/10folds_C1223/C3_fold{i}.txt' for i in range(10)] 

#2.2 load features
features = Features(info=info_list)

c1_test_scores  = []
c2h_scores = []
c2p_scores = []
c3_scores  = []

for foldn in range(10):
    print(f"fold{foldn}: load file => ",end="");sys.stdout.flush();
    c1_train = np.genfromtxt(c1_train_files[foldn],str)
    c1_test  = np.genfromtxt(c1_test_files[foldn],str)
    c2h = np.genfromtxt(c2h_files[foldn],str)
    c2p = np.genfromtxt(c2p_files[foldn],str)
    c3 = np.genfromtxt(c3_files[foldn],str)

    print("encode file =>",end="");sys.stdout.flush();#对蛋白进行编码
    X_c1_train, y_c1_train = c1_train[:,:2], c1_train[:,2].astype(np.float32)
    X_c1_test,  y_c1_test  = c1_test[:,:2],  c1_test[:,2].astype(np.float32)
    X_c2h,      y_c2h      = c2h[:,:2],      c2h[:,2].astype(np.float32)
    X_c2p,      y_c2p      = c2p[:,:2],      c2p[:,2].astype(np.float32)
    X_c3,       y_c3       = c3[:,:2],       c3[:,2].astype(np.float32)
    
    x_c1_train = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c1_train ]) 
    x_c1_test  = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c1_test  ]) 
    x_c2h      = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c2h      ]) 
    x_c2p      = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c2p      ]) 
    x_c3       = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c3       ])
    
    print("training...")
    model = dense_model(x_c1_train.shape[1])
    model.fit(x_c1_train,y_c1_train, epochs=30,batch_size=64,
            verbose=False,
            shuffle=True,
            workers=15)

    print("predicting..")
    y_c1_test_pred = model.predict(x_c1_test)
    y_c2h_pred     = model.predict(x_c2h)
    y_c2p_pred     = model.predict(x_c2p)
    y_c3_pred      = model.predict(x_c3)

    c1_test_score = multi_scores(y_c1_test, y_c1_test_pred, show=True,threshold=0.5)
    c2h_score     = multi_scores(y_c2h,     y_c2h_pred,     show=True,show_index=False,threshold=0.5)
    c2p_score     = multi_scores(y_c2p,     y_c2p_pred,     show=True,show_index=False,threshold=0.5)
    c3_score      = multi_scores(y_c3,      y_c3_pred,      show=True,show_index=False,threshold=0.5)
    
    c1_test_scores.append(c1_test_score)
    c2h_scores.append(c2h_score)
    c2p_scores.append(c2p_score)
    c3_scores.append(c3_score)

    #保存结果
    #save pred result
    with open(f"{output_dir}/c1_test_pred_{foldn}.txt","w") as f:
        for line in np.hstack([X_c1_test,y_c1_test.reshape(-1,1),y_c1_test_pred.reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    with open(f"{output_dir}/c2h_pred_{foldn}.txt","w") as f:
        for line in np.hstack([X_c2h, y_c2h.reshape(-1,1), y_c2h_pred.reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    with open(f"{output_dir}/c2p_pred_{foldn}.txt","w") as f:
        for line in np.hstack([X_c2p, y_c2p.reshape(-1,1), y_c2p_pred.reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    with open(f"{output_dir}/c3_pred_{foldn}.txt","w") as f:
        for line in np.hstack([X_c3, y_c3.reshape(-1,1), y_c3_pred.reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    #save pred score
    with open(f"{output_dir}/c1_test_score_{foldn}.txt","w") as f:
            f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n")
            f.write("\t".join([str(i) for i in c1_test_score]))

    with open(f"{output_dir}/c2h_score_{foldn}.txt","w") as f:
            f.write(f"TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n")
            f.write("\t".join([str(i) for i in c2h_score]))

    with open(f"{output_dir}/c2p_score_{foldn}.txt","w") as f:
            f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n")
            f.write("\t".join([str(i) for i in c2p_score]))

    with open(f"{output_dir}/c3_score_{foldn}.txt","w") as f:
            f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n")
            f.write("\t".join([str(i) for i in c3_score]))


print("10 fold C1223 average")
c1_test_scores = np.array(c1_test_scores)
fmat =  [1, 1,  1,  1,  3,  3,  3,  3,  3,  3,  3,      3]
with open(f"{output_dir}/c1_test_average_score.txt",'w') as f:
    line1 = f"TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c1_test_scores.mean(0),c1_test_scores.std(0))])
    print(line1,end="")
    print('\t'.join([f'{a:.{_}f}' for (_,a) in zip(fmat,c1_test_scores.mean(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

c2h_scores = np.array(c2h_scores)
with open(f"{output_dir}/c2h_average_score.txt",'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c2h_scores.mean(0),c2h_scores.std(0)) ])
    #print(line1)
    print('\t'.join([f'{a:.{_}f}' for (_,a) in zip(fmat,c2h_scores.mean(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

c2p_scores = np.array(c2p_scores)
with open(f"{output_dir}/c2p_average_score.txt",'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c2p_scores.mean(0),c2p_scores.std(0)) ])
    #print(line1)
    print('\t'.join([f'{a:.{_}f}' for (_,a) in zip(fmat,c2p_scores.mean(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

c3_scores = np.array(c3_scores)
with open(f"{output_dir}/c3_average_score.txt",'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c3_scores.mean(0),c3_scores.std(0))])
    #print(line1)
    print('\t'.join([f'{a:.{_}f}' for (_,a) in zip(fmat,c3_scores.mean(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

print("-----------------------------------")
