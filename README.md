# 1. Directory structure
```
.
├── data
│   ├── 10folds_C1223           # ppi of training and testing set .
│   ├── effector_ppi_database   # ppi database extract from effectork #doi: 10.1111/mpp.12965. 
│   └── sequences               # sequences of arabidopsis and pathogen effector proteins. 
│       ├── ara_and_eff.tar.xz  
│       ├── only_need.fasta     # protein sequences that only  ppear in the training and testing sets.
│       └── only_need.py        # script for generate 
├── features    # Script required for generating features
│   ├── AC
│   │   └── generate_AC_encode.py   # script for autocovariance encoding.
│   ├── AraNetNode2vec  #script for 
│   ├── AraNetStruc2vec
│   ├── CKSAAP
│   ├── CT
│   ├── doc2vec
│   ├── DPC
│   ├── EsmMean
│   └── prottrans
├── output
│   ├── model_state
│   └── pics
└── script      # script for train and test.
    ├── 10folds_NN_C1223.py
    ├── 10folds_RF_C1223.py
    ├── 10folds_SVM_C1223.py
    ├── 10folds_Xgboost_C1223.py    # Training and testing code for XGBoost
    ├── gen_Xgboost_model.py
    ├── RF_features.py
    └── run.sh  # execute command
```
# 2. Generate features
```
    # Run each feature generation script.
    # example 
    cd ./features/AC
    python generate_AC_encode.py
```

# 3. For XGBoost train and test.
```
    cd ./script/
    python 10folds_RF_C1223.py  EsmMean AraNetStruc2vec
```

# 4. Our web server
http://zzdlab.com/intersppi/arapathogen/index.php


