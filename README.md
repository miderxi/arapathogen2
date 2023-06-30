## 1. Directory structure
```
.
├── data
│   ├── 10folds_C1223               # PPIs for training and test sets.
│   ├── effector_ppi_database       # The ppi database extracted from effectwork #doi: 10.1111/mpp.12965. 
│   └── sequences                   # Sequences of Arabidopsis and pathogen effector proteins. 
│       ├── ara_and_eff.tar.xz      # Sequences of Arabidopsis protome and pathogen effector.
│       ├── only_need.fasta         # Protein sequences that appear only in the training set and the test set.
│       └── only_need.py            # Generating script for only_need.fasta. 
├── features    # Script required for generating features
│   ├── AraNetNode2vec              # Node2vec feature generation code.
│   ├── AraNetStruc2vec
│   │   ├── AraNetStruc2vec_256_v3.pkl  # Pre-trained struct2vec features.
│   │   ├── env_graph2vec.yml       # Conda configuration environment for struc2vec.
│   │   ├── gen_struc2vec_v3.py     # Code for generating struc2vec feature.
│   │   └── Intact_TAIR_protein_interaction.txt # Arabidopsis thaliana PPIs collected from UniProt and TAIR.
│   ├── AC                          # AC feature generation code.
│   ├── CT                          # CT feature generation code.
│   ├── DPC                         # DPC feature generation code.
│   ├── CKSAAP                      # CKSAAP feature generation code.
│   ├── EsmMean                     
│   │   ├── esm2_env.yml            # Conda configuration environment for EsmMean.
│   │   ├── extract2.py             # Script for generate EsmMena feature.
│   │   └── temp.sh                 # Bash code for running extract2.py.
│   └── prottrans                   # prottrans feature generation code.
├── output  # Prediction results and statistics.
│   ├── preds.tar.xz                # Prediction results of models.
│   ├── graph-embed-compare.ipynb   # Comparation of graph embedding.
│   ├── method-compare.ipynb        # Comparation of three plant-pathogen methods.
│   ├── ml-compare.ipynb            # Comparation of four ML algorithm.  
│   └── sequence-based-compare.ipynb# Comparation of various sequence-based encoding. 
└── script  # Script for train and test.
    ├── 10folds_NN_C1223.py         # Training and testing code for MLP.
    ├── 10folds_RF_C1223.py         # Training and testing code for RF.
    ├── 10folds_SVM_C1223.py        # Training and testing code for SVM.
    ├── 10folds_Xgboost_C1223.py    # Training and testing code for XGBoost.
    ├── gen_Xgboost_model.py        # Generate model with whole data.
    ├── RF_features.py              # Load feature.
    ├── run.sh                      # Execute command.
    └── script_env.yml              # Conda configuration environment.

```
## 2. Generate features
Run the generation code in the corresponding feature folder to generate the features.
```
# Example 
# Create the conda runtime environment from the provided yml file.
tar -xvf ./data/sequences/ara_and_eff.tar.xz
cd ./features/AC
python generate_AC_encode.py
```

## 3. For XGBoost train and test.
```
cd ./script/
python 10folds_RF_C1223.py  EsmMean AraNetStruc2vec
```

## 4. Our web server
http://zzdlab.com/intersppi/arapathogen/index.php


