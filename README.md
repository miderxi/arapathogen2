# 1. generate feature of EsmMean and AraNetSturc2vec
python ./features/EsmMean/extract2.py
python ./features/AraNetStruc2vec/gen_struc2vec_v3.py

# 2. run train and test
bash ./script/run.sh


# 3. visit our web server
http://zzdlab.com/intersppi/arapathogen/index.php


