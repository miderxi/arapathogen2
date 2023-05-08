import numpy as np

# 统计训练集于测试集样本数量
for foldn in range(10):
    c1_train = np.genfromtxt(f"./C1_fold{foldn}_train.txt",str)
    c1_test = np.genfromtxt(f"./C1_fold{foldn}_test.txt",str)
    c2h = np.genfromtxt(f"./C2h_fold{foldn}.txt",str)
    c2p = np.genfromtxt(f"./C2p_fold{foldn}.txt",str)
    c3 = np.genfromtxt(f"./C3_fold{foldn}.txt",str)

    
    #print("pos number:")
    a=len([_ for _ in c1_train if float(_[2])==1])
    b=len([_ for _ in c1_test if  float(_[2])==1])
    c=len([_ for _ in c2h if      float(_[2])==1])
    d=len([_ for _ in c2p if      float(_[2])==1])
    e=len([_ for _ in c3 if       float(_[2])==1])
    print(a,b,c,d,e)

    #print("neg number:")
    a=len([_ for _ in c1_train if float(_[2])==0])
    b=len([_ for _ in c1_test if float(_[2])==0])
    c=len([_ for _ in c2h if float(_[2])==0])
    d=len([_ for _ in c2p if float(_[2])==0])
    e=len([_ for _ in c3 if float(_[2])==0])
    print(a,b,c,d,e)
    
    #print("total number:")
    a=len(c1_train)
    b=len(c1_test )
    c=len(c2h )
    d=len(c2p )
    e=len(c3 )
    print(a,b,c,d,e)
    #break
