import numpy as np

class Features_C123:
    def __init__(self,info:list):
        class MyFeature():
            def __init__(self,pkl_files, allow_empty=False):
                self.data_loaded = False
                self.data_files = pkl_files
                self.allow_empty = allow_empty #host only feature,
                self.data = None
                self.data_shape = None

            def __getitem__(self,index):
                if not self.data_loaded:
                    self.data = dict()
                    for pkl_file in self.data_files:
                        self.data.update(np.load(pkl_file, allow_pickle=True))
                    self.data_loaded = True
                    self.data_shape = list(self.data.values())[0].shape 
                    if 'average' not in self.data.keys():
                        self.data['average'] = np.array(list(self.data.values())).mean(0)

                if not self.allow_empty:
                    v = self.data[index]
                else:
                    if index in self.data.keys():
                        v = self.data[index] 
                    else:
                        v = self.data['average']
                return v
               
        feature_EsmMean   = MyFeature(["../features/EsmMean/ara_and_eff_EsmMean_v2.pkl"])  #shape(1280,)
        feature_EsmMean   = MyFeature(["../features/EsmMean/ara_and_eff_only_need_esm2.pkl"])  #shape(2560,)
        feature_prottrans = MyFeature(["../features/prottrans/ara_and_eff_prottrans_v2.pkl"]) #shape (1024,)
        feature_prottrans = MyFeature(["../features/prottrans/ara_and_eff_only_need_prottrans.pkl"]) #shape (1024,)
        feature_doc2vec   = MyFeature(["../features/doc2vec/ara_and_eff_doc2vec_128.pkl"])  #shape (400,)
        
        feature_ac  = MyFeature(["../features/AC/ara_and_eff_AC_210.pkl"])  #shape(210,)
        feature_ct  = MyFeature(["../features/CT/ara_and_eff_CT_343.pkl"])  #shape (343,)
        feature_dpc = MyFeature(["../features/DPC/ara_and_eff_DPC_400.pkl"])                #shape (400,)
        feature_cksaap = MyFeature(["../features/CKSAAP/ara_and_eff_CKSAAP.pkl"])      #shape (1200,)
        
        feature_AraNetStruc2vec = MyFeature(["../features/AraNetStruc2vec/AraNetStruc2vec_256_v3.pkl"],allow_empty=True)
        feature_AraNetNode2vec  = MyFeature(["../features/AraNetNode2vec/AraNet_node2vec_256.pkl"],allow_empty=True) #
        
        #feature_EsmMsa    = MyFeature(["../features/esm-msa/ara_and_eff_esm-msa.pkl"])  #shape (768,)
        #feature_sublocation = MyFeature(["../features/sublocation/sublocaiton_34.pkl"],allow_empty=True)       
        #feature_DMINode2vec = MyFeature(['../features/dmi_node2vec/ara_and_eff_dmi_node2vec_128.pkl'],allow_empty=True)
        #feature_PSP    = MyFeature(["../features/SecondaryStructure/features_secd.pkl"])
        feature_AraNetProperty = MyFeature(["../features/AraNetProperty/AraNetProperty_7_v1.pkl"],allow_empty=True)
        #feature_AraGeo = MyFeature(["../features/gene_expression/geo_111.pkl"],allow_empty=True)

        self.info = info
        self.features={
            'EsmMean':feature_EsmMean,
            'prottrans':feature_prottrans,
            'doc2vec':feature_doc2vec,
            
            'ac':feature_ac,
            'ct':feature_ct,
            'dpc':feature_dpc,
            'cksaap':feature_cksaap,
            
            'AraNetNode2vec':feature_AraNetNode2vec,
            'AraNetStruc2vec':feature_AraNetStruc2vec,
            #'EsmMsa':feature_EsmMsa,
            #'PSP':feature_PSP,
            #'DMINode2vec':feature_DMINode2vec,
            #'stringNet':feature_StringAraNetStruc2vec
            'AraNetProperty':feature_AraNetProperty,
            #'AraGeo':feature_AraGeo,
            #'sublocation':feature_sublocation,
            }
    
    def get(self,index,foldn=None):
        return np.hstack([self.features[i][index] for i in self.info])

    def __getitem__(self,index):
        return np.hstack([self.features[i][index] for i in self.info])


