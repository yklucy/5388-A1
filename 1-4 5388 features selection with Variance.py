import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold

#read data from features preprocessing - MinMaxScaler
data_aftcat_aftif_aftmm = pd.read_csv(r"./1-3 5388 traindata_aftcat_aftif_aftmm.csv")
testdata_aftcat_aftmm = pd.read_csv(r"./1-3 5388 testdata_aftcat_aftmm.csv")

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^ feature selection - variance ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# author:           Kun Yan
# student number:   300259303
# data:             2021-10-03
# Python version:   3.9.7
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Step3:  select features on train dataset with VarianceThreshold() ---------------------
selector = VarianceThreshold(np.median(data_aftcat_aftif_aftmm.var().values)).fit(data_aftcat_aftif_aftmm)

data_aftcat_aftif_aftmm_aftfs = pd.DataFrame(selector.transform(data_aftcat_aftif_aftmm))
data_aftcat_aftif_aftmm_aftfs.columns = selector.get_feature_names_out()

testdata_aftcat_aftmm_aftfs = pd.DataFrame(selector.transform(testdata_aftcat_aftmm))
testdata_aftcat_aftmm_aftfs.columns = selector.get_feature_names_out()


#-----------output-------------------

data_aftcat_aftif_aftmm_aftfs.to_csv("./1-4 5388 traindata_aftcat_aftif_aftmm_aftfs.csv",index=0)
testdata_aftcat_aftmm_aftfs.to_csv("./1-4 5388 testdata_aftcat_aftmm_aftfs.csv",index=0)