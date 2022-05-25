import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ^^^^^^^^^^^^^^^^^^^^^^feature preprocessing - MinMaxScaler()^^^^^^^^^^^^^^^^^^
# author:           Kun Yan
# student number:   300259303
# data:             2021-10-03
# Python version:   3.9.7
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#read data sets from files after categorical feature preprocessing and anomaly detection
data_aftcat_aftif = pd.read_csv(r"./1-2 5388 traindata_aftcat_aftif.csv")
data_aftcat_aftif_y = pd.read_csv(r"./1-2 5388 traindata_aftcat_aftif_label.csv")

testdata_aftcat = pd.read_csv(r"./1-1 5388 testdata_aftcat.csv")

# step 2:  All features - MinMaxScaler()
# 
# numerical: 
# MinMaxScaler() - [0,1] / 
# MaxAbsScaler() - [0,1] / 
# PowerTransformer() - zero-mean, unit variance normalization / Box-Cox - to stablilize variance and minimize skewness
# QuantileTransformer(uniform output) - [0,1] - robust to outliers
# Both StandardScaler and MinMaxScaler are very sensitive to the presence of outliers.
# MaxAbsScaler therefore also suffers from the presence of large outliers.

def num_trans(X,X_test):
    fi = MinMaxScaler().fit(X)
    result = fi.transform(X)
    result_test = fi.transform(X_test)

    X = pd.DataFrame(result)
    X.columns = fi.get_feature_names_out()

    X_test = pd.DataFrame(result_test)
    X_test.columns = fi.get_feature_names_out()
    return X, X_test

# transform all feature with MinMaxScaler()
data_aftcat_aftif_aftmm, testdata_aftcat_aftmm = num_trans(data_aftcat_aftif,testdata_aftcat)


#-----------output-------------------

data_aftcat_aftif_aftmm.to_csv("./1-3 5388 traindata_aftcat_aftif_aftmm.csv",index=0)
testdata_aftcat_aftmm.to_csv("./1-3 5388 testdata_aftcat_aftmm.csv",index=0)
