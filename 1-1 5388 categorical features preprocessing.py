import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#read data from dataset
data = pd.read_csv(r"./traindata.csv")
testdata = pd.read_csv(r"./testdata.csv")


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^ categorical features preprocessing  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# author:           Kun Yan
# student number:   300259303
# data:             2021-10-03
# Python version:   3.9.7
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# Step1: drop 3 categorical features:
#              ID, num_outbound_cmds (value=0), is_host_login (value=0)
data.drop(["ID", "num_outbound_cmds", "is_host_login"],axis=1,inplace=True)
testdata.drop(["ID", "num_outbound_cmds", "is_host_login"],axis=1,inplace=True)


# step 2: categorical features:
# OneHotEncorder() - non importance, non order
# f2 (protocol_type),f3 (service),f4(flag) -OneHotEncoder to 80 columns
def cat_trans(X,X_test):
    t = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(X)

    result = t.transform(X).toarray()
    result_test = t.transform(X_test).toarray()

    X = pd.DataFrame(result)
    X.columns = t.get_feature_names_out()

    X_test = pd.DataFrame(result_test)
    X_test.columns = t.get_feature_names_out()
    return X, X_test

f_cat_org = pd.concat([data["protocol_type"],data["service"],data["flag"]],axis=1)
f_cat_test_org = pd.concat([testdata["protocol_type"],testdata["service"],testdata["flag"]],axis=1)

f_cat, f_cat_test = cat_trans(f_cat_org,f_cat_test_org)


################# drop and concat ##########
data.drop(["protocol_type", "service", "flag"],axis=1,inplace=True)
testdata.drop(["protocol_type", "service", "flag"],axis=1,inplace=True)

data_aftcat = pd.concat([data.iloc[:,0:-1],f_cat],axis=1)
data_y = data["Class"]
testdata_aftcat = pd.concat([testdata,f_cat_test],axis=1)


#-----------output-------------------

data_aftcat.to_csv("./1-1 5388 traindata_aftcat.csv",index=0)
data_y.to_csv("./1-1 5388 traindata_aftcat_label.csv",index=0)
data_y = pd.DataFrame(data_y)
testdata.to_csv("./1-1 5388 testdata_aftcat.csv",index=0)