from os import fsdecode
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate



#read data from dataset (after categorical feature preprocessing and anomaly detection(IF))
# train model: aftcat,aftif, CV(Pipeline(MinMaxScaler, selector, model))
data_aftcat_aftif = pd.read_csv(r"./1-2 5388 traindata_aftcat_aftif.csv")
data_aftcat_aftif_y = pd.read_csv(r"./1-2 5388 traindata_aftcat_aftif_label.csv")

# predict model: aftcat, aftif, aftmm, aftfs
data_aftcat_aftif_aftmm_aftfs = pd.read_csv(r"./1-4 5388 traindata_aftcat_aftif_aftmm_aftfs.csv")
testdata_aftcat_aftmm_aftfs = pd.read_csv(r"./1-4 5388 testdata_aftcat_aftmm_aftfs.csv") 

# function score_mean_std: calculate the average and std of scores
def score_mean_std(score_sf):
    index_a1 = []
    index_a2 = []

    for i in range(score_sf.shape[1]):
        index_a1.append(np.average(score_sf[score_sf.columns[i]]))
        index_a2.append(np.std(score_sf[score_sf.columns[i]]))

    index_a1 = pd.DataFrame(index_a1, index = score_sf.columns)
    index_a2 = pd.DataFrame(index_a2, index = score_sf.columns)
    #print(index_a1.T)
    #print(index_a2.T)
    re = pd.concat([index_a1.T,index_a2.T],axis=0)
    re.index = ["avg","std"]
    #print(re)
    return re

# -------------- train predictive model with pipeline: MinMaxScaler(), VarianceThreshold(), cross_validate()--------
def train_model(m, se, model, X, y):
    #print(model, "\n")
    pipe_steps = [('mm',m),('selector',se),('model',model)]
    id_pipeline = Pipeline(steps=pipe_steps)

    # evaluate the pipeline using the crossvalidation technique defined in cv

    # ------------------- score ------------------------------------
    scoring = {'f1', 'precision', 'accuracy',
            'recall', 'roc_auc'}

    score_sf = cross_validate(id_pipeline, X, y.values.ravel(),cv=10,scoring=scoring, n_jobs=-1)
    score_sf_re = score_mean_std(pd.DataFrame(score_sf))

    return score_sf,score_sf_re


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Prediction  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def pre_submission(model,X, y, X_test):
    print(model,"\n")
    model.fit(X, y.values.ravel())

    y_pre = model.predict(X_test).astype(int)
    print("sum:",y_pre.sum())

    pre = pd.DataFrame(columns=["ID","Class"])
    pre["ID"] = range(y_pre.shape[0])
    pre["Class"] = y_pre
    return pre


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^ train models: RF, GRB, MLP, DT ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# repeat several times for parameters tuning
# author:           Kun Yan
# student number:   300259303
# data:             2021-10-03
# Python version:   3.9.7
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

mm = MinMaxScaler()
selector = VarianceThreshold(np.median(data_aftcat_aftif.var().values))
model_rf = RandomForestClassifier(random_state = 90, min_samples_split=2,n_estimators=61,max_depth=24, max_features=27,min_samples_leaf=1, n_jobs=-1)
model_grb = GradientBoostingClassifier(max_features=3,learning_rate=0.1,n_estimators=130,min_samples_split=100,min_samples_leaf=7,max_depth=15,random_state = 10)

# -------------------- model: RF (repeat several times for parameters tuning)-----------------------
print(model_rf)
score_sf_rf,score_sf_re_rf = train_model(mm, selector, model_rf, data_aftcat_aftif, data_aftcat_aftif_y)
print("score_sf_re_rf: \n", pd.DataFrame(score_sf_re_rf))
RF = pd.DataFrame(score_sf_rf['test_f1'],columns = ["RF"])
#print(RF)

# -------------------- model: GRB (repeat several times for parameters tuning) -----------------------
print(model_grb)
score_sf_grb,score_sf_re_grb = train_model(mm, selector, model_grb, data_aftcat_aftif, data_aftcat_aftif_y)
print("score_sf_re_grb: \n", pd.DataFrame(score_sf_re_grb))
GRB = pd.DataFrame(score_sf_grb['test_f1'],columns = ["GRB"])
print(GRB)



# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Prediction^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# author:           Kun Yan
# student number:   300259303
# data:             2021-10-03
# Python version:   3.9.7
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#  ----------------------------- RF -------------------
pre_rf = pre_submission(model_rf,data_aftcat_aftif_aftmm_aftfs, data_aftcat_aftif_y, testdata_aftcat_aftmm_aftfs)
pre_rf.to_csv(r'./submission_rf.csv',index=None)

#  ----------------------------- GRB -------------------
pre_rf = pre_submission(model_grb,data_aftcat_aftif_aftmm_aftfs, data_aftcat_aftif_y, testdata_aftcat_aftmm_aftfs)
pre_rf.to_csv(r'./submission_grb.csv',index=None)