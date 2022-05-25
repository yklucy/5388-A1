import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
import scipy.stats as stats

#read data from dataset (after categorical feature preprocessing and anomaly detection(IF))
data_aftcat_aftif = pd.read_csv(r"./1-4 5388 traindata_aftcat_aftif.csv")
data_aftcat_aftif_y = pd.read_csv(r"./1-4 5388 traindata_aftcat_aftif_label.csv")

testdata_aftcat = pd.read_csv(r"./1-1 5388 testdata_aftcat.csv")

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


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^ train models: RF, GRB, MLP, DT ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# author:           Kun Yan
# student number:   300259303
# data:             2021-10-03
# Python version:   3.9.7
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

mm = MinMaxScaler()
selector = VarianceThreshold(np.median(data_aftcat_aftif.var().values))
model_rf = RandomForestClassifier(random_state = 90, min_samples_split=2,n_estimators=61,max_depth=24, max_features=27,min_samples_leaf=1, n_jobs=-1)
model_grb = GradientBoostingClassifier(max_features=3,learning_rate=0.1,n_estimators=130,min_samples_split=100,min_samples_leaf=7,max_depth=15,random_state = 10)
model_mlp = MLPClassifier(random_state=1, max_iter=10000,hidden_layer_sizes = (160,160), activation='tanh',solver='adam')
model_dt = DecisionTreeClassifier(random_state=10, criterion='gini', max_depth=18, max_features=25, min_impurity_decrease=0.0, min_samples_leaf=1, splitter='best')

# -------------- train predictive model with pipeline: MinMaxScaler(), VarianceThreshold(), cross_validate()--------
def train_model(m, se, model, X, y):
    print(model, "\n")
    pipe_steps = [('mm',m),('selector',se),('model',model)]
    id_pipeline = Pipeline(steps=pipe_steps)

    # evaluate the pipeline using the crossvalidation technique defined in cv

    # ------------------- score ------------------------------------
    scoring = {'f1', 'precision', 'accuracy',
            'recall', 'roc_auc'}

    score_sf = cross_validate(id_pipeline, X, y.values.ravel(),cv=10,scoring=scoring, n_jobs=-1)
    score_sf_re = score_mean_std(pd.DataFrame(score_sf))

    print("score_sf_re: \n", pd.DataFrame(score_sf_re))
    return score_sf

# -------------------- RF -----------------------
score_sf_rf = train_model(mm, selector, model_rf, data_aftcat_aftif, data_aftcat_aftif_y)
RF = pd.DataFrame(score_sf_rf['test_f1'],columns = ["RF"])
print(RF)

# -------------------- GRB -----------------------
score_sf_grb = train_model(mm, selector, model_grb, data_aftcat_aftif, data_aftcat_aftif_y)
GRB = pd.DataFrame(score_sf_grb['test_f1'],columns = ["GRB"])
print(GRB)

# -------------------- MLP -----------------------
score_sf_mlp = train_model(mm, selector, model_mlp, data_aftcat_aftif, data_aftcat_aftif_y)
MLP = pd.DataFrame(score_sf_mlp['test_f1'],columns = ["MLP"])
print(MLP)


# -------------------- DT -----------------------
score_sf_dt = train_model(mm, selector, model_dt, data_aftcat_aftif, data_aftcat_aftif_y)
DT = pd.DataFrame(score_sf_dt['test_f1'],columns = ["DT"])
print(DT)




# ^^^^ compare the four train models: RF, GRB, MLP, DT using Paired T-test^^^^^^^^^^^^^^^
# author:           Kun Yan
# student number:   300259303
# data:             2021-10-03
# Python version:   3.9.7
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# get the t-statistics and p-value of two algorithms based on the ten-fold cross validation
# a, b are original cross validation score array
def sig_diff(p_a_b):
    if p_a_b < 0.05:
        print("There is a significant difference between the two algorithms.\n")
    else:
        print("No significant difference!\n")

stat_RF_GRB, p_RF_GRB = stats.ttest_rel(np.array(RF),np.array(GRB))
print("The p-value of RF and GRB:", p_RF_GRB)
sig_diff(p_RF_GRB)

stat_RF_MLP, p_RF_MLP = stats.ttest_rel(np.array(RF),np.array(MLP))
print("The p-value of RF and MLP:", p_RF_MLP)
sig_diff(p_RF_MLP)

stat_MLP_GRB, p_MLP_GRB = stats.ttest_rel(np.array(MLP),np.array(GRB))
print("The p-value of MLP and GRB:", p_MLP_GRB)
sig_diff(p_MLP_GRB)

stat_RF_DT, p_RF_DT = stats.ttest_rel(np.array(RF),np.array(DT))
print("The p-value of RF and DT:", p_RF_DT)
sig_diff(p_RF_DT)

stat_GRB_DT, p_GRB_DT = stats.ttest_rel(np.array(GRB),np.array(DT))
print("The p-value of GRB and DT:", p_GRB_DT)
sig_diff(p_GRB_DT)

stat_MLP_DT, p_MLP_DT = stats.ttest_rel(np.array(MLP),np.array(DT))
print("The p-value of MLP and DT:", p_MLP_DT)
sig_diff(p_MLP_DT)