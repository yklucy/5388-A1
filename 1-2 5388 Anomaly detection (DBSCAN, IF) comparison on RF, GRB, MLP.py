import pandas as pd
from collections import Counter

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.preprocessing import MinMaxScaler

#read data from files after features preprocessing
data_aftcat = pd.read_csv(r"./1-1 5388 traindata_aftcat.csv")
data_y = pd.read_csv(r"./1-1 5388 traindata_aftcat_label.csv")


# ---------------- transform all feature with MinMaxScaler()---------------- 
mm = MinMaxScaler()
data_aftmm = pd.DataFrame(mm.fit_transform(data_aftcat))
data_aftmm.columns = mm.get_feature_names_out()


# ---------------- Split train datasets and test ---------------
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(data_aftmm, data_y, test_size=0.3, random_state=42)


# ---------------- function: dbscan_detection() ----------------
# function: dbscan_detection() - return X_new, y_new
def dbscan_detection(X, y):
    dbscan = DBSCAN(eps=0.1, min_samples=2, metric='cosine')
    # fit the data to DBSCAN
    y_pre = dbscan.fit_predict(X, y.values.ravel())
    # filter out predictions values = -1 since they are considered as anomalies
    mask = y_pre != -1
    out_dbscan = Counter(mask)[0]
    in_dbscan = Counter(mask)[1]
    print("Removed outliers:", out_dbscan)
    print("kept inliers:", in_dbscan)
    X_new, y_new = X[mask],y[mask]
    return X_new, y_new



# ---------------- function: if_detection() / Isolation Forest ----------------
# function: if_detection() - return X_new, y_new
def if_detection(X, y):
    isf = IsolationForest(contamination='auto',random_state=40)
    # fit the data to IF
    y_pre = isf.fit_predict(X, y.values.ravel())
    # filter out predictions values = -1, they are anomalies
    mask = y_pre != -1
    X_isf, y_isf = X[mask], y[mask]
    return X_isf, y_isf



# ---------------- function: dbscan_score() - test and print scores ---------------
# function:dbscan_score() - test and print scores
def dbscan_score(m, X, y, X_t, y_t, X_new, y_new):
    print(m)
    print("Score before Anomaly Detection (DBSCAN) \n")
    m.fit(X, y.values.ravel()) 
    # prediction from the model
    y_pre = m.predict(X_t)
    # score
    f1 = f1_score(y_t, y_pre)
    print("F1:", f1)
    print("\n")


    print("Score after Anomaly Detection (DBSCAN) \n")
    m.fit(X_new, y_new.values.ravel())
    y_pre_af = m.predict(X_t)
    f1_af = f1_score(y_t,y_pre_af)
    print("F1:", f1_af)
    print("\n")



# ---------------- function: if_score() - test and print ----------------
def if_score(m,X, y, X_new, y_new):
    print(m)
    # Step 1: Score before Anomaly Detection (IF) ---------------
    mae_before = mean_absolute_error(m.predict(X),y)
    print("The MAE before IF is:", mae_before)
    # fit the model on the new data set
    m.fit(X_new, y_new.values.ravel())

    # Step 2: Score after Anomaly Detection (IF) ---------------
    # compute the MAE
    mae_after = mean_absolute_error(m.predict(X),y)
    print("The MAE after IF is: ", mae_after)
    # compute the difference between the MAEs before and after IF
    diff = mae_before - mae_after
    print("the difference between the MAEs before and after IF:", diff)
    print("\n")



# ---------------- model: RF, GRB, MLP----------------
# fit the model (RF, GRB, MLP)
model_rf = RandomForestClassifier(random_state = 90, min_samples_split=2,n_estimators=61,max_depth=24, max_features=27,min_samples_leaf=1, n_jobs=-1)
model_grb = GradientBoostingClassifier(max_features=3,learning_rate=0.1,n_estimators=130,min_samples_split=100,min_samples_leaf=7,max_depth=15,random_state = 10)
model_mlp = MLPClassifier(random_state=1, max_iter=10000,hidden_layer_sizes = (160,160), activation='tanh',solver='adam')


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^ 1 Anomaly detection - DBSCAN ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# check if it is efficient for RF, MLP, GRB on the intrusion detection dataset
# author:           Kun Yan
# student number:   300259303
# data:             2021-10-03
# Python version:   3.9.7
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ---------------- 1 Anomaly Detection: DBSCAN---------------
X_train_new, y_train_new = dbscan_detection(X_train, y_train)

dbscan_score(model_rf, X_train, y_train, X_test, y_test, X_train_new, y_train_new)
dbscan_score(model_grb, X_train, y_train, X_test, y_test, X_train_new, y_train_new)
dbscan_score(model_mlp, X_train, y_train, X_test, y_test, X_train_new, y_train_new)


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^2 Anomaly detection - IF ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# check if it is efficient for RF, MLP, GRB on the intrusion detection dataset
# author:           Kun Yan
# student number:   300259303
# data:             2021-10-03
# Python version:   3.9.7
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ---------------- Anomaly Detection: IF ---------------
X_train_new_if, y_train_new_if = if_detection(data_aftmm, data_y)

if_score(model_rf, X_train, y_train, X_train_new_if, y_train_new_if)
if_score(model_grb, X_train, y_train, X_train_new_if, y_train_new_if)
if_score(model_mlp, X_train, y_train, X_train_new_if, y_train_new_if)



# ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Conclusion : DBSCAN and IF on RF, GRB and MLP^^^^^^^^^^^^^^^^^
#DBSCAN (no)
#   RF(-), GRB(+), MLP(+)
#    F1 (RF) before - highest
#
#IF (ok, RF)
#    RF(+, 0.0003105397), GRB(+, 0.00018632383), MLP(+, 0.00111794298)
#    MAE RF(after) =0.0
#    MAE RF(before) is the lowest
# 
#do IF for the dataset, then use RF
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#notes: redo Anomaly Detection: IF based on train dataset after categorical feature preprocessing
X_if, y_if = if_detection(data_aftcat, data_y)

#- - - - output - - - - - - -
X_if.to_csv("./1-4 5388 traindata_aftcat_aftif.csv",index=0)
y_if.to_csv("./1-4 5388 traindata_aftcat_aftif_label.csv",index=0)



