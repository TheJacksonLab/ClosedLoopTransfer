#
# Copyright 2023 The Board of Trustees of the University of Illinois. All Rights Reserved.
# Licensed under the terms of the MIT license (the "License")
# The License is included in the distribution as License.txt file.
# You may not use this file except in compliance with the License. 
# Software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and limitations under the License.
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# run command (python SVR_PermuImp.py) must be followed by 4 numbers designating which features are to be selected for down selection. "0 21" = Basic, "21 51" = TSO10, "51 84" = TDOS, "84 114" = SDOS. See runDownselect.bash for example. 
# Results contain some stochasticity due to permutation importance


def plot_figs(perm_imp,y, y_pred,best,C_reg,R2s,k,data_labels):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(data_labels,np.average(perm_imp[best],axis = 0))
    ax.set_ylabel('Permuation Importance')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.annotate("$R^2$ = {:.2f}, $C_r$ = {:.0f}".format(R2s[best],C_reg[best]), xy=(0,0.1),size=10)
    fig.tight_layout()
    fig.savefig("PermImp{}.png".format(k))
    plt.close()
    
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Arial"
    fig, ax = plt.subplots(figsize=(3,3))
    ax.scatter(y, y_pred[best], color='mediumvioletred')
    ax.plot([0,100],[0,100],color='gray')
    ax.annotate("$R^2$ = {:.2f}".format(R2s[best]), xy=(10,80),size=15)
    ax.annotate("$C_r$ = {:.0f}".format(C_reg[best]), xy=(10,60),size=15)
    ax.set_ylabel("Predicted T80",fontsize='x-large');
    ax.set_xlabel("Actual T80",fontsize='x-large');
    fig.tight_layout()
    fig.savefig("Performance_SVR_RBF{}.png".format(k))
    return


if len(sys.argv) != 5:
    print('Missing Arguments!')
    exit()
argvs = np.array(sys.argv[1:5],dtype=int)

df = pd.read_csv("OligomerFeatures_PreValidation_TSO10.csv")
molecs = 26
all_labels = np.array(df.columns.values[7:250])
data_labels = np.r_[all_labels[argvs[0]:argvs[1]],all_labels[argvs[2]:argvs[3]]]
alll_features = np.array(df.iloc[:molecs,7:250])
if argvs[3] == 0:
    all_features = alll_features[:,argvs[0]:argvs[1]]
else:
    a1 = alll_features[:,argvs[0]:argvs[1]]
    a2 = alll_features[:,argvs[2]:argvs[3]]
    all_features = np.concatenate((a1,a2),axis=1)
T80 = np.array(df.iloc[:molecs,5]) #SO=4, T80=5, SO*T80=6

#Scale X to unit variance and zero mean
st = StandardScaler()
X = st.fit_transform(all_features)
feats = X.shape[1]
y = T80

C_reg = np.array([30,100,300,1000,3000]) #reg. strength = 1/C_reg
C_regs = C_reg.shape[0]
y_pred = np.zeros((C_regs,molecs))
MSEs = np.zeros(C_regs)
R2s = np.zeros(C_regs)

perm_imp_final = np.zeros((feats,feats))
R2_final = np.zeros(feats)
out = "R2,Creg,Feature Delted\n"

#Divide into train and test set
for k in range((np.shape(X)[1]-1)):
    coeffs = np.zeros((C_regs,molecs,X.shape[1]),dtype = float)
    perm_imp = np.zeros((C_regs,molecs,X.shape[1]),dtype = float)
    for j in range(C_regs):
        for i in range(molecs):
            X_train = np.delete(X,i,0)
            X_test = [X[i]]
            y_train = np.delete(y,i,0)
            y_test = y[i]

            svr = SVR(kernel="rbf", C=C_reg[j])
            svr.fit(X_train,y_train)
            y_pred[j,i] = svr.predict(X_test)
            result = permutation_importance(svr,X,y)
            perm_imp[j,i] = result.importances_mean
        MSEs[j] = mean_squared_error(y,y_pred[j])
        R2s[j]  = r2_score(y,y_pred[j])
        
    #Plot the results
    best = 0
    for i in range(len(MSEs)):
        if MSEs[i] < MSEs[best]:
            best = i

    perm_imp_final[k,:(feats-k)] = np.average(perm_imp[best],axis = 0)
    R2_final[k] = R2s[best]
    plot_figs(perm_imp,y,y_pred,best,C_reg,R2s,k,data_labels)

    low = np.argmin(perm_imp_final[k,:(feats-k)])
    print("R2={:.2f},Creg={:.0f},{} Deleted".format(R2s[best],C_reg[best],data_labels[low]))
    out += "{:.2f},{:.0f},{}\n".format(R2s[best],C_reg[best],data_labels[low])
    X = np.delete(X,low,1)
    data_labels = np.delete(data_labels,low,0)

print("Best Model is {}, R2 = {:.2f}".format(np.argmax(R2_final),np.amax(R2_final)))

fileo = open("FeatureElimination.csv",'w')
fileo.write(out)
fileo.close()
