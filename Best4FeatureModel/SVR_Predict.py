#
# Copyright <year> The Board of Trustees of the University of Illinois. All Rights Reserved.
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
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

def plot_figs(perm_imp,y, y_pred,C_reg,R2s,data_labels):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(data_labels,np.average(perm_imp,axis = 0))
    ax.set_ylabel('Permuation Importance')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.annotate("$R^2$ = {:.2f}, $C_r$ = {:.0f}".format(R2s,C_reg[0]), xy=(0,0.1),size=10)
    fig.tight_layout()
    fig.savefig("PermImp.png")
    plt.close()
    
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Arial"
    fig, ax = plt.subplots(figsize=(3,3))
    ax.scatter(y, y_pred, color='mediumvioletred')
    ax.plot([0,100],[0,100],color='gray')
    ax.annotate("$R^2$ = {:.2f}".format(R2s), xy=(10,80),size=15)
    ax.annotate("$C_r$ = {:.0f}".format(C_reg[0]), xy=(10,60),size=15)
    ax.set_ylabel("Predicted T80",fontsize='x-large');
    ax.set_xlabel("Actual T80",fontsize='x-large');
    fig.tight_layout()
    fig.savefig("Performance_SVR_RBF.png")
    return


df = pd.read_csv("OligomerFeatures.csv")
molecs = 40
data_labels = np.array(df.columns.values[7:25])
all_features = np.array(df.iloc[:,7:25])
T80 = np.array(df.iloc[:molecs,5])

#Scale X to unit variance and zero mean
st = StandardScaler()
X = st.fit_transform(all_features)
feats = X.shape[1]
y = T80

C_reg = np.array([100]) #Selected when running all 4-feature models
C_regs = C_reg.shape[0]

perm_imp_final = np.zeros(feats)
R2_final = np.zeros(feats)

#Divide into train and test set
coeffs = np.zeros((C_regs,molecs,X.shape[1]),dtype = float)
perm_imp = np.zeros((C_regs,molecs,X.shape[1]),dtype = float)
X_train = X[:molecs]
y_train = y

svr = SVR(kernel="rbf", C=C_reg[0])
svr.fit(X_train,y_train)
y_pred = svr.predict(X)
result = permutation_importance(svr,X_train,y_train)
perm_imp = result.importances_mean
MSEs = mean_squared_error(T80,y_pred[:molecs])
R2s  = r2_score(T80,y_pred[:molecs])

plot_figs(perm_imp,T80,y_pred[:molecs],C_reg,R2s,data_labels)


fileo = open("PredictedT80.csv",'w')
out = ""
ml = np.array(df.columns.values[0:5])
for i in range(ml.shape[0]):
    out += '{},'.format(ml[i])
out += 'Predicted_T80\n'
    
md = np.array(df.iloc[:,0:5])
for i in range(md.shape[0]):
    for j in range(5):
        out += '{},'.format(md[i,j])
    out += '{}\n'.format(y_pred[i])
fileo.write(out)
fileo.close()
