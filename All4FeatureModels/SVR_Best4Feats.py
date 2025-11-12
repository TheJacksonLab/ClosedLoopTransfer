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
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

df = pd.read_csv("OligomerFeatures.csv") #molecules with data must be at the top
molecs = 40 #40 T80 measurements
data_labels = np.array(df.columns.values[7:250])
all_features = np.array(df.iloc[:molecs,7:250])
T80 = np.array(df.iloc[:molecs,5])

#Scale X to unit variance and zero mean
st = StandardScaler()
X = st.fit_transform(all_features)
feats = X.shape[1]
y = T80

C_reg = np.array([10,100,1000]) #reg. strength = 1/C_reg
C_regs = C_reg.shape[0]
MSEs = np.zeros(C_regs)
R2s = np.zeros(C_regs)
top_y_pred = np.zeros((feats**4,molecs))
top_C_reg = np.zeros(feats**4)
top_R2s = np.zeros(feats**4)
out = "FeatID,FeatA,FeatB,FeatC,FeatD,LOOV_R2,Creg\n"

#Select all 4-feature combinations
for k in range(feats):
    for l in range(k+1,feats):
        for m in range(l+1,feats):
            for n in range(m+1,feats):
                y_pred = np.zeros((C_regs,molecs))
                del_feats = np.arange(0,feats,dtype=int)
                del_feats = np.delete(del_feats,k,0)
                del_feats = np.delete(del_feats,l-1,0)
                del_feats = np.delete(del_feats,m-2,0)
                del_feats = np.delete(del_feats,n-3,0)
                X_temp = np.delete(X,del_feats,1) #4 unique features
                
                #LOOV over selected regularization strengths
                for j in range(C_regs):
                    for i in range(molecs):
                        X_train = np.delete(X_temp,i,0)
                        X_test = [X_temp[i]]
                        y_train = np.delete(y,i,0)
                        y_test = y[i]
                        
                        svr = SVR(kernel="rbf", C=C_reg[j])
                        svr.fit(X_train,y_train)
                        y_pred[j,i] = svr.predict(X_test)
                    MSEs[j] = mean_squared_error(y,y_pred[j])
                    R2s[j]  = r2_score(y,y_pred[j])
        
                #Select and store best regularization strength results
                bestC = 0
                for i in range(len(MSEs)):
                    if MSEs[i] < MSEs[bestC]:
                        bestC = i
                featID = k*feats**3+l*feats**2+m*feats**1+n
                top_y_pred[featID] = y_pred[bestC]
                top_C_reg[featID] = C_reg[bestC]
                top_R2s[featID] = R2s[bestC]
                print("{} R2={:.2f}, Creg={:.0f}".format(featID,R2s[bestC],C_reg[bestC]))
                out += "{},{},{},{},{},{:.2f},{:.0f}\n".format(featID,data_labels[k],data_labels[l],data_labels[m],data_labels[n],R2s[bestC],C_reg[bestC])

#find the best features
bestF = np.argmax(top_R2s)
bestFeatA = int(np.floor(bestF/feats**3))
bestF -= bestFeatA*feats**3
bestFeatB = int(np.floor(bestF/feats**2))
bestF -= bestFeatB*feats**2
bestFeatC = int(np.floor(bestF/feats**1))
bestF -= bestFeatC*feats**1
bestFeatD = bestF
bestF = np.argmax(top_R2s)

print("Best Model is {}, R2 = {:.2f}".format(bestF,np.amax(top_R2s)))

fileo = open("FourFeatureModels.csv",'w')
fileo.write(out)
fileo.close()

## Plot Best model performance ##
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Arial"
fig, ax = plt.subplots(figsize=(3,3))
ax.scatter(y, top_y_pred[bestF], color='mediumvioletred')
ax.plot([0,100],[0,100],color='gray')
ax.annotate("$R^2$ = {:.2f}".format(top_R2s[bestF]), xy=(10,80),size=15)
ax.annotate("$C_r$ = {:.0f}".format(top_C_reg[bestF]), xy=(10,60),size=15)
ax.set_ylabel("Predicted T80",fontsize='x-large');
ax.set_xlabel("Actual T80",fontsize='x-large');
fig.tight_layout()
fig.savefig("Top4FeatureModel_{}.png".format(bestF))
plt.close()
