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
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

def lin_no_b(x,m):
    return (x*m)

df = pd.read_csv("OligomerFeatures_S20O20_only.csv")
n = 44 #number of molecules with SO data, must be listed first
T_labels = df.columns.values[7:27]
Tstates = np.array(df.iloc[:n,7:27]) #Singlet States
T_osc =  np.array(df.iloc[:n,27:47]) #Singlet Oscilator Strengths
SO_true = np.array(df.iloc[:n,4])

df_trans = pd.read_csv("CB_solar_0.1eV.csv")
trans = np.array(df_trans.iloc[:,0:30]) #(1,30) array, 1.5-4.4 eV

x = np.arange(1.5,4.5,0.1) #eV (window set by CB trans spectrum
shift = np.arange(-2,1,0.1, dtype = float) #visually estimated to be ~ -0.7 eV
SO_raw = np.zeros((shift.shape[0],n),dtype = float) #estimated by trans*DOS
polyfit = np.zeros((shift.shape[0],2),dtype = float) #slope and intercepts
SO_pred = np.zeros((shift.shape[0],n),dtype = float) #predicted SO with extra linear best fit
R2 = np.zeros((shift.shape[0]),dtype = float)
total_abs = np.zeros((shift.shape[0],n,x.shape[0]),dtype = float)

width = 0.2 #eV
pref = 1/(width*(2*math.pi)**(0.5))
expf = -1/(2*width**2)
for s in range(shift.shape[0]):
    DOS = np.zeros((n,x.shape[0]),dtype = float)
    for i in range(n):
        for j in range(Tstates.shape[1]):
            DOS[i] += T_osc[i,j]*pref*np.exp(expf*(x-(Tstates[i,j]+shift[s]))**2)
            #DOS accounts for oscilator strengths and shift
    total_abs[s] = DOS*trans
    SO_raw[s] = np.sum(total_abs[s], axis=1) #integrated absorbance
    polyfit[s,0], cov = curve_fit(f=lin_no_b,xdata=SO_raw[s],ydata=SO_true,p0=[1],method='trf')
    SO_pred[s] = SO_raw[s]*polyfit[s,0]
    R2[s] = r2_score(SO_true,SO_pred[s])

best = 0
for i in range(R2.shape[0]):
    if R2[i] > R2[best]:
        best = i
fig, ax = plt.subplots()
ax.scatter(SO_raw[best], SO_true)
ax.plot([0,np.amax(SO_raw[best])],[polyfit[best,1],polyfit[best,1]+np.amax(SO_raw[best])*polyfit[best,0]],
        color='gray')
ax.annotate("R2 = {:.3f} for shift = {:.2f}".format(R2[best],shift[best]), xy=(0,12),size=15)
ax.annotate("m = {:.4f}, b = {:.4f} ".format(polyfit[best,0],polyfit[best,1]), xy=(0,10),size=15)
ax.set_ylabel("True SO");
ax.set_xlabel("Weighted DOS");
fig.savefig("Predictions.png")

fig, ax = plt.subplots()
ax.bar(shift,R2,width=0.08)
ax.annotate("R2 = {:.3f} for shift = {:.2f}".format(R2[best],shift[best]), xy=(np.amin(shift),0.9),size=15)
ax.set_ylabel("R2 Values");
ax.set_xlabel("Shift (eV)\nNegative = Red Shift");
ax.set_ylim(0,1)
fig.tight_layout()
fig.savefig("R2_vs_Shift.png")

##### Predict SO for all molecules #####

df = pd.read_csv("OligomerFeatures_S20O20_only.csv")
T_labels = df.columns.values[7:27]
Tstates = np.array(df.iloc[:,7:27])
n = Tstates.shape[0] #number of molecules
T_osc =  np.array(df.iloc[:,27:47])
DBA = np.array(df.iloc[:,0])[:,np.newaxis]

df_trans = pd.read_csv("CB_solar_0.1eV.csv")
trans = np.array(df_trans.iloc[:,0:30]) #(1,30) array, 1.5-4.4 eV

SO_raw = np.zeros((n),dtype = float) #estimated by trans*DOS
SO_pred = np.zeros((n),dtype = float) #predicted SO with extra linear bestfit
total_abs = np.zeros((n,x.shape[0]),dtype = float)

for s in range(1):
    DOS = np.zeros((n,x.shape[0]),dtype = float)
    for i in range(Tstates.shape[0]):
        for j in range(Tstates.shape[1]):
            DOS[i] += T_osc[i,j]*pref*np.exp(expf*(x-(Tstates[i,j]+shift[best]))**2)
    total_abs = DOS*trans
    SO_raw = np.sum(total_abs, axis=1) #integrated absorbance
    SO_pred = SO_raw*polyfit[best,0]+polyfit[best,1]

# Write SO Values #
SO_str = np.array(SO_pred, dtype = str)
f = open("All2200_SO_values.csv", 'w')
for i in range(DBA.shape[0]):
    f.write(DBA[i,0]+','+SO_str[i])
    f.write('\n')
f.close()
