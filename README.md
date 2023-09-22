# ClosedLoopTransfer

1. System Requirements:
Python Packages:
- matplotlib 3.5.3
- pandas 1.4.3
- numpy 1.21.5
- scipy 1.9.0
- sklearn 1.1.2
All python scripts were run on macOS Ventura 13.6

2. Installation Guide:
Standard installation of python packages (5 minutes)
Bayesian Optimizer (Gryffin) was installed and used as described in https://doi.org/10.1063/5.0048164

3. Demo and 4. Instructions for Use
All scripts are saved in separate folders with their appropriate data set (saved as .csv) to reproduce the analysis in the paper.

## All4FeatureModels ##
Instructions:
Select which features you want to run all 4-feature combinations over by removing extra features from OligomerFeatures.csv. Run SVR_Best4Feats.py
Expected Output:
A csv file "FourfeatureModels.csv" that contains all the model IDs, 4 features, LOOV_R2, and regularization strength. All 4 feature models with LOOV R2 > 0.70 derived from pairwise combinations of feature sets (see text) is included.
A png file graphing the predicted T80 values vs Actual T80 values of the best 4-feature model with its R2 and C_reg.
Expected run time:
< 1 minute for a small set of 10 features.
Reproduction:
Run SVR_Best4Feats.py on all pair-wise combinations of feature sets to identify the most common features (e.g. Figure S9) and the best set of features

## Best4FeatureModel ##
Instructions:
The best 4-feature model's features are saved in OligomerFeatures.csv. To predict T80 values for all molecules, run SVR_Predict.py
Expected Output:
All Predicted T80s will be listed in PredictedT80.csv. The Predicted T80 will be plotted against the Actual T80 values (not LOOV) in Performance_SVR_RBF.png along with the R2 and Cr. The permutation importance of each feature will be plotted in PermImp.png
Expected run time:
< 1 minute

## DownSelect ##
Instructions:
To identify a reduced set of predictive features from a larger dataset, identify the columns of the features in your larger dataset from OligomerFeatures_PreValidations_TSO10.csv (the script can handle two contiguous blocks of features). 'Run SVR_PermuImp.py A B C D', where A and C are numbers that correspond to the start of the two contiguous blocks, and B and D are numbers that correspond to the end of the two contiguous blocks. The script requires A B C D to all be present (you can use 0 0 for C D if you only want one contiguous block). Running run Downselect.bash will run all single & combinations of feature sets in separate folders ('BTOS' = Basic + TOS10, 'BT' = Basic + TDOS etc.)
Expected Output:
A series of PermImpX.png files showing the permutation importance for each model (X) as low-importance features are eliminated.
A series of Performance_SVR_RBFX.png files showing LOOV predictions of each model.A file, FeatureElimination.csv, recording the LOOV R2, Cr, and least important feature (deleted) for each model.
Reproduction: run Downselect.bash

