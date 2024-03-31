import load_data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# SKlearn imports
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn import feature_selection

## Preparing data
# Load and extract data
data = load_data.load_data()
X,y  = load_data.extract_data(data)

# Scaling data 
scaler  = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the dataset in train and test part
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)


## Variance Thresholding
# Determine variances per feature
variances = X_train.var()
variances_df = pd.DataFrame({'Feature Count': range(1, len(variances) + 1), 'Variance': variances})

# Set threshold
threshold    = 0.015
low_var_cols = variances[variances < threshold].index

X_train_selected = X_train.drop(columns=low_var_cols)

print('Amount of features after thresholding:',X_train_selected.shape[1])

# Plot histogram of variances
# sns.scatterplot(data=variances_df, x=range(0,X_train.shape[1]), y=variances, color='skyblue')
# plt.xlabel('Features')
# plt.ylabel('Variance')
# plt.axhline(y=threshold, color='skyblue', linestyle='--')
# plt.show()

## Perform RFE
# Instantiate an estimator
svc = svm.SVC(kernel="linear")

# Instantiate RFECV
rfecv = feature_selection.RFECV(
    estimator=svc, step=1,
    cv=model_selection.StratifiedKFold(4),
    scoring='roc_auc')
rfecv.fit(X_train_selected, y_train)

# Select Features
selected_features = rfecv.feature_names_in_
print('Amount of features after RFE',selected_features.shape[0])

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
plt.show()
