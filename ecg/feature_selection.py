import load_data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Classifiers
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

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
sns.scatterplot(data=variances_df, x=range(0,X_train.shape[1]), y=variances, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Variance')
plt.axhline(y=threshold, color='b', linestyle='--')
plt.show()


## Perform PCA
p = PCA() 
p = p.fit(X_train_selected)
X_train_selected = p.transform(X_train_selected)

