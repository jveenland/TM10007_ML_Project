import load_data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import svm
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




