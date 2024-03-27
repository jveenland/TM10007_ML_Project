import load_data
import numpy as np

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

# Load and extract data
data = load_data.load_data()
X,y  = load_data.extract_data(data)

# Perform PCA
p = PCA(n_components=X.shape[1]) # Hier gebruik ik nu het aantal kollomen als n_components, maar vgm klopt dat dus niet
p = p.fit(X)
X = p.transform(X)

# Split the dataset in train and test part
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, stratify=y)

# ## Classification
# lda = LinearDiscriminantAnalysis()
# lda = lda.fit(X_train, y_train)
# y_pred = lda.predict(X_train)

# print("Number of mislabeled points out of a total %d points : %d" % (X_train.shape[0], (y_train != y_pred).sum()))
