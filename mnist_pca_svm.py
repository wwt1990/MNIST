import os
from time import time
import pandas as pd
from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


FTRAIN = '/Users/tian/Documents/CNN/mnist/train.csv'
FTEST = '/Users/tian/Documents/CNN/mnist/test.csv'

train_df = read_csv(os.path.expanduser(FTRAIN))
# 42000, 785
# int64(785)
test_df = read_csv(os.path.expanduser(FTEST))
# 28000, 784
# int64(784)

X_train = train_df[train_df.columns[1:]].values
X_train = X_train.astype(np.float32) / 255

y_train = train_df[train_df.columns[0]]


X_test = test_df.values.astype(np.float32) / 255


def plot_samples(data = X_train, label = y_train, shapes = (28,28), limit = 41000, seeds = None):
    np.random.seed(seeds)
    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    axes = axes.ravel()
    m = np.random.randint(limit)
    for i in range(16):
        img = data[m+i].reshape(shapes)
        axes[i].imshow(img, cmap = 'gray')
        axes[i].set_title('number is {}'.format(label[m+i]))
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
    plt.tight_layout()
    plt.show()



# calculate eigenvalues ad eigenvectors
cov_mat = np.cov(X_train.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

# cum_var_exp[199]
fig = plt.figure(figsize = (6,6))
plt.plot(cum_var_exp, linestyle = 'solid', color = 'blue', linewidth = 3)
plt.vlines(200, -100, 200, color = 'red', linestyle = 'dashed')
plt.hlines(cum_var_exp[199], -100, 800, color = 'red', linestyle = 'dashed')
plt.xlim(0, 800)
plt.ylim(0, 105)
plt.show()

# Compute a PCA on dataset (treated as unlabeled dataset):
# unsupervised feature extraction / dimensionality reduction
X_train_set, X_valid_set, y_train_set, y_valid_set = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)
n_components = 225
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train_set)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, 28, 28))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train_set)
X_valid_pca = pca.transform(X_valid_set)
print("done in %0.3fs" % (time() - t0))

# Train a SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca[:1000], y_train_set[:1000])
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# Quantitative evaluation of the model quality on the valid set
print("Predicting people's names on the valid set")
t0 = time()
y_pred = clf.predict(X_valid_pca)
print("done in %0.3fs" % (time() - t0))

#print(classification_report(y_valid_set, y_pred, target_names=target_names))
print(confusion_matrix(y_valid_set, y_pred, labels=range(10)))
