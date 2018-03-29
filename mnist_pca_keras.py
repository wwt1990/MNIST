import os
from time import time
import pandas as pd
from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from sklearn.decomposition import PCA

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Activation, Dropout, Flatten
from keras.optimizers import RMSprop



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
y_train = y_train.astype("category")

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
# Create a list of (eigenvalue, eigenvector) tuples
#eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort the eigenvalue, eigenvector pair from high to low
#eig_pairs.sort(key = lambda x: x[0], reverse= True)
# Calculation of Explained Variance from the eigenvalues
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

# reduce X_train to fisrt 200 features
n_components = 225
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

X_train_pca = pca.transform(X_train)

plt.hist(pca.explained_variance_ratio_, bins=n_components, log=True)
pca.explained_variance_ratio_.sum()

plot_samples(data=X_train_pca, shapes = (15,15), seeds = 10)
plot_samples(data=X_train, seeds = 10)


# add one hidden layer
data = X_train_pca
# (42000, 784)
label = np_utils.to_categorical(y_train, 10)
# (42000, 10)
model = Sequential()
model.add(Dense(64, activation = 'relu', input_shape = (784, )))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = model.fit(data, label, validation_split = 0.2, shuffle=True, verbose=1, nb_epoch = 20, batch_size = 100)

# see how the model performs on the test data
testdata = X_test
y_pred = model.predict_classes(testdata)
testlabel = np_utils.to_categorical(y_pred, 10)

for i in range(10):
    plot_samples(testdata, testlabel, 27000)

plt.plot(hist.history['loss'], linewidth=3, label='train')
plt.plot(hist.history['val_loss'], linewidth=3, label='valid')
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# add cnn layers
data = X_train.reshape(-1,1,28,28)
# (42000, 1, 28, 28)
data, label = shuffle(data, label, random_state=42)
#plot_samples(data = data, label = label)
from keras import backend as K
K.set_image_dim_ordering('th')

model1 = Sequential()
model1.add(Convolution2D(64, 5, 5, activation='relu', input_shape = (1, 28, 28)))
model1.add(MaxPooling2D(pool_size = (2, 2)))
model1.add(Dropout(0.1))
#print(model1.output_shape)
# model1.add(Convolution2D(8, 3, 3, activation='relu'))
# model1.add(MaxPooling2D(pool_size = (2, 2)))
# model1.add(Dropout(0.2))

# model1.add(Convolution2D(16, 3, 3, activation='relu'))
# model1.add(MaxPooling2D(pool_size = (2, 2)))
# model1.add(Dropout(0.3))

model1.add(Flatten())
model1.add(Dense(64, activation = 'relu'))
# model1.add(Dropout(0.5))
model1.add(Dense(10, activation = 'softmax'))
#model1.summary()
# tried 'sgd', but low accuracy
# so tried 'rmsprop'
model1.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist1 = model1.fit(data, label, validation_split = 0.2, shuffle=True, verbose=1, nb_epoch = 10, batch_size = 100)

plt.plot(hist1.history['loss'], linewidth=3, label='train')
plt.plot(hist1.history['val_loss'], linewidth=3, label='valid')
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

y_pred = model1.predict_classes(testdata)
# 49s
testlabel = np_utils.to_categorical(y_pred, 10)

for i in range(10):
    plot_samples(testdata, testlabel, 27000)

print('The accuracy is {}'.format(hist1.history['acc'][-1]))
# open('model1_architecture.json', 'w').write(model1.to_json())
# model1.save_weights('model1_weights.h5')
