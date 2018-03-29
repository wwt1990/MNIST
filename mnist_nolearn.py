import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
train = pd.read_csv('/Users/Tiantian/Documents/CNN/minst/train.csv')
test = pd.read_csv('/Users/Tiantian/Documents/CNN/minst/test.csv')

X_train = train.ix[:, 1:].values.astype(np.float32)
y_train = train.ix[:, 0]
X_test = test.values.astype(np.float32)

# For convolutional layers, the default shape of data is bc01,
# i.e. batch size * color channels * image dimension 1 * image dimension 2.
X_train = X_train.reshape(-1, 1, 28, 28)

# try plotting some digits
fig, axes = plt.subplots(4,4,figsize=(6,6))
axes = axes.ravel()
for i in range(16):
    axes[i].imshow(X_train[100+i].reshape(28,28), cmap='gray', interpolation='none')
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title("Label: {}".format(y_train[100+i]))
    #axes[i].axis('off')
plt.tight_layout()
#plt.show()


# layers
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer, DenseLayer
from lasagne.layers import get_all_params
from lasagne.updates import adam
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet, TrainSplit, objective

layers0 = [
    (InputLayer, {'shape': (None, 1, 28, 28)}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
]

# net
net0 = NeuralNet(
    layers = layers0,
    max_epochs = 10,
    update = adam, # For 'adam', a small learning rate is best
    update_learning_rate = 0.0002,
    objective_l2 = 0.0025, # L2 regularization
    train_split = TrainSplit(eval_size = 0.25),
    verbose = 1
)
net0.fit(X_train, y_train)


# visualization
from nolearn.lasagne.visualize import draw_to_notebook, plot_loss
from nolearn.lasagne.visualize import plot_conv_weights, plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion, plot_saliency

draw_to_notebook(net0)
plot_loss(net0)
#plot helps determine if we are overfitting:
#If the train loss is much lower than the validation loss,
#we should probably do something to regularize the net.

# visualize layer weights
plot_conv_weights(net0.layers_[1], figsize = (4,4))
#If the weights just look like noise, we might have to do something
#(e.g. use more filters so that each can specialize better).

# visualize layers' activities
x = X_train[0:1] # an image in the bc01 format (so use X[0:1] instead of just X[0]).
plot_conv_activity(net0.layers_[1], x)

plot_occlusion(net0, X_train[:5], y_train[:5])
plot_saliency(net0, X_train[:5])


from nolearn.lasagne import PrintLayerInfo

layers1 = [
    (InputLayer, {'shape': (None, 1, 28, 28)}),

    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3)}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3)}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3)}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 96, 'filter_size': (3, 3)}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
]

net1 = NeuralNet(
    layers=layers1,
    update_learning_rate=0.01,
    verbose=2,
)

# To see information about the capacity and coverage of each layer,
# we need to set the verbosity of the net to a value of 2 and
# then initialize the net. We next pass the initialized net to PrintLayerInfo
# to see some useful information. By the way, we could also just call the
# fit method of the net to get the same outcome, but since we don't want
# to fit just now, we proceed as shown below.

net1.initialize()
layer_info = PrintLayerInfo()
layer_info(net1)
# This net is fine. The capacity never falls below 1/6, which would be 16.7%,
# and the coverage of the image never exceeds 100%. However,
# with only 4 convolutional layers, this net is not very deep and will
# properly not achieve the best possible results.

# if we use max pooling too often, the coverage will quickly
# exceed 100% and we cannot go sufficiently deep.

# Too little maxpooling
layers2 = [
    (InputLayer, {'shape': (None, 1, 28, 28)}),

    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3)}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3)}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3)}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3)}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3)}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3)}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3)}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3)}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3)}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3)}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
]

net2 = NeuralNet(
    layers=layers2,
    update_learning_rate=0.01,
    verbose=2,
)

net2.initialize()
layer_info(net2)

# Too much maxpooling
layers3 = [
    (InputLayer, {'shape': (None, 1, 28, 28)}),

    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
]

net3 = NeuralNet(
    layers=layers3,
    update_learning_rate=0.01,
    verbose=2,
)

net3.initialize()
layer_info(net3)
# This net uses too much maxpooling for too small an image.
# The later layers, colored in cyan, would cover more than 100% of the image.
# So this network is clearly also suboptimal.

layers4 = [
    (InputLayer, {'shape': (None, 1, 28, 28)}),

    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
]
net4 = NeuralNet(
    layers=layers4,
    update_learning_rate=0.01,
    verbose=2,
)

net4.initialize()
layer_info(net4)


# get more information by increasing the verbosity level beyond 2.
net4.verbose = 3
layer_info(net4)

# http://nbviewer.jupyter.org/github/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb
