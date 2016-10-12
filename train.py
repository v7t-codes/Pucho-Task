import ft_ex
import ebow
import numpy as np
from sklearn.neural_network import MLPClassfier

labels=np.load('score_train.npy')
feats= np.load('train_feats.npy')

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, \
            hidden_layer_sizes=(8, 4), random_state=1)
NN.fit(X, labels)
