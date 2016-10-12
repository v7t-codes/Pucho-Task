import ft_ex
import ebow
import numpy as np
from sklearn.neural_network import MLPClassfier


labels=np.load('score_train.npy')
feats= np.load('train_features.npy')
feats= np.delete(feats,0)  #removing the header vector
X=[]
for i in feats:
    i=np.array(i)
    X.append(i)
X= np.array(X)   #X is the input feature 2D vector input to the classifier
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, \
            hidden_layer_sizes=(8, 4), random_state=1)
