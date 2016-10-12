import ft_ex
import ebow
import numpy as np
from sklearn.neural_network import MLPClassfier

labels=np.load('/home/psi/PuchoTask/train_score.npy')
feats= np.load('/home/psi/PuchoTask/train_feats.npy')

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, \
            hidden_layer_sizes=(5, 10, 4), random_state=1)

NN.fit(feats, labels)

test_feats = np.load('/home/psi/PuchoTask/test_feats.npy')
test_ids = np.load('/home/psi/PuchoTask/test_ids.npy')

prediction = []

for i in test_feats:
    x = NN.predict(i)
    prediction.append(x[0])

prediction = np.array(prediction)
prediction = prediction.astype(np.int)
np.save('prediction.npy', prediction)
