import ft_ex
import numpy as np
from sklearn.neural_network import MLPClassifier
######
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
######

labels=np.load('/home/psi/PuchoTask/train_score.npy')
feats= np.load('/home/psi/PuchoTask/train_feats.npy')
test_score_ids = np.load('/home/psi/PuchoTask/test_score_ids.npy')
test_scores= np.load('/home/psi/PuchoTask/test_scores.npy')

#####
scaler.fit(feats)
feats = scaler.transform(feats)
#####

al=np.logspace(-5,3,5)
NN = MLPClassifier(solver='lbfgs', alpha=al[0], \
            hidden_layer_sizes=(18 , 30 , 15), random_state = 1, warm_start=True)

NN.fit(feats, labels)

test_feats = np.load('/home/psi/PuchoTask/test_feats.npy')
test_ids = np.load('/home/psi/PuchoTask/test_ids.npy')

prediction = []

#######
test_feats= scaler.transform(test_feats)
#######
for i in test_feats:
    x = NN.predict(i)
    prediction.append(x[0])

test_predictions = np.array(prediction)
test_predictions = test_predictions.astype(np.int)

###The PUBLIC LEADERBORD has 5732 entries and the PUBLIC LEADERBOARD solution with scores had only 5224
###The following code searches for the appropriate indices to match and check the scores

count = 0
indices = []
j=0
for i in test_score_ids:
    z= np.where(test_ids == i)
    indices.append(int(z[0]))

for i in indices:
    if(test_predictions[i]== test_scores[j]):
        count+=1
    j+=1
print 'The accuracy is',count,' out of ', len(test_scores)
