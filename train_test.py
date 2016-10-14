import ft_ex
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn import ensemble
######
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
######

labels=np.load('/home/psi/PuchoTask/train_score.npy')
feats= np.load('/home/psi/PuchoTask/train_feats.npy')
test_score_ids = np.load('/home/psi/PuchoTask/test_score_ids.npy')
test_scores= np.load('/home/psi/PuchoTask/test_score.npy')

#####
scaler.fit(feats)
feats = scaler.transform(feats)
#####

al=np.logspace(-5,3,5)
################################MODEL 1############################################
NN = MLPClassifier(solver='lbfgs', alpha=al[0], \
            hidden_layer_sizes=(15, 8), random_state = 1,  max_iter= 600 , activation ='logistic')
NN.fit(feats, labels)


##############################MODEL 2##############################################
RF = ensemble.RandomForestClassifier(n_estimators=20)
RF.fit(feats, labels)

test_feats = np.load('/home/psi/PuchoTask/test_feats.npy')
test_ids = np.load('/home/psi/PuchoTask/test_ids.npy')

prediction1 = []
prediction3 = []

#######
test_feats= scaler.transform(test_feats)
#######

for i in test_feats:
    x = NN.predict(i)
    z = RF.predict(i)
    prediction1.append(x[0])
    #prediction3.append(z[0])

test_predictions1 = np.array(prediction1)
test_predictions1 = test_predictions1.astype(np.int)

#test_predictions3 = np.array(prediction3)
#test_predictions3 = test_predictions3.astype(np.int)


###The PUBLIC LEADERBORD has 5732 entries and the PUBLIC LEADERBOARD solution with scores had only 5224
###The following code searches for the appropriate indices to match and check the scores

count = 0
indices = []
j=0
for i in test_score_ids:
    z= np.where(test_ids == i)
    indices.append(int(z[0]))

for i in indices:
    if(test_predictions1[i] == test_scores[j]):
        count+=1
    j+=1
print 'The accuracy  of the neural network is', count,' out of ', len(test_scores)

#i=0
#j=0
#count = 0
#for i in indices:
#    if(test_predictions3[i]== test_scores[j]):
#        count+=1
#    j+=1

#print 'The accuracy  of the RandomForest classifier is',count,' out of ', len(test_scores)
