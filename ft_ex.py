import ebow
import enchant as en
import numpy as np

"""
The following are the features used from the dataset:
1.Number of words
2.Unique words / words
3.Average word length
4.Spelling errors / words
5.Commas
6.Essay set
7.Correctly spelled words
8.Number of sentences
9.Number of double quotes
10.Number of unique parts of speech tags
"""
# train_essays = np.load('/home/psi/PuchoTask/Datasets/essay_train.npy')
# train_score = np.load('/home/psi/PuchoTask/train_score.npy')
# train_sets = np.load('/home/psi/PuchoTask/Datasets/essay_sets.npy')

# test_ids = np.load('/home/psi/PuchoTask/test_ids.npy')
# test_essays = np.load('/home/psi/PuchoTask/test_essays.npy')

d=en.Dict('en_US')
bow = {}
string = ''
# train_essays = np.delete(train_essays, 0) # the label
i=0


#To find the average word length from the bag of words representation
def avr_word_len(bow):
    num = 0
    den = len(bow)
    for i in bow:
        num+=len(i)
    avr_wrd_len = num/den
    return avr_wrd_len

#To count the total number of words
def no_words(bow):
    count= 0
    for i in bow:
        count+=bow[i]
    return count

#To count the number of unique words
def no_unq_words(bow):
    return len(bow)

#To count the number of commas from the essay string
def count_commas(string):
    return string.count(',')

#To count the number of spelling errors in an essay
def count_spell_error(string):
    string= string.lower()
    check= False
    error = 0
    for i in string:
        check = d.check(i)
        if check== True :
            error+=1
    return error


# To count the number unique of POS tags
def POS_count(string):
    x = []
    words = nltk.tokenize(string.split(' '))
    for i in words:
        _, tag = i
        x.append(tag)
    x = set(x) # creates a dict with only the unique tags
    return len(x)


def extract_feats(essays,sets):
    features = [[]]
    f=[]
    i=0
    for essay in essays:
        bow = ebow.essay_to_bow(essay)
        f1 = int(no_words(bow))
        f2 = int(no_unq_words(bow))
        f3 = int(avr_word_len(bow))
        f4 = int(count_spell_error(essay))
        f5 = int(count_commas(essay))
        f6 = int(sets[i])
        f = [f1,f2,f3,f4,f5,f6]
        f = np.array(f)
        features.append(f)
        i+=1
    features = np.array(features)
    features = np.delete(features, 0)
    return features

# test_feats = extract_feats(test_essays, test_sets)
# train_feats = extract_feats(train_essays, train_sets)
# AND save the above vectors as npy files
