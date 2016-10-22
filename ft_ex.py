import ebow
import enchant as en
import numpy as np
import nltk

"""

The following are the features used from the dataset:
1.Number of words
2.Unique words
3.Average word length
4.Spelling errors
5.Commas in essay
6.Essay set
7.Correctly spelled words
8.Number of sentences
9.Number of double quotes
10.Unique parts of speech tags count ----takes over 4 hours so dropped
11.No. of words per sentence
12.Average number syllables in a word

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
    return round(float(avr_wrd_len),3)

#To count the total number of words
def no_words(bow):
    count= 0
    for i in bow:
        count+=bow[i]
    return float(count)

#To count the number of unique words
def no_unq_words(bow):
    return float(len(bow))

#To count the number of commas from the essay string
def count_commas_dquots(string):
    return float(string.count(',')), float(string.count('"'))

#To count the number of spelling errors and the correctly spelled words in an essay
def count_spell_error(string):
    string = ((((string.replace(".", " ")).replace(","," ")).replace("?"," ")).replace('!',' '))
    string = nltk.word_tokenize(string.strip(":;"))
    check = False
    error = 0
    correct = 0
    for i in string:
        i = i.lower()
        check = d.check(i)
        if check== True :
            correct+=1
        else :
            error+=1
    return float(error), float(correct)

# To count the number unique of POS tags
def POS_count(string):
    x = []
    words = nltk.word_tokenize(string)
    tagged = nltk.pos_tag(words)
    x = [i[1] for i in tagged]
    x = set(x)      #creates a dict with only the unique tags
    del words
    del tagged
    return float(len(x))

#To count the number of sentences
def sent_count(string):
    cnt = len(string.split('.'))
    return cnt

#To count the number of syllables in a word
def syllables(word):
    count = 0
    vowels = 'aeiou'
    word = word.lower().strip(".:;?!")
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count+=1
    if count == 0:
        count +=1
    return count

def av_sylls(string):
    string = ((((string.replace(".", " ")).replace(","," ")).replace("?"," ")).replace('!',' '))
    words = nltk.word_tokenize(string)
    y = len(words)
    val = 0.0
    for j in words:
        if len(j)>1:
            val+= int(syllables(j))
    if y == 0:
        return 1.0
    x = val/y
    return round(x,3)

def extract_feats(essays,sets):
    features = [[]]
    f=[]
    i=0
    for essay in essays:
        bow = ebow.essay_to_bow(essay)
        f1 = no_words(bow)
        f2 = no_unq_words(bow)
        f3 = avr_word_len(bow)
        f4, f7 = count_spell_error(essay)
        f5 ,f9= count_commas_dquots(essay)
        f6 = int(sets[i])
        f8 = sent_count(essay)
        #f10 = POS_count(essay)
        f11 = round(f1/f8, 3)
        f12 = av_sylls(essay)
        f = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f11, f12]
        f = np.array(f)
        f = f.astype(np.float)
        features.append(f)
        i+=1
    features = np.array(features)
    features = np.delete(features, 0)
    return features

# test_feats = extract_feats(test_essays, test_sets)
# train_feats = extract_feats(train_essays, train_sets)
# AND save the above vectors as npy files
