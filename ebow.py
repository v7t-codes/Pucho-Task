import os
import sys
from stemming.porter2 import stem

def essay_to_bow(lyrics):
    # remove end of lines
    l_flat = lyrics.replace('\r', '\n').replace('\n', ' ').lower()
    l_flat = ' ' + l_flat + ' '
    #common special cases in English...
    l_flat = l_flat.replace("'m ", " am ")
    l_flat = l_flat.replace("'re ", " are ")
    l_flat = l_flat.replace("'ve ", " have ")
    l_flat = l_flat.replace("'d ", " would ")
    l_flat = l_flat.replace( "'ll ", " will ")
    l_flat = l_flat.replace(" he's ", " he is ")
    l_flat = l_flat.replace(" she's ", " she is ")
    l_flat = l_flat.replace(" it's ", " it is ")
    l_flat = l_flat.replace(" ain't ", " is not ")
    l_flat = l_flat.replace("n't ", " not ")
    l_flat = l_flat.replace("'s ", " ")
    # remove punctuations and othre characters except commas as it is one of the features we use to predict the score
    punctuation = (',',"'", '"', ';', ':', '.', '?', '!', '(', ')',
                   '{', '}', '/', '\\', '_', '|', '-', '@', '#', '*')
    for p in punctuation:
        l_flat = l_flat.replace(p, '')
    words = filter(lambda x: x.strip() != '', l_flat.split(' '))
    # stem words
    words = map(lambda x: stem(x), words)
    bow = {}
    for w in words:
        if not w in bow.keys():
            bow[w] = 1
        else:
            bow[w] += 1
    fake_words = ('>', '<', 'outro~')
    bowwords = bow.keys()
    for bw in bowwords:
        if bw in fake_words:
            bow.pop(bw)
        elif bw.find(']') >= 0:
            bow.pop(bw)
        elif bw.find('[') >= 0:
            bow.pop(bw)
    return bow
