# -*- coding: utf-8 -*-

""" Use DeepMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the DeepMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division
import example_helper
import json
import csv
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
###############for stock prediction
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

OUTPUT_PATH = 'test_sentences.csv'
'''
TEST_SENTENCES = [u'Georgia \'downs two Russian warplanes\' as countries move to brink of war',
                  u'I love how you never reply back..',
                  u'I love cruising with my homies',
                  u'I love messing with yo mind!!',
                  u'I love you and now you\'re just gone..',
                  u'This is shit',
                  u'This is the shit']
'''
##################################################################
#LOAD news for stock prediction
data = pd.read_csv('Combined_News_DJIA.csv')
headlines=[[0 for x in range(25)] for y in range(1)]#len(data.index)
for row in range(1):#len(data.index)
    for col in range(25):
        temp0=str(data.iloc[row,(col+2)])
        #temp0=temp0.lower()
        temp=HashingVectorizer().build_tokenizer()(temp0)
        #=========code below remove stopwords==================
        #temp=[s for s in temp if s not in stopwords]
        #headlines[row][col]=temp
        headlines[row][col]=(' '.join(word for word in temp))
'''
print ("data shape:",data.shape)#(1989,27)
print ("list shape:",(len(headlines),len(headlines[0])))#(1989,25)
print (headlines[0])
print (headlines[0][0])
'''
print (len(headlines[0][0]))
TEST_SENTENCES=headlines[0]#JUST TEST 25 HEADLINES
#TEST_SENTENCES=sum(headlines,[])

print ("len(TEST_SENTENCES):",len(TEST_SENTENCES))
print ("TEST_SENTENCES[1]:",TEST_SENTENCES[1])


##################################################################
def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

maxlen = 30
batch_size = 32

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)#FROM LIST OF TOKENS TO NUMBERS

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
model.summary()

print('Running predictions.')
prob = model.predict(tokenized)
####################################################################prob[] is the softmax output for 64 emojis
# Find top emojis for each sentence. Emoji ids (0-63)
# correspond to the mapping in emoji_overview.png 
# at the root of the DeepMoji repo.
'''
print('Writing results to {}'.format(OUTPUT_PATH))

scores = []
for i, t in enumerate(TEST_SENTENCES):
    t_tokens = tokenized[i]
    t_score = [t]
    t_prob = prob[i]
    ind_top = top_elements(t_prob, 64)#changed 5 to 64
    t_score.append(sum(t_prob[ind_top]))
    t_score.extend(ind_top)
    t_score.extend([t_prob[ind] for ind in ind_top])
    scores.append(t_score)
    print(t_score)


with open(OUTPUT_PATH,'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(['Text', 'Top5%',
                     'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4', 'Emoji_5',
                     'Pct_1', 'Pct_2', 'Pct_3', 'Pct_4', 'Pct_5'])
    for i, row in enumerate(scores):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))

'''

scores = []
for i,t in enumerate(TEST_SENTENCES):
    scores.append(prob[i])
    #print(t_score)
#print(scores[0])

#OUTPUT_PATH='t_prob64.csv'
OUTPUT_PATH='tt.csv'

print('Writing results to {}'.format(OUTPUT_PATH))
with open (OUTPUT_PATH,'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(['Emoji_0', 'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4',
                         'Emoji_5', 'Emoji_6', 'Emoji_7', 'Emoji_8', 'Emoji_9', 'Emoji_10',
                         'Emoji_11', 'Emoji_12', 'Emoji_13', 'Emoji_14', 'Emoji_15',
                         'Emoji_16', 'Emoji_17', 'Emoji_18', 'Emoji_19', 'Emoji_20',
                         'Emoji_21', 'Emoji_22', 'Emoji_23', 'Emoji_24', 'Emoji_25',
                         'Emoji_26', 'Emoji_27', 'Emoji_28', 'Emoji_29', 'Emoji_30',
                         'Emoji_31', 'Emoji_32', 'Emoji_33', 'Emoji_34', 'Emoji_35',
                         'Emoji_36', 'Emoji_37', 'Emoji_38', 'Emoji_39', 'Emoji_40',
                         'Emoji_41', 'Emoji_42', 'Emoji_43', 'Emoji_44', 'Emoji_45',
                         'Emoji_46', 'Emoji_47', 'Emoji_48', 'Emoji_49', 'Emoji_50',
                         'Emoji_51', 'Emoji_52', 'Emoji_53', 'Emoji_54', 'Emoji_55',
                         'Emoji_56', 'Emoji_57', 'Emoji_58', 'Emoji_59', 'Emoji_60',
                         'Emoji_61', 'Emoji_62', 'Emoji_63'])#in all 64 emojis
    for i, row in enumerate(scores):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))
