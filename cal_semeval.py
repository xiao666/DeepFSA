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



'''
#LOAD news for stock prediction
data = pd.read_csv('semeval_1.csv')
headlines=[[0 for x in range(1)] for y in range(len(data.index))]#len(data.index)
for row in range(len(data.index)):#len(data.index)
    for col in range(1):
        temp0=str(data.iloc[row,col])
        #temp0=temp0.lower()
        temp=HashingVectorizer().build_tokenizer()(temp0)
        #=========code below remove stopwords==================
        #temp=[s for s in temp if s not in stopwords]
        #headlines[row][col]=temp
        headlines[row][0]=(' '.join(word for word in temp))
'''
'''
with open('Microblog_Trainingdata.json','r') as f2:
    json_data=json.load(f2)
TEST_SENTENCES=[]
for i in range(len(json_data)):
    TEST_SENTENCES.append(json_data[i]['spans'])

'''
'''
TEST_SENTENCES=sum(headlines,[])
print ("len(TEST_SENTENCES):",len(TEST_SENTENCES))
print ("TEST_SENTENCES[1]:",TEST_SENTENCES[1])
'''
#TEST_SENTENCES=np.reshape(headlines,(1,-1))
'''
TEST_SENTENCES = [u'Georgia \'downs two Russian warplanes\' as countries move to brink of war',
                  u'I love how you never reply back..',
                  u'I love cruising with my homies',
                  u'I love messing with yo mind!!',
                  u'I love you and now you\'re just gone..',
                  u'This is shit',
                  u'This is the shit']
'''
data = pd.read_csv('Microblogs_Testdata_withscores.csv')

#TEST_SENTENCES=(' '.join(word for word in data[0]['spans__001']))
TEST_SENTENCES=[]
for i in range(len(data.index)):
    TEST_SENTENCES.append(str(data.iloc[i,4]))

print (len(TEST_SENTENCES))
print (TEST_SENTENCES[-1])
'''
temp0=str(data.iloc[0,4])
temp=HashingVectorizer().build_tokenizer()(temp0)
TEST_SENTENCES=(' '.join(word for word in temp))

'''
#print (TEST_SENTENCES)

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
scores = []
for i,t in enumerate(TEST_SENTENCES):
    scores.append(prob[i])

OUTPUT_PATH='microblog_test_64emoji_prob.csv'
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

