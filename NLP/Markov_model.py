import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
import re
import random
import math
# %%
# Read the text file into a DataFrame
AlPoe = pd.read_csv('PycharmProjects/NLP/datasets/allan_poe.txt', delimiter='/n', header=None,engine='python')
RobFrost = pd.read_csv('PycharmProjects/NLP/datasets/robert_frost.txt', delimiter='/n', header=None,engine='python')

# Display the DataFrame
print(AlPoe.shape)
print(RobFrost.shape)
#%% 80- 20 Split

le = len(AlPoe)

AlanTrain, AlanTest  = AlPoe[:int(le*0.8)] , AlPoe[-int(le*0.2):]

le = len(RobFrost)
RobTrain, RobTest = RobFrost[:int(le*0.8)] , RobFrost[-int(le*0.2):]
#%%
with open('PycharmProjects/NLP/datasets/allan_poe.txt', 'r', encoding='utf-8') as file:
    alan = file.read()

with open('PycharmProjects/NLP/datasets/robert_frost.txt', 'r', encoding='utf-8') as file:
    rob = file.read()

    
#alan = re.sub(" , [0-2][0-9]:[0-5][0-9]", "",alan)
#alan = re.sub("[,|!|.|?|\"}\][\']", "", alan)    
#alan = re.sub("\\n", " ", alan)   
#rob = re.sub("\\n", " ", rob)   
#%%

def tokenize(df):
    uniqueWords = []
    # for each row
    for index, row in df.iterrows():
        row = re.sub(" , [0-2][0-9]:[0-5][0-9]", "",str(row.values))
        row = re.sub("[,|!|.|?|\"}\][\']", "", row)
        words = [w.lower() for w in row.split()]

        for word in words:
            if word not in uniqueWords:
                uniqueWords.append(word)
    
    return uniqueWords

def tokenize_txt(txt):
    uniqueWords = []
    word2vec = []
    vec2word = []
    # for each row
    txt = re.sub("\\n", " ",txt)
    txt = re.sub("\\'", " ",txt)
    
    for index, row in enumerate(txt.split(" ")):
        row = cleanup(row)
        words = [w.lower() for w in row.split()]

        for word in words:
            if word not in uniqueWords:
                uniqueWords.append(word)
            word2vec.append(uniqueWords.index(word))
            vec2word.append(word)
        
    return uniqueWords, word2vec, vec2word
    
    
    
def UnknownIndex(df,table):
    uniqueWords = []
    # for each row
    for index, row in df.iterrows():
        row = re.sub(" , [0-2][0-9]:[0-5][0-9]", "",str(row.values))
        row = re.sub("[-,|!|.|?|\"}\][\']", "", row)
        words = [w.lower() for w in row.split()]

        for word in words:
            if word not in uniqueWords and word not in table:
                uniqueWords.append(word)
                
    
    return uniqueWords

def cleanup(item):
    item = re.sub(" , [0-2][0-9]:[0-5][0-9]", "",item)
    item = re.sub("[()-,|!|.|?|\"{}\][\']", "", item)
    return item


# tokenize sequence
AlanuniqueWords = tokenize(AlanTrain)
RobuniqueWords = tokenize(RobTrain)

AlanUnknownWords = UnknownIndex(AlanTest, AlanTrain)
RobUnknownWords = UnknownIndex(RobTest, RobTrain)


#%%
# correct 1362
un = tokenize_txt(alan)[0]
w2v = tokenize_txt(alan)[1]
v2w = tokenize_txt(alan)[2]

# should print true 
print(un.index('more') == w2v[-1])
    
#%%

def count_sequence(array, target):
    count = 0
    targ_len = len(target)
    
    for i in range(len(array) - targ_len+1):
        if array[i:i+targ_len] == target:
            count+=1
    
    return count
    



def markov_model(sequence, order=2,eps=1):
    #tokenize sequence
    unique,word2vec,vec2word = tokenize_txt(sequence)
    
    lenght = len(vec2word)
    A = np.empty((lenght, lenght))
    
    markov_model = {}  
    
    #loop from start to end-order
    for i, word in enumerate(word2vec[:-order]):
        con = []
        
        # add next words to context
        context = word2vec[i:i+order+1]
        
        next_char = word2vec[i+order]
        
        if context not in markov_model: 
            markov_model[context] = {}
        if next_char not in markov_model[context]:
            markov_model[context][next_char] = 0
        markov_model[context][next_char] += 1
    
    return markov_model
        
#%%
markov_model(alan)
