import pandas as pd
import sys
import os
import re
# sagemaker container path at run time
sys.path.append('/opt/ml/model/code')

MAX_NB_WORDS = 40000
MAX_SEQUENCE_LENGTH=150

def cleanup(ip,column=None):
    '''
    ip: list of strings at inference time or dataframe at training time
    '''
    if column:
        print("training ETL")
        ip[column] = ip.apply(lambda row: row[column].replace("\n"," "), axis=1)
        ip[column] = ip.apply(lambda row: re.sub('http://\S+|https://\S+', 'urls',row[column]).lower(),axis=1)
        ip[column] = ip.apply(lambda row: re.sub('[^A-Za-z\' ]+', '',row[column]).lower(), axis=1)
    else:
        print("inference ETL")
        ip= [i.replace("\n"," ") for i in ip]
        ip = [re.sub('http://\S+|https://\S+', 'url',i).lower() for i in ip]
        ip = [re.sub('[^A-Za-z\' ]+', '',i).lower() for i in ip]          
    return ip

def fitandtokenize(data):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    # sagemaker container path
    tokenizer = pd.read_pickle("/opt/ml/model/code/tokenizer.pkl")
    #convert each text into array of integers
    data = tokenizer.texts_to_sequences(data)
    data = pad_sequences(data, maxlen = MAX_SEQUENCE_LENGTH)
    return data

def get_tokenizer(data):
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words = MAX_NB_WORDS) 
    tokenizer.fit_on_texts(data)
    return tokenizer
    
def tokenize(data, tokenizer):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    data = tokenizer.texts_to_sequences(data)
    return pad_sequences(data, maxlen = MAX_SEQUENCE_LENGTH)
    
def get_word_index(data):
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(data)
    return tokenizer.word_index
