# add sys path to import etl
import sys
import os
dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, '..', 'inference'))

# standard library imports 
import re
import pandas as pd
import numpy as np
import boto3
import io
import pickle
import re

# custom library imports
from etl import cleanup

s3_client = boto3.client('s3')
src_bucket = os.getenv("BUCKET_NAME")

def npy_to_s3(nparray,bucket,key,name):
    '''
    uploading numpy array to S3
    '''
    # upload without using disk
    array_data = io.BytesIO()
    pickle.dump(nparray,array_data)
    array_data.seek(0)
    s3_client.upload_fileobj(array_data, bucket, key+'/'+name+'.pkl')
    
def read_data(inputcsv):
    '''
    Read input data for text pre processing
    '''
    df = pd.read_csv(inputcsv)
    df_processed = cleanup(df,"comment_text")
    df_processed = df_processed[0:100000]
    train_data = np.array(df_processed['comment_text'])
    train_labels = df_processed[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']].to_numpy()
    npy_to_s3(train_data,src_bucket,"toxic_comments","train_data")
    npy_to_s3(train_labels,src_bucket,"toxic_comments","train_labels")
    
read_data(f's3://{src_bucket}/toxic_comments/train.csv.zip')
