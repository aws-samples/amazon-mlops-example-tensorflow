import os
import json
import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
import etl
from sklearn.model_selection import train_test_split
import CustomModel
import io
import pickle
import boto3

def load_training_data(base_dir):
    x_train = np.load(os.path.join(base_dir, 'train_data.pkl'),allow_pickle=True)
    y_train = np.load(os.path.join(base_dir, 'train_labels.pkl'),allow_pickle=True)
    return x_train, y_train

def parse_args():

    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    return parser.parse_known_args()

def train_model(data,labels):
        
    ## Train your own embeddings
    model = CustomModel.define_network("none")
    model.compile(loss = "binary_crossentropy", optimizer = "adam", 
    metrics = [tf.keras.metrics.AUC(multi_label=True),
               tf.keras.metrics.MeanIoU(num_classes=6),
               CustomModel.MultiLabelPrecision()])
    
    ## Split data
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, shuffle = True, random_state = 123)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle = True, random_state = 123)

    print("Shape of train,test,val:",x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)
        
    ## Fit tokenizer on x_train and tokenize both train and test
    tokenizer = etl.get_tokenizer(x_train)
    x_train = etl.tokenize(x_train,tokenizer)
    x_test = etl.tokenize(x_test,tokenizer)
    x_val = etl.tokenize(x_val,tokenizer)
    save_tokenizer(tokenizer)
    
    ## Fit model and evaluate
    model.fit(x_train, y_train, batch_size = 2048, epochs = 2,validation_data = (x_val,y_val),verbose=2) 
    print("Model Trained. Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=2048,verbose=1)
    print("test loss, test auc, test_IoU, test_precision:", results)
    return model
    
def save_model(model,outputpath):
    print("saving model in path - ",outputpath)
    tf.saved_model.save(model,os.path.join(outputpath ,'1'))

def save_tokenizer(tokenizer):
    s3_client = boto3.client('s3')
    s3_sourcedir = os.environ['SM_MODULE_DIR']
    s3_bucket = s3_sourcedir.split("/")[2]
    array_data = io.BytesIO()
    pickle.dump(tokenizer,array_data)
    array_data.seek(0)
    s3_client.upload_fileobj(array_data,s3_bucket,"toxic_comments/tokenizer.pkl")


if __name__ == "__main__":
    args, unknown = parse_args()
    
    print("........Tensor flow version...........")
    print(tf.__version__)

    # 1) get training data
    train_data, train_labels = load_training_data(args.train)
    
    # 2) train model
    model = train_model(train_data,train_labels)
    
    # 3) save model
    save_model(model,args.model_dir)
    
    

    