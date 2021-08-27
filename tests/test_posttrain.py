import unittest
import pandas as pd
import os
import json 
import boto3

src_bucket = os.getenv("BUCKET_NAME")
env = os.getenv("STAGE","dev")
endpoint_name = f"toxic-comment-classifier-{env}"
client = boto3.client('sagemaker-runtime')

class PostTraining(unittest.TestCase):
    
    def __init__(self,*args, **kwargs):
        super(PostTraining, self).__init__(*args, **kwargs)
        self.traindata = pd.read_pickle(f"s3://{src_bucket}/toxic_comments/train_data.pkl")
        self.trainlabels = pd.read_pickle(f"s3://{src_bucket}/toxic_comments/train_labels.pkl")
    
    def test_endpoint(self):
        length = 5
        content_type = "application/json" 
        accept = "application/json"
        payload = json.dumps(self.traindata[0:length].tolist())
        
        response = client.invoke_endpoint(
        EndpointName=endpoint_name,  
        ContentType=content_type,
        Accept=accept,
        Body=payload
        )

        result = response['Body'].read().decode('utf-8')
        preds = json.loads(result)['predictions']
        self.assertEqual(len(preds),length)
        
        
if __name__ == '__main__':
    unittest.main()
