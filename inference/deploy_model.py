import sagemaker
from sagemaker.tensorflow import TensorFlowModel
import argparse
import os 

src_bucket = os.getenv("BUCKET_NAME")
job_name = os.getenv("job")
sm_role = os.getenv("SAGEMAKER_IAM_ROLE")
env = os.getenv("STAGE","dev")

## Set default bucket in sagemaker session
sess = sagemaker.session.Session(default_bucket=src_bucket)

# set model tarball path
modelpath = f's3://{src_bucket}/toxic_comments/{job_name}/output/model.tar.gz'

# create a tensorflow serving model
tensorflow_serving_model = TensorFlowModel(entry_point='inference.py',
              source_dir = "./inference",
              sagemaker_session = sess,
              model_data= modelpath,
              role=sm_role,
              framework_version='2.2')

# deploy model   
try:
    predictor = tensorflow_serving_model.deploy(initial_instance_count=1,instance_type='ml.c5.large',
                                                endpoint_name = f"toxic-comment-classifier-{env}")
    print("Creating Endpoint")
except Exception as e:
    print(e)
    print(f"Delete toxic-comment-classifier-{env} endpoint before retrying..")
    #predictor = tensorflow_serving_model.deploy(initial_instance_count=1,instance_type='ml.c5.large')
    ## TODO - Either delete and create new endpoint OR update endpoint
    ## Update endpoint in deploy method is deprecated in sagemaker version 2.x
    ## There is no easy way to update endpoints
    ## Find a workaround
    
