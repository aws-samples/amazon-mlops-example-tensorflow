
from sagemaker.tensorflow import TensorFlow
import os
import numpy as np

# set environment
src_bucket = os.getenv("BUCKET_NAME")
print(src_bucket)
sm_role = os.getenv("SAGEMAKER_IAM_ROLE")
artifact_path = f's3://{src_bucket}/toxic_comments'
instance = os.getenv("INSTANCE")

# create an estimator object
estimator = TensorFlow(
    entry_point="train/train.py",
    instance_count=1,
    dependencies=['train/requirements.txt','./inference/etl.py','train/CustomModel.py'],
    output_path = artifact_path,
    model_dir = False,
    code_location = artifact_path,
    #instance_type = "local",
    base_job_name = "comment-classification",
    instance_type = instance,
    framework_version="2.2",
    py_version="py37",
    role = sm_role,
    script_mode =True
)

# train the model
estimator.fit(artifact_path)

# get the training job name (prefixed with datetime)
training_job_name = estimator.latest_training_job.name

# write dynamic variables(model artifact location) to a .env file for later use in deploy stages
with open("dynamic_vars/.env", "w") as f:
    f.write("%s=%s\n%s=%s\n" %("job", training_job_name,"tokpath",f"{artifact_path}/tokenizer.pkl"))
f.close()
