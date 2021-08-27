source .env

# delete endpoint, endpoint config and associated model
aws sagemaker delete-endpoint --endpoint-name toxic-comment-classifier-${STAGE} 
model=$(aws sagemaker describe-endpoint-config --endpoint-config-name toxic-comment-classifier-${STAGE} --output text | awk '{print $5}') 
aws sagemaker delete-endpoint-config --endpoint-config-name toxic-comment-classifier-${STAGE} 
aws sagemaker delete-model --model-name ${model}

# delete resources created using cfn
aws cloudformation delete-stack --stack-name datascience-toxic-comments-pipeline-${STAGE}

# delete S3 bucket and all data
accountnum=$(aws sts get-caller-identity --query Account --output text)
aws s3 rb s3://${BUCKET_NAME}-${accountnum}  --force



