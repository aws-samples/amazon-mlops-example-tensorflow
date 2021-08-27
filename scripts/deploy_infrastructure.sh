#!/usr/bin/env bash

source .env

accountnum=$(aws sts get-caller-identity --query Account --output text)

s3bucketname=${BUCKET_NAME}-${accountnum}

zip -r code.zip . -x '*.git*' 'data/*' '.venv/*'

aws s3 cp code.zip s3://${s3bucketname}/ 

aws cloudformation deploy\
    --template-file "cloudformation/template.yaml"\
    --s3-bucket "$s3bucketname"\
    --s3-prefix "template"\
    --region "$AWS_REGION"\
    --stack-name "datascience-toxic-comments-pipeline-$STAGE"\
    --capabilities CAPABILITY_NAMED_IAM\
    --parameter-overrides\
    pSourceBucket=$s3bucketname\
    pRepositoryName=$REPOSITORY_NAME\
    pEnvironment=$STAGE\
    pinstance=$INSTANCE

# Cleanup zip file
rm -f code.zip
