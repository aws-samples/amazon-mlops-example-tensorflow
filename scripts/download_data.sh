#!/usr/bin/env bash

source .env

mkdir -p data

cd data 

kaggle competitions download -c jigsaw-toxic-comment-classification-challenge

unzip jigsaw-toxic-comment-classification-challenge.zip

accountnum=$(aws sts get-caller-identity --query Account --output text)

aws s3 mb s3://${BUCKET_NAME}-${accountnum} --region $AWS_REGION

aws s3 cp train.csv.zip s3://${BUCKET_NAME}-${accountnum}/toxic_comments/train.csv.zip
