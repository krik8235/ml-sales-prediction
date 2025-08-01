from dotenv import load_dotenv # type: ignore
load_dotenv(override=True)

import os
import boto3 # type: ignore

from src._utils.log import main_logger


S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
S3_MODEL_PREFIX = os.environ.get('S3_MODEL_PREFIX')
S3_DATA_PREFIX = os.environ.get('S3_DATA_PREFIX')

s3_client = boto3.client('s3')

def s3_upload(file_path: str, file_name: str, prefix = S3_MODEL_PREFIX):
    if s3_client:
        s3_key = f"{prefix}_{file_name}"
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        main_logger.info(f"PyTorch model uploaded to s3://{S3_BUCKET_NAME}/{s3_key}")

    else:
        main_logger.error('Failed to create an S3 client.')
