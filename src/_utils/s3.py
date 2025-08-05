import os
import boto3 # type: ignore
import io
import gzip
from botocore.exceptions import ClientError # type: ignore
from dotenv import load_dotenv # type: ignore

from src._utils.log import main_logger

load_dotenv(override=True)

S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
s3_client = boto3.client('s3')



def s3_upload(file_path: str):
    if s3_client:
        s3_key = f"{file_path}"
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        main_logger.info(f"file uploaded to s3://{S3_BUCKET_NAME}/{s3_key}")
    else:
        main_logger.error('failed to create an S3 client.')



def _get_latest_s3_file_key(prefix=""):
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)

        if 'Contents' not in response:
            main_logger.info(f"No objects found in prefix: s3://{S3_BUCKET_NAME}/{prefix}")
            return None

        # sort the objects by their 'LastModified' timestamp
        latest_file = max(response['Contents'], key=lambda obj: obj['LastModified'])
        return latest_file['Key']

    except ClientError as e:
        main_logger.error(f"An error occurred while listing S3 objects: {e}")
        return None
        


def s3_load(file_path = None, prefix = '', verbose: bool = False):
    buffer = None

    if not file_path:
        file_key = _get_latest_s3_file_key(prefix)

        if not file_key:
            main_logger.error("failed to find the latest file key. Returning None.")
            return None
        
        file_path = file_key

    try:
        if s3_client:
            if verbose: main_logger.info(f"loading from S3 bucket")
            obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_path)
            buffer = io.BytesIO(obj['Body'].read())
        else:
            if verbose: main_logger.info(f"loading {file_path} from local file system")
            
        if buffer is not None:
            buffer.seek(0)
            if buffer.read(2) == b'\x1f\x8b':
                if verbose: main_logger.info("file is gzipped. Decompressing...")
                buffer.seek(0)
                decompressed_buffer = io.BytesIO(gzip.decompress(buffer.read()))
                return decompressed_buffer
            else:
                if verbose: main_logger.info("File is not gzipped. Returning as-is.")
                buffer.seek(0)
                return buffer
    
    except ClientError as e:
        main_logger.error(f"failed to load file from S3: {e}")
        return None



def s3_bulk_upload():
    s3_upload(file_path='preprocessors/column_transformer.pkl')
    s3_upload(file_path='models/dfn_best/dfn_20250801203833.pth')
    s3_upload(file_path='data/processed/processed_df_20250731224220.csv')
    s3_upload(file_path='data/raw/online_retail.csv')
