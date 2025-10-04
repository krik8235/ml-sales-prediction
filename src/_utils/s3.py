import os
import boto3
import io
import gzip
import tempfile
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from src._utils.log import main_logger


def s3_upload(file_path: str, s3_key=None):
    load_dotenv(override=True)
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-sales-pred')
    s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION_NAME', 'us-east-1'))

    if s3_client:
        s3_key = s3_key if s3_key else file_path if file_path[0] != '/' else file_path[1:]
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        main_logger.info(f"file uploaded to s3://{S3_BUCKET_NAME}/{s3_key}")
    else:
        main_logger.error('failed to create an S3 client.')


def _get_latest_s3_file_key(prefix=""):
    load_dotenv(override=True)
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-sales-pred')
    s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION_NAME', 'us-east-1'))

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


def s3_extract(file_path = None, prefix = '', verbose: bool = False):
    load_dotenv(override=True)
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-sales-pred')
    s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION_NAME', 'us-east-1'))

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
                if verbose: main_logger.info("file is not gzipped. Returning as-is.")
                buffer.seek(0)
                return buffer

    except ClientError as e:
        main_logger.error(f"failed to load file from S3: {e}. return None")
        raise


def s3_extract_from_temp_file(file_path):
    load_dotenv(override=True)
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-sales-pred')
    s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION_NAME', 'us-east-1'))

    main_logger.info(f"... loading {file_path} from S3...")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            s3_client.download_file(Bucket=S3_BUCKET_NAME, Key=file_path, Filename=tmp_file.name)
            return tmp_file.name
    except ClientError as e:
        main_logger.error(f"Error downloading {file_path} from S3: {e}")
        raise
