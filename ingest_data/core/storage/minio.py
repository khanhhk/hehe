import os
from io import BytesIO
from typing import Tuple

from minio import Minio
from minio.error import S3Error

from utils.logger import FrameworkLogger, get_logger

logger: FrameworkLogger = get_logger()


def check_src_data(file_link: str) -> bool:
    """
    Check if a given file path exists on the local filesystem.

    Args:
        file_link (str): Path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.exists(file_link)


class MinioLoader:
    """
    Wrapper class for MinIO operations including upload/download via stream.
    """

    def __init__(
        self,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        secure: bool = False,
    ) -> None:
        """
        Initialize the MinIO client.

        Args:
            minio_endpoint (str): Endpoint for the MinIO server.
            minio_access_key (str): Access key for authentication.
            minio_secret_key (str): Secret key for authentication.
            secure (bool): Whether to use HTTPS (True) or HTTP (False).
        """
        self.client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=secure,
        )

    @staticmethod
    def get_info_from_minio(s3_path: str) -> Tuple[str, str]:
        """
        Parse the bucket and object key from an s3:// path.

        Args:
            s3_path (str): Full S3 path (e.g. s3://bucket/key).

        Returns:
            Tuple[str, str]: Tuple of (bucket_name, object_key).
        """
        s3_path = s3_path.replace("s3://", "")
        s3_bucket, s3_key = s3_path.split("/", 1)
        return s3_bucket, s3_key

    def upload_object_from_stream(
        self,
        s3_path: str,
        data_stream: BytesIO,
        data_length: int,
    ) -> None:
        """
        Upload an object from a BytesIO stream to MinIO.

        Args:
            s3_path (str): Target S3 path (e.g. s3://bucket/key).
            data_stream (BytesIO): Byte stream to upload.
            data_length (int): Length in bytes of the data stream.

        Raises:
            S3Error: If uploading fails.
        """
        s3_bucket, s3_key = self.get_info_from_minio(s3_path)
        if not self.client.bucket_exists(s3_bucket):
            self.client.make_bucket(s3_bucket)

        try:
            self.client.put_object(
                bucket_name=s3_bucket,
                object_name=s3_key,
                data=data_stream,
                length=data_length,
                content_type="application/octet-stream",
            )
            logger.info("Successfully uploaded data to '%s'", s3_path)
        except S3Error as e:
            logger.error("Failed to upload to MinIO: %s", e)
            raise

    def download_object_as_stream(self, s3_path: str) -> BytesIO:
        """
        Download an object from MinIO as a BytesIO stream.

        Args:
            s3_path (str): Target S3 path (e.g. s3://bucket/key).

        Returns:
            BytesIO: Stream of the downloaded object.

        Raises:
            S3Error: If download fails.
        """
        s3_bucket, s3_key = self.get_info_from_minio(s3_path)
        try:
            response = self.client.get_object(bucket_name=s3_bucket, object_name=s3_key)
            buffer = BytesIO(response.read())
            buffer.seek(0)
            logger.info("Successfully uploaded data to '%s'", s3_path)
            return buffer
        except S3Error as e:
            logger.error("Failed to download from MinIO: %s", e)
            raise
