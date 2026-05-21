"""
Created on Thu May 21 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

General helper routines.

"""

from botocore import UNSIGNED
from botocore.client import Config
from random import sample
from google.cloud import storage
from io import BytesIO
from zipfile import ZipFile

import boto3
import os
import pandas as pd
import shutil


# -- OS Utils ---
def list_dir(dir_path, extension=""):
    """
    Lists filenames in the given directory. If "extension" is provided,
    filenames ending with the given extension are returned.

    Parameters
    ----------
    dir_path : str
        Path to directory to be searched.
    extension : str, optional
       Extension of filenames to be returned. Default is an empty string.

    Returns
    -------
    List[str]
        Filenames in the given directory.
    """
    return [f for f in os.listdir(dir_path) if f.endswith(extension)]


def list_files_in_zip(zip_content):
    """
    Lists all files in a ZIP archive stored in a GCS bucket.

    Parameters
    ----------
    zip_content : str
        Content stored in a ZIP archive in the form of a string of bytes.

    Returns
    -------
    List[str]
        Filenames in a ZIP archive file.
    """
    with ZipFile(BytesIO(zip_content), "r") as zip_file:
        return zip_file.namelist()


def list_paths(dir_path, extension=None):
    """
    Lists paths of files in the given directory. If "extension" is provided,
    filenames ending with the given extension are returned.

    Parameters
    ----------
    dir_path : str
        Path to directory to be searched.
    extension : str, optional
        Extension of filenames to be returned. Default is None.

    Returns
    -------
    paths : List[str]
        Paths of files in the given directory.
    """
    paths = list()
    for f in list_dir(dir_path, extension=extension):
        paths.append(os.path.join(dir_path, f))
    return paths


def mkdir(dir_path, delete=False):
    """
    Creates a directory at the given path.

    Parameters
    ----------
    dir_path : str
        Path of directory to be created.
    delete : bool, optional
        Indication of whether to delete the directory if it already exists
        Default is False.
    """
    if delete:
        rmdir(dir_path)

    os.makedirs(dir_path, exist_ok=True)


def rmdir(dir_path):
    """
    Removes the given directory and all of its subdirectories.

    Parameters
    ----------
    dir_path : str
        Path to directory to be removed if it exists.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def rm_file(path):
    """
    Removes the file at the given path.

    Parameters
    ----------
    path : str
        Path to file to be removed.
    """
    if os.path.exists(path):
        os.remove(path)


# --- IO Utils ---
def read_json(path):
    """
    Reads JSON file located at the given path.

    Parameters
    ----------
    path : str
        Path to JSON file to be read.

    Returns
    -------
    dict
        Contents of JSON file.
    """
    return pd.read_json(path, typ="series")


def read_txt(path):
    """
    Reads txt file at the given path.

    Parameters
    ----------
    path : str
        Path to txt file.

    Returns
    -------
    str
        Text from the txt file.
    """
    if is_s3_path(path):
        return read_txt_from_s3(path)
    elif is_gcs_path(path):
        return read_txt_from_gcs(path)
    else:
        with open(path, "r") as f:
            return f.read()


def read_zip(zip_file, path):
    """
    Reads txt file contained in the given ZIP archive.

    Parameters
    ----------
    zip_file : ZipFile
        ZIP archive containing TXT file.

    Returns
    -------
    str
        Contents of a TXT file.
    """
    with zip_file.open(path) as f:
        return f.read().decode("utf-8")


# --- Cloud Utils ---
def parse_cloud_path(path):
    """
    Parses a cloud storage path into its bucket name and prefix. Supports
    paths of the form: "{scheme}://bucket_name/prefix" or without a scheme.

    Parameters
    ----------
    path : str
        Path to be parsed.

    Returns
    -------
    bucket_name : str
        Name of the bucket.
    prefix : str
        Cloud prefix.
    """
    # Split path
    path = path[len("s3://"):] if is_s3_path else path[len("gs://"):]
    parts = path.split("/", 1)

    # Extract bucket and prefix
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix


def list_cloud_paths(path, extension=""):
    """
    Lists all files in a GCS/S3 bucket with the given extension.

    Parameters
    ----------
    path : str
        Path to cloud prefix to be searched, must be in the format:
        f"{scheme}://{bucket_name}/{prefix}".
    extension : str, optional
        File extension of filenames to be listed. Default is an empty string.

    Returns
    -------
    List[str]
        Filenames stored at the GCS path with the given extension.
    """
    assert is_gcs_path(path) or is_s3_path(path)
    bucket_name, prefix = parse_cloud_path(path)
    list_fn = list_gcs_paths if is_gcs_path(path) else list_s3_paths
    return list_fn(bucket_name, prefix, extension=extension)


# -- GCS Utils --
def is_gcs_path(path):
    """
    Checks if the path is a GCS path.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path is a GCS path.
    """
    return path.startswith("gs://")


def list_gcs_paths(bucket_name, prefix, extension=""):
    """
    Lists paths at a GCS prefix with the given extension.

    Parameters
    ----------
    bucket_name : str
        Name of bucket containing prefix.
    prefix : str
        Path to location within bucket to be searched.
    extension : str, optional
        File extension of filenames to be listed. Default is an empty string.

    Returns
    -------
    List[str]
        Paths under the GCS prefix with the given extension.
    """
    bucket = storage.Client().bucket(bucket_name)
    paths = list()
    for name in [b.name for b in bucket.list_blobs(prefix=prefix)]:
        if extension in name:
            paths.append(os.path.join(f"gs://{bucket_name}", name))
    return paths


def list_gcs_subdirectories(bucket_name, prefix):
    """
    Lists all direct subdirectories of a given prefix in a GCS bucket.

    Parameters
    ----------
    bucket : str
        Name of bucket to be read from.
    prefix : str
        Path to directory in the specified bucket.

    Returns
    -------
    subdirs : List[str]
         Direct subdirectories.
    """
    # Load blobs
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter="/")
    [blob.name for blob in blobs]

    # Parse directory contents
    prefix_depth = len(prefix.split("/"))
    subdirs = list()
    for prefix in blobs.prefixes:
        is_dir = prefix.endswith("/")
        is_direct_subdir = len(prefix.split("/")) - 1 == prefix_depth
        if is_dir and is_direct_subdir:
            subdirs.append(prefix)
    return subdirs


def read_txt_from_gcs(path):
    """
    Reads a txt file stored in a GCS bucket.

    Parameters
    ----------
    path : str
        Path to txt file to be read.

    Returns
    -------
    str
        Contents of txt file.
    """
    bucket_name, subpath = parse_cloud_path(path)
    bucket = storage.Client().bucket(bucket_name)
    return bucket.blob(subpath).download_as_text()


# --- S3 Utils ---
def is_s3_path(path):
    """
    Checks if the given path is an S3 path.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path is an S3 path.
    """
    return path.startswith("s3://")


def list_s3_paths(bucket_name, prefix, extension=""):
    """
    Lists all object keys in a public S3 bucket under a given prefix,
    optionally filters by file extension.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket.
    prefix : str
        Prefix to search under.
    extension : str, optional
        File extension to filter by. Default is an empty string.

    Returns
    -------
    paths : List[str]
        S3 object keys that match the prefix and extension filter.
    """
    # Create an anonymous client for public buckets
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # List all objects under the prefix
    paths = list()
    if "Contents" in response:
        for obj in response["Contents"]:
            filename = obj["Key"]
            if filename.endswith(extension):
                path = os.path.join(f"s3://{bucket_name}", filename)
                paths.append(path)
    return paths


def read_txt_from_s3(path):
    """
    Reads a txt file stored in an S3 bucket.

    Parameters
    ----------
    path : str
        Path to txt file to be read.

    Returns
    -------
    str
        Contents of txt file.
    """
    bucket_name, subpath = parse_cloud_path(path)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    obj = s3.get_object(Bucket=bucket_name, Key=subpath)
    return obj["Body"].read().decode("utf-8")


# --- Miscellaneous ---
def sample_once(my_container):
    """
    Samples a single element from "my_container".

    Parameters
    ----------
    my_container : Container
        Container to be sampled from.

    Returns
    -------
    hashable
        Random element from the given container.
    """
    return sample(my_container, 1)[0]
