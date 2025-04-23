import abc
import asyncio
import collections.abc
import aiohttp.client_exceptions
import bz2
import copy
import collections
import dataclasses
import dill
import enum
import glob
import gzip
import itertools
import io
import inspect
import importlib
import joblib
import json
import lzma
import multiprocess
import os
import posixpath
import struct
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import queue
import re
import regex
import sys
import shutil
import tarfile
import math
import warnings
import xxhash
import xml.dom.minidom
import time
import requests
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Sequence as Sequence_
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime
from functools import total_ordering, partial, reduce
from glob import has_magic
from io import BytesIO
from importlib import import_module
from itertools import chain, groupby, islice
from operator import mul
from pathlib import Path, PurePath, PurePosixPath
from shutil import disk_usage
from typing import Optional, Union, Any, ClassVar, Callable, Iterable, TypeVar, Generator, Generic, Iterator
from types import FunctionType
from packaging import version
from urllib.parse import urlparse
from functools import wraps
from xml.etree import ElementTree as ET
from unittest.mock import patch

from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype

import fsspec
import huggingface_hub
import numpy as np
import multiprocessing
import pandas as pd
import PIL.Image
import PIL.ImageOps
import scipy.io as sio
import torch
import torch.utils.data
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.core import strip_protocol, url_to_fs
from fsspec.implementations.local import LocalFileSystem
from fsspec.utils import can_be_local
from filelock import FileLock as FileLock_
from filelock import UnixFileLock
from filelock import __version__ as _filelock_version
from fsspec.core import url_to_fs
from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub import HfFileSystem
from huggingface_hub.utils import get_session
from huggingface_hub.utils import insecure_hashlib
from huggingface_hub.utils import EntryNotFoundError
from multiprocessing import Pool, RLock
from multiprocessing import context
from tqdm.auto import tqdm as old_tqdm
from tqdm.contrib.concurrent import thread_map

import data_config

class BaseCompressedFileFileSystem(AbstractArchiveFileSystem):
    
    root_marker = ""
    protocol: str = (
        None  # protocol passed in prefix to the url. ex: "gzip", for gzip://file.txt::http://foo.bar/file.txt.gz
    )
    compression: str = None  # compression type in fsspec. ex: "gzip"
    extension: str = None  # extension of the filename to strip. ex: "".gz" to get file.txt from file.txt.gz

    def __init__(
        self, fo: str = "", target_protocol: Optional[str] = None, target_options: Optional[dict] = None, **kwargs
    ):
        super().__init__(self, **kwargs)
        self.fo = fo.__fspath__() if hasattr(fo, "__fspath__") else fo
        self._open_with_fsspec = partial(
            fsspec.open,
            self.fo,
            mode="rb",
            protocol=target_protocol,
            compression=self.compression,
            client_kwargs={
                "requote_redirect_url": False,  # see https://github.com/huggingface/datasets/pull/5459
                "trust_env": True,  # Enable reading proxy env variables.
                **(target_options or {}).pop("client_kwargs", {}),  # To avoid issues if it was already passed.
            },
            **(target_options or {}),
        )
        self.compressed_name = os.path.basename(self.fo.split("::")[0])
        self.uncompressed_name = (
            self.compressed_name[: self.compressed_name.rindex(".")]
            if "." in self.compressed_name
            else self.compressed_name
        )
        self.dir_cache = None

    @classmethod
    def _strip_protocol(cls, path):
        # compressed file paths are always relative to the archive root
        return super()._strip_protocol(path).lstrip("/")

    def _get_dirs(self):
        if self.dir_cache is None:
            f = {**self._open_with_fsspec().fs.info(self.fo), "name": self.uncompressed_name}
            self.dir_cache = {f["name"]: f}

    def cat(self, path: str):
        with self._open_with_fsspec().open() as f:
            return f.read()

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        path = self._strip_protocol(path)
        if mode != "rb":
            raise ValueError(f"Tried to read with mode {mode} on file {self.fo} opened with mode 'rb'")
        return self._open_with_fsspec().open()

class Bz2FileSystem(BaseCompressedFileFileSystem):
    protocol = "bz2"
    compression = "bz2"
    extension = ".bz2"

class GzipFileSystem(BaseCompressedFileFileSystem):
    protocol = "gzip"
    compression = "gzip"
    extension = ".gz"

class Lz4FileSystem(BaseCompressedFileFileSystem):
    protocol = "lz4"
    compression = "lz4"
    extension = ".lz4"

class XzFileSystem(BaseCompressedFileFileSystem):
    protocol = "xz"
    compression = "xz"
    extension = ".xz"

class ZstdFileSystem(BaseCompressedFileFileSystem):
    protocol = "zstd"
    compression = "zstd"
    extension = ".zst"

COMPRESSION_FILESYSTEMS: list[BaseCompressedFileFileSystem] = [
    Bz2FileSystem,
    GzipFileSystem,
    Lz4FileSystem,
    XzFileSystem,
    ZstdFileSystem,
]
T = TypeVar("T")
Y = TypeVar("Y")
P = TypeVar("P", str, Path)
Key = Union[int, str]
RowFormat = TypeVar("RowFormat")
ColumnFormat = TypeVar("ColumnFormat")
BatchFormat = TypeVar("BatchFormat")
type_ = type

__version__ = "3.5.0"
Manager = context._default_context.Manager
_VERSION_REG = re.compile(r"^(?P<major>\d+)" r"\.(?P<minor>\d+)" r"\.(?P<patch>\d+)$")
SingleOriginMetadata = Union[tuple[str, str], tuple[str], tuple[()]]
SINGLE_SLASH_AFTER_PROTOCOL_PATTERN = re.compile(r"(?<!:):/")
_split_re = r"^\w+(\.\w+)*$"
_uppercase_uppercase_re = re.compile(r"([A-Z]+)([A-Z][a-z])")
_lowercase_uppercase_re = re.compile(r"([a-z\d])([A-Z])")
_single_underscore_re = re.compile(r"(?<!_)_(?!_)")
_multiple_underscores_re = re.compile(r"(_{2,})")
_hf_datasets_progress_bars_disabled: bool = data_config.HF_DATASETS_DISABLE_PROGRESS_BARS or False
_FIELDS = '__dataclass_fields__'
_IMAGE_COMPRESSION_FORMATS: Optional[list[str]] = None
_NATIVE_BYTEORDER = "<" if sys.byteorder == "little" else ">"
_VALID_IMAGE_ARRAY_DTPYES = [
    np.dtype("|b1"),
    np.dtype("|u1"),
    np.dtype("<u2"),
    np.dtype(">u2"),
    np.dtype("<i2"),
    np.dtype(">i2"),
    np.dtype("<u4"),
    np.dtype(">u4"),
    np.dtype("<i4"),
    np.dtype(">i4"),
    np.dtype("<f4"),
    np.dtype(">f4"),
    np.dtype("<f8"),
    np.dtype(">f8"),
]
FILES_TO_IGNORE = [
    "README.md",
    "config.json",
    "dataset_info.json",
    "dataset_infos.json",
    "dummy_data.zip",
    "dataset_dict.json",
]

_SUB_SPEC_RE = re.compile(
    rf"""
^
 (?P<split>{_split_re[1:-1]})
 (\[
    ((?P<from>-?\d+)
     (?P<from_pct>%)?)?
    :
    ((?P<to>-?\d+)
     (?P<to_pct>%)?)?
 \])?(\((?P<rounding>[^\)]*)\))?
$
""",  # remove ^ and $
    re.X,
)

_ADDITION_SEP_RE = re.compile(r"\s*\+\s*")
BASE_KNOWN_EXTENSIONS = [
    "txt",
    "csv",
    "json",
    "jsonl",
    "tsv",
    "conll",
    "conllu",
    "orig",
    "parquet",
    "pkl",
    "pickle",
    "rel",
    "xml",
    "arrow",
]
COMPRESSION_EXTENSION_TO_PROTOCOL = {
    # single file compression
    **{fs_class.extension.lstrip("."): fs_class.protocol for fs_class in COMPRESSION_FILESYSTEMS},
    # archive compression
    "zip": "zip",
}
SINGLE_FILE_COMPRESSION_EXTENSION_TO_PROTOCOL = {
    fs_class.extension.lstrip("."): fs_class.protocol for fs_class in COMPRESSION_FILESYSTEMS
}
SINGLE_FILE_COMPRESSION_PROTOCOLS = {fs_class.protocol for fs_class in COMPRESSION_FILESYSTEMS}
MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL = {
    bytes.fromhex("504B0304"): "zip",
    bytes.fromhex("504B0506"): "zip",  # empty archive
    bytes.fromhex("504B0708"): "zip",  # spanned archive
    bytes.fromhex("425A68"): "bz2",
    bytes.fromhex("1F8B"): "gzip",
    bytes.fromhex("FD377A585A00"): "xz",
    bytes.fromhex("04224D18"): "lz4",
    bytes.fromhex("28B52FFD"): "zstd",
}
MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL = {b"Rar!": "rar"}
MAGIC_NUMBER_MAX_LENGTH = max(len(magic_number) for magic_number in chain(MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL, MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL))

def no_op_if_value_is_null(func):
    def wrapper(value):
        return func(value) if value is not None else None

    return wrapper

@dataclass
class Sequence:
    feature: Any
    length: int = -1
    id: Optional[str] = None
    dtype: ClassVar[str] = "list"
    pa_type: ClassVar[Any] = None
    _type: str = field(default="Sequence", init=False, repr=False)

@dataclass
class LargeList:
    feature: Any
    id: Optional[str] = None
    pa_type: ClassVar[Any] = None
    _type: str = field(default="LargeList", init=False, repr=False)

@dataclass
class Value:
    dtype: str
    id: Optional[str] = None
    pa_type: ClassVar[Any] = None
    _type: str = field(default="Value", init=False, repr=False)

    def __post_init__(self):
        if self.dtype == "double":  # fix inferred type
            self.dtype = "float64"
        if self.dtype == "float":  # fix inferred type
            self.dtype = "float32"
        self.pa_type = string_to_arrow(self.dtype)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value):
        if pa.types.is_boolean(self.pa_type):
            return bool(value)
        elif pa.types.is_integer(self.pa_type):
            return int(value)
        elif pa.types.is_floating(self.pa_type):
            return float(value)
        elif pa.types.is_string(self.pa_type):
            return str(value)
        else:
            return value

class _ArrayXD:
    def __post_init__(self):
        self.shape = tuple(self.shape)

    def __call__(self):
        pa_type = globals()[self.__class__.__name__ + "ExtensionType"](self.shape, self.dtype)
        return pa_type

    def encode_example(self, value):
        return value

@dataclass
class Array2D(_ArrayXD):
    shape: tuple
    dtype: str
    id: Optional[str] = None
    _type: str = field(default="Array2D", init=False, repr=False)

@dataclass
class Array3D(_ArrayXD):
    shape: tuple
    dtype: str
    id: Optional[str] = None
    _type: str = field(default="Array3D", init=False, repr=False)

@dataclass
class Array4D(_ArrayXD):
    shape: tuple
    dtype: str
    id: Optional[str] = None
    _type: str = field(default="Array4D", init=False, repr=False)

@dataclass
class Array5D(_ArrayXD):
    shape: tuple
    dtype: str
    id: Optional[str] = None
    _type: str = field(default="Array5D", init=False, repr=False)

@dataclass
class Image:
    mode: Optional[str] = None
    decode: bool = True
    id: Optional[str] = None
    dtype: ClassVar[str] = "PIL.Image.Image"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="Image", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, bytes, dict, np.ndarray, "PIL.Image.Image"]) -> dict:
        if isinstance(value, list):
            value = np.array(value)

        if isinstance(value, str):
            return {"path": value, "bytes": None}
        elif isinstance(value, bytes):
            return {"path": None, "bytes": value}
        elif isinstance(value, np.ndarray):
            # convert the image array to PNG/TIFF bytes
            return encode_np_array(value)
        elif isinstance(value, PIL.Image.Image):
            # convert the PIL image to bytes (default format is PNG/TIFF)
            return encode_pil_image(value)
        elif value.get("path") is not None and os.path.isfile(value["path"]):
            # we set "bytes": None to not duplicate the data if they're already available locally
            return {"bytes": None, "path": value.get("path")}
        elif value.get("bytes") is not None or value.get("path") is not None:
            # store the image bytes, and path is used to infer the image format using the file extension
            return {"bytes": value.get("bytes"), "path": value.get("path")}
        else:
            raise ValueError(f"An image sample should have one of 'path' or 'bytes' but they are missing or None in {value}.")

    def decode_example(self, value: dict, token_per_repo_id=None) -> "PIL.Image.Image":
        if not self.decode:
            raise RuntimeError("Decoding is disabled for this feature. Please use Image(decode=True) instead.")

        if token_per_repo_id is None:
            token_per_repo_id = {}

        path, bytes_ = value["path"], value["bytes"]
        if bytes_ is None:
            if path is None:
                raise ValueError(f"An image should have one of 'path' or 'bytes' but both are None in {value}.")
            else:
                if is_local_path(path):
                    image = PIL.Image.open(path)
                else:
                    source_url: str = path.split("::")[-1]
                    pattern = (data_config.HUB_DATASETS_URL if source_url.startswith(data_config.HF_ENDPOINT) else data_config.HUB_DATASETS_HFFS_URL)
                    source_url_fields = string_to_dict(source_url, pattern)
                    token = (token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None)
                    download_config = DownloadConfig(token=token)
                    with xopen(path, "rb", download_config=download_config) as f:
                        bytes_ = BytesIO(f.read())
                    image = PIL.Image.open(bytes_)
        else:
            image = PIL.Image.open(BytesIO(bytes_))
        image.load()  # to avoid "Too many open files" errors
        if image.getexif().get(PIL.Image.ExifTags.Base.Orientation) is not None:
            image = PIL.ImageOps.exif_transpose(image)
        if self.mode and self.mode != image.mode:
            image = image.convert(self.mode)
        return image

    def flatten(self) -> Union["FeatureType", dict[str, "FeatureType"]]:
        return (
            self
            if self.decode
            else {
                "bytes": Value("binary"),
                "path": Value("string"),
            }
        )

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray, pa.ListArray]) -> pa.StructArray:
        if pa.types.is_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
            storage = pa.StructArray.from_arrays([bytes_array, storage], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_list(storage.type):
            bytes_array = pa.array(
                [encode_np_array(np.array(arr))["bytes"] if arr is not None else None for arr in storage.to_pylist()],
                type=pa.binary(),
            )
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null()
            )
        return array_cast(storage, self.pa_type)

    def embed_storage(self, storage: pa.StructArray) -> pa.StructArray:
        @no_op_if_value_is_null
        def path_to_bytes(path):
            with xopen(path, "rb") as f:
                bytes_ = f.read()
            return bytes_

        bytes_array = pa.array([(path_to_bytes(x["path"]) if x["bytes"] is None else x["bytes"]) if x is not None else None for x in storage.to_pylist()], type=pa.binary())
        path_array = pa.array([os.path.basename(path) if path is not None else None for path in storage.field("path").to_pylist()], type=pa.string())
        storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null())
        return array_cast(storage, self.pa_type)

def keep_features_dicts_synced(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            self: "Features" = args[0]
            args = args[1:]
        else:
            self: "Features" = kwargs.pop("self")
        out = func(self, *args, **kwargs)
        assert hasattr(self, "_column_requires_decoding")
        self._column_requires_decoding = {col: require_decoding(feature) for col, feature in self.items()}
        return out

    wrapper._decorator_name_ = "_keep_dicts_synced"
    return wrapper

class Features(dict):
    def __init__(*args, **kwargs):
        if not args:
            raise TypeError("descriptor '__init__' of 'Features' object needs an argument")
        self, *args = args
        super(Features, self).__init__(*args, **kwargs)
        self._column_requires_decoding: dict[str, bool] = { # type: ignore
            col: require_decoding(feature) for col, feature in self.items()}

    __setitem__ = keep_features_dicts_synced(dict.__setitem__)
    __delitem__ = keep_features_dicts_synced(dict.__delitem__)
    update = keep_features_dicts_synced(dict.update)
    setdefault = keep_features_dicts_synced(dict.setdefault)
    pop = keep_features_dicts_synced(dict.pop)
    popitem = keep_features_dicts_synced(dict.popitem)
    clear = keep_features_dicts_synced(dict.clear)

    def __reduce__(self):
        return Features, (dict(self),)

    @property
    def type(self):
        return get_nested_type(self)

    @property
    def arrow_schema(self):
        hf_metadata = {"info": {"features": self.to_dict()}}
        return pa.schema(self.type).with_metadata({"huggingface": json.dumps(hf_metadata)})

    @classmethod
    def from_arrow_schema(cls, pa_schema: pa.Schema) -> "Features":
        metadata_features = Features()
        if pa_schema.metadata is not None and b"huggingface" in pa_schema.metadata:
            metadata = json.loads(pa_schema.metadata[b"huggingface"].decode())
            if "info" in metadata and "features" in metadata["info"] and metadata["info"]["features"] is not None:
                metadata_features = Features.from_dict(metadata["info"]["features"])
        metadata_features_schema = metadata_features.arrow_schema
        obj = {
            field.name: (
                metadata_features[field.name]
                if field.name in metadata_features and metadata_features_schema.field(field.name) == field
                else generate_from_arrow_type(field.type)
            )
            for field in pa_schema
        }
        return cls(**obj)

    @classmethod
    def from_dict(cls, dic) -> "Features":
        obj = generate_from_dict(dic)
        return cls(**obj)

    def to_dict(self):
        return asdict(self)

    def _to_yaml_list(self) -> list:
        yaml_data = self.to_dict()

        def simplify(feature: dict) -> dict:
            if not isinstance(feature, dict):
                raise TypeError(f"Expected a dict but got a {type(feature)}: {feature}")

            for list_type in ["large_list", "list", "sequence"]:
                if isinstance(feature.get(list_type), dict) and list(feature[list_type]) == ["dtype"]:
                    feature[list_type] = feature[list_type]["dtype"]
                if isinstance(feature.get(list_type), dict) and list(feature[list_type]) == ["struct"]:
                    feature[list_type] = feature[list_type]["struct"]
            if isinstance(feature.get("class_label"), dict) and isinstance(feature["class_label"].get("names"), list):
                feature["class_label"]["names"] = {str(label_id): label_name for label_id, label_name in enumerate(feature["class_label"]["names"])}
            return feature

        def to_yaml_inner(obj: Union[dict, list]) -> dict:
            if isinstance(obj, dict):
                _type = obj.pop("_type", None)
                if _type == "LargeList":
                    _feature = obj.pop("feature")
                    return simplify({"large_list": to_yaml_inner(_feature), **obj})
                elif _type == "Sequence":
                    _feature = obj.pop("feature")
                    return simplify({"sequence": to_yaml_inner(_feature), **obj})
                elif _type == "Value":
                    return obj
                elif _type and not obj:
                    return {"dtype": camelcase_to_snakecase(_type)}
                elif _type:
                    return {"dtype": simplify({camelcase_to_snakecase(_type): obj})}
                else:
                    return {"struct": [{"name": name, **to_yaml_inner(_feature)} for name, _feature in obj.items()]}
            elif isinstance(obj, list):
                return simplify({"list": simplify(to_yaml_inner(obj[0]))})
            elif isinstance(obj, tuple):
                return to_yaml_inner(list(obj))
            else:
                raise TypeError(f"Expected a dict or a list but got {type(obj)}: {obj}")

        def to_yaml_types(obj: dict) -> dict:
            if isinstance(obj, dict):
                return {k: to_yaml_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_yaml_types(v) for v in obj]
            elif isinstance(obj, tuple):
                return to_yaml_types(list(obj))
            else:
                return obj

        return to_yaml_types(to_yaml_inner(yaml_data)["struct"])

    @classmethod
    def _from_yaml_list(cls, yaml_data: list) -> "Features":
        yaml_data = copy.deepcopy(yaml_data)

        def unsimplify(feature: dict) -> dict:
            if not isinstance(feature, dict):
                raise TypeError(f"Expected a dict but got a {type(feature)}: {feature}")

            for list_type in ["large_list", "list", "sequence"]:
                if isinstance(feature.get(list_type), str):
                    feature[list_type] = {"dtype": feature[list_type]}

            if isinstance(feature.get("class_label"), dict) and isinstance(feature["class_label"].get("names"), dict):
                label_ids = sorted(feature["class_label"]["names"], key=int)
                if label_ids and [int(label_id) for label_id in label_ids] != list(range(int(label_ids[-1]) + 1)):
                    raise ValueError(f"ClassLabel expected a value for all label ids [0:{int(label_ids[-1]) + 1}] but some ids are missing.")
                feature["class_label"]["names"] = [feature["class_label"]["names"][label_id] for label_id in label_ids]
            return feature

        def from_yaml_inner(obj: Union[dict, list]) -> Union[dict, list]:
            if isinstance(obj, dict):
                if not obj:
                    return {}
                _type = next(iter(obj))
                if _type == "large_list":
                    _feature = unsimplify(obj).pop(_type)
                    return {"feature": from_yaml_inner(_feature), **obj, "_type": "LargeList"}
                if _type == "sequence":
                    _feature = unsimplify(obj).pop(_type)
                    return {"feature": from_yaml_inner(_feature), **obj, "_type": "Sequence"}
                if _type == "list":
                    return [from_yaml_inner(unsimplify(obj)[_type])]
                if _type == "struct":
                    return from_yaml_inner(obj["struct"])
                elif _type == "dtype":
                    if isinstance(obj["dtype"], str):
                        try:
                            Value(obj["dtype"])
                            return {**obj, "_type": "Value"}
                        except ValueError:
                            return {"_type": snakecase_to_camelcase(obj["dtype"])}
                    else:
                        return from_yaml_inner(obj["dtype"])
                else:
                    return {"_type": snakecase_to_camelcase(_type), **unsimplify(obj)[_type]}
            elif isinstance(obj, list):
                names = [_feature.pop("name") for _feature in obj]
                return {name: from_yaml_inner(_feature) for name, _feature in zip(names, obj)}
            else:
                raise TypeError(f"Expected a dict or a list but got {type(obj)}: {obj}")

        return cls.from_dict(from_yaml_inner(yaml_data))

    def encode_example(self, example):
        return encode_nested_example(self, cast_to_python_objects(example))

    def encode_column(self, column, column_name: str):
        column = cast_to_python_objects(column)
        return [encode_nested_example(self[column_name], obj, level=1) for obj in column]

    def encode_batch(self, batch):
        encoded_batch = {}
        if set(batch) != set(self):
            raise ValueError(f"Column mismatch between batch {set(batch)} and features {set(self)}")
        for key, column in batch.items():
            column = cast_to_python_objects(column)
            encoded_batch[key] = [encode_nested_example(self[key], obj, level=1) for obj in column]
        return encoded_batch

    def decode_example(self, example: dict, token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None):
        return {
            column_name: decode_nested_example(feature, value, token_per_repo_id=token_per_repo_id)
            if self._column_requires_decoding[column_name]
            else value
            for column_name, (feature, value) in zip_dict(
                {key: value for key, value in self.items() if key in example}, example
            )
        }

    def decode_column(self, column: list, column_name: str):
        return (
            [decode_nested_example(self[column_name], value) if value is not None else None for value in column]
            if self._column_requires_decoding[column_name]
            else column
        )

    def decode_batch(self, batch: dict, token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None):
        decoded_batch = {}
        for column_name, column in batch.items():
            decoded_batch[column_name] = (
                [
                    decode_nested_example(self[column_name], value, token_per_repo_id=token_per_repo_id)
                    if value is not None
                    else None
                    for value in column
                ]
                if self._column_requires_decoding[column_name]
                else column
            )
        return decoded_batch

    def copy(self) -> "Features":
        return copy.deepcopy(self)

    def reorder_fields_as(self, other: "Features") -> "Features":

        def recursive_reorder(source, target, stack=""):
            stack_position = " at " + stack[1:] if stack else ""
            if isinstance(target, Sequence):
                target = target.feature
                if isinstance(target, dict):
                    target = {k: [v] for k, v in target.items()}
                else:
                    target = [target]
            if isinstance(source, Sequence):
                sequence_kwargs = vars(source).copy()
                source = sequence_kwargs.pop("feature")
                if isinstance(source, dict):
                    source = {k: [v] for k, v in source.items()}
                    reordered = recursive_reorder(source, target, stack)
                    return Sequence({k: v[0] for k, v in reordered.items()}, **sequence_kwargs)
                else:
                    source = [source]
                    reordered = recursive_reorder(source, target, stack)
                    return Sequence(reordered[0], **sequence_kwargs)
            elif isinstance(source, dict):
                if not isinstance(target, dict):
                    raise ValueError(f"Type mismatch: between {source} and {target}" + stack_position)
                if sorted(source) != sorted(target):
                    raise ValueError(
                        f"Keys mismatch: between {source} (source) and {target} (target).\n"
                        f"{source.keys() - target.keys()} are missing from target "
                        f"and {target.keys() - source.keys()} are missing from source" + stack_position
                    )
                return {key: recursive_reorder(source[key], target[key], stack + f".{key}") for key in target}
            elif isinstance(source, list):
                if not isinstance(target, list):
                    raise ValueError(f"Type mismatch: between {source} and {target}" + stack_position)
                if len(source) != len(target):
                    raise ValueError(f"Length mismatch: between {source} and {target}" + stack_position)
                return [recursive_reorder(source[i], target[i], stack + ".<list>") for i in range(len(target))]
            elif isinstance(source, LargeList):
                if not isinstance(target, LargeList):
                    raise ValueError(f"Type mismatch: between {source} and {target}" + stack_position)
                return LargeList(recursive_reorder(source.feature, target.feature, stack))
            else:
                return source

        return Features(recursive_reorder(self, other))

    def flatten(self, max_depth=16) -> "Features":
        for depth in range(1, max_depth):
            no_change = True
            flattened = self.copy()
            for column_name, subfeature in self.items():
                if isinstance(subfeature, dict):
                    no_change = False
                    flattened.update({f"{column_name}.{k}": v for k, v in subfeature.items()})
                    del flattened[column_name]
                elif isinstance(subfeature, Sequence) and isinstance(subfeature.feature, dict):
                    no_change = False
                    flattened.update(
                        {
                            f"{column_name}.{k}": Sequence(v) if not isinstance(v, dict) else [v]
                            for k, v in subfeature.feature.items()
                        }
                    )
                    del flattened[column_name]
                elif hasattr(subfeature, "flatten") and subfeature.flatten() != subfeature:
                    no_change = False
                    flattened.update({f"{column_name}.{k}": v for k, v in subfeature.flatten().items()})
                    del flattened[column_name]
            self = flattened
            if no_change:
                break
        return self

class hf_tqdm(old_tqdm):
    def __init__(self, *args, **kwargs):
        if are_progress_bars_disabled():
            kwargs["disable"] = True
        super().__init__(*args, **kwargs)

    def __delattr__(self, attr: str) -> None:
        try:
            super().__delattr__(attr)
        except AttributeError:
            if attr != "_lock":
                raise

def are_progress_bars_disabled() -> bool:
    global _hf_datasets_progress_bars_disabled
    return _hf_datasets_progress_bars_disabled

@total_ordering
@dataclass
class Version:
    version_str: str
    description: Optional[str] = None
    major: Optional[Union[str, int]] = None
    minor: Optional[Union[str, int]] = None
    patch: Optional[Union[str, int]] = None

    def __post_init__(self):
        self.major, self.minor, self.patch = _str_to_version_tuple(self.version_str)

    def __repr__(self):
        return f"{self.tuple[0]}.{self.tuple[1]}.{self.tuple[2]}"

    @property
    def tuple(self):
        return self.major, self.minor, self.patch

    def _validate_operand(self, other):
        if isinstance(other, str):
            return Version(other)
        elif isinstance(other, Version):
            return other
        raise TypeError(f"{other} (type {type(other)}) cannot be compared to version.")

    def __eq__(self, other):
        try:
            other = self._validate_operand(other)
        except (TypeError, ValueError):
            return False
        else:
            return self.tuple == other.tuple

    def __lt__(self, other):
        other = self._validate_operand(other)
        return self.tuple < other.tuple

    def __hash__(self):
        return hash(_version_tuple_to_str(self.tuple))

    @classmethod
    def from_dict(cls, dic):
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dic.items() if k in field_names})

    def _to_yaml_string(self) -> str:
        return self.version_str

def _str_to_version_tuple(version_str):
    res = _VERSION_REG.match(version_str)
    if not res:
        raise ValueError(f"Invalid version '{version_str}'. Format should be x.y.z with {{x,y,z}} being digits.")
    return tuple(int(v) for v in [res.group("major"), res.group("minor"), res.group("patch")])

def _version_tuple_to_str(version_tuple):
    return ".".join(str(v) for v in version_tuple)

class InvalidConfigName(ValueError):
    pass

class NonStreamableDatasetError(Exception):
    pass

class Hasher:
    dispatch: dict = {}

    def __init__(self):
        self.m = xxhash.xxh64()

    @classmethod
    def hash_bytes(cls, value: Union[bytes, list[bytes]]) -> str:
        value = [value] if isinstance(value, bytes) else value
        m = xxhash.xxh64()
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    def hash(cls, value: Any) -> str:
        return cls.hash_bytes(dumps(value))

    def update(self, value: Any) -> None:
        header_for_update = f"=={type(value)}=="
        value_for_update = self.hash(value)
        self.m.update(header_for_update.encode("utf8"))
        self.m.update(value_for_update.encode("utf-8"))

    def hexdigest(self) -> str:
        return self.m.hexdigest()

class Pickler(dill.Pickler):
    dispatch = dill._dill.MetaCatchingDict(dill.Pickler.dispatch.copy())
    _legacy_no_dict_keys_sorting = False

    def save(self, obj, save_persistent_id=True):
        obj_type = type(obj)
        if obj_type not in self.dispatch:
            if obj_type is regex.Pattern:
                pklregister(obj_type)(_save_regexPattern)
            if "torch" in sys.modules:

                if issubclass(obj_type, torch.Tensor):
                    pklregister(obj_type)(_save_torchTensor)

                if obj_type is torch.Generator:
                    pklregister(obj_type)(_save_torchGenerator)

                if issubclass(obj_type, torch.nn.Module):
                    obj = getattr(obj, "_orig_mod", obj)
            
        if obj_type is FunctionType:
            obj = getattr(obj, "_torchdynamo_orig_callable", obj)
        dill.Pickler.save(self, obj, save_persistent_id=save_persistent_id)

    def _batch_setitems(self, items):
        if self._legacy_no_dict_keys_sorting:
            return super()._batch_setitems(items)
        try:
            items = sorted(items)
        except Exception:  # TypeError, decimal.InvalidOperation, etc.
            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        dill.Pickler._batch_setitems(self, items)

    def memoize(self, obj):
        if type(obj) is not str:  # noqa: E721
            dill.Pickler.memoize(self, obj)

def dump(obj, file):
    Pickler(file, recurse=True).dump(obj)

def dumps(obj):
    file = BytesIO()
    dump(obj, file)
    return file.getvalue()

def pklregister(t):
    def proxy(func):
        Pickler.dispatch[t] = func
        return func

    return proxy

def log(pickler, msg):
    dill._dill.logger.trace(pickler, msg)

def _save_regexPattern(pickler, obj):
    log(pickler, f"Re: {obj}")
    args = (obj.pattern, obj.flags)
    pickler.save_reduce(regex.compile, args, obj=obj)
    log(pickler, "# Re")

def _save_torchTensor(pickler, obj):
    def create_torchTensor(np_array, dtype=None):
        tensor = torch.from_numpy(np_array)
        if dtype:
            tensor = tensor.type(dtype)
        return tensor

    log(pickler, f"To: {obj}")
    if obj.dtype == torch.bfloat16:
        args = (obj.detach().to(torch.float).cpu().numpy(), torch.bfloat16)
    else:
        args = (obj.detach().cpu().numpy(),)
    pickler.save_reduce(create_torchTensor, args, obj=obj)
    log(pickler, "# To")

def _save_torchGenerator(pickler, obj):
    def create_torchGenerator(state):
        generator = torch.Generator()
        generator.set_state(state)
        return generator

    log(pickler, f"Ge: {obj}")
    args = (obj.get_state(),)
    pickler.save_reduce(create_torchGenerator, args, obj=obj)
    log(pickler, "# Ge")

@dataclass
class DownloadConfig:
    cache_dir: Optional[Union[str, Path]] = None
    force_download: bool = False
    resume_download: bool = False
    local_files_only: bool = False
    proxies: Optional[dict] = None
    user_agent: Optional[str] = None
    extract_compressed_file: bool = False
    force_extract: bool = False
    delete_extracted: bool = False
    extract_on_the_fly: bool = False
    use_etag: bool = True
    num_proc: Optional[int] = None
    max_retries: int = 1
    token: Optional[Union[str, bool]] = None
    storage_options: dict[str, Any] = field(default_factory=dict)
    download_desc: Optional[str] = None
    disable_tqdm: bool = False

    def copy(self) -> "DownloadConfig":
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})

    def __setattr__(self, name, value):
        if name == "token" and getattr(self, "storage_options", None) is not None:
            if "hf" not in self.storage_options:
                self.storage_options["hf"] = {"token": value, "endpoint": data_config.HF_ENDPOINT}
            elif getattr(self.storage_options["hf"], "token", None) is None:
                self.storage_options["hf"]["token"] = value
        super().__setattr__(name, value)

class DataFilesList(list[str]):

    def __init__(self, data_files: list[str], origin_metadata: list[SingleOriginMetadata]) -> None:
        super().__init__(data_files)
        self.origin_metadata = origin_metadata

    def __add__(self, other: "DataFilesList") -> "DataFilesList":
        return DataFilesList([*self, *other], self.origin_metadata + other.origin_metadata)

    @classmethod
    def from_hf_repo(
        cls,
        patterns: list[str],
        dataset_info: huggingface_hub.hf_api.DatasetInfo,
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = f"hf://datasets/{dataset_info.id}@{dataset_info.sha}/{base_path or ''}".rstrip("/")
        return cls.from_patterns(patterns, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config)

    @classmethod
    def from_local_or_remote(
        cls,
        patterns: list[str],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        return cls.from_patterns(patterns, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config)

    @classmethod
    def from_patterns(
        cls,
        patterns: list[str],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        data_files = []
        for pattern in patterns:
            try:
                data_files.extend(
                    resolve_pattern(
                        pattern,
                        base_path=base_path,
                        allowed_extensions=allowed_extensions,
                        download_config=download_config,
                    )
                )
            except FileNotFoundError:
                if not has_magic(pattern):
                    raise
        origin_metadata = _get_origin_metadata(data_files, download_config=download_config)
        return cls(data_files, origin_metadata)

    def filter(self, *, extensions: Optional[list[str]] = None, file_names: Optional[list[str]] = None) -> "DataFilesList":
        patterns = []
        if extensions:
            ext_pattern = "|".join(re.escape(ext) for ext in extensions)
            patterns.append(re.compile(f".*({ext_pattern})(\\..+)?$"))
        if file_names:
            fn_pattern = "|".join(re.escape(fn) for fn in file_names)
            patterns.append(re.compile(rf".*[\/]?({fn_pattern})$"))
        if patterns:
            return DataFilesList(
                [data_file for data_file in self if any(pattern.match(data_file) for pattern in patterns)],
                origin_metadata=self.origin_metadata,
            )
        else:
            return DataFilesList(list(self), origin_metadata=self.origin_metadata)

class DataFilesDict(dict[str, DataFilesList]):
    @classmethod
    def from_local_or_remote(
        cls,
        patterns: dict[str, Union[list[str], DataFilesList]],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesList)
                else DataFilesList.from_local_or_remote(
                    patterns_for_key,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    download_config=download_config,
                )
            )
        return out

    @classmethod
    def from_hf_repo(
        cls,
        patterns: dict[str, Union[list[str], DataFilesList]],
        dataset_info: huggingface_hub.hf_api.DatasetInfo,
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesList)
                else DataFilesList.from_hf_repo(
                    patterns_for_key,
                    dataset_info=dataset_info,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    download_config=download_config,
                )
            )
        return out

    @classmethod
    def from_patterns(
        cls,
        patterns: dict[str, Union[list[str], DataFilesList]],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesList)
                else DataFilesList.from_patterns(
                    patterns_for_key,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    download_config=download_config,
                )
            )
        return out

    def filter(self, *, extensions: Optional[list[str]] = None, file_names: Optional[list[str]] = None) -> "DataFilesDict":
        out = type(self)()
        for key, data_files_list in self.items():
            out[key] = data_files_list.filter(extensions=extensions, file_names=file_names)
        return out

class DataFilesPatternsList(list[str]):
    def __init__(self, patterns: list[str], allowed_extensions: list[Optional[list[str]]]):
        super().__init__(patterns)
        self.allowed_extensions = allowed_extensions

    def __add__(self, other):
        return DataFilesList([*self, *other], self.allowed_extensions + other.allowed_extensions)

    @classmethod
    def from_patterns(cls, patterns: list[str], allowed_extensions: Optional[list[str]] = None) -> "DataFilesPatternsList":
        return cls(patterns, [allowed_extensions] * len(patterns))

    def resolve(self, base_path: str, download_config: Optional[DownloadConfig] = None) -> "DataFilesList":
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        data_files = []
        for pattern, allowed_extensions in zip(self, self.allowed_extensions):
            try:
                data_files.extend(
                    resolve_pattern(
                        pattern,
                        base_path=base_path,
                        allowed_extensions=allowed_extensions,
                        download_config=download_config,
                    )
                )
            except FileNotFoundError:
                if not has_magic(pattern):
                    raise
        origin_metadata = _get_origin_metadata(data_files, download_config=download_config)
        return DataFilesList(data_files, origin_metadata)

    def filter_extensions(self, extensions: list[str]) -> "DataFilesPatternsList":
        return DataFilesPatternsList(self, [allowed_extensions + extensions for allowed_extensions in self.allowed_extensions])

class DataFilesPatternsDict(dict[str, DataFilesPatternsList]):
    @classmethod
    def from_patterns(
        cls, patterns: dict[str, list[str]], allowed_extensions: Optional[list[str]] = None
    ) -> "DataFilesPatternsDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesPatternsList)
                else DataFilesPatternsList.from_patterns(
                    patterns_for_key,
                    allowed_extensions=allowed_extensions,
                )
            )
        return out

    def resolve(self, base_path: str, download_config: Optional[DownloadConfig] = None) -> "DataFilesDict":
        out = DataFilesDict()
        for key, data_files_patterns_list in self.items():
            out[key] = data_files_patterns_list.resolve(base_path, download_config)
        return out

    def filter_extensions(self, extensions: list[str]) -> "DataFilesPatternsDict":
        out = type(self)()
        for key, data_files_patterns_list in self.items():
            out[key] = data_files_patterns_list.filter_extensions(extensions)
        return out

def xexists(urlpath: str, download_config: Optional[DownloadConfig] = None):
    main_hop, *rest_hops = _as_str(urlpath).split("::")
    if is_local_path(main_hop):
        return os.path.exists(main_hop)
    else:
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        main_hop, *rest_hops = urlpath.split("::")
        fs, *_ = url_to_fs(urlpath, **storage_options)
        return fs.exists(main_hop)

def xdirname(a):
    a, *b = str(a).split("::")
    if is_local_path(a):
        a = os.path.dirname(Path(a).as_posix())
    else:
        a = posixpath.dirname(a)
    if a.endswith(":"):
        a += "//"
    return "::".join([a] + b)

class xPath(type(Path())):
    def __str__(self):
        path_str = super().__str__()
        main_hop, *rest_hops = path_str.split("::")
        if is_local_path(main_hop):
            return main_hop
        path_as_posix = path_str.replace("\\", "/")
        path_as_posix = SINGLE_SLASH_AFTER_PROTOCOL_PATTERN.sub("://", path_as_posix)
        path_as_posix += "//" if path_as_posix.endswith(":") else ""  # Add slashes to root of the protocol
        return path_as_posix

    def exists(self, download_config: Optional[DownloadConfig] = None):
        return xexists(str(self), download_config=download_config)

    def glob(self, pattern, download_config: Optional[DownloadConfig] = None):
        posix_path = self.as_posix()
        main_hop, *rest_hops = posix_path.split("::")
        if is_local_path(main_hop):
            yield from Path(main_hop).glob(pattern)
        else:
            if rest_hops:
                urlpath = rest_hops[0]
                urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
                storage_options = {urlpath.split("://")[0]: storage_options}
                posix_path = "::".join([main_hop, urlpath, *rest_hops[1:]])
            else:
                storage_options = None
            fs, *_ = url_to_fs(xjoin(posix_path, pattern), **(storage_options or {}))
            globbed_paths = fs.glob(xjoin(main_hop, pattern))
            for globbed_path in globbed_paths:
                yield type(self)("::".join([f"{fs.protocol}://{globbed_path}"] + rest_hops))

    def rglob(self, pattern, **kwargs):
        return self.glob("**/" + pattern, **kwargs)

    @property
    def parent(self) -> "xPath":
        return type(self)(xdirname(self.as_posix()))

    @property
    def name(self) -> str:
        return PurePosixPath(self.as_posix().split("::")[0]).name

    @property
    def stem(self) -> str:
        return PurePosixPath(self.as_posix().split("::")[0]).stem

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.as_posix().split("::")[0]).suffix

    def open(self, *args, **kwargs):
        return xopen(str(self), *args, **kwargs)

    def joinpath(self, *p: tuple[str, ...]) -> "xPath":
        return type(self)(xjoin(self.as_posix(), *p))

    def __truediv__(self, p: str) -> "xPath":
        return self.joinpath(p)

    def with_suffix(self, suffix):
        main_hop, *rest_hops = str(self).split("::")
        if is_local_path(main_hop):
            return type(self)(str(super().with_suffix(suffix)))
        return type(self)("::".join([type(self)(PurePosixPath(main_hop).with_suffix(suffix)).as_posix()] + rest_hops))

def _prepare_single_hop_path_and_storage_options(urlpath: str, download_config: Optional[DownloadConfig] = None) -> tuple[str, dict[str, dict[str, Any]]]:
    token = None if download_config is None else download_config.token
    if urlpath.startswith(data_config.HF_ENDPOINT) and "/resolve/" in urlpath:
        urlpath = "hf://" + urlpath[len(data_config.HF_ENDPOINT) + 1 :].replace("/resolve/", "@", 1)
    protocol = urlpath.split("://")[0] if "://" in urlpath else "file"
    if download_config is not None and protocol in download_config.storage_options:
        storage_options = download_config.storage_options[protocol].copy()
    elif download_config is not None and protocol not in download_config.storage_options:
        storage_options = {
            option_name: option_value
            for option_name, option_value in download_config.storage_options.items()
            if option_name not in fsspec.available_protocols()
        }
    else:
        storage_options = {}
    if protocol in {"http", "https"}:
        client_kwargs = storage_options.pop("client_kwargs", {})
        storage_options["client_kwargs"] = {"trust_env": True, **client_kwargs}  # Enable reading proxy env variables
        if "drive.google.com" in urlpath:
            response = get_session().head(urlpath, timeout=10)
            for k, v in response.cookies.items():
                if k.startswith("download_warning"):
                    urlpath += "&confirm=" + v
                    cookies = response.cookies
                    storage_options = {"cookies": cookies, **storage_options}
            if "confirm=" not in urlpath:
                urlpath += "&confirm=t"
        if urlpath.startswith("https://raw.githubusercontent.com/"):
            headers = storage_options.pop("headers", {})
            storage_options["headers"] = {"Accept-Encoding": "identity", **headers}
    elif protocol == "hf":
        storage_options = {
            "token": token,
            "endpoint": data_config.HF_ENDPOINT,
            **storage_options,
        }
        if data_config.HF_HUB_VERSION < version.parse("0.21.0"):
            storage_options["block_size"] = "default"
    if storage_options:
        storage_options = {protocol: storage_options}
    return urlpath, storage_options

def _prepare_path_and_storage_options(urlpath: str, download_config: Optional[DownloadConfig] = None) -> tuple[str, dict[str, dict[str, Any]]]:
    prepared_urlpath = []
    prepared_storage_options = {}
    for hop in urlpath.split("::"):
        hop, storage_options = _prepare_single_hop_path_and_storage_options(hop, download_config=download_config)
        prepared_urlpath.append(hop)
        prepared_storage_options.update(storage_options)
    return "::".join(prepared_urlpath), storage_options

def _get_single_origin_metadata(data_file: str, download_config: Optional[DownloadConfig] = None) -> SingleOriginMetadata:
    data_file, storage_options = _prepare_path_and_storage_options(data_file, download_config=download_config)
    fs, *_ = url_to_fs(data_file, **storage_options)
    if isinstance(fs, HfFileSystem):
        resolved_path = fs.resolve_path(data_file)
        return resolved_path.repo_id, resolved_path.revision
    elif isinstance(fs, HTTPFileSystem) and data_file.startswith(data_config.HF_ENDPOINT):
        hffs = HfFileSystem(endpoint=data_config.HF_ENDPOINT, token=download_config.token)
        data_file = "hf://" + data_file[len(data_config.HF_ENDPOINT) + 1 :].replace("/resolve/", "@", 1)
        resolved_path = hffs.resolve_path(data_file)
        return resolved_path.repo_id, resolved_path.revision
    info = fs.info(data_file)
    for key in ["ETag", "etag", "mtime"]:
        if key in info:
            return (str(info[key]),)
    return ()

def _get_origin_metadata(
    data_files: list[str],
    download_config: Optional[DownloadConfig] = None,
    max_workers: Optional[int] = None,
) -> list[SingleOriginMetadata]:
    max_workers = max_workers if max_workers is not None else data_config.HF_DATASETS_MULTITHREADING_MAX_WORKERS
    return thread_map(
        partial(_get_single_origin_metadata, download_config=download_config),
        data_files,
        max_workers=max_workers,
        tqdm_class=hf_tqdm,
        desc="Resolving data files",
        disable=len(data_files) <= 16 or None,
    )

def is_remote_url(url_or_filename: str) -> bool:
    return urlparse(url_or_filename).scheme != "" and not os.path.ismount(urlparse(url_or_filename).scheme + ":/")

def is_relative_path(url_or_filename: str) -> bool:
    return urlparse(url_or_filename).scheme == "" and not os.path.isabs(url_or_filename)

def is_local_path(url_or_filename: str) -> bool:
    return urlparse(url_or_filename).scheme == "" or os.path.ismount(urlparse(url_or_filename).scheme + ":/")

def xjoin(a, *p):
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.join(a, *p)
    else:
        a = posixpath.join(a, *p)
        return "::".join([a] + b)

def xbasename(a):
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.basename(Path(a).as_posix())
    else:
        return posixpath.basename(a)

def _is_inside_unrequested_special_dir(matched_rel_path: str, pattern: str) -> bool:
    data_dirs_to_ignore_in_path = [part for part in PurePath(matched_rel_path).parent.parts if part.startswith("__")]
    data_dirs_to_ignore_in_pattern = [part for part in PurePath(pattern).parent.parts if part.startswith("__")]
    return len(data_dirs_to_ignore_in_path) != len(data_dirs_to_ignore_in_pattern)

def _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(matched_rel_path: str, pattern: str) -> bool:
    hidden_directories_in_path = [part for part in PurePath(matched_rel_path).parts if part.startswith(".") and not set(part) == {"."}]
    hidden_directories_in_pattern = [part for part in PurePath(pattern).parts if part.startswith(".") and not set(part) == {"."}]
    return len(hidden_directories_in_path) != len(hidden_directories_in_pattern)

def resolve_pattern(
    pattern: str,
    base_path: str,
    allowed_extensions: Optional[list[str]] = None,
    download_config: Optional[DownloadConfig] = None,
) -> list[str]:
    if is_relative_path(pattern):
        pattern = xjoin(base_path, pattern)
    elif is_local_path(pattern):
        base_path = os.path.splitdrive(pattern)[0] + os.sep
    else:
        base_path = ""
    pattern, storage_options = _prepare_path_and_storage_options(pattern, download_config=download_config)
    fs, fs_pattern = url_to_fs(pattern, **storage_options)
    files_to_ignore = set(FILES_TO_IGNORE) - {xbasename(pattern)}
    protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
    protocol_prefix = protocol + "://" if protocol != "file" else ""
    glob_kwargs = {}
    if protocol == "hf" and data_config.HF_HUB_VERSION >= version.parse("0.20.0"):
        glob_kwargs["expand_info"] = False
    matched_paths = [
        filepath if filepath.startswith(protocol_prefix) else protocol_prefix + filepath
        for filepath, info in fs.glob(pattern, detail=True, **glob_kwargs).items()
        if (info["type"] == "file" or (info.get("islink") and os.path.isfile(os.path.realpath(filepath))))
        and (xbasename(filepath) not in files_to_ignore)
        and not _is_inside_unrequested_special_dir(filepath, fs_pattern)
        and not _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(filepath, fs_pattern)
    ]  # ignore .ipynb and __pycache__, but keep /../
    if allowed_extensions is not None:
        out = [
            filepath
            for filepath in matched_paths
            if any("." + suffix in allowed_extensions for suffix in xbasename(filepath).split(".")[1:])
        ]
        if len(out) < len(matched_paths):
            invalid_matched_files = list(set(matched_paths) - set(out))
            print(f"Some files matched the pattern '{pattern}' but don't have valid data file extensions: {invalid_matched_files}")
    else:
        out = matched_paths
    if not out:
        error_msg = f"Unable to find '{pattern}'"
        if allowed_extensions is not None:
            error_msg += f" with any supported extension {list(allowed_extensions)}"
        raise FileNotFoundError(error_msg)
    return out

def is_dataclass(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    return hasattr(cls, _FIELDS)

def asdict(obj):
    def _is_dataclass_instance(obj):
        return is_dataclass(obj) and not isinstance(obj, type)

    def _asdict_inner(obj):
        if _is_dataclass_instance(obj):
            result = {}
            for f in fields(obj):
                value = _asdict_inner(getattr(obj, f.name))
                if not f.init or value != f.default or f.metadata.get("include_in_asdict_even_if_is_default", False):
                    result[f.name] = value
            return result
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return type(obj)(*[_asdict_inner(v) for v in obj])
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_asdict_inner(v) for v in obj)
        elif isinstance(obj, dict):
            return {_asdict_inner(k): _asdict_inner(v) for k, v in obj.items()}
        else:
            return copy.deepcopy(obj)

    if not isinstance(obj, dict) and not _is_dataclass_instance(obj):
        raise TypeError(f"{obj} is not a dict or a dataclass")

    return _asdict_inner(obj)

def camelcase_to_snakecase(name):
    name = _uppercase_uppercase_re.sub(r"\1_\2", name)
    name = _lowercase_uppercase_re.sub(r"\1_\2", name)
    return name.lower()

def snakecase_to_camelcase(name):
    name = _single_underscore_re.split(name)
    name = [_multiple_underscores_re.split(n) for n in name]
    return "".join(n.capitalize() for n in itertools.chain.from_iterable(name) if n != "")

def string_to_arrow(datasets_dtype: str) -> pa.DataType:
    def _dtype_error_msg(dtype, pa_dtype, examples=None, urls=None):
        msg = f"{dtype} is not a validly formatted string representation of the pyarrow {pa_dtype} type."
        if examples:
            examples = ", ".join(examples[:-1]) + " or " + examples[-1] if len(examples) > 1 else examples[0]
            msg += f"\nValid examples include: {examples}."
        if urls:
            urls = ", ".join(urls[:-1]) + " and " + urls[-1] if len(urls) > 1 else urls[0]
            msg += f"\nFor more insformation, see: {urls}."
        return msg

    if datasets_dtype in pa.__dict__:
        return pa.__dict__[datasets_dtype]()

    if (datasets_dtype + "_") in pa.__dict__:
        return pa.__dict__[datasets_dtype + "_"]()

    timestamp_matches = re.search(r"^timestamp\[(.*)\]$", datasets_dtype)
    if timestamp_matches:
        timestamp_internals = timestamp_matches.group(1)
        internals_matches = re.search(r"^(s|ms|us|ns),\s*tz=([a-zA-Z0-9/_+\-:]*)$", timestamp_internals)
        if timestamp_internals in ["s", "ms", "us", "ns"]:
            return pa.timestamp(timestamp_internals)
        elif internals_matches:
            return pa.timestamp(internals_matches.group(1), internals_matches.group(2))
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "timestamp",
                    examples=["timestamp[us]", "timestamp[us, tz=America/New_York"],
                    urls=["https://arrow.apache.org/docs/python/generated/pyarrow.timestamp.html"],
                )
            )

    duration_matches = re.search(r"^duration\[(.*)\]$", datasets_dtype)
    if duration_matches:
        duration_internals = duration_matches.group(1)
        if duration_internals in ["s", "ms", "us", "ns"]:
            return pa.duration(duration_internals)
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "duration",
                    examples=["duration[s]", "duration[us]"],
                    urls=["https://arrow.apache.org/docs/python/generated/pyarrow.duration.html"],
                )
            )

    time_matches = re.search(r"^time(.*)\[(.*)\]$", datasets_dtype)
    if time_matches:
        time_internals_bits = time_matches.group(1)
        if time_internals_bits == "32":
            time_internals_unit = time_matches.group(2)
            if time_internals_unit in ["s", "ms"]:
                return pa.time32(time_internals_unit)
            else:
                raise ValueError(f"{time_internals_unit} is not a valid unit for the pyarrow time32 type. Supported units: s (second) and ms (millisecond).")
        elif time_internals_bits == "64":
            time_internals_unit = time_matches.group(2)
            if time_internals_unit in ["us", "ns"]:
                return pa.time64(time_internals_unit)
            else:
                raise ValueError(f"{time_internals_unit} is not a valid unit for the pyarrow time64 type. Supported units: us (microsecond) and ns (nanosecond).")
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "time",
                    examples=["time32[s]", "time64[us]"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.time32.html",
                        "https://arrow.apache.org/docs/python/generated/pyarrow.time64.html",
                    ],
                )
            )

    decimal_matches = re.search(r"^decimal(.*)\((.*)\)$", datasets_dtype)
    if decimal_matches:
        decimal_internals_bits = decimal_matches.group(1)
        if decimal_internals_bits == "128":
            decimal_internals_precision_and_scale = re.search(r"^(\d+),\s*(-?\d+)$", decimal_matches.group(2))
            if decimal_internals_precision_and_scale:
                precision = decimal_internals_precision_and_scale.group(1)
                scale = decimal_internals_precision_and_scale.group(2)
                return pa.decimal128(int(precision), int(scale))
            else:
                raise ValueError(
                    _dtype_error_msg(
                        datasets_dtype,
                        "decimal128",
                        examples=["decimal128(10, 2)", "decimal128(4, -2)"],
                        urls=["https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html"],
                    )
                )
        elif decimal_internals_bits == "256":
            decimal_internals_precision_and_scale = re.search(r"^(\d+),\s*(-?\d+)$", decimal_matches.group(2))
            if decimal_internals_precision_and_scale:
                precision = decimal_internals_precision_and_scale.group(1)
                scale = decimal_internals_precision_and_scale.group(2)
                return pa.decimal256(int(precision), int(scale))
            else:
                raise ValueError(
                    _dtype_error_msg(
                        datasets_dtype,
                        "decimal256",
                        examples=["decimal256(30, 2)", "decimal256(38, -4)"],
                        urls=["https://arrow.apache.org/docs/python/generated/pyarrow.decimal256.html"],
                    )
                )
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "decimal",
                    examples=["decimal128(12, 3)", "decimal256(40, 6)"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html",
                        "https://arrow.apache.org/docs/python/generated/pyarrow.decimal256.html",
                    ],
                )
            )

    raise ValueError(
        f"Neither {datasets_dtype} nor {datasets_dtype + '_'} seems to be a pyarrow data type. "
        f"Please make sure to use a correct data type, see: "
        f"https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions"
    )

def zip_dict(*dicts):
    for key in unique_values(itertools.chain(*dicts)):  # set merge all keys
        yield key, tuple(d[key] for d in dicts)

def unique_values(values):
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            yield value

def decode_nested_example(schema, obj, token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None):
    if isinstance(schema, dict):
        return (
            {k: decode_nested_example(sub_schema, sub_obj) for k, (sub_schema, sub_obj) in zip_dict(schema, obj)}
            if obj is not None
            else None
        )
    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if decode_nested_example(sub_schema, first_elmt) != first_elmt:
                    return [decode_nested_example(sub_schema, o) for o in obj]
            return list(obj)
    elif isinstance(schema, LargeList):
        if obj is None:
            return None
        else:
            sub_schema = schema.feature
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if decode_nested_example(sub_schema, first_elmt) != first_elmt:
                    return [decode_nested_example(sub_schema, o) for o in obj]
            return list(obj)
    elif isinstance(schema, Sequence):
        if isinstance(schema.feature, dict):
            return {k: decode_nested_example([schema.feature[k]], obj[k]) for k in schema.feature}
        else:
            return decode_nested_example([schema.feature], obj)
    elif hasattr(schema, "decode_example") and getattr(schema, "decode", True):
        return schema.decode_example(obj, token_per_repo_id=token_per_repo_id) if obj is not None else None
    return obj

FeatureType = Union[
    dict,
    list,
    tuple,
    Value,
    LargeList,
    Sequence,
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    Image,
]

_FEATURE_TYPES: dict[str, FeatureType] = {
    Value.__name__: Value,
    LargeList.__name__: LargeList,
    Sequence.__name__: Sequence,
    Array2D.__name__: Array2D,
    Array3D.__name__: Array3D,
    Array4D.__name__: Array4D,
    Array5D.__name__: Array5D,
    Image.__name__: Image
}

def _check_non_null_non_empty_recursive(obj, schema: Optional[FeatureType] = None) -> bool:
    if obj is None:
        return False
    elif isinstance(obj, (list, tuple)) and (schema is None or isinstance(schema, (list, tuple, LargeList, Sequence))):
        if len(obj) > 0:
            if schema is None:
                pass
            elif isinstance(schema, (list, tuple)):
                schema = schema[0]
            else:
                schema = schema.feature
            return _check_non_null_non_empty_recursive(obj[0], schema)
        else:
            return False
    else:
        return True

def image_to_bytes(image: "PIL.Image.Image") -> bytes:
    buffer = BytesIO()
    if image.format in list_image_compression_formats():
        format = image.format
    else:
        format = "PNG" if image.mode in ["1", "L", "LA", "RGB", "RGBA"] else "TIFF"
    image.save(buffer, format=format)
    return buffer.getvalue()

def list_image_compression_formats() -> list[str]:
    global _IMAGE_COMPRESSION_FORMATS
    if _IMAGE_COMPRESSION_FORMATS is None:
        PIL.Image.init()
        _IMAGE_COMPRESSION_FORMATS = list(set(PIL.Image.OPEN.keys()) & set(PIL.Image.SAVE.keys()))
    return _IMAGE_COMPRESSION_FORMATS

def encode_pil_image(image: "PIL.Image.Image") -> dict:
    if hasattr(image, "filename") and image.filename != "":
        return {"path": image.filename, "bytes": None}
    else:
        return {"path": None, "bytes": image_to_bytes(image)}

def encode_np_array(array: np.ndarray) -> dict:
    dtype = array.dtype
    dtype_byteorder = dtype.byteorder if dtype.byteorder != "=" else _NATIVE_BYTEORDER
    dtype_kind = dtype.kind
    dtype_itemsize = dtype.itemsize
    dest_dtype = None

    if array.shape[2:]:
        if dtype_kind not in ["u", "i"]:
            raise TypeError(f"Unsupported array dtype {dtype} for image encoding. Only {dest_dtype} is supported for multi-channel arrays.")
        dest_dtype = np.dtype("|u1")
    elif dtype in _VALID_IMAGE_ARRAY_DTPYES:
        dest_dtype = dtype
    else:  # Downcast the type within the kind (np.can_cast(from_type, to_type, casting="same_kind") doesn't behave as expected, so do it manually)
        while dtype_itemsize >= 1:
            dtype_str = dtype_byteorder + dtype_kind + str(dtype_itemsize)
            if np.dtype(dtype_str) in _VALID_IMAGE_ARRAY_DTPYES:
                dest_dtype = np.dtype(dtype_str)
                break
            else:
                dtype_itemsize //= 2
        if dest_dtype is None:
            raise TypeError(f"Cannot downcast dtype {dtype} to a valid image dtype. Valid image dtypes: {_VALID_IMAGE_ARRAY_DTPYES}")

    image = PIL.Image.fromarray(array.astype(dest_dtype))
    return {"path": None, "bytes": image_to_bytes(image)}

def string_to_dict(string: str, pattern: str) -> Optional[dict[str, str]]:
    pattern = re.sub(r"{([^:}]+)(?::[^}]+)?}", r"{\1}", pattern)  # remove format specifiers, e.g. {rank:05d} -> {rank}
    regex = re.sub(r"{(.+?)}", r"(?P<_\1>.+)", pattern)
    result = re.search(regex, string)
    if result is None:
        return None
    values = list(result.groups())
    keys = re.findall(r"{(.+?)}", pattern)
    _dict = dict(zip(keys, values))
    return _dict

def _as_str(path: Union[str, Path, xPath]):
    return str(path) if isinstance(path, xPath) else str(xPath(str(path)))

def xopen(file: str, mode="r", *args, download_config: Optional[DownloadConfig] = None, **kwargs):
    file_str = _as_str(file)
    main_hop, *rest_hops = file_str.split("::")
    if is_local_path(main_hop):
        kwargs.pop("block_size", None)
        return open(main_hop, mode, *args, **kwargs)
    file, storage_options = _prepare_path_and_storage_options(file_str, download_config=download_config)
    kwargs = {**kwargs, **(storage_options or {})}
    try:
        file_obj = fsspec.open(file, mode=mode, *args, **kwargs).open()
    except ValueError as e:
        if str(e) == "Cannot seek streaming HTTP file":
            raise NonStreamableDatasetError(
                "Streaming is not possible for this dataset because data host server doesn't support HTTP range "
                "requests. You can still load this dataset in non-streaming mode by passing `streaming=False` (default)"
            ) from e
        else:
            raise
    except FileNotFoundError:
        if file.startswith(data_config.HF_ENDPOINT):
            raise FileNotFoundError(file + "\nIf the repo is private or gated, make sure to log in with `huggingface-cli login`.") from None
        else:
            raise
    file_obj = _add_retries_to_file_obj_read_method(file_obj)
    return file_obj

def _add_retries_to_file_obj_read_method(file_obj):
    read = file_obj.read
    max_retries = data_config.STREAMING_READ_MAX_RETRIES

    def read_with_retries(*args, **kwargs):
        disconnect_err = None
        for retry in range(1, max_retries + 1):
            try:
                out = read(*args, **kwargs)
                break
            except (
                aiohttp.client_exceptions.ClientError,
                asyncio.TimeoutError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as err:
                disconnect_err = err
                time.sleep(data_config.STREAMING_READ_RETRY_INTERVAL)
        else:
            raise ConnectionError("Server Disconnected") from disconnect_err
        return out

    try:
        file_obj.read = read_with_retries
    except AttributeError:  # read-only attribute
        orig_file_obj = file_obj
        file_obj = io.RawIOBase()
        file_obj.read = read_with_retries
        file_obj.__getattr__ = lambda _, attr: getattr(orig_file_obj, attr)
    return file_obj

def _wrap_for_chunked_arrays(func):
    def wrapper(array, *args, **kwargs):
        if isinstance(array, pa.ChunkedArray):
            return pa.chunked_array([func(chunk, *args, **kwargs) for chunk in array.chunks])
        else:
            return func(array, *args, **kwargs)

    return wrapper

def _are_list_values_of_length(array: pa.ListArray, length: int) -> bool:
    return pc.all(pc.equal(array.value_lengths(), length)).as_py() or array.null_count == len(array)

def _storage_type(type: pa.DataType) -> pa.DataType:
    if isinstance(type, pa.ExtensionType):
        return _storage_type(type.storage_type)
    elif isinstance(type, pa.StructType):
        return pa.struct([pa.field(field.name, _storage_type(field.type)) for field in type])
    elif isinstance(type, pa.ListType):
        return pa.list_(_storage_type(type.value_type))
    elif isinstance(type, pa.FixedSizeListType):
        return pa.list_(_storage_type(type.value_type), type.list_size)
    return type

def _short_str(value: Any) -> str:
    out = str(value)
    if len(out) > 3000:
        out = out[:1500] + "\n...\n" + out[-1500:]
    return out

def _combine_list_array_offsets_with_mask(array: pa.ListArray) -> pa.Array:
    offsets = array.offsets
    if array.null_count > 0:
        offsets = pa.concat_arrays([pc.replace_with_mask(offsets[:-1], array.is_null(), pa.nulls(len(array), pa.int32())), offsets[-1:]])
    return offsets

@_wrap_for_chunked_arrays
def array_cast(array: pa.Array, pa_type: pa.DataType, allow_primitive_to_str: bool = True, allow_decimal_to_str: bool = True) -> Union[pa.Array, pa.FixedSizeListArray, pa.ListArray, pa.StructArray, pa.ExtensionArray]:
    _c = partial(array_cast, allow_primitive_to_str=allow_primitive_to_str, allow_decimal_to_str=allow_decimal_to_str)
    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if isinstance(pa_type, pa.ExtensionType):
        return pa_type.wrap_array(_c(array, pa_type.storage_type))
    elif array.type == pa_type:
        return array
    elif pa.types.is_struct(array.type):
        if pa.types.is_struct(pa_type) and ({field.name for field in pa_type} == {field.name for field in array.type}):
            if array.type.num_fields == 0:
                return array
            arrays = [_c(array.field(field.name), field.type) for field in pa_type]
            return pa.StructArray.from_arrays(arrays, fields=list(pa_type), mask=array.is_null())
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        if pa.types.is_fixed_size_list(pa_type):
            if _are_list_values_of_length(array, pa_type.list_size):
                if array.null_count > 0:
                    array_type = array.type
                    storage_type = _storage_type(array_type)
                    if array_type != storage_type:
                        array = _c(array, storage_type)
                        array = pc.list_slice(array, 0, pa_type.list_size, return_fixed_size_list=True)
                        array = _c(array, array_type)
                    else:
                        array = pc.list_slice(array, 0, pa_type.list_size, return_fixed_size_list=True)
                    array_values = array.values
                    return pa.FixedSizeListArray.from_arrays(_c(array_values, pa_type.value_type), pa_type.list_size, mask=array.is_null())
                else:
                    array_values = array.values[array.offset * pa_type.list_size : (array.offset + len(array)) * pa_type.list_size]
                    return pa.FixedSizeListArray.from_arrays(_c(array_values, pa_type.value_type), pa_type.list_size)
        elif pa.types.is_list(pa_type):
            array_offsets = _combine_list_array_offsets_with_mask(array)
            return pa.ListArray.from_arrays(array_offsets, _c(array.values, pa_type.value_type))
        elif pa.types.is_large_list(pa_type):
            array_offsets = _combine_list_array_offsets_with_mask(array)
            return pa.LargeListArray.from_arrays(array_offsets, _c(array.values, pa_type.value_type))
    elif pa.types.is_fixed_size_list(array.type):
        if pa.types.is_fixed_size_list(pa_type):
            if pa_type.list_size == array.type.list_size:
                array_values = array.values[array.offset * array.type.list_size : (array.offset + len(array)) * array.type.list_size]
                return pa.FixedSizeListArray.from_arrays(_c(array_values, pa_type.value_type), pa_type.list_size, mask=array.is_null())
        elif pa.types.is_list(pa_type):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.ListArray.from_arrays(array_offsets, _c(array.values, pa_type.value_type), mask=array.is_null())
        elif pa.types.is_large_list(pa_type):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.LargeListArray.from_arrays(array_offsets, _c(array.values, pa_type.value_type), mask=array.is_null())
    else:
        if pa.types.is_string(pa_type):
            if not allow_primitive_to_str and pa.types.is_primitive(array.type):
                raise TypeError(
                    f"Couldn't cast array of type {_short_str(array.type)} to {_short_str(pa_type)} "
                    f"since allow_primitive_to_str is set to {allow_primitive_to_str} "
                )
            if not allow_decimal_to_str and pa.types.is_decimal(array.type):
                raise TypeError(
                    f"Couldn't cast array of type {_short_str(array.type)} to {_short_str(pa_type)} "
                    f"and allow_decimal_to_str is set to {allow_decimal_to_str}"
                )
        if pa.types.is_null(pa_type) and not pa.types.is_null(array.type):
            raise TypeError(f"Couldn't cast array of type {_short_str(array.type)} to {_short_str(pa_type)}")
        return array.cast(pa_type)
    raise TypeError(f"Couldn't cast array of type {_short_str(array.type)} to {_short_str(pa_type)}")

def require_decoding(feature: FeatureType, ignore_decode_attribute: bool = False) -> bool:
    if isinstance(feature, dict):
        return any(require_decoding(f) for f in feature.values())
    elif isinstance(feature, (list, tuple)):
        return require_decoding(feature[0])
    elif isinstance(feature, LargeList):
        return require_decoding(feature.feature)
    elif isinstance(feature, Sequence):
        return require_decoding(feature.feature)
    else:
        return hasattr(feature, "decode_example") and (getattr(feature, "decode", True) if not ignore_decode_attribute else True)
    
def get_nested_type(schema: FeatureType) -> pa.DataType:
    if isinstance(schema, Features):
        return pa.struct({key: get_nested_type(schema[key]) for key in schema})  # Features is subclass of dict, and dict order is deterministic since Python 3.6
    elif isinstance(schema, dict):
        return pa.struct({key: get_nested_type(schema[key]) for key in schema})  # however don't sort on struct types since the order matters
    elif isinstance(schema, (list, tuple)):
        if len(schema) != 1:
            raise ValueError("When defining list feature, you should just provide one example of the inner type")
        value_type = get_nested_type(schema[0])
        return pa.list_(value_type)
    elif isinstance(schema, LargeList):
        value_type = get_nested_type(schema.feature)
        return pa.large_list(value_type)
    elif isinstance(schema, Sequence):
        value_type = get_nested_type(schema.feature)
        if isinstance(schema.feature, dict):
            data_type = pa.struct({f.name: pa.list_(f.type, schema.length) for f in value_type})
        else:
            data_type = pa.list_(value_type, schema.length)
        return data_type
    return schema()

def generate_from_arrow_type(pa_type: pa.DataType) -> FeatureType:
    if isinstance(pa_type, pa.StructType):
        return {field.name: generate_from_arrow_type(field.type) for field in pa_type}
    elif isinstance(pa_type, pa.FixedSizeListType):
        return Sequence(feature=generate_from_arrow_type(pa_type.value_type), length=pa_type.list_size)
    elif isinstance(pa_type, pa.ListType):
        feature = generate_from_arrow_type(pa_type.value_type)
        if isinstance(feature, (dict, tuple, list)):
            return [feature]
        return Sequence(feature=feature)
    elif isinstance(pa_type, pa.LargeListType):
        feature = generate_from_arrow_type(pa_type.value_type)
        return LargeList(feature=feature)
    elif isinstance(pa_type, _ArrayXDExtensionType):
        array_feature = [None, None, Array2D, Array3D, Array4D, Array5D][pa_type.ndims]
        return array_feature(shape=pa_type.shape, dtype=pa_type.value_type)
    elif isinstance(pa_type, pa.DataType):
        return Value(dtype=_arrow_to_datasets_dtype(pa_type))
    else:
        raise ValueError(f"Cannot convert {pa_type} to a Feature type.")
    
def _arrow_to_datasets_dtype(arrow_type: pa.DataType) -> str:
    if pa.types.is_null(arrow_type):
        return "null"
    elif pa.types.is_boolean(arrow_type):
        return "bool"
    elif pa.types.is_int8(arrow_type):
        return "int8"
    elif pa.types.is_int16(arrow_type):
        return "int16"
    elif pa.types.is_int32(arrow_type):
        return "int32"
    elif pa.types.is_int64(arrow_type):
        return "int64"
    elif pa.types.is_uint8(arrow_type):
        return "uint8"
    elif pa.types.is_uint16(arrow_type):
        return "uint16"
    elif pa.types.is_uint32(arrow_type):
        return "uint32"
    elif pa.types.is_uint64(arrow_type):
        return "uint64"
    elif pa.types.is_float16(arrow_type):
        return "float16"  # pyarrow dtype is "halffloat"
    elif pa.types.is_float32(arrow_type):
        return "float32"  # pyarrow dtype is "float"
    elif pa.types.is_float64(arrow_type):
        return "float64"  # pyarrow dtype is "double"
    elif pa.types.is_time32(arrow_type):
        return f"time32[{pa.type_for_alias(str(arrow_type)).unit}]"
    elif pa.types.is_time64(arrow_type):
        return f"time64[{pa.type_for_alias(str(arrow_type)).unit}]"
    elif pa.types.is_timestamp(arrow_type):
        if arrow_type.tz is None:
            return f"timestamp[{arrow_type.unit}]"
        elif arrow_type.tz:
            return f"timestamp[{arrow_type.unit}, tz={arrow_type.tz}]"
        else:
            raise ValueError(f"Unexpected timestamp object {arrow_type}.")
    elif pa.types.is_date32(arrow_type):
        return "date32"  # pyarrow dtype is "date32[day]"
    elif pa.types.is_date64(arrow_type):
        return "date64"  # pyarrow dtype is "date64[ms]"
    elif pa.types.is_duration(arrow_type):
        return f"duration[{arrow_type.unit}]"
    elif pa.types.is_decimal128(arrow_type):
        return f"decimal128({arrow_type.precision}, {arrow_type.scale})"
    elif pa.types.is_decimal256(arrow_type):
        return f"decimal256({arrow_type.precision}, {arrow_type.scale})"
    elif pa.types.is_binary(arrow_type):
        return "binary"
    elif pa.types.is_large_binary(arrow_type):
        return "large_binary"
    elif pa.types.is_string(arrow_type):
        return "string"
    elif pa.types.is_large_string(arrow_type):
        return "large_string"
    elif pa.types.is_dictionary(arrow_type):
        return _arrow_to_datasets_dtype(arrow_type.value_type)
    else:
        raise ValueError(f"Arrow type {arrow_type} does not have a datasets dtype equivalent.")
    
def _is_zero_copy_only(pa_type: pa.DataType, unnest: bool = False) -> bool:
    def _unnest_pa_type(pa_type: pa.DataType) -> pa.DataType:
        if pa.types.is_list(pa_type):
            return _unnest_pa_type(pa_type.value_type)
        return pa_type

    if unnest:
        pa_type = _unnest_pa_type(pa_type)
    return pa.types.is_primitive(pa_type) and not (pa.types.is_boolean(pa_type) or pa.types.is_temporal(pa_type))

class _ArrayXDExtensionType(pa.ExtensionType):
    ndims: Optional[int] = None

    def __init__(self, shape: tuple, dtype: str):
        if self.ndims is None or self.ndims <= 1:
            raise ValueError("You must instantiate an array type with a value for dim that is > 1")
        if len(shape) != self.ndims:
            raise ValueError(f"shape={shape} and ndims={self.ndims} don't match")
        for dim in range(1, self.ndims):
            if shape[dim] is None:
                raise ValueError(f"Support only dynamic size on first dimension. Got: {shape}")
        self.shape = tuple(shape)
        self.value_type = dtype
        self.storage_dtype = self._generate_dtype(self.value_type)
        pa.ExtensionType.__init__(self, self.storage_dtype, f"{self.__class__.__module__}.{self.__class__.__name__}")

    def __arrow_ext_serialize__(self):
        return json.dumps((self.shape, self.value_type)).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        args = json.loads(serialized)
        return cls(*args)

    def __reduce__(self):
        return self.__arrow_ext_deserialize__, (self.storage_type, self.__arrow_ext_serialize__())

    def __hash__(self):
        return hash((self.__class__, self.shape, self.value_type))

    def __arrow_ext_class__(self):
        return ArrayExtensionArray

    def _generate_dtype(self, dtype):
        dtype = string_to_arrow(dtype)
        for d in reversed(self.shape):
            dtype = pa.list_(dtype)
        return dtype

    def to_pandas_dtype(self):
        return PandasArrayExtensionDtype(self.value_type)

class ArrayExtensionArray(pa.ExtensionArray):
    def __array__(self):
        zero_copy_only = _is_zero_copy_only(self.storage.type, unnest=True)
        return self.to_numpy(zero_copy_only=zero_copy_only)

    def __getitem__(self, i):
        return self.storage[i]

    def to_numpy(self, zero_copy_only=True):
        storage: pa.ListArray = self.storage
        null_mask = storage.is_null().to_numpy(zero_copy_only=False)

        if self.type.shape[0] is not None:
            size = 1
            null_indices = np.arange(len(storage))[null_mask] - np.arange(np.sum(null_mask))

            for i in range(self.type.ndims):
                size *= self.type.shape[i]
                storage = storage.flatten()
            numpy_arr = storage.to_numpy(zero_copy_only=zero_copy_only)
            numpy_arr = numpy_arr.reshape(len(self) - len(null_indices), *self.type.shape)

            if len(null_indices):
                numpy_arr = np.insert(numpy_arr.astype(np.float64), null_indices, np.nan, axis=0)

        else:
            shape = self.type.shape
            ndims = self.type.ndims
            arrays = []
            first_dim_offsets = np.array([off.as_py() for off in storage.offsets])
            for i, is_null in enumerate(null_mask):
                if is_null:
                    arrays.append(np.nan)
                else:
                    storage_el = storage[i : i + 1]
                    first_dim = first_dim_offsets[i + 1] - first_dim_offsets[i]
                    for _ in range(ndims):
                        storage_el = storage_el.flatten()

                    numpy_arr = storage_el.to_numpy(zero_copy_only=zero_copy_only)
                    arrays.append(numpy_arr.reshape(first_dim, *shape[1:]))

            if len(np.unique(np.diff(first_dim_offsets))) > 1:
                numpy_arr = np.empty(len(arrays), dtype=object)
                numpy_arr[:] = arrays
            else:
                numpy_arr = np.array(arrays)

        return numpy_arr

    def to_pylist(self):
        zero_copy_only = _is_zero_copy_only(self.storage.type, unnest=True)
        numpy_arr = self.to_numpy(zero_copy_only=zero_copy_only)
        if self.type.shape[0] is None and numpy_arr.dtype == object:
            return [arr.tolist() for arr in numpy_arr.tolist()]
        else:
            return numpy_arr.tolist()

class PandasArrayExtensionDtype(PandasExtensionDtype):
    _metadata = "value_type"

    def __init__(self, value_type: Union["PandasArrayExtensionDtype", np.dtype]):
        self._value_type = value_type

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]):
        if isinstance(array, pa.ChunkedArray):
            array = array.type.wrap_array(pa.concat_arrays([chunk.storage for chunk in array.chunks]))
        zero_copy_only = _is_zero_copy_only(array.storage.type, unnest=True)
        numpy_arr = array.to_numpy(zero_copy_only=zero_copy_only)
        return PandasArrayExtensionArray(numpy_arr)

    @classmethod
    def construct_array_type(cls):
        return PandasArrayExtensionArray

    @property
    def type(self) -> type:
        return np.ndarray

    @property
    def kind(self) -> str:
        return "O"

    @property
    def name(self) -> str:
        return f"array[{self.value_type}]"

    @property
    def value_type(self) -> np.dtype:
        return self._value_type

class PandasArrayExtensionArray(PandasExtensionArray):
    def __init__(self, data: np.ndarray, copy: bool = False):
        self._data = data if not copy else np.array(data)
        self._dtype = PandasArrayExtensionDtype(data.dtype)

    def __array__(self, dtype=None):
        if dtype == np.dtype(object):
            out = np.empty(len(self._data), dtype=object)
            for i in range(len(self._data)):
                out[i] = self._data[i]
            return out
        if dtype is None:
            return self._data
        else:
            return self._data.astype(dtype)

    def copy(self, deep: bool = False) -> "PandasArrayExtensionArray":
        return PandasArrayExtensionArray(self._data, copy=True)

    @classmethod
    def _from_sequence(cls, scalars, dtype: Optional[PandasArrayExtensionDtype] = None, copy: bool = False) -> "PandasArrayExtensionArray":
        if len(scalars) > 1 and all(isinstance(x, np.ndarray) and x.shape == scalars[0].shape and x.dtype == scalars[0].dtype for x in scalars):
            data = np.array(scalars, dtype=dtype if dtype is None else dtype.value_type, copy=copy)
        else:
            data = np.empty(len(scalars), dtype=object)
            data[:] = scalars
        return cls(data, copy=copy)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence_["PandasArrayExtensionArray"]) -> "PandasArrayExtensionArray":
        if len(to_concat) > 1 and all(va._data.shape == to_concat[0]._data.shape and va._data.dtype == to_concat[0]._data.dtype for va in to_concat):
            data = np.vstack([va._data for va in to_concat])
        else:
            data = np.empty(len(to_concat), dtype=object)
            data[:] = [va._data for va in to_concat]
        return cls(data, copy=False)

    @property
    def dtype(self) -> PandasArrayExtensionDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def isna(self) -> np.ndarray:
        return np.array([pd.isna(arr).any() for arr in self._data])

    def __setitem__(self, key: Union[int, slice, np.ndarray], value: Any) -> None:
        raise NotImplementedError()

    def __getitem__(self, item: Union[int, slice, np.ndarray]) -> Union[np.ndarray, "PandasArrayExtensionArray"]:
        if isinstance(item, int):
            return self._data[item]
        return PandasArrayExtensionArray(self._data[item], copy=False)

    def take(self, indices: Sequence_[int], allow_fill: bool = False, fill_value: bool = None) -> "PandasArrayExtensionArray":
        indices: np.ndarray = np.asarray(indices, dtype=int)
        if allow_fill:
            fill_value = (self.dtype.na_value if fill_value is None else np.asarray(fill_value, dtype=self.dtype.value_type))
            mask = indices == -1
            if (indices < -1).any():
                raise ValueError("Invalid value in `indices`, must be all >= -1 for `allow_fill` is True")
            elif len(self) > 0:
                pass
            elif not np.all(mask):
                raise IndexError("Invalid take for empty PandasArrayExtensionArray, must be all -1.")
            else:
                data = np.array([fill_value] * len(indices), dtype=self.dtype.value_type)
                return PandasArrayExtensionArray(data, copy=False)
        took = self._data.take(indices, axis=0)
        if allow_fill and mask.any():
            took[mask] = [fill_value] * np.sum(mask)
        return PandasArrayExtensionArray(took, copy=False)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> np.ndarray:
        if not isinstance(other, PandasArrayExtensionArray):
            raise NotImplementedError(f"Invalid type to compare to: {type(other)}")
        return (self._data == other._data).all()

def generate_from_dict(obj: Any):
    if isinstance(obj, list):
        return [generate_from_dict(value) for value in obj]
    if "_type" not in obj or isinstance(obj["_type"], dict):
        return {key: generate_from_dict(value) for key, value in obj.items()}
    obj = dict(obj)
    _type = obj.pop("_type")
    class_type = _FEATURE_TYPES.get(_type, None) or globals().get(_type, None)

    if class_type is None:
        raise ValueError(f"Feature type '{_type}' not found. Available feature types: {list(_FEATURE_TYPES.keys())}")

    if class_type == LargeList:
        feature = obj.pop("feature")
        return LargeList(feature=generate_from_dict(feature), **obj)
    if class_type == Sequence:
        feature = obj.pop("feature")
        return Sequence(feature=generate_from_dict(feature), **obj)

    field_names = {f.name for f in fields(class_type)}
    return class_type(**{k: v for k, v in obj.items() if k in field_names})

def cast_to_python_objects(obj: Any, only_1d_for_numpy=False, optimize_list_casting=True) -> Any:
    return _cast_to_python_objects(obj, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0]

def _cast_to_python_objects(obj: Any, only_1d_for_numpy: bool, optimize_list_casting: bool) -> tuple[Any, bool]:
    
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj[()], True
        elif not only_1d_for_numpy or obj.ndim == 1:
            return obj, False
        else:
            return [_cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0] for x in obj], True
    elif data_config.TORCH_AVAILABLE and "torch" in sys.modules and isinstance(obj, torch.Tensor):
        if obj.dtype == torch.bfloat16:
            return _cast_to_python_objects(obj.detach().to(torch.float).cpu().numpy(), only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0], True
        if obj.ndim == 0:
            return obj.detach().cpu().numpy()[()], True
        elif not only_1d_for_numpy or obj.ndim == 1:
            return obj.detach().cpu().numpy(), True
        else:
            return [_cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0] for x in obj.detach().cpu().numpy()], True
    elif data_config.PIL_AVAILABLE and "PIL" in sys.modules and isinstance(obj, PIL.Image.Image):
        return encode_pil_image(obj), True
    elif isinstance(obj, pd.Series):
        return _cast_to_python_objects(obj.tolist(), only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0], True
    elif isinstance(obj, pd.DataFrame):
        return (
            {
                key: _cast_to_python_objects(value, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0] for key, value in obj.to_dict("series").items()
            },
            True,
        )
    elif isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime(), True
    elif isinstance(obj, pd.Timedelta):
        return obj.to_pytimedelta(), True
    elif isinstance(obj, Mapping):
        has_changed = not isinstance(obj, dict)
        output = {}
        for k, v in obj.items():
            casted_v, has_changed_v = _cast_to_python_objects(v, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)
            has_changed |= has_changed_v
            output[k] = casted_v
        return output if has_changed else obj, has_changed
    elif hasattr(obj, "__array__"):
        if np.isscalar(obj):
            return obj, False
        else:
            return _cast_to_python_objects(obj.__array__(), only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0], True
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            for first_elmt in obj:
                if _check_non_null_non_empty_recursive(first_elmt):
                    break
            _, has_changed_first_elmt = _cast_to_python_objects(first_elmt, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)
            if has_changed_first_elmt or not optimize_list_casting:
                return [_cast_to_python_objects(elmt, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0] for elmt in obj], True
            else:
                if isinstance(obj, (list, tuple)):
                    return obj, False
                else:
                    return list(obj), True
        else:
            return obj, False
    else:
        return obj, False

def encode_nested_example(schema, obj, level=0):
    if isinstance(schema, dict):
        if level == 0 and obj is None:
            raise ValueError("Got None but expected a dictionary instead")
        return ({k: encode_nested_example(schema[k], obj.get(k), level=level + 1) for k in schema} if obj is not None else None)

    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        elif isinstance(obj, np.ndarray):
            return encode_nested_example(schema, obj.tolist())
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if encode_nested_example(sub_schema, first_elmt, level=level + 1) != first_elmt:
                    return [encode_nested_example(sub_schema, o, level=level + 1) for o in obj]
            return list(obj)
    elif isinstance(schema, LargeList):
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                sub_schema = schema.feature
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if encode_nested_example(sub_schema, first_elmt, level=level + 1) != first_elmt:
                    return [encode_nested_example(sub_schema, o, level=level + 1) for o in obj]
            return list(obj)
    elif isinstance(schema, Sequence):
        if obj is None:
            return None
        if isinstance(schema.feature, dict):
            list_dict = {}
            if isinstance(obj, (list, tuple)):
                for k in schema.feature:
                    list_dict[k] = [encode_nested_example(schema.feature[k], o.get(k), level=level + 1) for o in obj]
                return list_dict
            else:
                for k in schema.feature:
                    list_dict[k] = ([encode_nested_example(schema.feature[k], o, level=level + 1) for o in obj[k]] if k in obj else None)
                return list_dict
        if isinstance(obj, str):  # don't interpret a string as a list
            raise ValueError(f"Got a string but expected a list instead: '{obj}'")
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, schema.feature):
                        break
                if (not (isinstance(first_elmt, list) or np.isscalar(first_elmt)) or encode_nested_example(schema.feature, first_elmt, level=level + 1) != first_elmt):
                    return [encode_nested_example(schema.feature, o, level=level + 1) for o in obj]
            return list(obj)

    elif hasattr(schema, "encode_example"):
        return schema.encode_example(obj) if obj is not None else None
    return obj

class SplitBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_read_instruction(self, split_dict):
        raise NotImplementedError("Abstract method")

    def __eq__(self, other):
        if isinstance(other, (NamedSplit, str)):
            return False
        raise NotImplementedError("Equality is not implemented between merged/sub splits.")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        return _SplitMerged(self, other)

    def subsplit(self, arg=None, k=None, percent=None, weighted=None):  # pylint: disable=redefined-outer-name
        if sum(bool(x) for x in (arg, k, percent, weighted)) != 1:
            raise ValueError("Only one argument of subsplit should be set.")

        if isinstance(arg, int):
            k = arg
        elif isinstance(arg, slice):
            percent = arg
        elif isinstance(arg, list):
            weighted = arg

        if not (k or percent or weighted):
            raise ValueError(f"Invalid split argument {arg}. Only list, slice and int supported. One of k, weighted or percent should be set to a non empty value.")

        def assert_slices_coverage(slices):
            assert sum((list(range(*s.indices(100))) for s in slices), []) == list(range(100))

        if k:
            if not 0 < k <= 100:
                raise ValueError(f"Subsplit k should be between 0 and 100, got {k}")
            shift = 100 // k
            slices = [slice(i * shift, (i + 1) * shift) for i in range(k)]
            slices[-1] = slice(slices[-1].start, 100)
            assert_slices_coverage(slices)
            return tuple(_SubSplit(self, s) for s in slices)
        elif percent:
            return _SubSplit(self, percent)
        elif weighted:
            total = sum(weighted)
            weighted = [100 * x // total for x in weighted]
            start = 0
            stop = 0
            slices = []
            for v in weighted:
                stop += v
                slices.append(slice(start, stop))
                start = stop
            slices[-1] = slice(slices[-1].start, 100)
            assert_slices_coverage(slices)
            return tuple(_SubSplit(self, s) for s in slices)
        else:
            raise ValueError("Could not determine the split")

class _SplitMerged(SplitBase):
    def __init__(self, split1, split2):
        self._split1 = split1
        self._split2 = split2

    def get_read_instruction(self, split_dict):
        read_instruction1 = self._split1.get_read_instruction(split_dict)
        read_instruction2 = self._split2.get_read_instruction(split_dict)
        return read_instruction1 + read_instruction2

    def __repr__(self):
        return f"({repr(self._split1)} + {repr(self._split2)})"
    
class _SubSplit(SplitBase):
    def __init__(self, split, slice_value):
        self._split = split
        self._slice_value = slice_value

    def get_read_instruction(self, split_dict):
        return self._split.get_read_instruction(split_dict)[self._slice_value]

    def __repr__(self):
        slice_str = "{start}:{stop}"
        if self._slice_value.step is not None:
            slice_str += ":{step}"
        slice_str = slice_str.format(
            start="" if self._slice_value.start is None else self._slice_value.start,
            stop="" if self._slice_value.stop is None else self._slice_value.stop,
            step=self._slice_value.step,
        )
        return f"{repr(self._split)}(datasets.percent[{slice_str}])"

class NamedSplit(SplitBase):
    def __init__(self, name):
        self._name = name
        split_names_from_instruction = [split_instruction.split("[")[0] for split_instruction in name.split("+")]
        for split_name in split_names_from_instruction:
            if not re.match(_split_re, split_name):
                raise ValueError(f"Split name should match '{_split_re}' but got '{split_name}'.")

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"NamedSplit({self._name!r})"

    def __eq__(self, other):
        if isinstance(other, NamedSplit):
            return self._name == other._name  # pylint: disable=protected-access
        elif isinstance(other, SplitBase):
            return False
        elif isinstance(other, str):  # Other should be string
            return self._name == other
        else:
            return False

    def __lt__(self, other):
        return self._name < other._name  # pylint: disable=protected-access

    def __hash__(self):
        return hash(self._name)

    def get_read_instruction(self, split_dict):
        return SplitReadInstruction(split_dict[self._name])

class NamedSplitAll(NamedSplit):
    def __init__(self):
        super().__init__("all")

    def __repr__(self):
        return "NamedSplitAll()"

    def get_read_instruction(self, split_dict):
        read_instructions = [SplitReadInstruction(s) for s in split_dict.values()]
        return sum(read_instructions, SplitReadInstruction())

class Split:
    TRAIN = NamedSplit("train")
    TEST = NamedSplit("test")
    VALIDATION = NamedSplit("validation")
    ALL = NamedSplitAll()

    def __new__(cls, name):
        return NamedSplitAll() if name == "all" else NamedSplit(name)

SlicedSplitInfo = collections.namedtuple(
    "SlicedSplitInfo",
    [
        "split_info",
        "slice_value",
    ],
)  

class NonMutableDict(dict):
    def __init__(self, *args, **kwargs):
        self._error_msg = kwargs.pop("error_msg", "Try to overwrite existing key: {key}")
        if kwargs:
            raise ValueError("NonMutableDict cannot be initialized with kwargs.")
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(self._error_msg.format(key=key))
        return super().__setitem__(key, value)

    def update(self, other):
        if any(k in self for k in other):
            raise ValueError(self._error_msg.format(key=set(self) & set(other)))
        return super().update(other)
    
class SplitReadInstruction:
    def __init__(self, split_info=None):
        self._splits = NonMutableDict(error_msg="Overlap between splits. Split {key} has been added with itself.")

        if split_info:
            self.add(SlicedSplitInfo(split_info=split_info, slice_value=None))

    def add(self, sliced_split):
        self._splits[sliced_split.split_info.name] = sliced_split

    def __add__(self, other):
        split_instruction = SplitReadInstruction()
        split_instruction._splits.update(self._splits)  # pylint: disable=protected-access
        split_instruction._splits.update(other._splits)  # pylint: disable=protected-access
        return split_instruction

    def __getitem__(self, slice_value):
        split_instruction = SplitReadInstruction()
        for v in self._splits.values():
            if v.slice_value is not None:
                raise ValueError(f"Trying to slice Split {v.split_info.name} which has already been sliced")
            v = v._asdict()
            v["slice_value"] = slice_value
            split_instruction.add(SlicedSplitInfo(**v))
        return split_instruction

    def get_list_sliced_split_info(self):
        return list(self._splits.values())

@dataclass(frozen=True)
class FileInstructions:
    num_examples: int
    file_instructions: list[dict]

@dataclass
class SplitInfo:
    name: str = dataclasses.field(default="", metadata={"include_in_asdict_even_if_is_default": True})
    num_bytes: int = dataclasses.field(default=0, metadata={"include_in_asdict_even_if_is_default": True})
    num_examples: int = dataclasses.field(default=0, metadata={"include_in_asdict_even_if_is_default": True})
    shard_lengths: Optional[list[int]] = None

    dataset_name: Optional[str] = dataclasses.field(default=None, metadata={"include_in_asdict_even_if_is_default": True})

    @property
    def file_instructions(self):
        return make_file_instructions(name=self.dataset_name, split_infos=[self], instruction=str(self.name)).file_instructions

@dataclass
class SubSplitInfo:
    instructions: FileInstructions

    @property
    def num_examples(self):
        return self.instructions.num_examples

    @property
    def file_instructions(self):
        return self.instructions.file_instructions

class SplitDict(dict):
    def __init__(self, *args, dataset_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name

    def __getitem__(self, key: Union[SplitBase, str]):
        if str(key) in self:
            return super().__getitem__(str(key))
        else:
            instructions = make_file_instructions(name=self.dataset_name, split_infos=self.values(), instruction=key)
            return SubSplitInfo(instructions)

    def __setitem__(self, key: Union[SplitBase, str], value: SplitInfo):
        if key != value.name:
            raise ValueError(f"Cannot add elem. (key mismatch: '{key}' != '{value.name}')")
        super().__setitem__(key, value)

    def add(self, split_info: SplitInfo):
        if split_info.name in self:
            raise ValueError(f"Split {split_info.name} already present")
        split_info.dataset_name = self.dataset_name
        super().__setitem__(split_info.name, split_info)

    @property
    def total_num_examples(self):
        return sum(s.num_examples for s in self.values())

    @classmethod
    def from_split_dict(cls, split_infos: Union[list, dict], dataset_name: Optional[str] = None):
        if isinstance(split_infos, dict):
            split_infos = list(split_infos.values())

        if dataset_name is None:
            dataset_name = split_infos[0].get("dataset_name") if split_infos else None

        split_dict = cls(dataset_name=dataset_name)

        for split_info in split_infos:
            if isinstance(split_info, dict):
                split_info = SplitInfo(**split_info)
            split_dict.add(split_info)

        return split_dict

    def to_split_dict(self):
        out = []
        for split_name, split_info in self.items():
            split_info = copy.deepcopy(split_info)
            split_info.name = split_name
            out.append(split_info)
        return out

    def copy(self):
        return SplitDict.from_split_dict(self.to_split_dict(), self.dataset_name)

    def _to_yaml_list(self) -> list:
        out = [asdict(s) for s in self.to_split_dict()]
        for split_info_dict in out:
            split_info_dict.pop("shard_lengths", None)
        for split_info_dict in out:
            split_info_dict.pop("dataset_name", None)
        return out

    @classmethod
    def _from_yaml_list(cls, yaml_data: list) -> "SplitDict":
        return cls.from_split_dict(yaml_data)

def filename_prefix_for_name(name):
    if os.path.basename(name) != name:
        raise ValueError(f"Should be a dataset name, not a path: {name}")
    return camelcase_to_snakecase(name)

def filename_prefix_for_split(name, split):
    if os.path.basename(name) != name:
        raise ValueError(f"Should be a dataset name, not a path: {name}")
    if not re.match(_split_re, split):
        raise ValueError(f"Split name should match '{_split_re}'' but got '{split}'.")
    return f"{filename_prefix_for_name(name)}-{split}"

def filenames_for_dataset_split(path, dataset_name, split, filetype_suffix=None, shard_lengths=None):
    prefix = filename_prefix_for_split(dataset_name, split)
    prefix = os.path.join(path, prefix)

    if shard_lengths:
        num_shards = len(shard_lengths)
        filenames = [f"{prefix}-{shard_id:05d}-of-{num_shards:05d}" for shard_id in range(num_shards)]
        if filetype_suffix:
            filenames = [filename + f".{filetype_suffix}" for filename in filenames]
        return filenames
    else:
        filename = prefix
        if filetype_suffix:
            filename += f".{filetype_suffix}"
        return [filename]

def make_file_instructions(
    name: str,
    split_infos: list["SplitInfo"],
    instruction: Union[str, "ReadInstruction"],
    filetype_suffix: Optional[str] = None,
    prefix_path: Optional[str] = None,
) -> FileInstructions:
    if not isinstance(name, str):
        raise TypeError(f"Expected str 'name', but got: {type(name).__name__}")
    elif not name:
        raise ValueError("Expected non-empty str 'name'")
    name2len = {info.name: info.num_examples for info in split_infos}
    name2shard_lengths = {info.name: info.shard_lengths for info in split_infos}
    name2filenames = {
        info.name: filenames_for_dataset_split(
            path=prefix_path,
            dataset_name=name,
            split=info.name,
            filetype_suffix=filetype_suffix,
            shard_lengths=name2shard_lengths[info.name],
        )
        for info in split_infos
    }
    if not isinstance(instruction, ReadInstruction):
        instruction = ReadInstruction.from_spec(instruction)
    absolute_instructions = instruction.to_absolute(name2len)

    file_instructions = []
    num_examples = 0
    for abs_instr in absolute_instructions:
        split_length = name2len[abs_instr.splitname]
        filenames = name2filenames[abs_instr.splitname]
        shard_lengths = name2shard_lengths[abs_instr.splitname]
        from_ = 0 if abs_instr.from_ is None else abs_instr.from_
        to = split_length if abs_instr.to is None else abs_instr.to
        if shard_lengths is None:  # not sharded
            for filename in filenames:
                take = to - from_
                if take == 0:
                    continue
                num_examples += take
                file_instructions.append({"filename": filename, "skip": from_, "take": take})
        else:  # sharded
            index_start = 0  # Beginning (included) of moving window.
            index_end = 0  # End (excluded) of moving window.
            for filename, shard_length in zip(filenames, shard_lengths):
                index_end += shard_length
                if from_ < index_end and to > index_start:  # There is something to take.
                    skip = from_ - index_start if from_ > index_start else 0
                    take = to - index_start - skip if to < index_end else -1
                    if take == 0:
                        continue
                    file_instructions.append({"filename": filename, "skip": skip, "take": take})
                    num_examples += shard_length - skip if take == -1 else take
                index_start += shard_length
    return FileInstructions(num_examples=num_examples, file_instructions=file_instructions)

def _str_to_read_instruction(spec):
    res = _SUB_SPEC_RE.match(spec)
    if not res:
        raise ValueError(f"Unrecognized instruction format: {spec}")
    unit = "%" if res.group("from_pct") or res.group("to_pct") else "abs"
    return ReadInstruction(
        split_name=res.group("split"),
        rounding=res.group("rounding"),
        from_=int(res.group("from")) if res.group("from") else None,
        to=int(res.group("to")) if res.group("to") else None,
        unit=unit,
    )

@dataclass(frozen=True)
class _RelativeInstruction:
    splitname: str
    from_: Optional[int] = None  # int (starting index) or None if no lower boundary.
    to: Optional[int] = None  # int (ending index) or None if no upper boundary.
    unit: Optional[str] = None
    rounding: Optional[str] = None

    def __post_init__(self):
        if self.unit is not None and self.unit not in ["%", "abs"]:
            raise ValueError("unit must be either % or abs")
        if self.rounding is not None and self.rounding not in ["closest", "pct1_dropremainder"]:
            raise ValueError("rounding must be either closest or pct1_dropremainder")
        if self.unit != "%" and self.rounding is not None:
            raise ValueError("It is forbidden to specify rounding if not using percent slicing.")
        if self.unit == "%" and self.from_ is not None and abs(self.from_) > 100:
            raise ValueError("Percent slice boundaries must be > -100 and < 100.")
        if self.unit == "%" and self.to is not None and abs(self.to) > 100:
            raise ValueError("Percent slice boundaries must be > -100 and < 100.")
        self.__dict__["rounding"] = "closest" if self.rounding is None and self.unit == "%" else self.rounding

@dataclass(frozen=True)
class _AbsoluteInstruction:
    splitname: str
    from_: int  # uint (starting index).
    to: int  # uint (ending index).

class ReadInstruction:
    def _init(self, relative_instructions):
        # Private initializer.
        self._relative_instructions = relative_instructions

    @classmethod
    def _read_instruction_from_relative_instructions(cls, relative_instructions):
        result = cls.__new__(cls)
        result._init(relative_instructions)  # pylint: disable=protected-access
        return result

    def __init__(self, split_name, rounding=None, from_=None, to=None, unit=None):
        self._init([_RelativeInstruction(split_name, from_, to, unit, rounding)])

    @classmethod
    def from_spec(cls, spec):
        spec = str(spec)  # Need to convert to str in case of NamedSplit instance.
        subs = _ADDITION_SEP_RE.split(spec)
        if not subs:
            raise ValueError(f"No instructions could be built out of {spec}")
        instruction = _str_to_read_instruction(subs[0])
        return sum((_str_to_read_instruction(sub) for sub in subs[1:]), instruction)

    def to_spec(self):
        rel_instr_specs = []
        for rel_instr in self._relative_instructions:
            rel_instr_spec = rel_instr.splitname
            if rel_instr.from_ is not None or rel_instr.to is not None:
                from_ = rel_instr.from_
                to = rel_instr.to
                unit = rel_instr.unit
                rounding = rel_instr.rounding
                unit = unit if unit == "%" else ""
                from_ = str(from_) + unit if from_ is not None else ""
                to = str(to) + unit if to is not None else ""
                slice_str = f"[{from_}:{to}]"
                rounding_str = (f"({rounding})" if unit == "%" and rounding is not None and rounding != "closest" else "")
                rel_instr_spec += slice_str + rounding_str
            rel_instr_specs.append(rel_instr_spec)
        return "+".join(rel_instr_specs)

    def __add__(self, other):
        if not isinstance(other, ReadInstruction):
            raise TypeError("ReadInstruction can only be added to another ReadInstruction obj.")
        self_ris = self._relative_instructions
        other_ris = other._relative_instructions  # pylint: disable=protected-access
        if (self_ris[0].unit != "abs" and other_ris[0].unit != "abs" and self._relative_instructions[0].rounding != other_ris[0].rounding):
            raise ValueError("It is forbidden to sum ReadInstruction instances with different rounding values.")
        return self._read_instruction_from_relative_instructions(self_ris + other_ris)

    def __str__(self):
        return self.to_spec()

    def __repr__(self):
        return f"ReadInstruction({self._relative_instructions})"

    def to_absolute(self, name2len):
        return [_rel_to_abs_instr(rel_instr, name2len) for rel_instr in self._relative_instructions]

def _rel_to_abs_instr(rel_instr, name2len):
    pct_to_abs = _pct_to_abs_closest if rel_instr.rounding == "closest" else _pct_to_abs_pct1
    split = rel_instr.splitname
    if split not in name2len:
        raise ValueError(f'Unknown split "{split}". Should be one of {list(name2len)}.')
    num_examples = name2len[split]
    from_ = rel_instr.from_
    to = rel_instr.to
    if rel_instr.unit == "%":
        from_ = 0 if from_ is None else pct_to_abs(from_, num_examples)
        to = num_examples if to is None else pct_to_abs(to, num_examples)
    else:
        from_ = 0 if from_ is None else from_
        to = num_examples if to is None else to
    if from_ < 0:
        from_ = max(num_examples + from_, 0)
    if to < 0:
        to = max(num_examples + to, 0)
    from_ = min(from_, num_examples)
    to = min(to, num_examples)
    return _AbsoluteInstruction(split, from_, to)

def _pct_to_abs_pct1(boundary, num_examples):
    if num_examples < 100:
        raise ValueError('Using "pct1_dropremainder" rounding on a split with less than 100 elements is forbidden: it always results in an empty dataset.')
    return boundary * math.trunc(num_examples / 100.0)

def _pct_to_abs_closest(boundary, num_examples):
    return int(round(boundary * num_examples / 100.0))

@dataclass
class PostProcessedInfo:
    features: Optional[Features] = None
    resources_checksums: Optional[dict] = None

    def __post_init__(self):
        if self.features is not None and not isinstance(self.features, Features):
            self.features = Features.from_dict(self.features)

    @classmethod
    def from_dict(cls, post_processed_info_dict: dict) -> "PostProcessedInfo":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in post_processed_info_dict.items() if k in field_names})

@dataclass
class SupervisedKeysData:
    input: str = ""
    output: str = ""

class FileLock(FileLock_):
    MAX_FILENAME_LENGTH = 255

    def __init__(self, lock_file, *args, **kwargs):
        if "mode" not in kwargs and version.parse(_filelock_version) >= version.parse("3.10.0"):
            umask = os.umask(0o666)
            os.umask(umask)
            kwargs["mode"] = 0o666 & ~umask
        lock_file = self.hash_filename_if_too_long(lock_file)
        super().__init__(lock_file, *args, **kwargs)

    @classmethod
    def hash_filename_if_too_long(cls, path: str) -> str:
        path = os.path.abspath(os.path.expanduser(path))
        filename = os.path.basename(path)
        max_filename_length = cls.MAX_FILENAME_LENGTH
        if issubclass(cls, UnixFileLock):
            max_filename_length = min(max_filename_length, os.statvfs(os.path.dirname(path)).f_namemax)
        if len(filename) > max_filename_length:
            dirname = os.path.dirname(path)
            hashed_filename = str(hash(filename))
            new_filename = (filename[: max_filename_length - len(hashed_filename) - 8] + "..." + hashed_filename + ".lock")
            return os.path.join(dirname, new_filename)
        else:
            return path

class _PatchedModuleObj:
    def __init__(self, module, attrs=None):
        attrs = attrs or []
        if module is not None:
            for key in module.__dict__:
                if key in attrs or not key.startswith("__"):
                    setattr(self, key, getattr(module, key))
        self._original_module = module._original_module if isinstance(module, _PatchedModuleObj) else module

class patch_submodule:
    _active_patches = []

    def __init__(self, obj, target: str, new, attrs=None):
        self.obj = obj
        self.target = target
        self.new = new
        self.key = target.split(".")[0]
        self.original = {}
        self.attrs = attrs or []

    def __enter__(self):
        *submodules, target_attr = self.target.split(".")

        for i in range(len(submodules)):
            try:
                submodule = import_module(".".join(submodules[: i + 1]))
            except ModuleNotFoundError:
                continue
            for attr in self.obj.__dir__():
                obj_attr = getattr(self.obj, attr)
                if obj_attr is submodule or (isinstance(obj_attr, _PatchedModuleObj) and obj_attr._original_module is submodule):
                    self.original[attr] = obj_attr
                    setattr(self.obj, attr, _PatchedModuleObj(obj_attr, attrs=self.attrs))
                    patched = getattr(self.obj, attr)
                    for key in submodules[i + 1 :]:
                        setattr(patched, key, _PatchedModuleObj(getattr(patched, key, None), attrs=self.attrs))
                        patched = getattr(patched, key)
                    setattr(patched, target_attr, self.new)

        if submodules:  # if it's an attribute of a submodule like "os.path.join"
            try:
                attr_value = getattr(import_module(".".join(submodules)), target_attr)
            except (AttributeError, ModuleNotFoundError):
                return
            for attr in self.obj.__dir__():
                if getattr(self.obj, attr) is attr_value:
                    self.original[attr] = getattr(self.obj, attr)
                    setattr(self.obj, attr, self.new)
        elif target_attr in globals()["__builtins__"]:  # if it'a s builtin like "open"
            self.original[target_attr] = globals()["__builtins__"][target_attr]
            setattr(self.obj, target_attr, self.new)
        else:
            raise RuntimeError(f"Tried to patch attribute {target_attr} instead of a submodule.")

    def __exit__(self, *exc_info):
        for attr in list(self.original):
            setattr(self.obj, attr, self.original.pop(attr))

    def start(self):
        self.__enter__()
        self._active_patches.append(self)

    def stop(self):
        try:
            self._active_patches.remove(self)
        except ValueError:
            return None

        return self.__exit__()

def _get_path_extension(path: str) -> str:
    extension = path.split(".")[-1]
    for symb in "?-_":
        extension = extension.split(symb)[0]
    return extension

def _get_extraction_protocol_with_magic_number(f) -> Optional[str]:
    try:
        f.seek(0)
    except (AttributeError, io.UnsupportedOperation):
        return None
    magic_number = f.read(MAGIC_NUMBER_MAX_LENGTH)
    f.seek(0)
    for i in range(MAGIC_NUMBER_MAX_LENGTH):
        compression = MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL.get(magic_number[: MAGIC_NUMBER_MAX_LENGTH - i])
        if compression is not None:
            return compression
        compression = MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL.get(magic_number[: MAGIC_NUMBER_MAX_LENGTH - i])
        if compression is not None:
            raise NotImplementedError(f"Compression protocol '{compression}' not implemented.")

def _get_extraction_protocol(urlpath: str, download_config: Optional[DownloadConfig] = None) -> Optional[str]:
    urlpath = str(urlpath)
    path = urlpath.split("::")[0]
    extension = _get_path_extension(path)
    if (extension in BASE_KNOWN_EXTENSIONS or extension in ["tgz", "tar"] or path.endswith((".tar.gz", ".tar.bz2", ".tar.xz"))):
        return None
    elif extension in COMPRESSION_EXTENSION_TO_PROTOCOL:
        return COMPRESSION_EXTENSION_TO_PROTOCOL[extension]
    urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
    try:
        with fsspec.open(urlpath, **(storage_options or {})) as f:
            return _get_extraction_protocol_with_magic_number(f)
    except FileNotFoundError:
        if urlpath.startswith(data_config.HF_ENDPOINT):
            raise FileNotFoundError(urlpath + "\nIf the repo is private or gated, make sure to log in with `huggingface-cli login`.") from None
        else:
            raise

def xlistdir(path: str, download_config: Optional[DownloadConfig] = None) -> list[str]:
    main_hop, *rest_hops = _as_str(path).split("::")
    if is_local_path(main_hop):
        return os.listdir(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split("::")
        fs, *_ = url_to_fs(path, **storage_options)
        inner_path = main_hop.split("://")[-1]
        if inner_path.strip("/") and not fs.isdir(inner_path):
            raise FileNotFoundError(f"Directory doesn't exist: {path}")
        paths = fs.listdir(inner_path, detail=False)
        return [os.path.basename(path.rstrip("/")) for path in paths]

def xglob(urlpath, *, recursive=False, download_config: Optional[DownloadConfig] = None):
    main_hop, *rest_hops = _as_str(urlpath).split("::")
    if is_local_path(main_hop):
        return glob.glob(main_hop, recursive=recursive)
    else:
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        main_hop, *rest_hops = urlpath.split("::")
        fs, *_ = url_to_fs(urlpath, **storage_options)
        inner_path = main_hop.split("://")[1]
        globbed_paths = fs.glob(inner_path)
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[-1]
        return ["::".join([f"{protocol}://{globbed_path}"] + rest_hops) for globbed_path in globbed_paths]

def xisfile(path, download_config: Optional[DownloadConfig] = None) -> bool:
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.isfile(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split("::")
        fs, *_ = url_to_fs(path, **storage_options)
        return fs.isfile(main_hop)

def xgetsize(path, download_config: Optional[DownloadConfig] = None) -> int:
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.getsize(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split("::")
        fs, *_ = fs, *_ = url_to_fs(path, **storage_options)
        try:
            size = fs.size(main_hop)
        except EntryNotFoundError:
            raise FileNotFoundError(f"No such file: {path}")
        if size is None:
            with xopen(path, download_config=download_config) as f:
                size = len(f.read())
        return size

def xisdir(path, download_config: Optional[DownloadConfig] = None) -> bool:
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.isdir(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split("::")
        fs, *_ = fs, *_ = url_to_fs(path, **storage_options)
        inner_path = main_hop.split("://")[-1]
        if not inner_path.strip("/"):
            return True
        return fs.isdir(inner_path)

def xgzip_open(filepath_or_buffer, *args, download_config: Optional[DownloadConfig] = None, **kwargs):
    if hasattr(filepath_or_buffer, "read"):
        return gzip.open(filepath_or_buffer, *args, **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        return gzip.open(xopen(filepath_or_buffer, "rb", download_config=download_config), *args, **kwargs)
    
def xet_parse(source, parser=None, download_config: Optional[DownloadConfig] = None):
    if hasattr(source, "read"):
        return ET.parse(source, parser=parser)
    else:
        with xopen(source, "rb", download_config=download_config) as f:
            return ET.parse(f, parser=parser)

def xwalk(urlpath, download_config: Optional[DownloadConfig] = None, **kwargs):
    main_hop, *rest_hops = _as_str(urlpath).split("::")
    if is_local_path(main_hop):
        yield from os.walk(main_hop, **kwargs)
    else:
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        main_hop, *rest_hops = urlpath.split("::")
        fs, *_ = url_to_fs(urlpath, **storage_options)
        inner_path = main_hop.split("://")[-1]
        if inner_path.strip("/") and not fs.isdir(inner_path):
            return []
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[-1]
        for dirpath, dirnames, filenames in fs.walk(inner_path, **kwargs):
            yield "::".join([f"{protocol}://{dirpath}"] + rest_hops), dirnames, filenames

def xrelpath(path, start=None):
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.relpath(main_hop, start=start) if start else os.path.relpath(main_hop)
    else:
        return posixpath.relpath(main_hop, start=str(start).split("::")[0]) if start else os.path.relpath(main_hop)

def xsplit(a):
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.split(Path(a).as_posix())
    else:
        a, tail = posixpath.split(a)
        return "::".join([a + "//" if a.endswith(":") else a] + b), tail

def xsplitext(a):
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.splitext(Path(a).as_posix())
    else:
        a, ext = posixpath.splitext(a)
        return "::".join([a] + b), ext    

def xnumpy_load(filepath_or_buffer, *args, download_config: Optional[DownloadConfig] = None, **kwargs):
    if hasattr(filepath_or_buffer, "read"):
        return np.load(filepath_or_buffer, *args, **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        return np.load(xopen(filepath_or_buffer, "rb", download_config=download_config), *args, **kwargs)

def xpandas_read_csv(filepath_or_buffer, download_config: Optional[DownloadConfig] = None, **kwargs):
    if hasattr(filepath_or_buffer, "read"):
        return pd.read_csv(filepath_or_buffer, **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        if kwargs.get("compression", "infer") == "infer":
            kwargs["compression"] = _get_extraction_protocol(filepath_or_buffer, download_config=download_config)
        return pd.read_csv(xopen(filepath_or_buffer, "rb", download_config=download_config), **kwargs)

def xpandas_read_excel(filepath_or_buffer, download_config: Optional[DownloadConfig] = None, **kwargs):
    if hasattr(filepath_or_buffer, "read"):
        try:
            return pd.read_excel(filepath_or_buffer, **kwargs)
        except ValueError:  # Cannot seek streaming HTTP file
            return pd.read_excel(BytesIO(filepath_or_buffer.read()), **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        try:
            return pd.read_excel(xopen(filepath_or_buffer, "rb", download_config=download_config), **kwargs)
        except ValueError:  # Cannot seek streaming HTTP file
            return pd.read_excel(BytesIO(xopen(filepath_or_buffer, "rb", download_config=download_config).read()), **kwargs)

def xpyarrow_parquet_read_table(filepath_or_buffer, download_config: Optional[DownloadConfig] = None, **kwargs):
    if hasattr(filepath_or_buffer, "read"):
        return pq.read_table(filepath_or_buffer, **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        return pq.read_table(xopen(filepath_or_buffer, mode="rb", download_config=download_config), **kwargs)
    
def xsio_loadmat(filepath_or_buffer, download_config: Optional[DownloadConfig] = None, **kwargs):
    if hasattr(filepath_or_buffer, "read"):
        return sio.loadmat(filepath_or_buffer, **kwargs)
    else:
        return sio.loadmat(xopen(filepath_or_buffer, "rb", download_config=download_config), **kwargs)
    
def xxml_dom_minidom_parse(filename_or_file, download_config: Optional[DownloadConfig] = None, **kwargs):
    if hasattr(filename_or_file, "read"):
        return xml.dom.minidom.parse(filename_or_file, **kwargs)
    else:
        with xopen(filename_or_file, "rb", download_config=download_config) as f:
            return xml.dom.minidom.parse(f, **kwargs)

class classproperty(property):  # pylint: disable=invalid-name
    def __get__(self, obj, objtype=None):
        return self.fget.__get__(None, objtype)()
    
def rename(fs: fsspec.AbstractFileSystem, src: str, dst: str):
    if not is_remote_filesystem(fs):
        shutil.move(fs._strip_protocol(src), fs._strip_protocol(dst))
    else:
        fs.mv(src, dst, recursive=True)

def is_remote_filesystem(fs: fsspec.AbstractFileSystem) -> bool:
    return not isinstance(fs, LocalFileSystem)

class DownloadMode(enum.Enum):
    REUSE_DATASET_IF_EXISTS = "reuse_dataset_if_exists"
    REUSE_CACHE_IF_EXISTS = "reuse_cache_if_exists"
    FORCE_REDOWNLOAD = "force_redownload"

class VerificationMode(enum.Enum):
    ALL_CHECKS = "all_checks"
    BASIC_CHECKS = "basic_checks"
    NO_CHECKS = "no_checks"

class NestedDataStructure:
    def __init__(self, data=None):
        self.data = data if data is not None else []

    def flatten(self, data=None):
        data = data if data is not None else self.data
        if isinstance(data, dict):
            return self.flatten(list(data.values()))
        elif isinstance(data, (list, tuple)):
            return [flattened for item in data for flattened in self.flatten(item)]
        else:
            return [data]

def get_size_checksum_dict(path: str, record_checksum: bool = True) -> dict:
    if record_checksum:
        m = insecure_hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                m.update(chunk)
            checksum = m.hexdigest()
    else:
        checksum = None
    return {"num_bytes": os.path.getsize(path), "checksum": checksum}

def stack_multiprocessing_download_progress_bars():
    return patch.dict(os.environ, {"HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS": "1"})

def iter_batched(iterable: Iterable[T], n: int) -> Iterable[list[T]]:
    if n < 1:
        raise ValueError(f"Invalid batch size {n}")
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def _single_map_nested(args):
    function, data_struct, batched, batch_size, types, rank, disable_tqdm, desc = args

    if not isinstance(data_struct, dict) and not isinstance(data_struct, types):
        if batched:
            return function([data_struct])[0]
        else:
            return function(data_struct)
    if (
        batched
        and not isinstance(data_struct, dict)
        and isinstance(data_struct, types)
        and all(not isinstance(v, (dict, types)) for v in data_struct)
    ):
        return [mapped_item for batch in iter_batched(data_struct, batch_size) for mapped_item in function(batch)]

    if rank is not None and not disable_tqdm and any("notebook" in tqdm_cls.__name__ for tqdm_cls in hf_tqdm.__mro__):
        print(" ", end="", flush=True)

    pbar_iterable = data_struct.items() if isinstance(data_struct, dict) else data_struct
    pbar_desc = (desc + " " if desc is not None else "") + "#" + str(rank) if rank is not None else desc
    with hf_tqdm(pbar_iterable, disable=disable_tqdm, position=rank, unit="obj", desc=pbar_desc) as pbar:
        if isinstance(data_struct, dict):
            return {k: _single_map_nested((function, v, batched, batch_size, types, None, True, None)) for k, v in pbar}
        else:
            mapped = [_single_map_nested((function, v, batched, batch_size, types, None, True, None)) for v in pbar]
            if isinstance(data_struct, list):
                return mapped
            elif isinstance(data_struct, tuple):
                return tuple(mapped)
            else:
                return np.array(mapped)

def experimental(fn: Callable) -> Callable:
    @wraps(fn)
    def _inner_fn(*args, **kwargs):
        warnings.warn(
            (f"'{fn.__name__}' is experimental and might be subject to breaking changes in the future."),
            UserWarning,
        )
        return fn(*args, **kwargs)

    return _inner_fn

class ParallelBackendConfig:
    backend_name = None

def _map_with_multiprocessing_pool(function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func):
    num_proc = num_proc if num_proc <= len(iterable) else len(iterable)
    split_kwds = []  # We organize the splits ourselve (contiguous splits)
    for index in range(num_proc):
        div = len(iterable) // num_proc
        mod = len(iterable) % num_proc
        start = div * index + min(index, mod)
        end = start + div + (1 if index < mod else 0)
        split_kwds.append((function, iterable[start:end], batched, batch_size, types, index, disable_tqdm, desc))

    if len(iterable) != sum(len(i[1]) for i in split_kwds):
        raise ValueError(f"Error dividing inputs iterable among processes. Total number of objects {len(iterable)}, length: {sum(len(i[1]) for i in split_kwds)}")

    initargs, initializer = None, None
    if not disable_tqdm:
        initargs, initializer = (RLock(),), hf_tqdm.set_lock
    with Pool(num_proc, initargs=initargs, initializer=initializer) as pool:
        mapped = pool.map(single_map_nested_func, split_kwds)
    mapped = [obj for proc_res in mapped for obj in proc_res]

    return mapped

def _map_with_joblib(function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func):
    with joblib.parallel_backend(ParallelBackendConfig.backend_name, n_jobs=num_proc):
        return joblib.Parallel()(joblib.delayed(single_map_nested_func)((function, obj, batched, batch_size, types, None, True, None)) for obj in iterable)

@experimental
def parallel_map(function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func):
    if ParallelBackendConfig.backend_name is None:
        return _map_with_multiprocessing_pool(function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func)

    return _map_with_joblib(function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func)

def map_nested(
    function: Callable[[Any], Any],
    data_struct: Any,
    dict_only: bool = False,
    map_list: bool = True,
    map_tuple: bool = False,
    map_numpy: bool = False,
    num_proc: Optional[int] = None,
    parallel_min_length: int = 2,
    batched: bool = False,
    batch_size: Optional[int] = 1000,
    types: Optional[tuple] = None,
    disable_tqdm: bool = True,
    desc: Optional[str] = None,
) -> Any:
    if types is None:
        types: list = []
        if not dict_only:
            if map_list:
                types.append(list)
            if map_tuple:
                types.append(tuple)
            if map_numpy:
                types.append(np.ndarray)
        types = tuple(types)

    if not isinstance(data_struct, dict) and not isinstance(data_struct, types):
        if batched:
            data_struct = [data_struct]
        mapped = function(data_struct)
        if batched:
            mapped = mapped[0]
        return mapped

    iterable = list(data_struct.values()) if isinstance(data_struct, dict) else data_struct

    if num_proc is None:
        num_proc = 1
    if any(isinstance(v, types) and len(v) > len(iterable) for v in iterable):
        mapped = [
            map_nested(
                function=function,
                data_struct=obj,
                num_proc=num_proc,
                parallel_min_length=parallel_min_length,
                batched=batched,
                batch_size=batch_size,
                types=types,
            )
            for obj in iterable
        ]
    elif num_proc != -1 and num_proc <= 1 or len(iterable) < parallel_min_length:
        if batched:
            if batch_size is None or batch_size <= 0:
                batch_size = max(len(iterable) // num_proc + int(len(iterable) % num_proc > 0), 1)
            iterable = list(iter_batched(iterable, batch_size))
        mapped = [_single_map_nested((function, obj, batched, batch_size, types, None, True, None)) for obj in hf_tqdm(iterable, disable=disable_tqdm, desc=desc)]
        if batched:
            mapped = [mapped_item for mapped_batch in mapped for mapped_item in mapped_batch]
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".* is experimental and might be subject to breaking changes in the future\\.$",
                category=UserWarning,
            )
            if batched:
                if batch_size is None or batch_size <= 0:
                    batch_size = len(iterable) // num_proc + int(len(iterable) % num_proc > 0)
                iterable = list(iter_batched(iterable, batch_size))
            mapped = parallel_map(function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, _single_map_nested)
            if batched:
                mapped = [mapped_item for mapped_batch in mapped for mapped_item in mapped_batch]

    if isinstance(data_struct, dict):
        return dict(zip(data_struct.keys(), mapped))
    else:
        if isinstance(data_struct, list):
            return mapped
        elif isinstance(data_struct, tuple):
            return tuple(mapped)
        else:
            return np.array(mapped)

def url_or_path_join(base_name: str, *pathnames: str) -> str:
    if is_remote_url(base_name):
        return posixpath.join(base_name, *(str(pathname).replace(os.sep, "/").lstrip("/") for pathname in pathnames))
    else:
        return Path(base_name, *pathnames).as_posix()

def get_datasets_user_agent(user_agent: Optional[Union[str, dict]] = None) -> str:
    ua = f"datasets/{__version__}"
    ua += f"; python/{data_config.PY_VERSION}"
    ua += f"; huggingface_hub/{huggingface_hub.__version__}"
    ua += f"; pyarrow/{data_config.PYARROW_VERSION}"
    if data_config.TORCH_AVAILABLE:
        ua += f"; torch/{data_config.TORCH_VERSION}"
    if data_config.TF_AVAILABLE:
        ua += f"; tensorflow/{data_config.TF_VERSION}"
    if data_config.JAX_AVAILABLE:
        ua += f"; jax/{data_config.JAX_VERSION}"
    if isinstance(user_agent, dict):
        ua += f"; {'; '.join(f'{k}/{v}' for k, v in user_agent.items())}"
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua

class TqdmCallback(fsspec.callbacks.TqdmCallback):
    def __init__(self, tqdm_kwargs=None, *args, **kwargs):
        if data_config.FSSPEC_VERSION < version.parse("2024.2.0"):
            super().__init__(tqdm_kwargs, *args, **kwargs)
            self._tqdm = hf_tqdm  # replace tqdm module by datasets.utils.tqdm module
        else:
            kwargs["tqdm_cls"] = hf_tqdm.tqdm
            super().__init__(tqdm_kwargs, *args, **kwargs)

def _raise_if_offline_mode_is_enabled(msg: Optional[str] = None):
    if data_config.HF_HUB_OFFLINE:
        raise huggingface_hub.errors.OfflineModeIsEnabled("Offline mode is enabled." if msg is None else "Offline mode is enabled. " + str(msg))

def fsspec_get(url, temp_file, storage_options=None, desc=None, disable_tqdm=False):
    _raise_if_offline_mode_is_enabled(f"Tried to reach {url}")
    fs, path = url_to_fs(url, **(storage_options or {}))
    callback = TqdmCallback(
        tqdm_kwargs={
            "desc": desc or "Downloading",
            "unit": "B",
            "unit_scale": True,
            "position": multiprocessing.current_process()._identity[-1]  # contains the ranks of subprocesses
            if os.environ.get("HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS") == "1"
            and multiprocessing.current_process()._identity
            else None,
            "disable": disable_tqdm,
        }
    )
    fs.get_file(path, temp_file.name, callback=callback)

def hash_url_to_filename(url, etag=None):
    url_bytes = url.encode("utf-8")
    url_hash = insecure_hashlib.sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = insecure_hashlib.sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    if url.endswith(".py"):
        filename += ".py"

    return filename

def get_authentication_headers_for_url(url: str, token: Optional[Union[str, bool]] = None) -> dict:
    if url.startswith(data_config.HF_ENDPOINT):
        return huggingface_hub.utils.build_hf_headers(token=token, library_name="datasets", library_version=__version__)
    else:
        return {}
    
def fsspec_head(url, storage_options=None):
    _raise_if_offline_mode_is_enabled(f"Tried to reach {url}")
    fs, path = url_to_fs(url, **(storage_options or {}))
    return fs.info(path)

def get_from_cache(
    url,
    cache_dir=None,
    force_download=False,
    user_agent=None,
    use_etag=True,
    token=None,
    storage_options=None,
    download_desc=None,
    disable_tqdm=False,
) -> str:
    if storage_options is None:
        storage_options = {}
    if cache_dir is None:
        cache_dir = data_config.HF_DATASETS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    response = None
    etag = None

    filename = hash_url_to_filename(url, etag=None)
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and not force_download and not use_etag:
        return cache_path

    headers = get_authentication_headers_for_url(url, token=token)
    if user_agent is not None:
        headers["user-agent"] = user_agent

    response = fsspec_head(url, storage_options=storage_options)
    etag = (response.get("ETag", None) or response.get("etag", None)) if use_etag else None

    filename = hash_url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and not force_download:
        return cache_path

    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        if os.path.exists(cache_path) and not force_download:
            return cache_path

        incomplete_path = cache_path + ".incomplete"

        @contextmanager
        def temp_file_manager(mode="w+b"):
            with open(incomplete_path, mode) as f:
                yield f

        with temp_file_manager() as temp_file:
            fsspec_get(url, temp_file, storage_options=storage_options, desc=download_desc, disable_tqdm=disable_tqdm)

        shutil.move(temp_file.name, cache_path)
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(meta, meta_file)

    return cache_path

def relative_to_absolute_path(path: P) -> P:
    abs_path_str = os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))
    return Path(abs_path_str) if isinstance(path, Path) else abs_path_str

class BaseExtractor(ABC):
    @classmethod
    @abstractmethod
    def is_extractable(cls, path: Union[Path, str], **kwargs) -> bool: ...

    @staticmethod
    @abstractmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None: ...

class MagicNumberBaseExtractor(BaseExtractor, ABC):
    magic_numbers: list[bytes] = []

    @staticmethod
    def read_magic_number(path: Union[Path, str], magic_number_length: int):
        with open(path, "rb") as f:
            return f.read(magic_number_length)

    @classmethod
    def is_extractable(cls, path: Union[Path, str], magic_number: bytes = b"") -> bool:
        if not magic_number:
            magic_number_length = max(len(cls_magic_number) for cls_magic_number in cls.magic_numbers)
            try:
                magic_number = cls.read_magic_number(path, magic_number_length)
            except OSError:
                return False
        return any(magic_number.startswith(cls_magic_number) for cls_magic_number in cls.magic_numbers)

class TarExtractor(BaseExtractor):
    @classmethod
    def is_extractable(cls, path: Union[Path, str], **kwargs) -> bool:
        return tarfile.is_tarfile(path)

    @staticmethod
    def safemembers(members, output_path):
        
        def resolved(path: str) -> str:
            return os.path.realpath(os.path.abspath(path))

        def badpath(path: str, base: str) -> bool:
            return not resolved(os.path.join(base, path)).startswith(base)

        def badlink(info, base: str) -> bool:
            tip = resolved(os.path.join(base, os.path.dirname(info.name)))
            return badpath(info.linkname, base=tip)

        base = resolved(output_path)

        for finfo in members:
            if badpath(finfo.name, base):
                print(f"Extraction of {finfo.name} is blocked (illegal path)")
            elif finfo.issym() and badlink(finfo, base):
                print(f"Extraction of {finfo.name} is blocked: Symlink to {finfo.linkname}")
            elif finfo.islnk() and badlink(finfo, base):
                print(f"Extraction of {finfo.name} is blocked: Hard link to {finfo.linkname}")
            else:
                yield finfo

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        os.makedirs(output_path, exist_ok=True)
        tar_file = tarfile.open(input_path)
        tar_file.extractall(output_path, members=TarExtractor.safemembers(tar_file, output_path))
        tar_file.close()

class GzipExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b"\x1f\x8b"]

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        with gzip.open(input_path, "rb") as gzip_file:
            with open(output_path, "wb") as extracted_file:
                shutil.copyfileobj(gzip_file, extracted_file)

class ZipExtractor(MagicNumberBaseExtractor):
    magic_numbers = [
        b"PK\x03\x04",
        b"PK\x05\x06",  # empty archive
        b"PK\x07\x08",  # spanned archive
    ]

    @classmethod
    def is_extractable(cls, path: Union[Path, str], magic_number: bytes = b"") -> bool:
        if super().is_extractable(path, magic_number=magic_number):
            return True
        try:
            from zipfile import (
                _CD_SIGNATURE,
                _ECD_DISK_NUMBER,
                _ECD_DISK_START,
                _ECD_ENTRIES_TOTAL,
                _ECD_OFFSET,
                _ECD_SIZE,
                _EndRecData,
                sizeCentralDir,
                stringCentralDir,
                structCentralDir,
            )

            with open(path, "rb") as fp:
                endrec = _EndRecData(fp)
                if endrec:
                    if endrec[_ECD_ENTRIES_TOTAL] == 0 and endrec[_ECD_SIZE] == 0 and endrec[_ECD_OFFSET] == 0:
                        return True  # Empty zipfiles are still zipfiles
                    elif endrec[_ECD_DISK_NUMBER] == endrec[_ECD_DISK_START]:
                        fp.seek(endrec[_ECD_OFFSET])  # Central directory is on the same disk
                        if fp.tell() == endrec[_ECD_OFFSET] and endrec[_ECD_SIZE] >= sizeCentralDir:
                            data = fp.read(sizeCentralDir)  # CD is where we expect it to be
                            if len(data) == sizeCentralDir:
                                centdir = struct.unpack(structCentralDir, data)  # CD is the right size
                                if centdir[_CD_SIGNATURE] == stringCentralDir:
                                    return True  # First central directory entry  has correct magic number
            return False
        except Exception:  # catch all errors in case future python versions change the zipfile internals
            return False

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        os.makedirs(output_path, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zip_file:
            zip_file.extractall(output_path)
            zip_file.close()

class XzExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b"\xfd\x37\x7a\x58\x5a\x00"]

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        with lzma.open(input_path) as compressed_file:
            with open(output_path, "wb") as extracted_file:
                shutil.copyfileobj(compressed_file, extracted_file)

class Bzip2Extractor(MagicNumberBaseExtractor):
    magic_numbers = [b"\x42\x5a\x68"]

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        with bz2.open(input_path, "rb") as compressed_file:
            with open(output_path, "wb") as extracted_file:
                shutil.copyfileobj(compressed_file, extracted_file)

class Extractor:
    extractors: dict[str, type[BaseExtractor]] = {
        "tar": TarExtractor,
        "gzip": GzipExtractor,
        "zip": ZipExtractor,
        "xz": XzExtractor,
        "bz2": Bzip2Extractor,
    }

    @classmethod
    def _get_magic_number_max_length(cls):
        return max(
            len(extractor_magic_number)
            for extractor in cls.extractors.values()
            if issubclass(extractor, MagicNumberBaseExtractor)
            for extractor_magic_number in extractor.magic_numbers
        )

    @staticmethod
    def _read_magic_number(path: Union[Path, str], magic_number_length: int):
        try:
            return MagicNumberBaseExtractor.read_magic_number(path, magic_number_length=magic_number_length)
        except OSError:
            return b""

    @classmethod
    def is_extractable(cls, path: Union[Path, str], return_extractor: bool = False) -> bool:
        extractor_format = cls.infer_extractor_format(path)
        if extractor_format:
            return True if not return_extractor else (True, cls.extractors[extractor_format])
        return False if not return_extractor else (False, None)

    @classmethod
    def infer_extractor_format(cls, path: Union[Path, str]) -> Optional[str]:  # <Added version="2.4.0"/>
        magic_number_max_length = cls._get_magic_number_max_length()
        magic_number = cls._read_magic_number(path, magic_number_max_length)
        for extractor_format, extractor in cls.extractors.items():
            if extractor.is_extractable(path, magic_number=magic_number):
                return extractor_format

    @classmethod
    def extract(cls, input_path: Union[Path, str], output_path: Union[Path, str], extractor_format: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lock_path = str(Path(output_path).with_suffix(".lock"))
        with FileLock(lock_path):
            shutil.rmtree(output_path, ignore_errors=True)
            extractor = cls.extractors[extractor_format]
            return extractor.extract(input_path, output_path)

class ExtractManager:
    def __init__(self, cache_dir: Optional[str] = None):
        self.extract_dir = (os.path.join(cache_dir, data_config.EXTRACTED_DATASETS_DIR) if cache_dir else data_config.EXTRACTED_DATASETS_PATH)
        self.extractor = Extractor

    def _get_output_path(self, path: str) -> str:
        abs_path = os.path.abspath(path)
        return os.path.join(self.extract_dir, hash_url_to_filename(abs_path))

    def _do_extract(self, output_path: str, force_extract: bool) -> bool:
        return force_extract or (not os.path.isfile(output_path) and not (os.path.isdir(output_path) and os.listdir(output_path)))

    def extract(self, input_path: str, force_extract: bool = False) -> str:
        extractor_format = self.extractor.infer_extractor_format(input_path)
        if not extractor_format:
            return input_path
        output_path = self._get_output_path(input_path)
        if self._do_extract(output_path, force_extract):
            self.extractor.extract(input_path, output_path, extractor_format)
        return output_path

def cached_path(url_or_filename, download_config=None, **download_kwargs) -> str:
    if download_config is None:
        download_config = DownloadConfig(**download_kwargs)

    cache_dir = download_config.cache_dir or data_config.DOWNLOADED_DATASETS_PATH
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    if can_be_local(url_or_filename):
        url_or_filename = strip_protocol(url_or_filename)

    if is_remote_url(url_or_filename):
        url_or_filename, storage_options = _prepare_path_and_storage_options(url_or_filename, download_config=download_config)
        if url_or_filename.startswith("hf://"):
            resolved_path = huggingface_hub.HfFileSystem(endpoint=data_config.HF_ENDPOINT, token=download_config.token).resolve_path(url_or_filename)
            try:
                output_path = huggingface_hub.HfApi(
                    endpoint=data_config.HF_ENDPOINT,
                    token=download_config.token,
                    library_name="datasets",
                    library_version=__version__,
                    user_agent=get_datasets_user_agent(download_config.user_agent),
                ).hf_hub_download(
                    repo_id=resolved_path.repo_id,
                    repo_type=resolved_path.repo_type,
                    revision=resolved_path.revision,
                    filename=resolved_path.path_in_repo,
                    force_download=download_config.force_download,
                    proxies=download_config.proxies,
                )
            except (
                huggingface_hub.utils.RepositoryNotFoundError,
                huggingface_hub.utils.EntryNotFoundError,
                huggingface_hub.utils.RevisionNotFoundError,
                huggingface_hub.utils.GatedRepoError,
            ) as e:
                raise FileNotFoundError(str(e)) from e
        else:
            output_path = get_from_cache(
                url_or_filename,
                cache_dir=cache_dir,
                force_download=download_config.force_download,
                user_agent=download_config.user_agent,
                use_etag=download_config.use_etag,
                token=download_config.token,
                storage_options=storage_options,
                download_desc=download_config.download_desc,
                disable_tqdm=download_config.disable_tqdm,
            )
    elif os.path.exists(url_or_filename):
        output_path = url_or_filename
    elif is_local_path(url_or_filename):
        raise FileNotFoundError(f"Local file {url_or_filename} doesn't exist")
    else:
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")

    if output_path is None:
        return output_path

    if download_config.extract_compressed_file:
        if download_config.extract_on_the_fly:
            protocol = _get_extraction_protocol(output_path, download_config=download_config)
            extension = _get_path_extension(url_or_filename.split("::")[0])
            if (protocol and extension not in ["tgz", "tar"] and not url_or_filename.split("::")[0].endswith((".tar.gz", ".tar.bz2", ".tar.xz"))):
                output_path = relative_to_absolute_path(output_path)
                if protocol in SINGLE_FILE_COMPRESSION_PROTOCOLS:
                    inner_file = os.path.basename(output_path)
                    inner_file = inner_file[: inner_file.rindex(".")] if "." in inner_file else inner_file
                    output_path = f"{protocol}://{inner_file}::{output_path}"
                else:
                    output_path = f"{protocol}://::{output_path}"
                return output_path

        output_path = ExtractManager(cache_dir=download_config.cache_dir).extract(output_path, force_extract=download_config.force_extract)
    return relative_to_absolute_path(output_path)

class tracked_str(str):
    origins = {}

    def set_origin(self, origin: str):
        if super().__repr__() not in self.origins:
            self.origins[super().__repr__()] = origin

    def get_origin(self):
        return self.origins.get(super().__repr__(), str(self))

    def __repr__(self) -> str:
        if super().__repr__() not in self.origins or self.origins[super().__repr__()] == self:
            return super().__repr__()
        else:
            return f"{str(self)} (origin={self.origins[super().__repr__()]})"

class TrackedIterableFromGenerator(collections.abc.Iterable):
    def __init__(self, generator, *args):
        super().__init__()
        self.generator = generator
        self.args = args
        self.last_item = None

    def __iter__(self):
        for x in self.generator(*self.args):
            self.last_item = x
            yield x
        self.last_item = None

    def __repr__(self) -> str:
        if self.last_item is None:
            return super().__repr__()
        else:
            return f"{self.__class__.__name__}(current={self.last_item})"

    def __reduce__(self):
        return (self.__class__, (self.generator, *self.args))

class ArchiveIterable(TrackedIterableFromGenerator):
    @staticmethod
    def _iter_tar(f):
        stream = tarfile.open(fileobj=f, mode="r|*")
        for tarinfo in stream:
            file_path = tarinfo.name
            if not tarinfo.isreg():
                continue
            if file_path is None:
                continue
            if os.path.basename(file_path).startswith((".", "__")):
                # skipping hidden files
                continue
            file_obj = stream.extractfile(tarinfo)
            yield file_path, file_obj
            stream.members = []
        del stream

    @staticmethod
    def _iter_zip(f):
        zipf = zipfile.ZipFile(f)
        for member in zipf.infolist():
            file_path = member.filename
            if member.is_dir():
                continue
            if file_path is None:
                continue
            if os.path.basename(file_path).startswith((".", "__")):
                # skipping hidden files
                continue
            file_obj = zipf.open(member)
            yield file_path, file_obj

    @classmethod
    def _iter_from_fileobj(cls, f) -> Generator[tuple, None, None]:
        compression = _get_extraction_protocol_with_magic_number(f)
        if compression == "zip":
            yield from cls._iter_zip(f)
        else:
            yield from cls._iter_tar(f)

    @classmethod
    def _iter_from_urlpath(cls, urlpath: str, download_config: Optional[DownloadConfig] = None) -> Generator[tuple, None, None]:
        compression = _get_extraction_protocol(urlpath, download_config=download_config)
        with xopen(urlpath, "rb", download_config=download_config, block_size=0) as f:
            if compression == "zip":
                yield from cls._iter_zip(f)
            else:
                yield from cls._iter_tar(f)

    @classmethod
    def from_buf(cls, fileobj) -> "ArchiveIterable":
        return cls(cls._iter_from_fileobj, fileobj)

    @classmethod
    def from_urlpath(cls, urlpath_or_buf, download_config: Optional[DownloadConfig] = None) -> "ArchiveIterable":
        return cls(cls._iter_from_urlpath, urlpath_or_buf, download_config)

class FilesIterable(TrackedIterableFromGenerator):
    @classmethod
    def _iter_from_urlpaths(cls, urlpaths: Union[str, list[str]], download_config: Optional[DownloadConfig] = None) -> Generator[str, None, None]:
        if not isinstance(urlpaths, list):
            urlpaths = [urlpaths]
        for urlpath in urlpaths:
            if xisfile(urlpath, download_config=download_config):
                yield urlpath
            elif xisdir(urlpath, download_config=download_config):
                for dirpath, dirnames, filenames in xwalk(urlpath, download_config=download_config):
                    # in-place modification to prune the search
                    dirnames[:] = sorted([dirname for dirname in dirnames if not dirname.startswith((".", "__"))])
                    if xbasename(dirpath).startswith((".", "__")):
                        continue
                    for filename in sorted(filenames):
                        if filename.startswith((".", "__")):
                            continue
                        yield xjoin(dirpath, filename)
            else:
                raise FileNotFoundError(urlpath)

    @classmethod
    def from_urlpaths(cls, urlpaths, download_config: Optional[DownloadConfig] = None) -> "FilesIterable":
        return cls(cls._iter_from_urlpaths, urlpaths, download_config)

class DownloadManager:
    is_streaming = False

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        download_config: Optional[DownloadConfig] = None,
        base_path: Optional[str] = None,
        record_checksums=True,
    ):
        self._dataset_name = dataset_name
        self._data_dir = data_dir
        self._base_path = base_path or os.path.abspath(".")
        self._recorded_sizes_checksums: dict[str, dict[str, Optional[Union[int, str]]]] = {}
        self.record_checksums = record_checksums
        self.download_config = download_config or DownloadConfig()
        self.downloaded_paths = {}
        self.extracted_paths = {}

    @property
    def manual_dir(self):
        return self._data_dir

    @property
    def downloaded_size(self):
        return sum(checksums_dict["num_bytes"] for checksums_dict in self._recorded_sizes_checksums.values())

    def _record_sizes_checksums(self, url_or_urls: NestedDataStructure, downloaded_path_or_paths: NestedDataStructure):
        delay = 5
        for url, path in hf_tqdm(list(zip(url_or_urls.flatten(), downloaded_path_or_paths.flatten())), delay=delay, desc="Computing checksums"):
            self._recorded_sizes_checksums[str(url)] = get_size_checksum_dict(path, record_checksum=self.record_checksums)

    def download(self, url_or_urls):
        download_config = self.download_config.copy()
        download_config.extract_compressed_file = False
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading data"

        download_func = partial(self._download_batched, download_config=download_config)

        start_time = datetime.now()
        with stack_multiprocessing_download_progress_bars():
            downloaded_path_or_paths = map_nested(
                download_func,
                url_or_urls,
                map_tuple=True,
                num_proc=download_config.num_proc,
                desc="Downloading data files",
                batched=True,
                batch_size=-1,
            )
        duration = datetime.now() - start_time
        print(f"Downloading took {duration.total_seconds() // 60} min")
        url_or_urls = NestedDataStructure(url_or_urls)
        downloaded_path_or_paths = NestedDataStructure(downloaded_path_or_paths)
        self.downloaded_paths.update(dict(zip(url_or_urls.flatten(), downloaded_path_or_paths.flatten())))

        start_time = datetime.now()
        self._record_sizes_checksums(url_or_urls, downloaded_path_or_paths)
        duration = datetime.now() - start_time
        print(f"Checksum Computation took {duration.total_seconds() // 60} min")

        return downloaded_path_or_paths.data

    def _download_batched(
        self,
        url_or_filenames: list[str],
        download_config: DownloadConfig,
    ) -> list[str]:
        if len(url_or_filenames) >= 16:
            download_config = download_config.copy()
            download_config.disable_tqdm = True
            download_func = partial(self._download_single, download_config=download_config)

            fs: fsspec.AbstractFileSystem
            path = str(url_or_filenames[0])
            if is_relative_path(path):
                path = url_or_path_join(self._base_path, path)
            fs, path = url_to_fs(path, **download_config.storage_options)
            size = 0
            try:
                size = fs.info(path).get("size", 0)
            except Exception:
                pass
            max_workers = (
                data_config.HF_DATASETS_MULTITHREADING_MAX_WORKERS if size < (20 << 20) else 1
            )  # enable multithreading if files are small

            return thread_map(
                download_func,
                url_or_filenames,
                desc=download_config.download_desc or "Downloading",
                unit="files",
                position=multiprocessing.current_process()._identity[-1]  # contains the ranks of subprocesses
                if os.environ.get("HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS") == "1"
                and multiprocessing.current_process()._identity
                else None,
                max_workers=max_workers,
                tqdm_class=hf_tqdm,
            )
        else:
            return [self._download_single(url_or_filename, download_config=download_config) for url_or_filename in url_or_filenames]

    def _download_single(self, url_or_filename: str, download_config: DownloadConfig) -> str:
        url_or_filename = str(url_or_filename)
        if is_relative_path(url_or_filename):
            # append the relative path to the base_path
            url_or_filename = url_or_path_join(self._base_path, url_or_filename)
        out = cached_path(url_or_filename, download_config=download_config)
        out = tracked_str(out)
        out.set_origin(url_or_filename)
        return out

    def iter_archive(self, path_or_buf: Union[str, io.BufferedReader]):
        
        if hasattr(path_or_buf, "read"):
            return ArchiveIterable.from_buf(path_or_buf)
        else:
            return ArchiveIterable.from_urlpath(path_or_buf)

    def iter_files(self, paths: Union[str, list[str]]):
        return FilesIterable.from_urlpaths(paths)

    def extract(self, path_or_paths):

        download_config = self.download_config.copy()
        download_config.extract_compressed_file = True
        extract_func = partial(self._download_single, download_config=download_config)
        extracted_paths = map_nested(
            extract_func,
            path_or_paths,
            num_proc=download_config.num_proc,
            desc="Extracting data files",
        )
        path_or_paths = NestedDataStructure(path_or_paths)
        extracted_paths = NestedDataStructure(extracted_paths)
        self.extracted_paths.update(dict(zip(path_or_paths.flatten(), extracted_paths.flatten())))
        return extracted_paths.data

    def download_and_extract(self, url_or_urls):
        return self.extract(self.download(url_or_urls))

    def get_recorded_sizes_checksums(self):
        return self._recorded_sizes_checksums.copy()

    def delete_extracted_files(self):
        paths_to_delete = set(self.extracted_paths.values()) - set(self.downloaded_paths.values())
        for key, path in list(self.extracted_paths.items()):
            if path in paths_to_delete and os.path.isfile(path):
                os.remove(path)
                del self.extracted_paths[key]

    def manage_extracted_files(self):
        if self.download_config.delete_extracted:
            self.delete_extracted_files()

def has_sufficient_disk_space(needed_bytes, directory="."):
    try:
        free_bytes = disk_usage(os.path.abspath(directory)).free
    except OSError:
        return True
    return needed_bytes < free_bytes

def size_str(size_in_bytes):
    if not size_in_bytes:
        return "Unknown size"

    _NAME_LIST = [("PiB", 2**50), ("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)]

    size_in_bytes = float(size_in_bytes)
    for name, size_bytes in _NAME_LIST:
        value = size_in_bytes / size_bytes
        if value >= 1.0:
            return f"{value:.2f} {name}"
    return f"{int(size_in_bytes)} bytes"

@contextmanager
def temporary_assignment(obj, attr, value):
    original = getattr(obj, attr, None)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, original)

class DatasetsError(Exception):
    """Base class for exceptions in this library."""

class DatasetBuildError(DatasetsError):
    pass

class ManualDownloadError(DatasetBuildError):
    pass

class ChecksumVerificationError(DatasetsError):
    """Error raised during checksums verifications of downloaded files."""

class UnexpectedDownloadedFileError(ChecksumVerificationError):
    """Some downloaded files were not expected."""

class ExpectedMoreDownloadedFilesError(ChecksumVerificationError):
    """Some files were supposed to be downloaded but were not."""

class NonMatchingChecksumError(ChecksumVerificationError):
    """The downloaded file checksum don't match the expected checksum."""

class SplitsVerificationError(DatasetsError):
    """Error raised during splits verifications."""

class UnexpectedSplitsError(SplitsVerificationError):
    """The expected splits of the downloaded file is missing."""

class ExpectedMoreSplitsError(SplitsVerificationError):
    """Some recorded splits are missing."""

class NonMatchingSplitsSizesError(SplitsVerificationError):
    """The splits sizes don't match the expected splits sizes."""

class FileFormatError(DatasetBuildError):
    pass

class DatasetGenerationError(DatasetBuildError):
    pass

def verify_checksums(expected_checksums: Optional[dict], recorded_checksums: dict, verification_name=None):
    if expected_checksums is None:
        return
    if len(set(expected_checksums) - set(recorded_checksums)) > 0:
        raise ExpectedMoreDownloadedFilesError(str(set(expected_checksums) - set(recorded_checksums)))
    if len(set(recorded_checksums) - set(expected_checksums)) > 0:
        raise UnexpectedDownloadedFileError(str(set(recorded_checksums) - set(expected_checksums)))
    bad_urls = [url for url in expected_checksums if expected_checksums[url] != recorded_checksums[url]]
    for_verification_name = " for " + verification_name if verification_name is not None else ""
    if len(bad_urls) > 0:
        raise NonMatchingChecksumError(f"Checksums didn't match{for_verification_name}:\n {bad_urls}\n Set `verification_mode='no_checks'` to skip checksums verification and ignore this error")

class DuplicatedKeysError(Exception):
    def __init__(self, key, duplicate_key_indices, fix_msg=""):
        self.key = key
        self.duplicate_key_indices = duplicate_key_indices
        self.fix_msg = fix_msg
        self.prefix = "Found multiple examples generated with the same key"
        if len(duplicate_key_indices) <= 20:
            self.err_msg = f"\nThe examples at index {', '.join(duplicate_key_indices)} have the key {key}"
        else:
            self.err_msg = f"\nThe examples at index {', '.join(duplicate_key_indices[:20])}... ({len(duplicate_key_indices) - 20} more) have the key {key}"
        self.suffix = "\n" + fix_msg if fix_msg else ""
        super().__init__(f"{self.prefix}{self.err_msg}{self.suffix}")

def verify_splits(expected_splits: Optional[dict], recorded_splits: dict):
    if expected_splits is None:
        return
    if len(set(expected_splits) - set(recorded_splits)) > 0:
        raise ExpectedMoreSplitsError(str(set(expected_splits) - set(recorded_splits)))
    if len(set(recorded_splits) - set(expected_splits)) > 0:
        raise UnexpectedSplitsError(str(set(recorded_splits) - set(expected_splits)))
    bad_splits = [
        {"expected": expected_splits[name], "recorded": recorded_splits[name]}
        for name in expected_splits
        if expected_splits[name].num_examples != recorded_splits[name].num_examples
    ]
    if len(bad_splits) > 0:
        raise NonMatchingSplitsSizesError(str(bad_splits))

class ReadInstruction:
    def _init(self, relative_instructions):
        self._relative_instructions = relative_instructions

    @classmethod
    def _read_instruction_from_relative_instructions(cls, relative_instructions):
        result = cls.__new__(cls)
        result._init(relative_instructions)  # pylint: disable=protected-access
        return result

    def __init__(self, split_name, rounding=None, from_=None, to=None, unit=None):
        self._init([_RelativeInstruction(split_name, from_, to, unit, rounding)])

    @classmethod
    def from_spec(cls, spec):
        spec = str(spec)  # Need to convert to str in case of NamedSplit instance.
        subs = _ADDITION_SEP_RE.split(spec)
        if not subs:
            raise ValueError(f"No instructions could be built out of {spec}")
        instruction = _str_to_read_instruction(subs[0])
        return sum((_str_to_read_instruction(sub) for sub in subs[1:]), instruction)

    def to_spec(self):
        rel_instr_specs = []
        for rel_instr in self._relative_instructions:
            rel_instr_spec = rel_instr.splitname
            if rel_instr.from_ is not None or rel_instr.to is not None:
                from_ = rel_instr.from_
                to = rel_instr.to
                unit = rel_instr.unit
                rounding = rel_instr.rounding
                unit = unit if unit == "%" else ""
                from_ = str(from_) + unit if from_ is not None else ""
                to = str(to) + unit if to is not None else ""
                slice_str = f"[{from_}:{to}]"
                rounding_str = (
                    f"({rounding})" if unit == "%" and rounding is not None and rounding != "closest" else ""
                )
                rel_instr_spec += slice_str + rounding_str
            rel_instr_specs.append(rel_instr_spec)
        return "+".join(rel_instr_specs)

    def __add__(self, other):
        if not isinstance(other, ReadInstruction):
            raise TypeError("ReadInstruction can only be added to another ReadInstruction obj.")
        self_ris = self._relative_instructions
        other_ris = other._relative_instructions  # pylint: disable=protected-access
        if (self_ris[0].unit != "abs" and other_ris[0].unit != "abs" and self._relative_instructions[0].rounding != other_ris[0].rounding):
            raise ValueError("It is forbidden to sum ReadInstruction instances with different rounding values.")
        return self._read_instruction_from_relative_instructions(self_ris + other_ris)

    def __str__(self):
        return self.to_spec()

    def __repr__(self):
        return f"ReadInstruction({self._relative_instructions})"

    def to_absolute(self, name2len):
        return [_rel_to_abs_instr(rel_instr, name2len) for rel_instr in self._relative_instructions]

def _deepcopy(x, memo: dict):
    cls = x.__class__
    result = cls.__new__(cls)
    memo[id(x)] = result
    for k, v in x.__dict__.items():
        setattr(result, k, copy.deepcopy(v, memo))
    return result

def _interpolation_search(arr: list[int], x: int) -> int:
    i, j = 0, len(arr) - 1
    while i < j and arr[i] <= x < arr[j]:
        k = i + ((j - i) * (x - arr[i]) // (arr[j] - arr[i]))
        if arr[k] <= x < arr[k + 1]:
            return k
        elif arr[k] < x:
            i, j = k + 1, j
        else:
            i, j = i, k
    raise IndexError(f"Invalid query '{x}' for size {arr[-1] if len(arr) else 'none'}.")

def _in_memory_arrow_table_from_file(filename: str) -> pa.Table:
    in_memory_stream = pa.input_stream(filename)
    opened_stream = pa.ipc.open_stream(in_memory_stream)
    pa_table = opened_stream.read_all()
    return pa_table

def _in_memory_arrow_table_from_buffer(buffer: pa.Buffer) -> pa.Table:
    stream = pa.BufferReader(buffer)
    opened_stream = pa.ipc.open_stream(stream)
    table = opened_stream.read_all()
    return table

def table_flatten(table: pa.Table):
    features = Features.from_arrow_schema(table.schema)
    if any(hasattr(subfeature, "flatten") and subfeature.flatten() == subfeature for subfeature in features.values()):
        flat_arrays = []
        flat_column_names = []
        for field in table.schema:
            array = table.column(field.name)
            subfeature = features[field.name]
            if pa.types.is_struct(field.type) and (not hasattr(subfeature, "flatten") or subfeature.flatten() != subfeature):
                flat_arrays.extend(array.flatten())
                flat_column_names.extend([f"{field.name}.{subfield.name}" for subfield in field.type])
            else:
                flat_arrays.append(array)
                flat_column_names.append(field.name)
        flat_table = pa.Table.from_arrays(flat_arrays, names=flat_column_names)
    else:
        flat_table = table.flatten()
    flat_features = features.flatten(max_depth=2)
    flat_features = Features({column_name: flat_features[column_name] for column_name in flat_table.column_names})
    return flat_table.replace_schema_metadata(flat_features.arrow_schema.metadata)

@_wrap_for_chunked_arrays
def cast_array_to_feature(array: pa.Array, feature: "FeatureType", allow_primitive_to_str: bool = True, allow_decimal_to_str: bool = True) -> pa.Array:
    _c = partial(cast_array_to_feature, allow_primitive_to_str=allow_primitive_to_str, allow_decimal_to_str=allow_decimal_to_str)

    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if hasattr(feature, "cast_storage"):
        return feature.cast_storage(array)

    elif pa.types.is_struct(array.type):
        if isinstance(feature, Sequence) and isinstance(feature.feature, dict):
            sequence_kwargs = vars(feature).copy()
            feature = sequence_kwargs.pop("feature")
            feature = {name: Sequence(subfeature, **sequence_kwargs) for name, subfeature in feature.items()}
        if isinstance(feature, dict) and (array_fields := {field.name for field in array.type}) <= set(feature):
            null_array = pa.array([None] * len(array))
            arrays = [_c(array.field(name) if name in array_fields else null_array, subfeature) for name, subfeature in feature.items()]
            return pa.StructArray.from_arrays(arrays, names=list(feature), mask=array.is_null())
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        if isinstance(feature, list):
            casted_array_values = _c(array.values, feature[0])
            if pa.types.is_list(array.type) and casted_array_values.type == array.values.type:
                return array
            else:
                array_offsets = _combine_list_array_offsets_with_mask(array)
                return pa.ListArray.from_arrays(array_offsets, casted_array_values)
        elif isinstance(feature, LargeList):
            casted_array_values = _c(array.values, feature.feature)
            if pa.types.is_large_list(array.type) and casted_array_values.type == array.values.type:
                return array
            else:
                array_offsets = _combine_list_array_offsets_with_mask(array)
                return pa.LargeListArray.from_arrays(array_offsets, casted_array_values)
        elif isinstance(feature, Sequence):
            if feature.length > -1:
                if _are_list_values_of_length(array, feature.length):
                    if array.null_count > 0:
                        array_type = array.type
                        storage_type = _storage_type(array_type)
                        if array_type != storage_type:
                            array = array_cast(array, storage_type, allow_primitive_to_str=allow_primitive_to_str, allow_decimal_to_str=allow_decimal_to_str)
                            array = pc.list_slice(array, 0, feature.length, return_fixed_size_list=True)
                            array = array_cast(array, array_type, allow_primitive_to_str=allow_primitive_to_str, allow_decimal_to_str=allow_decimal_to_str)
                        else:
                            array = pc.list_slice(array, 0, feature.length, return_fixed_size_list=True)
                        array_values = array.values
                        casted_array_values = _c(array_values, feature.feature)
                        return pa.FixedSizeListArray.from_arrays(casted_array_values, feature.length, mask=array.is_null())
                    else:
                        array_values = array.values[array.offset * feature.length : (array.offset + len(array)) * feature.length]
                        return pa.FixedSizeListArray.from_arrays(_c(array_values, feature.feature), feature.length)
            else:
                casted_array_values = _c(array.values, feature.feature)
                if pa.types.is_list(array.type) and casted_array_values.type == array.values.type:
                    return array
                else:
                    array_offsets = _combine_list_array_offsets_with_mask(array)
                    return pa.ListArray.from_arrays(array_offsets, casted_array_values)
    elif pa.types.is_fixed_size_list(array.type):
        if isinstance(feature, list):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.ListArray.from_arrays(array_offsets, _c(array.values, feature[0]), mask=array.is_null())
        elif isinstance(feature, LargeList):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.LargeListArray.from_arrays(array_offsets, _c(array.values, feature.feature), mask=array.is_null())
        elif isinstance(feature, Sequence):
            if feature.length > -1:
                if feature.length == array.type.list_size:
                    array_values = array.values[array.offset * array.type.list_size : (array.offset + len(array)) * array.type.list_size]
                    casted_array_values = _c(array_values, feature.feature)
                    return pa.FixedSizeListArray.from_arrays(casted_array_values, feature.length, mask=array.is_null())
            else:
                array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
                return pa.ListArray.from_arrays(array_offsets, _c(array.values, feature.feature), mask=array.is_null())
    if pa.types.is_null(array.type):
        return array_cast(
            array,
            get_nested_type(feature),
            allow_primitive_to_str=allow_primitive_to_str,
            allow_decimal_to_str=allow_decimal_to_str,
        )
    elif not isinstance(feature, (Sequence, dict, list, tuple)):
        return array_cast(
            array,
            feature(),
            allow_primitive_to_str=allow_primitive_to_str,
            allow_decimal_to_str=allow_decimal_to_str,
        )
    raise TypeError(f"Couldn't cast array of type\n{_short_str(array.type)}\nto\n{_short_str(feature)}")

class CastError(ValueError):

    def __init__(self, *args, table_column_names: list[str], requested_column_names: list[str]) -> None:
        super().__init__(*args)
        self.table_column_names = table_column_names
        self.requested_column_names = requested_column_names

    def __reduce__(self):
        return partial(CastError, table_column_names=self.table_column_names, requested_column_names=self.requested_column_names), ()

    def details(self):
        new_columns = set(self.table_column_names) - set(self.requested_column_names)
        missing_columns = set(self.requested_column_names) - set(self.table_column_names)
        if new_columns and missing_columns:
            return f"there are {len(new_columns)} new columns ({_short_str(new_columns)}) and {len(missing_columns)} missing columns ({_short_str(missing_columns)})."
        elif new_columns:
            return f"there are {len(new_columns)} new columns ({_short_str(new_columns)})"
        else:
            return f"there are {len(missing_columns)} missing columns ({_short_str(missing_columns)})"

def cast_table_to_schema(table: pa.Table, schema: pa.Schema):
    features = Features.from_arrow_schema(schema)
    table_column_names = set(table.column_names)
    if not table_column_names <= set(schema.names):
        raise CastError(
            f"Couldn't cast\n{_short_str(table.schema)}\nto\n{_short_str(features)}\nbecause column names don't match",
            table_column_names=table.column_names,
            requested_column_names=list(features),
        )
    arrays = [
        cast_array_to_feature(
            table[name] if name in table_column_names else pa.array([None] * len(table), type=schema.field(name).type),
            feature,
        )
        for name, feature in features.items()
    ]
    return pa.Table.from_arrays(arrays, schema=schema)

def table_cast(table: pa.Table, schema: pa.Schema):
    if table.schema != schema:
        return cast_table_to_schema(table, schema)
    elif table.schema.metadata != schema.metadata:
        return table.replace_schema_metadata(schema.metadata)
    else:
        return table

class IndexedTableMixin:
    def __init__(self, table: pa.Table):
        self._schema: pa.Schema = table.schema
        self._batches: list[pa.RecordBatch] = [recordbatch for recordbatch in table.to_batches() if len(recordbatch) > 0]
        self._offsets: np.ndarray = np.cumsum([0] + [len(b) for b in self._batches], dtype=np.int64)

    def fast_gather(self, indices: Union[list[int], np.ndarray]) -> pa.Table:
        if not len(indices):
            raise ValueError("Indices must be non-empty")
        batch_indices = np.searchsorted(self._offsets, indices, side="right") - 1
        return pa.Table.from_batches(
            [
                self._batches[batch_idx].slice(i - self._offsets[batch_idx], 1)
                for batch_idx, i in zip(batch_indices, indices)
            ],
            schema=self._schema,
        )

    def fast_slice(self, offset=0, length=None) -> pa.Table:
        if offset < 0:
            raise IndexError("Offset must be non-negative")
        elif offset >= self._offsets[-1] or (length is not None and length <= 0):
            return pa.Table.from_batches([], schema=self._schema)
        i = _interpolation_search(self._offsets, offset)
        if length is None or length + offset >= self._offsets[-1]:
            batches = self._batches[i:]
            batches[0] = batches[0].slice(offset - self._offsets[i])
        else:
            j = _interpolation_search(self._offsets, offset + length - 1)
            batches = self._batches[i : j + 1]
            batches[-1] = batches[-1].slice(0, offset + length - self._offsets[j])
            batches[0] = batches[0].slice(offset - self._offsets[i])
        return pa.Table.from_batches(batches, schema=self._schema)

class Table(IndexedTableMixin):
    def __init__(self, table: pa.Table):
        super().__init__(table)
        self.table = table

    def __deepcopy__(self, memo: dict):
        memo[id(self.table)] = self.table
        memo[id(self._batches)] = list(self._batches)
        return _deepcopy(self, memo)

    def validate(self, *args, **kwargs):
        return self.table.validate(*args, **kwargs)

    def equals(self, *args, **kwargs):
        args = tuple(arg.table if isinstance(arg, Table) else arg for arg in args)
        kwargs = {k: v.table if isinstance(v, Table) else v for k, v in kwargs}
        return self.table.equals(*args, **kwargs)

    def to_batches(self, *args, **kwargs):
        return self.table.to_batches(*args, **kwargs)

    def to_pydict(self, *args, **kwargs):
        return self.table.to_pydict(*args, **kwargs)

    def to_pylist(self, *args, **kwargs):
        return self.table.to_pylist(*args, **kwargs)

    def to_pandas(self, *args, **kwargs):
        return self.table.to_pandas(*args, **kwargs)

    def to_string(self, *args, **kwargs):
        return self.table.to_string(*args, **kwargs)

    def to_reader(self, max_chunksize: Optional[int] = None):
        return self.table.to_reader(max_chunksize=max_chunksize)

    def field(self, *args, **kwargs):
        return self.table.field(*args, **kwargs)

    def column(self, *args, **kwargs):
        return self.table.column(*args, **kwargs)

    def itercolumns(self, *args, **kwargs):
        return self.table.itercolumns(*args, **kwargs)

    @property
    def schema(self):
        return self.table.schema

    @property
    def columns(self):
        return self.table.columns

    @property
    def num_columns(self):
        return self.table.num_columns

    @property
    def num_rows(self):
        return self.table.num_rows

    @property
    def shape(self):
        return self.table.shape

    @property
    def nbytes(self):
        return self.table.nbytes

    @property
    def column_names(self):
        return self.table.column_names

    def __eq__(self, other):
        return self.equals(other)

    def __getitem__(self, i):
        return self.table[i]

    def __len__(self):
        return len(self.table)

    def __repr__(self):
        return self.table.__repr__().replace("pyarrow.Table", self.__class__.__name__)

    def __str__(self):
        return self.table.__str__().replace("pyarrow.Table", self.__class__.__name__)

    def slice(self, *args, **kwargs):
        raise NotImplementedError()

    def filter(self, *args, **kwargs):
        raise NotImplementedError()

    def flatten(self, *args, **kwargs):
        raise NotImplementedError()

    def combine_chunks(self, *args, **kwargs):
        raise NotImplementedError()

    def cast(self, *args, **kwargs):
        raise NotImplementedError()

    def replace_schema_metadata(self, *args, **kwargs):
        raise NotImplementedError()

    def add_column(self, *args, **kwargs):
        raise NotImplementedError()

    def append_column(self, *args, **kwargs):
        raise NotImplementedError()

    def remove_column(self, *args, **kwargs):
        raise NotImplementedError()

    def set_column(self, *args, **kwargs):
        raise NotImplementedError()

    def rename_columns(self, *args, **kwargs):
        raise NotImplementedError()

    def drop(self, *args, **kwargs):
        raise NotImplementedError()

    def select(self, *args, **kwargs):
        raise NotImplementedError()

class TableBlock(Table):
    pass

class InMemoryTable(TableBlock):
    @classmethod
    def from_file(cls, filename: str):
        table = _in_memory_arrow_table_from_file(filename)
        return cls(table)

    @classmethod
    def from_buffer(cls, buffer: pa.Buffer):
        table = _in_memory_arrow_table_from_buffer(buffer)
        return cls(table)

    @classmethod
    def from_pandas(cls, *args, **kwargs):
        return cls(pa.Table.from_pandas(*args, **kwargs))

    @classmethod
    def from_arrays(cls, *args, **kwargs):
        return cls(pa.Table.from_arrays(*args, **kwargs))

    @classmethod
    def from_pydict(cls, *args, **kwargs):
        return cls(pa.Table.from_pydict(*args, **kwargs))

    @classmethod
    def from_pylist(cls, mapping, *args, **kwargs):
        return cls(pa.Table.from_pylist(mapping, *args, **kwargs))

    @classmethod
    def from_batches(cls, *args, **kwargs):
        return cls(pa.Table.from_batches(*args, **kwargs))

    def slice(self, offset=0, length=None):
        return InMemoryTable(self.fast_slice(offset=offset, length=length))

    def filter(self, *args, **kwargs):
        return InMemoryTable(self.table.filter(*args, **kwargs))

    def flatten(self, *args, **kwargs):
        return InMemoryTable(table_flatten(self.table, *args, **kwargs))

    def combine_chunks(self, *args, **kwargs):
        return InMemoryTable(self.table.combine_chunks(*args, **kwargs))

    def cast(self, *args, **kwargs):
        return InMemoryTable(table_cast(self.table, *args, **kwargs))

    def replace_schema_metadata(self, *args, **kwargs):
        return InMemoryTable(self.table.replace_schema_metadata(*args, **kwargs))

    def add_column(self, *args, **kwargs):
        return InMemoryTable(self.table.add_column(*args, **kwargs))

    def append_column(self, *args, **kwargs):
        return InMemoryTable(self.table.append_column(*args, **kwargs))

    def remove_column(self, *args, **kwargs):
        return InMemoryTable(self.table.remove_column(*args, **kwargs))

    def set_column(self, *args, **kwargs):
        return InMemoryTable(self.table.set_column(*args, **kwargs))

    def rename_columns(self, *args, **kwargs):
        return InMemoryTable(self.table.rename_columns(*args, **kwargs))

    def drop(self, *args, **kwargs):
        return InMemoryTable(self.table.drop(*args, **kwargs))

    def select(self, *args, **kwargs):
        return InMemoryTable(self.table.select(*args, **kwargs))

def concat_tables(tables: list[Table], axis: int = 0) -> Table:
    tables = list(tables)
    if len(tables) == 1:
        return tables[0]
    return ConcatenationTable.from_tables(tables, axis=axis)

TableBlockContainer = TypeVar("TableBlockContainer", TableBlock, list[TableBlock], list[list[TableBlock]])

class ConcatenationTable(Table):
    def __init__(self, table: pa.Table, blocks: list[list[TableBlock]]):
        super().__init__(table)
        self.blocks = blocks
        for subtables in blocks:
            for subtable in subtables:
                if not isinstance(subtable, TableBlock):
                    raise TypeError(f"The blocks of a ConcatenationTable must be InMemoryTable or MemoryMappedTable objects, but got {_short_str(subtable)}.")

    def __getstate__(self):
        return {"blocks": self.blocks, "schema": self.table.schema}

    def __setstate__(self, state):
        blocks = state["blocks"]
        schema = state["schema"]
        table = self._concat_blocks_horizontally_and_vertically(blocks)
        if schema is not None and table.schema != schema:
            empty_table = pa.Table.from_batches([], schema=schema)
            table = pa.concat_tables([table, empty_table], promote_options="default")
        ConcatenationTable.__init__(self, table, blocks=blocks)

    @staticmethod
    def _concat_blocks(blocks: list[Union[TableBlock, pa.Table]], axis: int = 0) -> pa.Table:
        pa_tables = [table.table if hasattr(table, "table") else table for table in blocks]
        if axis == 0:
            return pa.concat_tables(pa_tables, promote_options="default")
        elif axis == 1:
            for i, table in enumerate(pa_tables):
                if i == 0:
                    pa_table = table
                else:
                    for name, col in zip(table.column_names, table.columns):
                        pa_table = pa_table.append_column(name, col)
            return pa_table
        else:
            raise ValueError("'axis' must be either 0 or 1")

    @classmethod
    def _concat_blocks_horizontally_and_vertically(cls, blocks: list[list[TableBlock]]) -> pa.Table:
        pa_tables_to_concat_vertically = []
        for i, tables in enumerate(blocks):
            if not tables:
                continue
            pa_table_horizontally_concatenated = cls._concat_blocks(tables, axis=1)
            pa_tables_to_concat_vertically.append(pa_table_horizontally_concatenated)
        return cls._concat_blocks(pa_tables_to_concat_vertically, axis=0)

    @classmethod
    def _merge_blocks(cls, blocks: TableBlockContainer, axis: Optional[int] = None) -> TableBlockContainer:
        if axis is not None:
            merged_blocks = []
            for is_in_memory, block_group in groupby(blocks, key=lambda x: isinstance(x, InMemoryTable)):
                if is_in_memory:
                    block_group = [InMemoryTable(cls._concat_blocks(list(block_group), axis=axis))]
                merged_blocks += list(block_group)
        else:  # both
            merged_blocks = [cls._merge_blocks(row_block, axis=1) for row_block in blocks]
            if all(len(row_block) == 1 for row_block in merged_blocks):
                merged_blocks = cls._merge_blocks([block for row_block in merged_blocks for block in row_block], axis=0)
        return merged_blocks

    @classmethod
    def _consolidate_blocks(cls, blocks: TableBlockContainer) -> TableBlockContainer:
        if isinstance(blocks, TableBlock):
            return blocks
        elif isinstance(blocks[0], TableBlock):
            return cls._merge_blocks(blocks, axis=0)
        else:
            return cls._merge_blocks(blocks)

    @classmethod
    def from_blocks(cls, blocks: TableBlockContainer) -> "ConcatenationTable":
        blocks = cls._consolidate_blocks(blocks)
        if isinstance(blocks, TableBlock):
            table = blocks
            return cls(table.table, [[table]])
        elif isinstance(blocks[0], TableBlock):
            table = cls._concat_blocks(blocks, axis=0)
            blocks = [[t] for t in blocks]
            return cls(table, blocks)
        else:
            table = cls._concat_blocks_horizontally_and_vertically(blocks)
            return cls(table, blocks)

    @classmethod
    def from_tables(cls, tables: list[Union[pa.Table, Table]], axis: int = 0) -> "ConcatenationTable":
        def to_blocks(table: Union[pa.Table, Table]) -> list[list[TableBlock]]:
            if isinstance(table, pa.Table):
                return [[InMemoryTable(table)]]
            elif isinstance(table, ConcatenationTable):
                return copy.deepcopy(table.blocks)
            else:
                return [[table]]

        def _slice_row_block(row_block: list[TableBlock], length: int) -> tuple[list[TableBlock], list[TableBlock]]:
            sliced = [table.slice(0, length) for table in row_block]
            remainder = [table.slice(length, len(row_block[0]) - length) for table in row_block]
            return sliced, remainder

        def _split_both_like(result: list[list[TableBlock]], blocks: list[list[TableBlock]]) -> tuple[list[list[TableBlock]], list[list[TableBlock]]]:
            result, blocks = list(result), list(blocks)
            new_result, new_blocks = [], []
            while result and blocks:
                if len(result[0][0]) > len(blocks[0][0]):
                    new_blocks.append(blocks[0])
                    sliced, result[0] = _slice_row_block(result[0], len(blocks.pop(0)[0]))
                    new_result.append(sliced)
                elif len(result[0][0]) < len(blocks[0][0]):
                    new_result.append(result[0])
                    sliced, blocks[0] = _slice_row_block(blocks[0], len(result.pop(0)[0]))
                    new_blocks.append(sliced)
                else:
                    new_result.append(result.pop(0))
                    new_blocks.append(blocks.pop(0))
            if result or blocks:
                raise ValueError("Failed to concatenate on axis=1 because tables don't have the same number of rows")
            return new_result, new_blocks

        def _extend_blocks(result: list[list[TableBlock]], blocks: list[list[TableBlock]], axis: int = 0) -> list[list[TableBlock]]:
            if axis == 0:
                result.extend(blocks)
            elif axis == 1:
                # We make sure each row_block have the same num_rows
                result, blocks = _split_both_like(result, blocks)
                for i, row_block in enumerate(blocks):
                    result[i].extend(row_block)
            return result

        blocks = to_blocks(tables[0])
        for table in tables[1:]:
            table_blocks = to_blocks(table)
            blocks = _extend_blocks(blocks, table_blocks, axis=axis)
        return cls.from_blocks(blocks)

    @property
    def _slices(self):
        offset = 0
        for tables in self.blocks:
            length = len(tables[0])
            yield (offset, length)
            offset += length

    def slice(self, offset=0, length=None):
        table = self.table.slice(offset, length=length)
        length = length if length is not None else self.num_rows - offset
        blocks = []
        for tables in self.blocks:
            n_rows = len(tables[0])
            if length == 0:
                break
            elif n_rows <= offset:
                offset = offset - n_rows
            elif n_rows <= offset + length:
                blocks.append([t.slice(offset) for t in tables])
                length, offset = length + offset - n_rows, 0
            else:
                blocks.append([t.slice(offset, length) for t in tables])
                length, offset = 0, 0
        return ConcatenationTable(table, blocks)

    def filter(self, mask, *args, **kwargs):
        table = self.table.filter(mask, *args, **kwargs)
        blocks = []
        for (offset, length), tables in zip(self._slices, self.blocks):
            submask = mask.slice(offset, length)
            blocks.append([t.filter(submask, *args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def flatten(self, *args, **kwargs):
        table = table_flatten(self.table, *args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.flatten(*args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def combine_chunks(self, *args, **kwargs):
        table = self.table.combine_chunks(*args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.combine_chunks(*args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def cast(self, target_schema, *args, **kwargs):
        table = table_cast(self.table, target_schema, *args, **kwargs)
        target_features = Features.from_arrow_schema(target_schema)
        blocks = []
        for subtables in self.blocks:
            new_tables = []
            fields = list(target_schema)
            for subtable in subtables:
                subfields = []
                for name in subtable.column_names:
                    subfields.append(fields.pop(next(i for i, field in enumerate(fields) if field.name == name)))
                subfeatures = Features({subfield.name: target_features[subfield.name] for subfield in subfields})
                subschema = subfeatures.arrow_schema
                new_tables.append(subtable.cast(subschema, *args, **kwargs))
            blocks.append(new_tables)
        return ConcatenationTable(table, blocks)

    def replace_schema_metadata(self, *args, **kwargs):
        table = self.table.replace_schema_metadata(*args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.replace_schema_metadata(*args, **kwargs) for t in tables])
        return ConcatenationTable(table, self.blocks)

    def add_column(self, *args, **kwargs):
        raise NotImplementedError()

    def append_column(self, *args, **kwargs):
        raise NotImplementedError()

    def remove_column(self, i, *args, **kwargs):
        table = self.table.remove_column(i, *args, **kwargs)
        name = self.table.column_names[i]
        blocks = []
        for tables in self.blocks:
            blocks.append([t.remove_column(t.column_names.index(name), *args, **kwargs) if name in t.column_names else t for t in tables])
        return ConcatenationTable(table, blocks)

    def set_column(self, *args, **kwargs):
        raise NotImplementedError()

    def rename_columns(self, names, *args, **kwargs):
        table = self.table.rename_columns(names, *args, **kwargs)
        names = dict(zip(self.table.column_names, names))
        blocks = []
        for tables in self.blocks:
            blocks.append([t.rename_columns([names[name] for name in t.column_names], *args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def drop(self, columns, *args, **kwargs):
        table = self.table.drop(columns, *args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.drop([c for c in columns if c in t.column_names], *args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def select(self, columns, *args, **kwargs):
        table = self.table.select(columns, *args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.select([c for c in columns if c in t.column_names], *args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

Replay = tuple[str, tuple, dict]

def _memory_mapped_record_batch_reader_from_file(filename: str) -> pa.RecordBatchStreamReader:
    memory_mapped_stream = pa.memory_map(filename)
    return pa.ipc.open_stream(memory_mapped_stream)

def _memory_mapped_arrow_table_from_file(filename: str) -> pa.Table:
    opened_stream = _memory_mapped_record_batch_reader_from_file(filename)
    pa_table = opened_stream.read_all()
    return pa_table

class MemoryMappedTable(TableBlock):
    def __init__(self, table: pa.Table, path: str, replays: Optional[list[Replay]] = None):
        super().__init__(table)
        self.path = os.path.abspath(path)
        self.replays: list[Replay] = replays if replays is not None else []

    @classmethod
    def from_file(cls, filename: str, replays=None):
        table = _memory_mapped_arrow_table_from_file(filename)
        table = cls._apply_replays(table, replays)
        return cls(table, filename, replays)

    def __getstate__(self):
        return {"path": self.path, "replays": self.replays}

    def __setstate__(self, state):
        path = state["path"]
        replays = state["replays"]
        table = _memory_mapped_arrow_table_from_file(path)
        table = self._apply_replays(table, replays)
        MemoryMappedTable.__init__(self, table, path=path, replays=replays)

    @staticmethod
    def _apply_replays(table: pa.Table, replays: Optional[list[Replay]] = None) -> pa.Table:
        if replays is not None:
            for name, args, kwargs in replays:
                if name == "cast":
                    table = table_cast(table, *args, **kwargs)
                elif name == "flatten":
                    table = table_flatten(table, *args, **kwargs)
                else:
                    table = getattr(table, name)(*args, **kwargs)
        return table

    def _append_replay(self, replay: Replay) -> list[Replay]:
        replays = copy.deepcopy(self.replays)
        replays.append(replay)
        return replays

    def slice(self, offset=0, length=None):
        replay = ("slice", (offset, length), {})
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.fast_slice(offset=offset, length=length), self.path, replays)

    def filter(self, *args, **kwargs):
        replay = ("filter", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.filter(*args, **kwargs), self.path, replays)

    def flatten(self, *args, **kwargs):
        replay = ("flatten", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(table_flatten(self.table, *args, **kwargs), self.path, replays)

    def combine_chunks(self, *args, **kwargs):
        replay = ("combine_chunks", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.combine_chunks(*args, **kwargs), self.path, replays)

    def cast(self, *args, **kwargs):
        replay = ("cast", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(table_cast(self.table, *args, **kwargs), self.path, replays)

    def replace_schema_metadata(self, *args, **kwargs):
        replay = ("replace_schema_metadata", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.replace_schema_metadata(*args, **kwargs), self.path, replays)

    def add_column(self, *args, **kwargs):
        replay = ("add_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.add_column(*args, **kwargs), self.path, replays)

    def append_column(self, *args, **kwargs):
        replay = ("append_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.append_column(*args, **kwargs), self.path, replays)

    def remove_column(self, *args, **kwargs):
        replay = ("remove_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.remove_column(*args, **kwargs), self.path, replays)

    def set_column(self, *args, **kwargs):
        replay = ("set_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.set_column(*args, **kwargs), self.path, replays)

    def rename_columns(self, *args, **kwargs):
        replay = ("rename_columns", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.rename_columns(*args, **kwargs), self.path, replays)

    def drop(self, *args, **kwargs):
        replay = ("drop", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.drop(*args, **kwargs), self.path, replays)

    def select(self, *args, **kwargs):
        replay = ("select", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.select(*args, **kwargs), self.path, replays)

def make_file_instructions(
    name: str,
    split_infos: list["SplitInfo"],
    instruction: Union[str, "ReadInstruction"],
    filetype_suffix: Optional[str] = None,
    prefix_path: Optional[str] = None,
) -> FileInstructions:
    if not isinstance(name, str):
        raise TypeError(f"Expected str 'name', but got: {type(name).__name__}")
    elif not name:
        raise ValueError("Expected non-empty str 'name'")
    name2len = {info.name: info.num_examples for info in split_infos}
    name2shard_lengths = {info.name: info.shard_lengths for info in split_infos}
    name2filenames = {
        info.name: filenames_for_dataset_split(
            path=prefix_path,
            dataset_name=name,
            split=info.name,
            filetype_suffix=filetype_suffix,
            shard_lengths=name2shard_lengths[info.name],
        )
        for info in split_infos
    }
    if not isinstance(instruction, ReadInstruction):
        instruction = ReadInstruction.from_spec(instruction)
    absolute_instructions = instruction.to_absolute(name2len)

    file_instructions = []
    num_examples = 0
    for abs_instr in absolute_instructions:
        split_length = name2len[abs_instr.splitname]
        filenames = name2filenames[abs_instr.splitname]
        shard_lengths = name2shard_lengths[abs_instr.splitname]
        from_ = 0 if abs_instr.from_ is None else abs_instr.from_
        to = split_length if abs_instr.to is None else abs_instr.to
        if shard_lengths is None:  # not sharded
            for filename in filenames:
                take = to - from_
                if take == 0:
                    continue
                num_examples += take
                file_instructions.append({"filename": filename, "skip": from_, "take": take})
        else:  # sharded
            index_start = 0  # Beginning (included) of moving window.
            index_end = 0  # End (excluded) of moving window.
            for filename, shard_length in zip(filenames, shard_lengths):
                index_end += shard_length
                if from_ < index_end and to > index_start:  # There is something to take.
                    skip = from_ - index_start if from_ > index_start else 0
                    take = to - index_start - skip if to < index_end else -1
                    if take == 0:
                        continue
                    file_instructions.append({"filename": filename, "skip": skip, "take": take})
                    num_examples += shard_length - skip if take == -1 else take
                index_start += shard_length
    return FileInstructions(num_examples=num_examples, file_instructions=file_instructions)

class InvalidKeyError(Exception):
    def __init__(self, hash_data):
        self.prefix = "\nFAILURE TO GENERATE DATASET: Invalid key type detected"
        self.err_msg = f"\nFound Key {hash_data} of type {type(hash_data)}"
        self.suffix = "\nKeys should be either str, int or bytes type"
        super().__init__(f"{self.prefix}{self.err_msg}{self.suffix}")

def _as_bytes(hash_data: Union[str, int, bytes]) -> bytes:
    if isinstance(hash_data, bytes):
        return hash_data
    elif isinstance(hash_data, str):
        hash_data = hash_data.replace("\\", "/")
    elif isinstance(hash_data, int):
        hash_data = str(hash_data)
    else:
        raise InvalidKeyError(hash_data)

    return hash_data.encode("utf-8")

class KeyHasher:
    def __init__(self, hash_salt: str):
        self._split_md5 = insecure_hashlib.md5(_as_bytes(hash_salt))

    def hash(self, key: Union[str, int, bytes]) -> int:
        md5 = self._split_md5.copy()
        byte_key = _as_bytes(key)
        md5.update(byte_key)
        return int(md5.hexdigest(), 16)

def _visit(feature: FeatureType, func: Callable[[FeatureType], Optional[FeatureType]]) -> FeatureType:
    if isinstance(feature, Features):
        out = func(Features({k: _visit(f, func) for k, f in feature.items()}))
    elif isinstance(feature, dict):
        out = func({k: _visit(f, func) for k, f in feature.items()})
    elif isinstance(feature, (list, tuple)):
        out = func([_visit(feature[0], func)])
    elif isinstance(feature, LargeList):
        out = func(LargeList(_visit(feature.feature, func)))
    elif isinstance(feature, Sequence):
        out = func(Sequence(_visit(feature.feature, func), length=feature.length))
    else:
        out = func(feature)
    return feature if out is None else out

def get_writer_batch_size(features: Optional[Features]) -> Optional[int]:
    if not features:
        return None

    batch_size = np.inf

    def set_batch_size(feature: FeatureType) -> None:
        nonlocal batch_size
        if isinstance(feature, Image):
            batch_size = min(batch_size, data_config.PARQUET_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS)
        elif isinstance(feature, Value) and feature.dtype == "binary":
            batch_size = min(batch_size, data_config.PARQUET_ROW_GROUP_SIZE_FOR_BINARY_DATASETS)

    _visit(features, set_batch_size)

    return None if batch_size is np.inf else batch_size

class DuplicatedKeysError(Exception):
    def __init__(self, key, duplicate_key_indices, fix_msg=""):
        self.key = key
        self.duplicate_key_indices = duplicate_key_indices
        self.fix_msg = fix_msg
        self.prefix = "Found multiple examples generated with the same key"
        if len(duplicate_key_indices) <= 20:
            self.err_msg = f"\nThe examples at index {', '.join(duplicate_key_indices)} have the key {key}"
        else:
            self.err_msg = f"\nThe examples at index {', '.join(duplicate_key_indices[:20])}... ({len(duplicate_key_indices) - 20} more) have the key {key}"
        self.suffix = "\n" + fix_msg if fix_msg else ""
        super().__init__(f"{self.prefix}{self.err_msg}{self.suffix}")

def generate_from_arrow_type(pa_type: pa.DataType) -> FeatureType:
    if isinstance(pa_type, pa.StructType):
        return {field.name: generate_from_arrow_type(field.type) for field in pa_type}
    elif isinstance(pa_type, pa.FixedSizeListType):
        return Sequence(feature=generate_from_arrow_type(pa_type.value_type), length=pa_type.list_size)
    elif isinstance(pa_type, pa.ListType):
        feature = generate_from_arrow_type(pa_type.value_type)
        if isinstance(feature, (dict, tuple, list)):
            return [feature]
        return Sequence(feature=feature)
    elif isinstance(pa_type, pa.LargeListType):
        feature = generate_from_arrow_type(pa_type.value_type)
        return LargeList(feature=feature)
    elif isinstance(pa_type, _ArrayXDExtensionType):
        array_feature = [None, None, Array2D, Array3D, Array4D, Array5D][pa_type.ndims]
        return array_feature(shape=pa_type.shape, dtype=pa_type.value_type)
    elif isinstance(pa_type, pa.DataType):
        return Value(dtype=_arrow_to_datasets_dtype(pa_type))
    else:
        raise ValueError(f"Cannot convert {pa_type} to a Feature type.")

def first_non_null_value(iterable):
    for i, value in enumerate(iterable):
        if value is not None:
            return i, value
    return -1, None

def contains_any_np_array(data: Any):
    if isinstance(data, np.ndarray):
        return True
    elif isinstance(data, list):
        return contains_any_np_array(first_non_null_value(data)[1])
    else:
        return False

def list_of_pa_arrays_to_pyarrow_listarray(l_arr: list[Optional[pa.Array]]) -> pa.ListArray:
    null_mask = np.array([arr is None for arr in l_arr])
    null_indices = np.arange(len(null_mask))[null_mask] - np.arange(np.sum(null_mask))
    l_arr = [arr for arr in l_arr if arr is not None]
    offsets = np.cumsum([0] + [len(arr) for arr in l_arr], dtype=object) 
    offsets = np.insert(offsets, null_indices, None)
    offsets = pa.array(offsets, type=pa.int32())
    values = pa.concat_arrays(l_arr)
    return pa.ListArray.from_arrays(offsets, values)

def numpy_to_pyarrow_listarray(arr: np.ndarray, type: pa.DataType = None) -> pa.ListArray:
    arr = np.array(arr)
    values = pa.array(arr.flatten(), type=type)
    for i in range(arr.ndim - 1):
        n_offsets = reduce(mul, arr.shape[: arr.ndim - i - 1], 1)
        step_offsets = arr.shape[arr.ndim - i - 1]
        offsets = pa.array(np.arange(n_offsets + 1) * step_offsets, type=pa.int32())
        values = pa.ListArray.from_arrays(offsets, values)
    return values

def any_np_array_to_pyarrow_listarray(data: Union[np.ndarray, list], type: pa.DataType = None) -> pa.ListArray:
    if isinstance(data, np.ndarray):
        return numpy_to_pyarrow_listarray(data, type=type)
    elif isinstance(data, list):
        return list_of_pa_arrays_to_pyarrow_listarray([any_np_array_to_pyarrow_listarray(i, type=type) for i in data])

def to_pyarrow_listarray(data: Any, pa_type: _ArrayXDExtensionType) -> pa.Array:
    if contains_any_np_array(data):
        return any_np_array_to_pyarrow_listarray(data, type=pa_type.value_type)
    else:
        return pa.array(data, pa_type.storage_dtype)

def list_of_np_array_to_pyarrow_listarray(l_arr: list[np.ndarray], type: pa.DataType = None) -> pa.ListArray:
    if len(l_arr) > 0:
        return list_of_pa_arrays_to_pyarrow_listarray([numpy_to_pyarrow_listarray(arr, type=type) if arr is not None else None for arr in l_arr])
    else:
        return pa.array([], type=type)

class TypedSequence:
    def __init__(
        self,
        data: Iterable,
        type: Optional[FeatureType] = None,
        try_type: Optional[FeatureType] = None,
        optimized_int_type: Optional[FeatureType] = None,
    ):
        if type is not None and try_type is not None:
            raise ValueError("You cannot specify both type and try_type")
        self.data = data
        self.type = type
        self.try_type = try_type  # is ignored if it doesn't match the data
        self.optimized_int_type = optimized_int_type
        self.trying_type = self.try_type is not None
        self.trying_int_optimization = optimized_int_type is not None and type is None and try_type is None
        self._inferred_type = None

    def get_inferred_type(self) -> FeatureType:
        if self._inferred_type is None:
            self._inferred_type = generate_from_arrow_type(pa.array(self).type)
        return self._inferred_type

    @staticmethod
    def _infer_custom_type_and_encode(data: Iterable) -> tuple[Iterable, Optional[FeatureType]]:
        _, non_null_value = first_non_null_value(data)
        if isinstance(non_null_value, PIL.Image.Image):
            return [Image().encode_example(value) if value is not None else None for value in data], Image()
        return data, None

    def __arrow_array__(self, type: Optional[pa.DataType] = None):
        if type is not None:
            raise ValueError("TypedSequence is supposed to be used with pa.array(typed_sequence, type=None)")
        del type  # make sure we don't use it
        data = self.data
        if self.type is None and self.try_type is None:
            data, self._inferred_type = self._infer_custom_type_and_encode(data)
        if self._inferred_type is None:
            type = self.try_type if self.trying_type else self.type
        else:
            type = self._inferred_type
        pa_type = get_nested_type(type) if type is not None else None
        optimized_int_pa_type = (get_nested_type(self.optimized_int_type) if self.optimized_int_type is not None else None)
        trying_cast_to_python_objects = False
        try:
            if isinstance(pa_type, _ArrayXDExtensionType):
                storage = to_pyarrow_listarray(data, pa_type)
                return pa.ExtensionArray.from_storage(pa_type, storage)

            if isinstance(data, np.ndarray):
                out = numpy_to_pyarrow_listarray(data)
            elif isinstance(data, list) and data and isinstance(first_non_null_value(data)[1], np.ndarray):
                out = list_of_np_array_to_pyarrow_listarray(data)
            else:
                trying_cast_to_python_objects = True
                out = pa.array(cast_to_python_objects(data, only_1d_for_numpy=True))
            if self.trying_int_optimization:
                if pa.types.is_int64(out.type):
                    out = out.cast(optimized_int_pa_type)
                elif pa.types.is_list(out.type):
                    if pa.types.is_int64(out.type.value_type):
                        out = array_cast(out, pa.list_(optimized_int_pa_type))
                    elif pa.types.is_list(out.type.value_type) and pa.types.is_int64(out.type.value_type.value_type):
                        out = array_cast(out, pa.list_(pa.list_(optimized_int_pa_type)))
            elif type is not None:
                out = cast_array_to_feature(out, type, allow_primitive_to_str=not self.trying_type, allow_decimal_to_str=not self.trying_type)
            return out
        except (
            TypeError,
            pa.lib.ArrowInvalid,
            pa.lib.ArrowNotImplementedError,
        ) as e:  # handle type errors and overflows
            if not self.trying_type and isinstance(e, pa.lib.ArrowNotImplementedError):
                raise

            if self.trying_type:
                try:  # second chance
                    if isinstance(data, np.ndarray):
                        return numpy_to_pyarrow_listarray(data)
                    elif isinstance(data, list) and data and any(isinstance(value, np.ndarray) for value in data):
                        return list_of_np_array_to_pyarrow_listarray(data)
                    else:
                        trying_cast_to_python_objects = True
                        return pa.array(cast_to_python_objects(data, only_1d_for_numpy=True))
                except pa.lib.ArrowInvalid as e:
                    if "overflow" in str(e):
                        raise OverflowError(f"There was an overflow with type {type_(data)}. Try to reduce writer_batch_size to have batches smaller than 2GB.\n({e})") from None
                    elif self.trying_int_optimization and "not in range" in str(e):
                        return out
                    elif trying_cast_to_python_objects and "Could not convert" in str(e):
                        out = pa.array(cast_to_python_objects(data, only_1d_for_numpy=True, optimize_list_casting=False))
                        if type is not None:
                            out = cast_array_to_feature(out, type, allow_primitive_to_str=True, allow_decimal_to_str=True)
                        return out
                    else:
                        raise
            elif "overflow" in str(e):
                raise OverflowError(f"There was an overflow with type {type_(data)}. Try to reduce writer_batch_size to have batches smaller than 2GB.\n({e})") from None
            elif self.trying_int_optimization and "not in range" in str(e):
                return out
            elif trying_cast_to_python_objects and "Could not convert" in str(e):
                out = pa.array(cast_to_python_objects(data, only_1d_for_numpy=True, optimize_list_casting=False))
                if type is not None:
                    out = cast_array_to_feature(out, type, allow_primitive_to_str=True, allow_decimal_to_str=True)
                return out
            else:
                raise

class OptimizedTypedSequence(TypedSequence):
    def __init__(
        self,
        data,
        type: Optional[FeatureType] = None,
        try_type: Optional[FeatureType] = None,
        col: Optional[str] = None,
        optimized_int_type: Optional[FeatureType] = None,
    ):
        optimized_int_type_by_col = {
            "attention_mask": Value("int8"),  # binary tensor
            "special_tokens_mask": Value("int8"),
            "input_ids": Value("int32"),  # typical vocab size: 0-50k (max ~500k, never > 1M)
            "token_type_ids": Value(
                "int8"
            ),  # binary mask; some (XLNetModel) use an additional token represented by a 2
        }
        if type is None and try_type is None:
            optimized_int_type = optimized_int_type_by_col.get(col, None)
        super().__init__(data, type=type, try_type=try_type, optimized_int_type=optimized_int_type)

def require_storage_cast(feature: FeatureType) -> bool:
    if isinstance(feature, dict):
        return any(require_storage_cast(f) for f in feature.values())
    elif isinstance(feature, (list, tuple)):
        return require_storage_cast(feature[0])
    elif isinstance(feature, LargeList):
        return require_storage_cast(feature.feature)
    elif isinstance(feature, Sequence):
        return require_storage_cast(feature.feature)
    else:
        return hasattr(feature, "cast_storage")

def require_storage_embed(feature: FeatureType) -> bool:
    if isinstance(feature, dict):
        return any(require_storage_cast(f) for f in feature.values())
    elif isinstance(feature, (list, tuple)):
        return require_storage_cast(feature[0])
    elif isinstance(feature, LargeList):
        return require_storage_cast(feature.feature)
    elif isinstance(feature, Sequence):
        return require_storage_cast(feature.feature)
    else:
        return hasattr(feature, "embed_storage")

@_wrap_for_chunked_arrays
def embed_array_storage(array: pa.Array, feature: "FeatureType"):
    _e = embed_array_storage

    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if hasattr(feature, "embed_storage"):
        return feature.embed_storage(array)
    elif pa.types.is_struct(array.type):
        if isinstance(feature, Sequence) and isinstance(feature.feature, dict):
            feature = {name: Sequence(subfeature, length=feature.length) for name, subfeature in feature.feature.items()}
        if isinstance(feature, dict):
            arrays = [_e(array.field(name), subfeature) for name, subfeature in feature.items()]
            return pa.StructArray.from_arrays(arrays, names=list(feature), mask=array.is_null())
    elif pa.types.is_list(array.type):
        array_offsets = _combine_list_array_offsets_with_mask(array)
        if isinstance(feature, list):
            return pa.ListArray.from_arrays(array_offsets, _e(array.values, feature[0]))
        if isinstance(feature, Sequence) and feature.length == -1:
            return pa.ListArray.from_arrays(array_offsets, _e(array.values, feature.feature))
    elif pa.types.is_large_list(array.type):
        array_offsets = _combine_list_array_offsets_with_mask(array)
        return pa.LargeListArray.from_arrays(array_offsets, _e(array.values, feature.feature))
    elif pa.types.is_fixed_size_list(array.type):
        if isinstance(feature, Sequence) and feature.length > -1:
            array_values = array.values[array.offset * array.type.list_size : (array.offset + len(array)) * array.type.list_size]
            embedded_array_values = _e(array_values, feature.feature)
            return pa.FixedSizeListArray.from_arrays(embedded_array_values, feature.length, mask=array.is_null())
    if not isinstance(feature, (Sequence, dict, list, tuple)):
        return array
    raise TypeError(f"Couldn't embed array of type\n{_short_str(array.type)}\nwith\n{_short_str(feature)}")

def embed_table_storage(table: pa.Table):
    features = Features.from_arrow_schema(table.schema)
    arrays = [embed_array_storage(table[name], feature) if require_storage_embed(feature) else table[name] for name, feature in features.items()]
    return pa.Table.from_arrays(arrays, schema=features.arrow_schema)

class SchemaInferenceError(ValueError):
    pass

def convert_file_size_to_int(size: Union[int, str]) -> int:
    if isinstance(size, int):
        return size
    if size.upper().endswith("PIB"):
        return int(size[:-3]) * (2**50)
    if size.upper().endswith("TIB"):
        return int(size[:-3]) * (2**40)
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("PB"):
        int_size = int(size[:-2]) * (10**15)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("TB"):
        int_size = int(size[:-2]) * (10**12)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError(f"`size={size}` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")

def _write_generator_to_queue(queue: queue.Queue, func: Callable[..., Iterable[Y]], kwargs: dict) -> int:
    for i, result in enumerate(func(**kwargs)):
        queue.put(result)
    return i

def _get_pool_pid(pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool]) -> set[int]:
    return {f.pid for f in pool._pool}

class Empty(Exception): ...

def iflatmap_unordered(
    pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool],
    func: Callable[..., Iterable[Y]],
    *,
    kwargs_iterable: Iterable[dict],
) -> Iterable[Y]:
    initial_pool_pid = _get_pool_pid(pool)
    pool_changed = False
    manager_cls = Manager if isinstance(pool, multiprocessing.pool.Pool) else multiprocess.Manager
    with manager_cls() as manager:
        queue = manager.Queue()
        async_results = [pool.apply_async(_write_generator_to_queue, (queue, func, kwargs)) for kwargs in kwargs_iterable]
        try:
            while True:
                try:
                    yield queue.get(timeout=0.05)
                except Empty:
                    if all(async_result.ready() for async_result in async_results) and queue.empty():
                        break
                if _get_pool_pid(pool) != initial_pool_pid:
                    pool_changed = True
                    raise RuntimeError("One of the subprocesses has abruptly died during map operation. To debug the error, disable multiprocessing.")
        finally:
            if not pool_changed:
                [async_result.get(timeout=0.05) for async_result in async_results]

def is_small_dataset(dataset_size):
    if dataset_size and data_config.IN_MEMORY_MAX_SIZE:
        return dataset_size < data_config.IN_MEMORY_MAX_SIZE
    else:
        return False

def estimate_dataset_size(paths):
    return sum(path.stat().st_size for path in paths)

def list_table_cache_files(table: Table) -> list[str]:
    if isinstance(table, ConcatenationTable):
        cache_files = []
        for subtables in table.blocks:
            for subtable in subtables:
                cache_files += list_table_cache_files(subtable)
        return cache_files
    elif isinstance(table, MemoryMappedTable):
        return [table.path]
    else:
        return []

class BaseArrowExtractor(Generic[RowFormat, ColumnFormat, BatchFormat]):
    def extract_row(self, pa_table: pa.Table) -> RowFormat:
        raise NotImplementedError

    def extract_column(self, pa_table: pa.Table) -> ColumnFormat:
        raise NotImplementedError

    def extract_batch(self, pa_table: pa.Table) -> BatchFormat:
        raise NotImplementedError

def _unnest(py_dict: dict[str, list[T]]) -> dict[str, T]:
    return {key: array[0] for key, array in py_dict.items()}

def _is_array_with_nulls(pa_array: pa.Array) -> bool:
    return pa_array.null_count > 0

class SimpleArrowExtractor(BaseArrowExtractor[pa.Table, pa.Array, pa.Table]):
    def extract_row(self, pa_table: pa.Table) -> pa.Table:
        return pa_table

    def extract_column(self, pa_table: pa.Table) -> pa.Array:
        return pa_table.column(0)

    def extract_batch(self, pa_table: pa.Table) -> pa.Table:
        return pa_table

class PythonArrowExtractor(BaseArrowExtractor[dict, list, dict]):
    def extract_row(self, pa_table: pa.Table) -> dict:
        return _unnest(pa_table.to_pydict())

    def extract_column(self, pa_table: pa.Table) -> list:
        return pa_table.column(0).to_pylist()

    def extract_batch(self, pa_table: pa.Table) -> dict:
        return pa_table.to_pydict()

class NumpyArrowExtractor(BaseArrowExtractor[dict, np.ndarray, dict]):
    def __init__(self, **np_array_kwargs):
        self.np_array_kwargs = np_array_kwargs

    def extract_row(self, pa_table: pa.Table) -> dict:
        return _unnest(self.extract_batch(pa_table))

    def extract_column(self, pa_table: pa.Table) -> np.ndarray:
        return self._arrow_array_to_numpy(pa_table[pa_table.column_names[0]])

    def extract_batch(self, pa_table: pa.Table) -> dict:
        return {col: self._arrow_array_to_numpy(pa_table[col]) for col in pa_table.column_names}

    def _arrow_array_to_numpy(self, pa_array: pa.Array) -> np.ndarray:
        if isinstance(pa_array, pa.ChunkedArray):
            if isinstance(pa_array.type, _ArrayXDExtensionType):
                zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
                array: list = [row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)]
            else:
                zero_copy_only = _is_zero_copy_only(pa_array.type) and all(not _is_array_with_nulls(chunk) for chunk in pa_array.chunks)
                array: list = [row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)]
        else:
            if isinstance(pa_array.type, _ArrayXDExtensionType):
                zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
                array: list = pa_array.to_numpy(zero_copy_only=zero_copy_only)
            else:
                zero_copy_only = _is_zero_copy_only(pa_array.type) and not _is_array_with_nulls(pa_array)
                array: list = pa_array.to_numpy(zero_copy_only=zero_copy_only).tolist()

        if len(array) > 0:
            if any((isinstance(x, np.ndarray) and (x.dtype == object or x.shape != array[0].shape)) or (isinstance(x, float) and np.isnan(x)) for x in array):
                if np.lib.NumpyVersion(np.__version__) >= "2.0.0b1":
                    return np.asarray(array, dtype=object)
                return np.array(array, copy=False, dtype=object)
        if np.lib.NumpyVersion(np.__version__) >= "2.0.0b1":
            return np.asarray(array)
        else:
            return np.array(array, copy=False)

def pandas_types_mapper(dtype):
    if isinstance(dtype, _ArrayXDExtensionType):
        return PandasArrayExtensionDtype(dtype.value_type)
    
class PandasArrowExtractor(BaseArrowExtractor[pd.DataFrame, pd.Series, pd.DataFrame]):
    def extract_row(self, pa_table: pa.Table) -> pd.DataFrame:
        return pa_table.slice(length=1).to_pandas(types_mapper=pandas_types_mapper)

    def extract_column(self, pa_table: pa.Table) -> pd.Series:
        return pa_table.select([0]).to_pandas(types_mapper=pandas_types_mapper)[pa_table.column_names[0]]

    def extract_batch(self, pa_table: pa.Table) -> pd.DataFrame:
        return pa_table.to_pandas(types_mapper=pandas_types_mapper)

class PythonFeaturesDecoder:
    def __init__(self, features: Optional[Features], token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None):
        self.features = features
        self.token_per_repo_id = token_per_repo_id

    def decode_row(self, row: dict) -> dict:
        return self.features.decode_example(row, token_per_repo_id=self.token_per_repo_id) if self.features else row

    def decode_column(self, column: list, column_name: str) -> list:
        return self.features.decode_column(column, column_name) if self.features else column

    def decode_batch(self, batch: dict) -> dict:
        return self.features.decode_batch(batch) if self.features else batch

class PandasFeaturesDecoder:
    def __init__(self, features: Optional[Features]):
        self.features = features

    def decode_row(self, row: pd.DataFrame) -> pd.DataFrame:
        decode = (
            {
                column_name: no_op_if_value_is_null(partial(decode_nested_example, feature))
                for column_name, feature in self.features.items()
                if self.features._column_requires_decoding[column_name]
            }
            if self.features
            else {}
        )
        if decode:
            row[list(decode.keys())] = row.transform(decode)
        return row

    def decode_column(self, column: pd.Series, column_name: str) -> pd.Series:
        decode = (no_op_if_value_is_null(partial(decode_nested_example, self.features[column_name])) if self.features and column_name in self.features and self.features._column_requires_decoding[column_name] else None)
        if decode:
            column = column.transform(decode)
        return column

    def decode_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        return self.decode_row(batch)

class Formatter(Generic[RowFormat, ColumnFormat, BatchFormat]):
    simple_arrow_extractor = SimpleArrowExtractor
    python_arrow_extractor = PythonArrowExtractor
    numpy_arrow_extractor = NumpyArrowExtractor
    pandas_arrow_extractor = PandasArrowExtractor

    def __init__(
        self,
        features: Optional[Features] = None,
        token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None
    ):
        self.features = features
        self.token_per_repo_id = token_per_repo_id
        self.python_features_decoder = PythonFeaturesDecoder(self.features, self.token_per_repo_id)
        self.pandas_features_decoder = PandasFeaturesDecoder(self.features)

    def __call__(self, pa_table: pa.Table, query_type: str) -> Union[RowFormat, ColumnFormat, BatchFormat]:
        if query_type == "row":
            return self.format_row(pa_table)
        elif query_type == "column":
            return self.format_column(pa_table)
        elif query_type == "batch":
            return self.format_batch(pa_table)

    def format_row(self, pa_table: pa.Table) -> RowFormat:
        raise NotImplementedError

    def format_column(self, pa_table: pa.Table) -> ColumnFormat:
        raise NotImplementedError

    def format_batch(self, pa_table: pa.Table) -> BatchFormat:
        raise NotImplementedError

class TensorFormatter(Formatter[RowFormat, ColumnFormat, BatchFormat]):
    def recursive_tensorize(self, data_struct: dict):
        raise NotImplementedError

class TableFormatter(Formatter[RowFormat, ColumnFormat, BatchFormat]):
    table_type: str
    column_type: str

class ArrowFormatter(TableFormatter[pa.Table, pa.Array, pa.Table]):
    table_type = "arrow table"
    column_type = "arrow array"

    def format_row(self, pa_table: pa.Table) -> pa.Table:
        return self.simple_arrow_extractor().extract_row(pa_table)

    def format_column(self, pa_table: pa.Table) -> pa.Array:
        return self.simple_arrow_extractor().extract_column(pa_table)

    def format_batch(self, pa_table: pa.Table) -> pa.Table:
        return self.simple_arrow_extractor().extract_batch(pa_table)
    
class LazyDict(MutableMapping):
    def __init__(self, pa_table: pa.Table, formatter: "Formatter"):
        self.pa_table = pa_table
        self.formatter = formatter

        self.data = dict.fromkeys(pa_table.column_names)
        self.keys_to_format = set(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        value = self.data[key]
        if key in self.keys_to_format:
            value = self.format(key)
            self.data[key] = value
            self.keys_to_format.remove(key)
        return value

    def __setitem__(self, key, value):
        if key in self.keys_to_format:
            self.keys_to_format.remove(key)
        self.data[key] = value

    def __delitem__(self, key) -> None:
        if key in self.keys_to_format:
            self.keys_to_format.remove(key)
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, key):
        return key in self.data

    def __repr__(self):
        self._format_all()
        return repr(self.data)

    def __or__(self, other):
        if isinstance(other, LazyDict):
            inst = self.copy()
            other = other.copy()
            other._format_all()
            inst.keys_to_format -= other.data.keys()
            inst.data = inst.data | other.data
            return inst
        if isinstance(other, dict):
            inst = self.copy()
            inst.keys_to_format -= other.keys()
            inst.data = inst.data | other
            return inst
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, LazyDict):
            inst = self.copy()
            other = other.copy()
            other._format_all()
            inst.keys_to_format -= other.data.keys()
            inst.data = other.data | inst.data
            return inst
        if isinstance(other, dict):
            inst = self.copy()
            inst.keys_to_format -= other.keys()
            inst.data = other | inst.data
            return inst
        return NotImplemented

    def __ior__(self, other):
        if isinstance(other, LazyDict):
            other = other.copy()
            other._format_all()
            self.keys_to_format -= other.data.keys()
            self.data |= other.data
        else:
            self.keys_to_format -= other.keys()
            self.data |= other
        return self

    def __copy__(self):
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        inst.__dict__["data"] = self.__dict__["data"].copy()
        inst.__dict__["keys_to_format"] = self.__dict__["keys_to_format"].copy()
        return inst

    def copy(self):
        import copy

        return copy.copy(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        raise NotImplementedError

    def format(self, key):
        raise NotImplementedError

    def _format_all(self):
        for key in self.keys_to_format:
            self.data[key] = self.format(key)
        self.keys_to_format.clear()

class LazyRow(LazyDict):
    def format(self, key):
        return self.formatter.format_column(self.pa_table.select([key]))[0]

class LazyBatch(LazyDict):
    def format(self, key):
        return self.formatter.format_column(self.pa_table.select([key]))

class PythonFormatter(Formatter[Mapping, list, Mapping]):
    def __init__(self, features=None, lazy=False, token_per_repo_id=None):
        super().__init__(features, token_per_repo_id)
        self.lazy = lazy

    def format_row(self, pa_table: pa.Table) -> Mapping:
        if self.lazy:
            return LazyRow(pa_table, self)
        row = self.python_arrow_extractor().extract_row(pa_table)
        row = self.python_features_decoder.decode_row(row)
        return row

    def format_column(self, pa_table: pa.Table) -> list:
        column = self.python_arrow_extractor().extract_column(pa_table)
        column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        return column

    def format_batch(self, pa_table: pa.Table) -> Mapping:
        if self.lazy:
            return LazyBatch(pa_table, self)
        batch = self.python_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        return batch

_FORMAT_TYPES: dict[Optional[str], type[Formatter]] = {}
_FORMAT_TYPES_ALIASES: dict[Optional[str], str] = {}
_FORMAT_TYPES_ALIASES_UNAVAILABLE: dict[Optional[str], Exception] = {}

def get_format_type_from_alias(format_type: Optional[str]) -> Optional[str]:
    if format_type in _FORMAT_TYPES_ALIASES:
        return _FORMAT_TYPES_ALIASES[format_type]
    else:
        return format_type

def get_formatter(format_type: Optional[str], **format_kwargs) -> Formatter:
    format_type = get_format_type_from_alias(format_type)
    if format_type in _FORMAT_TYPES:
        return _FORMAT_TYPES[format_type](**format_kwargs)
    if format_type in _FORMAT_TYPES_ALIASES_UNAVAILABLE:
        raise _FORMAT_TYPES_ALIASES_UNAVAILABLE[format_type]
    else:
        raise ValueError(f"Format type should be one of {list(_FORMAT_TYPES.keys())}, but got '{format_type}'")

def table_iter(table: Table, batch_size: int, drop_last_batch=False) -> Iterator[pa.Table]:
    chunks_buffer = []
    chunks_buffer_size = 0
    for chunk in table.to_reader(max_chunksize=batch_size):
        if len(chunk) == 0:
            continue
        elif chunks_buffer_size + len(chunk) < batch_size:
            chunks_buffer.append(chunk)
            chunks_buffer_size += len(chunk)
            continue
        elif chunks_buffer_size + len(chunk) == batch_size:
            chunks_buffer.append(chunk)
            yield pa.Table.from_batches(chunks_buffer)
            chunks_buffer = []
            chunks_buffer_size = 0
        else:
            cropped_chunk_length = batch_size - chunks_buffer_size
            chunks_buffer.append(chunk.slice(0, cropped_chunk_length))
            yield pa.Table.from_batches(chunks_buffer)
            chunks_buffer = [chunk.slice(cropped_chunk_length, len(chunk) - cropped_chunk_length)]
            chunks_buffer_size = len(chunk) - cropped_chunk_length
    if not drop_last_batch and chunks_buffer:
        yield pa.Table.from_batches(chunks_buffer)

def _raise_bad_key_type(key: Any):
    raise TypeError(f"Wrong key type: '{key}' of type '{type(key)}'. Expected one of int, slice, range, str or Iterable.")

def key_to_query_type(key: Union[int, slice, range, str, Iterable]) -> str:
    if isinstance(key, int):
        return "row"
    elif isinstance(key, str):
        return "column"
    elif isinstance(key, (slice, range, Iterable)):
        return "batch"
    _raise_bad_key_type(key)

def format_table(
    table: Table,
    key: Union[int, slice, range, str, Iterable],
    formatter: Formatter,
    format_columns: Optional[list] = None,
    output_all_columns=False,
):
    if isinstance(table, Table):
        pa_table = table.table
    else:
        pa_table = table
    query_type = key_to_query_type(key)
    python_formatter = PythonFormatter(features=formatter.features)
    if format_columns is None:
        return formatter(pa_table, query_type=query_type)
    elif query_type == "column":
        if key in format_columns:
            return formatter(pa_table, query_type)
        else:
            return python_formatter(pa_table, query_type=query_type)
    else:
        pa_table_to_format = pa_table.drop(col for col in pa_table.column_names if col not in format_columns)
        formatted_output = formatter(pa_table_to_format, query_type=query_type)
        if output_all_columns:
            if isinstance(formatted_output, MutableMapping):
                pa_table_with_remaining_columns = pa_table.drop(col for col in pa_table.column_names if col in format_columns)
                remaining_columns_dict = python_formatter(pa_table_with_remaining_columns, query_type=query_type)
                formatted_output.update(remaining_columns_dict)
            else:
                raise TypeError(f"Custom formatting function must return a dict to work with output_all_columns=True, but got {formatted_output}")
        return formatted_output

def _check_valid_column_key(key: str, columns: list[str]) -> None:
    if key not in columns:
        raise KeyError(f"Column {key} not in the dataset. Current columns in the dataset: {columns}")

def _check_valid_index_key(key: Union[int, slice, range, Iterable], size: int) -> None:
    if isinstance(key, int):
        if (key < 0 and key + size < 0) or (key >= size):
            raise IndexError(f"Invalid key: {key} is out of bounds for size {size}")
        return
    elif isinstance(key, slice):
        pass
    elif isinstance(key, range):
        if len(key) > 0:
            _check_valid_index_key(max(key), size=size)
            _check_valid_index_key(min(key), size=size)
    elif isinstance(key, Iterable):
        if len(key) > 0:
            _check_valid_index_key(int(max(key)), size=size)
            _check_valid_index_key(int(min(key)), size=size)
    else:
        _raise_bad_key_type(key)

def _is_range_contiguous(key: range) -> bool:
    return key.step == 1 and key.stop >= key.start

def _query_table_with_indices_mapping(table: Table, key: Union[int, slice, range, str, Iterable], indices: Table) -> pa.Table:
    if isinstance(key, int):
        key = indices.fast_slice(key % indices.num_rows, 1).column(0)[0].as_py()
        return _query_table(table, key)
    if isinstance(key, slice):
        key = range(*key.indices(indices.num_rows))
    if isinstance(key, range):
        if _is_range_contiguous(key) and key.start >= 0:
            return _query_table(table, [i.as_py() for i in indices.fast_slice(key.start, key.stop - key.start).column(0)])
        else:
            pass  # treat as an iterable
    if isinstance(key, str):
        table = table.select([key])
        return _query_table(table, indices.column(0).to_pylist())
    if isinstance(key, Iterable):
        return _query_table(table, [indices.fast_slice(i, 1).column(0)[0].as_py() for i in key])

    _raise_bad_key_type(key)

def _query_table(table: Table, key: Union[int, slice, range, str, Iterable]) -> pa.Table:
    if isinstance(key, int):
        return table.fast_slice(key % table.num_rows, 1)
    if isinstance(key, slice):
        key = range(*key.indices(table.num_rows))
    if isinstance(key, range):
        if _is_range_contiguous(key) and key.start >= 0:
            return table.fast_slice(key.start, key.stop - key.start)
        else:
            pass  # treat as an iterable
    if isinstance(key, str):
        return table.table.drop([column for column in table.column_names if column != key])
    if isinstance(key, Iterable):
        key = np.fromiter(key, np.int64)
        if len(key) == 0:
            return table.table.slice(0, 0)
        return table.fast_gather(key % table.num_rows)

    _raise_bad_key_type(key)

def query_table(table: Table, key: Union[int, slice, range, str, Iterable], indices: Optional[Table] = None) -> pa.Table:
    if not isinstance(key, (int, slice, range, str, Iterable)):
        try:
            key = operator.index(key)
        except TypeError:
            _raise_bad_key_type(key)
    if isinstance(key, str):
        _check_valid_column_key(key, table.column_names)
    else:
        size = indices.num_rows if indices is not None else table.num_rows
        _check_valid_index_key(key, size)
    if indices is None:
        pa_subtable = _query_table(table, key)
    else:
        pa_subtable = _query_table_with_indices_mapping(table, key, indices=indices)
    return pa_subtable

class NonExistentDatasetError(Exception):
    pass

class DatasetTransformationNotAllowedError(Exception):
    pass

def _check_if_features_can_be_aligned(features_list: list[Features]):
    name2feature = {}
    for features in features_list:
        for k, v in features.items():
            if k not in name2feature or (isinstance(name2feature[k], Value) and name2feature[k].dtype == "null"):
                name2feature[k] = v

    for features in features_list:
        for k, v in features.items():
            if isinstance(v, dict) and isinstance(name2feature[k], dict):
                _check_if_features_can_be_aligned([name2feature[k], v])
            elif not (isinstance(v, Value) and v.dtype == "null") and name2feature[k] != v:
                raise ValueError(f'The features can\'t be aligned because the key {k} of features {features} has unexpected type - {v} (expected either {name2feature[k]} or Value("null").')

def _align_features(features_list: list[Features]) -> list[Features]:
    name2feature = {}
    for features in features_list:
        for k, v in features.items():
            if k in name2feature and isinstance(v, dict):
                name2feature[k] = _align_features([name2feature[k], v])[0]
            elif k not in name2feature or (isinstance(name2feature[k], Value) and name2feature[k].dtype == "null"):
                name2feature[k] = v

    return [Features({k: name2feature[k] for k in features.keys()}) for features in features_list]

def get_indices_from_mask_function(
    function: Callable,
    batched: bool,
    with_indices: bool,
    with_rank: bool,
    input_columns: Optional[Union[str, list[str]]],
    indices_mapping: Optional[Table] = None,
    *args,
    **fn_kwargs,
):
    if batched:
        *inputs, indices, rank = args
        additional_args = ()
        if with_indices:
            additional_args += (indices,)
        if with_rank:
            additional_args += (rank,)
        mask = function(*inputs, *additional_args, **fn_kwargs)
        if isinstance(mask, (pa.Array, pa.ChunkedArray)):
            mask = mask.to_pylist()
    else:
        *inputs, indices, rank = args
        mask = []
        if input_columns is None:
            batch: dict = inputs[0]
            num_examples = len(batch[next(iter(batch.keys()))])
            for i in range(num_examples):
                example = {key: batch[key][i] for key in batch}
                additional_args = ()
                if with_indices:
                    additional_args += (indices[i],)
                if with_rank:
                    additional_args += (rank,)
                mask.append(function(example, *additional_args, **fn_kwargs))
        else:
            columns: list[list] = inputs
            num_examples = len(columns[0])
            for i in range(num_examples):
                input = [column[i] for column in columns]
                additional_args = ()
                if with_indices:
                    additional_args += (indices[i],)
                if with_rank:
                    additional_args += (rank,)
                mask.append(function(*input, *additional_args, **fn_kwargs))
    indices_array = [i for i, to_keep in zip(indices, mask) if to_keep]
    if indices_mapping is not None:
        indices_array = pa.array(indices_array, type=pa.uint64())
        indices_array = indices_mapping.column(0).take(indices_array)
        indices_array = indices_array.to_pylist()
    return {"indices": indices_array}

async def async_get_indices_from_mask_function(
    function: Callable,
    batched: bool,
    with_indices: bool,
    with_rank: bool,
    input_columns: Optional[Union[str, list[str]]],
    indices_mapping: Optional[Table] = None,
    *args,
    **fn_kwargs,
):
    if batched:
        *inputs, indices, rank = args
        additional_args = ()
        if with_indices:
            additional_args += (indices,)
        if with_rank:
            additional_args += (rank,)
        mask = await function(*inputs, *additional_args, **fn_kwargs)
        if isinstance(mask, (pa.Array, pa.ChunkedArray)):
            mask = mask.to_pylist()
    else:
        *inputs, indices, rank = args
        mask = []
        if input_columns is None:
            batch: dict = inputs[0]
            num_examples = len(batch[next(iter(batch.keys()))])
            for i in range(num_examples):
                example = {key: batch[key][i] for key in batch}
                additional_args = ()
                if with_indices:
                    additional_args += (indices[i],)
                if with_rank:
                    additional_args += (rank,)
                mask.append(await function(example, *additional_args, **fn_kwargs))
        else:
            columns: list[list] = inputs
            num_examples = len(columns[0])
            for i in range(num_examples):
                input = [column[i] for column in columns]
                additional_args = ()
                if with_indices:
                    additional_args += (indices[i],)
                if with_rank:
                    additional_args += (rank,)
                mask.append(await function(*input, *additional_args, **fn_kwargs))
    indices_array = [i for i, to_keep in zip(indices, mask) if to_keep]
    if indices_mapping is not None:
        indices_array = pa.array(indices_array, type=pa.uint64())
        indices_array = indices_mapping.column(0).take(indices_array)
        indices_array = indices_array.to_pylist()
    return {"indices": indices_array}

def _check_valid_indices_value(index, size):
    if (index < 0 and index + size < 0) or (index >= size):
        raise IndexError(f"Index {index} out of range for dataset of size {size}.")

def approximate_mode(class_counts, n_draws, rng):
    continuous = n_draws * class_counts / class_counts.sum()
    floored = np.floor(continuous)
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        for value in values:
            (inds,) = np.where(remainder == value)
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(np.int64)

def stratified_shuffle_split_generate_indices(y, n_train, n_test, rng, n_splits=10):
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = classes.shape[0]
    class_counts = np.bincount(y_indices)
    if np.min(class_counts) < 2:
        raise ValueError("Minimum class count error")
    if n_train < n_classes:
        raise ValueError("The train_size = %d should be greater or equal to the number of classes = %d" % (n_train, n_classes))
    if n_test < n_classes:
        raise ValueError("The test_size = %d should be greater or equal to the number of classes = %d" % (n_test, n_classes))
    class_indices = np.split(np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1])
    for _ in range(n_splits):
        n_i = approximate_mode(class_counts, n_train, rng)
        class_counts_remaining = class_counts - n_i
        t_i = approximate_mode(class_counts_remaining, n_test, rng)
        train = []
        test = []
        for i in range(n_classes):
            permutation = rng.permutation(class_counts[i])
            perm_indices_class_i = class_indices[i].take(permutation, mode="clip")
            train.extend(perm_indices_class_i[: n_i[i]])
            test.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])
        train = rng.permutation(train)
        test = rng.permutation(test)

        yield train, test

def table_visitor(table: pa.Table, function: Callable[[pa.Array], None]):
    features = Features.from_arrow_schema(table.schema)
    def _visit(array, feature):
        if isinstance(array, pa.ChunkedArray):
            for chunk in array.chunks:
                _visit(chunk, feature)
        else:
            if isinstance(array, pa.ExtensionArray):
                array = array.storage
            function(array, feature)
            if pa.types.is_struct(array.type) and not hasattr(feature, "cast_storage"):
                if isinstance(feature, Sequence) and isinstance(feature.feature, dict):
                    feature = {
                        name: Sequence(subfeature, length=feature.length)
                        for name, subfeature in feature.feature.items()
                    }
                for name, subfeature in feature.items():
                    _visit(array.field(name), subfeature)
            elif pa.types.is_list(array.type):
                if isinstance(feature, list):
                    _visit(array.values, feature[0])
                elif isinstance(feature, Sequence):
                    _visit(array.values, feature.feature)

    for name, feature in features.items():
        _visit(table[name], feature)

def _maybe_add_torch_iterable_dataset_parent_class(cls):
    if torch.utils.data.IterableDataset not in cls.__bases__:
        cls.__bases__ += (torch.utils.data.IterableDataset,)

def _maybe_share_with_torch_persistent_workers(value: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value.share_memory_()
    else:
        return torch.tensor(value).share_memory_()

def _convert_to_arrow(
    iterable: Iterable[tuple[Key, dict]],
    batch_size: int,
    drop_last_batch: bool = False,
) -> Iterator[tuple[Key, pa.Table]]:
    if batch_size is None or batch_size <= 0:
        yield ("all", pa.Table.from_pylist(cast_to_python_objects([example for _, example in iterable], only_1d_for_numpy=True)))
        return
    iterator = iter(iterable)
    for key, example in iterator:
        iterator_batch = islice(iterator, batch_size - 1)
        key_examples_list = [(key, example)] + list(iterator_batch)
        if len(key_examples_list) < batch_size and drop_last_batch:
            return
        keys, examples = zip(*key_examples_list)
        new_key = "_".join(str(key) for key in keys)
        yield new_key, pa.Table.from_pylist(cast_to_python_objects(examples, only_1d_for_numpy=True))

def identity_func(x):
    return x

def _batch_to_examples(batch: dict[str, list]) -> Iterator[dict[str, Any]]:
    n_examples = 0 if len(batch) == 0 else len(batch[next(iter(batch))])
    for i in range(n_examples):
        yield {col: array[i] for col, array in batch.items()}

def _examples_to_batch(examples: list[dict[str, Any]]) -> dict[str, list]:
    cols = {col: None for example in examples for col in example}
    arrays = [[example.get(col) for example in examples] for col in cols]
    return dict(zip(cols, arrays))

def _apply_feature_types_on_example(example: dict, features: Features, token_per_repo_id: dict[str, Union[str, bool, None]]) -> dict:
    example = dict(example)
    for column_name in features:
        if column_name not in example:
            example[column_name] = None
    encoded_example = features.encode_example(example)
    decoded_example = features.decode_example(encoded_example, token_per_repo_id=token_per_repo_id)
    return decoded_example

def cast_table_to_features(table: pa.Table, features: "Features"):
    if sorted(table.column_names) != sorted(features):
        raise CastError(
            f"Couldn't cast\n{_short_str(table.schema)}\nto\n{_short_str(features)}\nbecause column names don't match",
            table_column_names=table.column_names, requested_column_names=list(features),
        )
    arrays = [cast_array_to_feature(table[name], feature) for name, feature in features.items()]
    return pa.Table.from_arrays(arrays, schema=features.arrow_schema)

def _shuffle_gen_kwargs(rng: np.random.Generator, gen_kwargs: dict) -> dict:
    list_sizes = {len(value) for value in gen_kwargs.values() if isinstance(value, list)}
    indices_per_size = {}
    for size in list_sizes:
        indices_per_size[size] = list(range(size))
        rng.shuffle(indices_per_size[size])
    shuffled_kwargs = dict(gen_kwargs)
    for key, value in shuffled_kwargs.items():
        if isinstance(value, list):
            shuffled_kwargs[key] = [value[i] for i in indices_per_size[len(value)]]
    return shuffled_kwargs

def _number_of_shards_in_gen_kwargs(gen_kwargs: dict) -> int:
    lists_lengths = {key: len(value) for key, value in gen_kwargs.items() if isinstance(value, list)}
    if len(set(lists_lengths.values())) > 1:
        raise RuntimeError(
            "Sharding is ambiguous for this dataset: we found several data sources lists of different lengths, and we don't know over which list we should parallelize:\n"
            + "\n".join(f"\t- key {key} has length {length}" for key, length in lists_lengths.items())
            + "\nTo fix this, check the 'gen_kwargs' and make sure to use lists only for data sources, and use tuples otherwise. In the end there should only be one single list, or several lists with the same length."
        )
    max_length = max(lists_lengths.values(), default=0)
    return max(1, max_length)

def _distribute_shards(num_shards: int, max_num_jobs: int) -> list[range]:
    shards_indices_per_group = []
    for group_idx in range(max_num_jobs):
        num_shards_to_add = num_shards // max_num_jobs + (group_idx < (num_shards % max_num_jobs))
        if num_shards_to_add == 0:
            break
        start = shards_indices_per_group[-1].stop if shards_indices_per_group else 0
        shard_indices = range(start, start + num_shards_to_add)
        shards_indices_per_group.append(shard_indices)
    return shards_indices_per_group

def _split_gen_kwargs(gen_kwargs: dict, max_num_jobs: int) -> list[dict]:
    num_shards = _number_of_shards_in_gen_kwargs(gen_kwargs)
    if num_shards == 1:
        return [dict(gen_kwargs)]
    else:
        shard_indices_per_group = _distribute_shards(num_shards=num_shards, max_num_jobs=max_num_jobs)
        return [{key: [value[shard_idx] for shard_idx in shard_indices_per_group[group_idx]] if isinstance(value, list) else value for key, value in gen_kwargs.items()} for group_idx in range(len(shard_indices_per_group))]

def _merge_gen_kwargs(gen_kwargs_list: list[dict]) -> dict:
    return {
        key: [value for gen_kwargs in gen_kwargs_list for value in gen_kwargs[key]]
        if isinstance(gen_kwargs_list[0][key], list)
        else gen_kwargs_list[0][key]
        for key in gen_kwargs_list[0]
    }

class _BaseExamplesIterable:
    def __init__(self) -> None:
        self._state_dict: Optional[Union[list, dict]] = None

    def __iter__(self) -> Iterator[tuple[Key, dict]]:
        raise NotImplementedError(f"{type(self)} doesn't implement __iter__ yet")

    @property
    def iter_arrow(self) -> Optional[Callable[[], Iterator[tuple[Key, pa.Table]]]]:
        return None

    @property
    def is_typed(self) -> bool:
        return False

    @property
    def features(self) -> Optional[Features]:
        return None

    def shuffle_data_sources(self, generator: np.random.Generator) -> "_BaseExamplesIterable":
        raise NotImplementedError(f"{type(self)} doesn't implement shuffle_data_sources yet")

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "_BaseExamplesIterable":
        raise NotImplementedError(f"{type(self)} doesn't implement shard_data_sources yet")

    def split_shard_indices_by_worker(self, num_shards: int, index: int, contiguous=True) -> list[int]:
        if contiguous:
            div = self.num_shards // num_shards
            mod = self.num_shards % num_shards
            start = div * index + min(index, mod)
            end = start + div + (1 if index < mod else 0)
            return list(range(start, end))
        else:
            return list(range(index, self.num_shards, num_shards))

    @property
    def num_shards(self) -> int:
        raise NotImplementedError(f"{type(self)} doesn't implement num_shards yet")

    def _init_state_dict(self) -> dict:
        raise NotImplementedError(f"{type(self)} doesn't implement _init_state_dict yet")

    def load_state_dict(self, state_dict: dict) -> dict:
        def _inner_load_state_dict(state, new_state):
            if new_state is not None and isinstance(state, dict):
                for key in new_state:
                    state[key] = _inner_load_state_dict(state[key], new_state[key])
                return state
            elif new_state is not None and isinstance(state, list):
                for i in range(len(state)):
                    state[i] = _inner_load_state_dict(state[i], new_state[i])
                return state
            return new_state

        return _inner_load_state_dict(self._state_dict, state_dict)

    def state_dict(self) -> dict:
        if self._state_dict:
            return copy.deepcopy(self._state_dict)
        raise RuntimeError("State dict is not initialized, please call ex_iterable._init_state_dict() first.")

@dataclass
class FormattingConfig:
    format_type: Optional[str]

    @property
    def is_table(self) -> bool:
        return isinstance(get_formatter(self.format_type), TableFormatter)

    @property
    def is_tensor(self) -> bool:
        return isinstance(get_formatter(self.format_type), TensorFormatter)

class FormattedExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        ex_iterable: _BaseExamplesIterable,
        formatting: Optional[FormattingConfig],
        features: Optional[Features],
        token_per_repo_id: dict[str, Union[str, bool, None]],
    ):
        super().__init__()
        self.ex_iterable = ex_iterable
        self._features = features
        self.formatting = formatting
        self.token_per_repo_id = token_per_repo_id

    @property
    def iter_arrow(self):
        if self.ex_iterable.iter_arrow and (not self.formatting or self.formatting.is_table):
            return self._iter_arrow

    @property
    def is_typed(self):
        return self.ex_iterable.is_typed or self._features is not None

    @property
    def features(self):
        return self._features

    def _init_state_dict(self) -> dict:
        self._state_dict = self.ex_iterable._init_state_dict()
        return self._state_dict

    def __iter__(self):
        if not self.formatting or self.formatting.is_table:
            formatter = PythonFormatter(features=self._features if not self.ex_iterable.is_typed else None)
        else:
            formatter = get_formatter(
                self.formatting.format_type,
                features=self._features if not self.ex_iterable.is_typed else None,
                token_per_repo_id=self.token_per_repo_id,
            )
        if self.ex_iterable.iter_arrow:
            for key, pa_table in self._iter_arrow():
                batch = formatter.format_batch(pa_table)
                for example in _batch_to_examples(batch):
                    yield key, example
        else:
            format_dict = (
                formatter.recursive_tensorize
                if isinstance(formatter, TensorFormatter)
                else None  # cast in case features is None
            )
            for key, example in self.ex_iterable:
                if self.features and not self.ex_iterable.is_typed:
                    example = _apply_feature_types_on_example(example, self.features, token_per_repo_id=self.token_per_repo_id)
                if format_dict:
                    example = format_dict(example)
                yield key, example

    def _iter_arrow(self) -> Iterator[tuple[Key, pa.Table]]:
        if not self.features:
            yield from self.ex_iterable._iter_arrow()
        for key, pa_table in self.ex_iterable._iter_arrow():
            columns = set(pa_table.column_names)
            schema = self.features.arrow_schema
            for column_name in self.features:
                if column_name not in columns:
                    col = pa.NullArray.from_buffers(pa.null(), len(pa_table), [None])
                    pa_table = pa_table.append_column(column_name, col)
            if pa_table.schema != schema:
                pa_table = cast_table_to_features(pa_table, self.features)
            yield key, pa_table

    def shuffle_data_sources(self, generator: np.random.Generator) -> "FormattedExamplesIterable":
        return FormattedExamplesIterable(
            self.ex_iterable.shuffle_data_sources(generator),
            features=self.features,
            token_per_repo_id=self.token_per_repo_id,
            formatting=self.formatting,
        )

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "FormattedExamplesIterable":
        return FormattedExamplesIterable(
            self.ex_iterable.shard_data_sources(num_shards, index, contiguous=contiguous),
            features=self.features,
            token_per_repo_id=self.token_per_repo_id,
            formatting=self.formatting,
        )

    @property
    def num_shards(self) -> int:
        return self.ex_iterable.num_shards

class RebatchedArrowExamplesIterable(_BaseExamplesIterable):
    def __init__(self, ex_iterable: _BaseExamplesIterable, batch_size: Optional[int], drop_last_batch: bool = False):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch

    @property
    def iter_arrow(self):
        return self._iter_arrow

    @property
    def is_typed(self):
        return self.ex_iterable.is_typed

    @property
    def features(self):
        return self.ex_iterable.features

    def _init_state_dict(self) -> dict:
        self._state_dict = {
            "examples_iterable": self.ex_iterable._init_state_dict(),
            "previous_state": None,
            "batch_idx": 0,
            "num_chunks_since_previous_state": 0,
            "cropped_chunk_length": 0,
            "type": self.__class__.__name__,
        }
        return self._state_dict

    def __iter__(self):
        yield from self.ex_iterable

    def _iter_arrow(self) -> Iterator[tuple[Key, pa.Table]]:
        if self._state_dict and self._state_dict["previous_state"]:
            self.ex_iterable.load_state_dict(self._state_dict["previous_state"])
        if self.ex_iterable.iter_arrow:
            iterator = self.ex_iterable.iter_arrow()
        else:
            iterator = _convert_to_arrow(self.ex_iterable, batch_size=1)
        if self.batch_size is None or self.batch_size <= 0:
            if self._state_dict and self._state_dict["batch_idx"] > 0:
                return
            all_pa_table = pa.concat_tables([pa_table for _, pa_table in iterator])
            if self._state_dict:
                self._state_dict["batch_idx"] = 1
            yield "all", all_pa_table
            return
        keys_buffer = []
        chunks_buffer = []
        chunks_buffer_size = 0
        num_chunks_to_skip = self._state_dict["num_chunks_since_previous_state"] if self._state_dict else 0
        chunk_length_to_crop = self._state_dict["cropped_chunk_length"] if self._state_dict else 0
        if self._state_dict:
            previous_state = self.ex_iterable.state_dict()
            self._state_dict["previous_state"] = previous_state
        for key, pa_table in iterator:
            for num_chunks_since_previous_state, chunk in enumerate(pa_table.to_reader(max_chunksize=self.batch_size)):
                if num_chunks_to_skip > 1:
                    num_chunks_to_skip -= 1
                    continue
                elif num_chunks_to_skip == 1 and chunk_length_to_crop == 0:
                    num_chunks_to_skip -= 1
                    continue
                elif num_chunks_to_skip == 1 and chunk_length_to_crop > 0:
                    chunk = chunk.slice(chunk_length_to_crop, len(chunk) - chunk_length_to_crop)
                    num_chunks_to_skip = 0
                    chunk_length_to_crop = 0
                if len(chunk) == 0:
                    continue

                if chunks_buffer_size + len(chunk) < self.batch_size:
                    keys_buffer.append(key)
                    chunks_buffer.append(chunk)
                    chunks_buffer_size += len(chunk)
                    continue
                elif chunks_buffer_size + len(chunk) == self.batch_size:
                    keys_buffer.append(key)
                    chunks_buffer.append(chunk)
                    new_key = "_".join(str(_key) for _key in keys_buffer)
                    if self._state_dict:
                        self._state_dict["batch_idx"] += 1
                        self._state_dict["num_chunks_since_previous_state"] += len(chunks_buffer)
                        self._state_dict["cropped_chunk_length"] = 0
                    yield new_key, pa.Table.from_batches(chunks_buffer)
                    keys_buffer = []
                    chunks_buffer = []
                    chunks_buffer_size = 0
                    if self._state_dict:
                        self._state_dict["previous_state"] = previous_state
                        self._state_dict["num_chunks_since_previous_state"] = num_chunks_since_previous_state + 1
                else:
                    cropped_chunk_length = self.batch_size - chunks_buffer_size
                    keys_buffer.append(f"{key}[:{cropped_chunk_length}]")
                    chunks_buffer.append(chunk.slice(0, cropped_chunk_length))
                    new_key = "_".join(str(_key) for _key in keys_buffer)
                    if self._state_dict:
                        self._state_dict["batch_idx"] += 1
                        self._state_dict["num_chunks_since_previous_state"] += len(chunks_buffer)
                        self._state_dict["cropped_chunk_length"] = cropped_chunk_length
                    yield new_key, pa.Table.from_batches(chunks_buffer)
                    keys_buffer = [f"{key}[{cropped_chunk_length}:]"]
                    chunks_buffer = [chunk.slice(cropped_chunk_length, len(chunk) - cropped_chunk_length)]
                    chunks_buffer_size = len(chunk) - cropped_chunk_length
                    if self._state_dict:
                        self._state_dict["previous_state"] = previous_state
                        self._state_dict["num_chunks_since_previous_state"] = num_chunks_since_previous_state
            if self._state_dict:
                previous_state = self.ex_iterable.state_dict()
        if not self.drop_last_batch and chunks_buffer:
            new_key = "_".join(str(_key) for _key in keys_buffer)
            if self._state_dict:
                self._state_dict["previous_state"] = previous_state
                self._state_dict["batch_idx"] += 1
                self._state_dict["num_chunks_since_previous_state"] = 0
                self._state_dict["cropped_chunk_length"] = 0
            yield new_key, pa.Table.from_batches(chunks_buffer)

    def shuffle_data_sources(self, generator: np.random.Generator) -> "RebatchedArrowExamplesIterable":
        return RebatchedArrowExamplesIterable(self.ex_iterable.shuffle_data_sources(generator), self.batch_size, self.drop_last_batch)

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "RebatchedArrowExamplesIterable":
        return RebatchedArrowExamplesIterable(
            self.ex_iterable.shard_data_sources(num_shards, index, contiguous=contiguous),
            self.batch_size,
            self.drop_last_batch,
        )

    @property
    def num_shards(self) -> int:
        return self.ex_iterable.num_shards

class StepExamplesIterable(_BaseExamplesIterable):
    def __init__(self, ex_iterable: _BaseExamplesIterable, step: int, offset: int):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.step = step
        self.offset = offset

    @property
    def is_typed(self):
        return self.ex_iterable.is_typed

    @property
    def features(self):
        return self.ex_iterable.features

    def _init_state_dict(self) -> dict:
        self._state_dict = self.ex_iterable._init_state_dict()
        return self._state_dict

    def __iter__(self):
        ex_iterator = iter(self.ex_iterable)
        while True:
            batch = list(islice(ex_iterator, self.step))
            if len(batch) > self.offset:
                yield batch[self.offset]
            else:
                break

    def shuffle_data_sources(self, generator: np.random.Generator) -> "StepExamplesIterable":
        return StepExamplesIterable(self.ex_iterable.shuffle_data_sources(generator), step=self.step, offset=self.offset)

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "StepExamplesIterable":
        return StepExamplesIterable(
            self.ex_iterable.shard_data_sources(num_shards, index, contiguous=contiguous),
            step=self.step,
            offset=self.offset,
        )

    @property
    def num_shards(self) -> int:
        return self.ex_iterable.num_shards

class ArrowExamplesIterable(_BaseExamplesIterable):
    def __init__(self, generate_tables_fn: Callable[..., tuple[Key, pa.Table]], kwargs: dict):
        super().__init__()
        self.generate_tables_fn = generate_tables_fn
        self.kwargs = kwargs

    @property
    def iter_arrow(self):
        return self._iter_arrow

    def _init_state_dict(self) -> dict:
        self._state_dict = {"shard_idx": 0, "shard_example_idx": 0, "type": self.__class__.__name__}
        return self._state_dict

    def __iter__(self):
        formatter = PythonFormatter()
        shard_idx_start = self._state_dict["shard_idx"] if self._state_dict else 0
        for gen_kwags in islice(_split_gen_kwargs(self.kwargs, max_num_jobs=self.num_shards), shard_idx_start, None):
            shard_example_idx_start = self._state_dict["shard_example_idx"] if self._state_dict else 0
            shard_example_idx = 0
            for key, pa_table in self.generate_tables_fn(**gen_kwags):
                if shard_example_idx + len(pa_table) <= shard_example_idx_start:
                    shard_example_idx += len(pa_table)
                    continue
                for pa_subtable in pa_table.to_reader(max_chunksize=data_config.ARROW_READER_BATCH_SIZE_IN_DATASET_ITER):
                    formatted_batch = formatter.format_batch(pa_subtable)
                    for example in _batch_to_examples(formatted_batch):
                        if shard_example_idx >= shard_example_idx_start:
                            if self._state_dict:
                                self._state_dict["shard_example_idx"] += 1
                            yield key, example
                        shard_example_idx += 1
            if self._state_dict:
                self._state_dict["shard_idx"] += 1
                self._state_dict["shard_example_idx"] = 0

    def _iter_arrow(self):
        shard_idx_start = self._state_dict["shard_idx"] if self._state_dict else 0
        for gen_kwags in islice(_split_gen_kwargs(self.kwargs, max_num_jobs=self.num_shards), shard_idx_start, None):
            shard_example_idx_start = self._state_dict["shard_example_idx"] if self._state_dict else 0
            shard_example_idx = 0
            for key, pa_table in self.generate_tables_fn(**gen_kwags):
                shard_example_idx += len(pa_table)
                if shard_example_idx <= shard_example_idx_start:
                    continue
                if self._state_dict:
                    self._state_dict["shard_example_idx"] += len(pa_table)
                yield key, pa_table
            if self._state_dict:
                self._state_dict["shard_idx"] += 1
                self._state_dict["shard_example_idx"] = 0

    def shuffle_data_sources(self, generator: np.random.Generator) -> "ArrowExamplesIterable":
        return ShuffledDataSourcesArrowExamplesIterable(self.generate_tables_fn, self.kwargs, generator)

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "ArrowExamplesIterable":
        gen_kwargs_list = _split_gen_kwargs(self.kwargs, max_num_jobs=self.num_shards)
        shard_indices = self.split_shard_indices_by_worker(num_shards, index, contiguous=contiguous)
        requested_gen_kwargs = _merge_gen_kwargs([gen_kwargs_list[i] for i in shard_indices])
        return ArrowExamplesIterable(self.generate_tables_fn, requested_gen_kwargs)

    @property
    def num_shards(self) -> int:
        return _number_of_shards_in_gen_kwargs(self.kwargs)

class ShuffledDataSourcesArrowExamplesIterable(ArrowExamplesIterable):
    def __init__(self, generate_tables_fn: Callable[..., tuple[Key, pa.Table]], kwargs: dict, generator: np.random.Generator,):
        super().__init__(generate_tables_fn, kwargs)
        self.generator = copy.deepcopy(generator)

    def _init_state_dict(self) -> dict:
        self._state_dict = {"shard_idx": 0, "shard_example_idx": 0, "type": self.__class__.__name__}
        return self._state_dict

    def __iter__(self):
        rng = copy.deepcopy(self.generator)
        kwargs_with_shuffled_shards = _shuffle_gen_kwargs(rng, self.kwargs)
        formatter = PythonFormatter()
        shard_idx_start = self._state_dict["shard_idx"] if self._state_dict else 0
        for gen_kwags in islice(_split_gen_kwargs(kwargs_with_shuffled_shards, max_num_jobs=self.num_shards), shard_idx_start, None):
            shard_example_idx_start = self._state_dict["shard_example_idx"] if self._state_dict else 0
            shard_example_idx = 0
            for key, pa_table in self.generate_tables_fn(**gen_kwags):
                if shard_example_idx + len(pa_table) <= shard_example_idx_start:
                    shard_example_idx += len(pa_table)
                    continue
                for pa_subtable in pa_table.to_reader(max_chunksize=data_config.ARROW_READER_BATCH_SIZE_IN_DATASET_ITER):
                    formatted_batch = formatter.format_batch(pa_subtable)
                    for example in _batch_to_examples(formatted_batch):
                        if shard_example_idx >= shard_example_idx_start:
                            if self._state_dict:
                                self._state_dict["shard_example_idx"] += 1
                            yield key, example
                        shard_example_idx += 1
            if self._state_dict:
                self._state_dict["shard_idx"] += 1
                self._state_dict["shard_example_idx"] = 0

    def _iter_arrow(self):
        rng = copy.deepcopy(self.generator)
        kwargs_with_shuffled_shards = _shuffle_gen_kwargs(rng, self.kwargs)
        shard_idx_start = self._state_dict["shard_idx"] if self._state_dict else 0
        for gen_kwags in islice(_split_gen_kwargs(kwargs_with_shuffled_shards, max_num_jobs=self.num_shards), shard_idx_start, None):
            shard_example_idx_start = self._state_dict["shard_example_idx"] if self._state_dict else 0
            shard_example_idx = 0
            for key, pa_table in self.generate_tables_fn(**gen_kwags):
                shard_example_idx += len(pa_table)
                if shard_example_idx <= shard_example_idx_start:
                    continue
                if self._state_dict:
                    self._state_dict["shard_example_idx"] += len(pa_table)
                yield key, pa_table
            if self._state_dict:
                self._state_dict["shard_idx"] += 1
                self._state_dict["shard_example_idx"] = 0

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "ArrowExamplesIterable":
        rng = copy.deepcopy(self.generator)
        kwargs_with_shuffled_shards = _shuffle_gen_kwargs(rng, self.kwargs)
        return ArrowExamplesIterable(self.generate_tables_fn, kwargs_with_shuffled_shards).shard_data_sources(num_shards, index, contiguous=contiguous)

def read_schema_from_file(filename: str) -> pa.Schema:
    with pa.memory_map(filename) as memory_mapped_stream:
        schema = pa.ipc.open_stream(memory_mapped_stream).schema
    return schema

def _table_output_to_arrow(output) -> pa.Table:
    if isinstance(output, pa.Table):
        return output
    if isinstance(output, (pd.DataFrame, pd.Series)):
        return pa.Table.from_pandas(output)
    return output

class MappedExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        ex_iterable: _BaseExamplesIterable,
        function: Callable,
        with_indices: bool = False,
        input_columns: Optional[list[str]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[list[str]] = None,
        fn_kwargs: Optional[dict] = None,
        formatting: Optional["FormattingConfig"] = None,
        features: Optional[Features] = None,
        max_num_running_async_map_functions_in_parallel: Optional[int] = None,
    ):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.function = function
        self.batched = batched
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch
        self.remove_columns = remove_columns
        self.with_indices = with_indices
        self.input_columns = input_columns
        self.fn_kwargs = fn_kwargs or {}
        self.formatting = formatting  # required for iter_arrow
        self._features = features
        self.max_num_running_async_map_functions_in_parallel = (max_num_running_async_map_functions_in_parallel or data_config.MAX_NUM_RUNNING_ASYNC_MAP_FUNCTIONS_IN_PARALLEL)
        if formatting and formatting.is_table:
            if not isinstance(ex_iterable, RebatchedArrowExamplesIterable):
                raise ValueError(f"The {formatting.format_type.capitalize()}-formatted {type(self).__name__} has underlying iterable that is a {type(ex_iterable).__name__} instead of a RebatchedArrowExamplesIterable.")
            elif ex_iterable.batch_size != (batch_size if batched else 1):
                raise ValueError(f"The {formatting.format_type.capitalize()}-formatted {type(self).__name__} has batch_size={batch_size if batched else 1} which is different from {ex_iterable.batch_size=} from its underlying iterable.")
        self._owned_loops_and_tasks: list[tuple[asyncio.AbstractEventLoop, list[asyncio.Task]]] = []

    @property
    def iter_arrow(self):
        if self.formatting and self.formatting.is_table:
            return self._iter_arrow

    @property
    def is_typed(self):
        return self.features is not None  # user has extracted features

    @property
    def features(self):
        return self._features

    def _init_state_dict(self) -> dict:
        self._state_dict = {
            "examples_iterable": self.ex_iterable._init_state_dict(),
            "previous_state": None,
            "num_examples_since_previous_state": 0,
            "previous_state_example_idx": 0,
            "type": self.__class__.__name__,
        }
        return self._state_dict

    def __iter__(self):
        if self.formatting and self.formatting.is_table:
            formatter = PythonFormatter()
            for key, pa_table in self._iter_arrow(max_chunksize=1):
                yield key, formatter.format_row(pa_table)
        else:
            yield from self._iter()

    def _iter(self):
        current_idx = self._state_dict["previous_state_example_idx"] if self._state_dict else 0
        if self._state_dict and self._state_dict["previous_state"]:
            self.ex_iterable.load_state_dict(self._state_dict["previous_state"])
            num_examples_to_skip = self._state_dict["num_examples_since_previous_state"]
        else:
            num_examples_to_skip = 0
        iterator = iter(self.ex_iterable)

        if self.formatting:
            formatter = get_formatter(self.formatting.format_type)
            format_dict = formatter.recursive_tensorize if isinstance(formatter, TensorFormatter) else None
        else:
            format_dict = None

        def iter_batched_inputs():
            nonlocal current_idx
            for key, example in iterator:
                iterator_batch = (iterator if self.batch_size is None or self.batch_size <= 0 else islice(iterator, self.batch_size - 1))
                key_examples_list = [(key, example)] + list(iterator_batch)
                keys, examples = zip(*key_examples_list)
                key = "_".join(str(key) for key in keys)
                if (
                    self.drop_last_batch
                    and self.batch_size is not None
                    and self.batch_size > 0
                    and len(examples) < self.batch_size
                ):  # ignore last batch
                    return
                batch = _examples_to_batch(examples)
                batch = format_dict(batch) if format_dict else batch
                indices = [current_idx + i for i in range(len(key_examples_list))]
                current_idx += len(indices)
                yield indices, (key, batch)

        def iter_inputs():
            nonlocal current_idx
            for key, example in iterator:
                example = dict(example)
                current_idx += 1
                yield current_idx - 1, (key, example)

        def validate_function_output(processed_inputs):
            if self.batched and processed_inputs:
                first_col = next(iter(processed_inputs))
                bad_cols = [col for col in processed_inputs if len(processed_inputs[col]) != len(processed_inputs[first_col])]
                if bad_cols:
                    raise ValueError(f"Column lengths mismatch: columns {bad_cols} have length {[len(processed_inputs[col]) for col in bad_cols]} while {first_col} has length {len(processed_inputs[first_col])}.")

        def prepare_inputs(key_example, indices):
            key, example = key_example
            fn_args = [example] if self.input_columns is None else [example[col] for col in self.input_columns]
            additional_args = ()
            if self.with_indices:
                fn_args += (indices,)
            inputs = dict(example)
            return inputs, fn_args, additional_args, self.fn_kwargs

        def prepare_outputs(key_example, inputs, processed_inputs):
            validate_function_output(processed_inputs)
            if self.remove_columns:
                for c in self.remove_columns:
                    if c in inputs:
                        del inputs[c]
                    if processed_inputs is key_example[1] and c in processed_inputs:
                        del processed_inputs[c]
            transformed_inputs = {**inputs, **processed_inputs}
            return transformed_inputs

        def apply_function(key_example, indices):
            inputs, fn_args, additional_args, fn_kwargs = prepare_inputs(key_example, indices)
            processed_inputs = self.function(*fn_args, *additional_args, **fn_kwargs)
            return prepare_outputs(key_example, inputs, processed_inputs)

        async def async_apply_function(key_example, indices):
            inputs, fn_args, additional_args, fn_kwargs = prepare_inputs(key_example, indices)
            processed_inputs = await self.function(*fn_args, *additional_args, **fn_kwargs)
            return prepare_outputs(key_example, inputs, processed_inputs)

        tasks: list[asyncio.Task] = []
        if inspect.iscoroutinefunction(self.function):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
            self._owned_loops_and_tasks.append((loop, tasks))
        else:
            loop = None

        def iter_outputs():
            nonlocal tasks, loop
            inputs_iterator = iter_batched_inputs() if self.batched else iter_inputs()
            if inspect.iscoroutinefunction(self.function):
                if self._state_dict:
                    previous_state = self.ex_iterable.state_dict()
                    self._state_dict["previous_state"] = previous_state
                    previous_state_task = None
                    previous_state_example_idx = self._state_dict["previous_state_example_idx"]
                indices: Union[list[int], list[list[int]]] = []
                for i, key_example in inputs_iterator:
                    indices.append(i)
                    tasks.append(loop.create_task(async_apply_function(key_example, i)))
                    if len(tasks) >= self.max_num_running_async_map_functions_in_parallel:
                        _, pending = loop.run_until_complete(asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED))
                        while tasks and len(pending) >= self.max_num_running_async_map_functions_in_parallel:
                            _, pending = loop.run_until_complete(asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED))
                    if len(tasks) >= 10 * self.max_num_running_async_map_functions_in_parallel:
                        loop.run_until_complete(tasks[0])
                    while tasks and tasks[0].done():
                        i, task = indices.pop(0), tasks.pop(0)
                        yield i, task.result()
                        if self._state_dict and task is previous_state_task:
                            self._state_dict["previous_state"] = previous_state
                            self._state_dict["num_examples_since_previous_state"] = 0
                            self._state_dict["previous_state_example_idx"] = previous_state_example_idx
                            previous_state, previous_state_task = None, None
                    if self._state_dict and previous_state_task is None and tasks:
                        previous_state = self.ex_iterable.state_dict()
                        previous_state_task = tasks[-1]
                        previous_state_example_idx = current_idx
                while tasks:
                    yield indices[0], loop.run_until_complete(tasks[0])
                    indices.pop(0), tasks.pop(0)
            else:
                if self._state_dict:
                    if self.batched:
                        self._state_dict["previous_state"] = self.ex_iterable.state_dict()
                        self._state_dict["num_examples_since_previous_state"] = 0
                        self._state_dict["previous_state_example_idx"] = current_idx
                for i, key_example in inputs_iterator:
                    if self._state_dict:
                        if not self.batched:
                            self._state_dict["previous_state_example_idx"] = current_idx
                    yield i, apply_function(key_example, i)
                    if self._state_dict:
                        if self.batched:
                            self._state_dict["previous_state"] = self.ex_iterable.state_dict()
                            self._state_dict["num_examples_since_previous_state"] = 0
                            self._state_dict["previous_state_example_idx"] = current_idx

        try:
            outputs = iter_outputs()
            if self.batched:
                outputs = (
                    (key, transformed_example)
                    for key, transformed_batch in outputs
                    for transformed_example in _batch_to_examples(transformed_batch)
                )
            for key, transformed_example in outputs:
                if self._state_dict and self._state_dict["previous_state"] is not None:
                    self._state_dict["num_examples_since_previous_state"] += 1
                if num_examples_to_skip > 0:
                    num_examples_to_skip -= 1
                    continue
                yield key, transformed_example
        except (Exception, KeyboardInterrupt):
            if loop:
                for task in tasks:
                    task.cancel(msg="KeyboardInterrupt")
                try:
                    loop.run_until_complete(asyncio.gather(*tasks))
                except (asyncio.CancelledError, ValueError):
                    pass
            raise

    def _iter_arrow(self, max_chunksize: Optional[int] = None) -> Iterator[tuple[Key, pa.Table]]:
        formatter: TableFormatter = get_formatter(self.formatting.format_type) if self.formatting else ArrowFormatter()
        if self.ex_iterable.iter_arrow:
            iterator = self.ex_iterable.iter_arrow()
        else:
            iterator = _convert_to_arrow(
                self.ex_iterable,
                batch_size=self.batch_size if self.batched else 1,
                drop_last_batch=self.drop_last_batch,
            )
        if self._state_dict and self._state_dict["previous_state"]:
            self.ex_iterable.load_state_dict(self._state_dict["previous_state"])
            num_examples_to_skip = self._state_dict["num_examples_since_previous_state"]
        else:
            num_examples_to_skip = 0
        if self._state_dict and max_chunksize is not None:
            self._state_dict["previous_state"] = self.ex_iterable.state_dict()
            self._state_dict["num_examples_since_previous_state"] = 0
        current_idx = self._state_dict["previous_state_example_idx"] if self._state_dict else 0
        for key, pa_table in iterator:
            if (self.batched and self.batch_size is not None and len(pa_table) < self.batch_size and self.drop_last_batch):
                return
            function_args = ([formatter.format_batch(pa_table)] if self.input_columns is None else [pa_table[col] for col in self.input_columns])
            if self.with_indices:
                if self.batched:
                    function_args.append([current_idx + i for i in range(len(pa_table))])
                else:
                    function_args.append(current_idx)
            output = self.function(*function_args, **self.fn_kwargs)
            output_table = _table_output_to_arrow(output)
            if not isinstance(output_table, pa.Table):
                raise TypeError(f"Provided `function` which is applied to {formatter.table_type} returns a variable of type {type(output)}. Make sure provided `function` returns a {formatter.table_type} to update the dataset.")
            if self.remove_columns:
                for column in self.remove_columns:
                    if column in output_table.column_names:
                        output_table = output_table.remove_column(output_table.column_names.index(column))
            if max_chunksize is None:
                current_idx += len(pa_table)
                if self._state_dict:
                    self._state_dict["previous_state_example_idx"] += len(pa_table)
                yield key, output_table
            else:
                for i, pa_subtable in enumerate(output_table.to_reader(max_chunksize=max_chunksize)):
                    current_idx += 1
                    if self._state_dict:
                        self._state_dict["num_examples_since_previous_state"] += 1
                    if num_examples_to_skip > 0:
                        num_examples_to_skip -= 1
                        continue
                    yield f"{key}_{i}", pa_subtable
                if self._state_dict:
                    self._state_dict["previous_state"] = self.ex_iterable.state_dict()
                    self._state_dict["num_examples_since_previous_state"] = 0
                    self._state_dict["previous_state_example_idx"] += len(pa_table)

    def shuffle_data_sources(self, generator: np.random.Generator) -> "MappedExamplesIterable":
        return MappedExamplesIterable(
            self.ex_iterable.shuffle_data_sources(generator),
            function=self.function,
            with_indices=self.with_indices,
            input_columns=self.input_columns,
            batched=self.batched,
            batch_size=self.batch_size,
            drop_last_batch=self.drop_last_batch,
            remove_columns=self.remove_columns,
            fn_kwargs=self.fn_kwargs,
            formatting=self.formatting,
            features=self.features,
            max_num_running_async_map_functions_in_parallel=self.max_num_running_async_map_functions_in_parallel,
        )

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "MappedExamplesIterable":
        return MappedExamplesIterable(
            self.ex_iterable.shard_data_sources(num_shards, index, contiguous=contiguous),
            function=self.function,
            with_indices=self.with_indices,
            input_columns=self.input_columns,
            batched=self.batched,
            batch_size=self.batch_size,
            drop_last_batch=self.drop_last_batch,
            remove_columns=self.remove_columns,
            fn_kwargs=self.fn_kwargs,
            formatting=self.formatting,
            features=self.features,
            max_num_running_async_map_functions_in_parallel=self.max_num_running_async_map_functions_in_parallel,
        )

    @property
    def num_shards(self) -> int:
        return self.ex_iterable.num_shards

def _add_mask(
    input: Union[dict, pa.Table],
    mask: Union[bool, list, pa.Array, pa.ChunkedArray, pa.BooleanScalar],
    mask_column_name: str,
):
    if isinstance(input, pa.Table):
        if not isinstance(mask, (list, pa.Array, pa.ChunkedArray)):
            mask = pa.array([mask], type=pa.bool_())
        return input.append_column(mask_column_name, mask)
    else:
        return {mask_column_name: mask}

def add_mask(mask_function: Callable, input: Union[dict, pa.Table], *args, mask_column_name: str, **kwargs):
    mask = mask_function(input, *args, **kwargs)
    return _add_mask(input, mask, mask_column_name)

async def async_add_mask(mask_function: Callable, input: Union[dict, pa.Table], *args, mask_column_name: str, **kwargs):
    mask = await mask_function(input, *args, **kwargs)
    return _add_mask(input, mask, mask_column_name)

class FilteredExamplesIterable(MappedExamplesIterable):
    mask_column_name = "===MASK==="

    def __init__(
        self,
        ex_iterable: _BaseExamplesIterable,
        function: Callable,
        with_indices: bool = False,
        input_columns: Optional[list[str]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
        formatting: Optional["FormattingConfig"] = None,
    ):
        self.mask_function = function
        if ex_iterable.is_typed:
            features = Features({**ex_iterable.features, self.mask_column_name: Value("bool")})
        else:
            features = None
        super().__init__(
            ex_iterable=ex_iterable,
            function=partial(async_add_mask if inspect.iscoroutinefunction(function) else add_mask, function, mask_column_name=self.mask_column_name),
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            fn_kwargs=fn_kwargs,
            formatting=formatting,
            features=features,
        )

    def _iter(self):
        for key, example in super()._iter():
            example = dict(example)
            if example.pop(self.mask_column_name):
                yield key, example

    def _iter_arrow(self, max_chunksize: Optional[int] = None):
        for key, pa_table in super()._iter_arrow(max_chunksize=max_chunksize):
            mask = pa_table[self.mask_column_name]
            yield key, pa_table.drop(self.mask_column_name).filter(mask)

    def shuffle_data_sources(self, seed: Optional[int]) -> "FilteredExamplesIterable":
        return FilteredExamplesIterable(
            self.ex_iterable.shuffle_data_sources(seed),
            function=self.mask_function,
            with_indices=self.with_indices,
            input_columns=self.input_columns,
            batched=self.batched,
            batch_size=self.batch_size,
            fn_kwargs=self.fn_kwargs,
            formatting=self.formatting,
        )

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "FilteredExamplesIterable":
        return FilteredExamplesIterable(
            self.ex_iterable.shard_data_sources(num_shards, index, contiguous=contiguous),
            function=self.mask_function,
            with_indices=self.with_indices,
            input_columns=self.input_columns,
            batched=self.batched,
            batch_size=self.batch_size,
            fn_kwargs=self.fn_kwargs,
            formatting=self.formatting,
        )

    @property
    def num_shards(self) -> int:
        return self.ex_iterable.num_shards

class BufferShuffledExamplesIterable(_BaseExamplesIterable):
    def __init__(self, ex_iterable: _BaseExamplesIterable, buffer_size: int, generator: np.random.Generator):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.buffer_size = buffer_size
        self.generator = generator

    @property
    def is_typed(self):
        return self.ex_iterable.is_typed

    @property
    def features(self):
        return self.ex_iterable.features

    def _init_state_dict(self) -> dict:
        self._state_dict = self.ex_iterable._init_state_dict()
        self._original_state_dict = self.state_dict()
        return self._state_dict

    def load_state_dict(self, state_dict: dict) -> dict:
        return super().load_state_dict(state_dict)

    @staticmethod
    def _iter_random_indices(rng: np.random.Generator, buffer_size: int, random_batch_size=1000) -> Iterator[int]:
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    def __iter__(self):
        buffer_size = self.buffer_size
        rng = copy.deepcopy(self.generator)
        indices_iterator = self._iter_random_indices(rng, buffer_size)
        mem_buffer = []
        for x in self.ex_iterable:
            if len(mem_buffer) == buffer_size:  # if the buffer is full, pick and example from it
                i = next(indices_iterator)
                yield mem_buffer[i]
                mem_buffer[i] = x  # replace the picked example by a new one
            else:  # otherwise, keep filling the buffer
                mem_buffer.append(x)
        rng.shuffle(mem_buffer)
        yield from mem_buffer

    def shuffle_data_sources(self, generator: np.random.Generator) -> "BufferShuffledExamplesIterable":
        return BufferShuffledExamplesIterable(self.ex_iterable.shuffle_data_sources(generator), buffer_size=self.buffer_size, generator=generator)

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "BufferShuffledExamplesIterable":
        return BufferShuffledExamplesIterable(self.ex_iterable.shard_data_sources(num_shards, index, contiguous=contiguous), buffer_size=self.buffer_size, generator=self.generator)

    @property
    def num_shards(self) -> int:
        return self.ex_iterable.num_shards

class SkipExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        ex_iterable: _BaseExamplesIterable,
        n: int,
        block_sources_order_when_shuffling: bool = True,
        split_when_sharding: bool = True,
    ):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.n = n
        self.block_sources_order_when_shuffling = block_sources_order_when_shuffling
        self.split_when_sharding = split_when_sharding

    @property
    def is_typed(self):
        return self.ex_iterable.is_typed

    @property
    def features(self):
        return self.ex_iterable.features

    def _init_state_dict(self) -> dict:
        self._state_dict = {
            "skipped": False,
            "examples_iterable": self.ex_iterable._init_state_dict(),
            "type": self.__class__.__name__,
        }
        return self._state_dict

    def __iter__(self):
        ex_iterable_idx_start = 0 if self._state_dict and self._state_dict["skipped"] else self.n
        if self._state_dict:
            self._state_dict["skipped"] = True
        yield from islice(self.ex_iterable, ex_iterable_idx_start, None)

    @staticmethod
    def split_number(num, n):
        quotient = num // n
        remainder = num % n
        result = [quotient] * n
        for i in range(remainder):
            result[i] += 1
        return result

    def shuffle_data_sources(self, generator: np.random.Generator) -> "SkipExamplesIterable":
        if self.block_sources_order_when_shuffling:
            return self
        else:
            return SkipExamplesIterable(
                self.ex_iterable.shuffle_data_sources(generator),
                n=self.n,
                block_sources_order_when_shuffling=self.block_sources_order_when_shuffling,
                split_when_sharding=self.split_when_sharding,
            )

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "SkipExamplesIterable":
        if self.split_when_sharding:
            return SkipExamplesIterable(
                self.ex_iterable.shard_data_sources(num_shards, index, contiguous=contiguous),
                n=self.split_number(self.n, num_shards)[index],
                block_sources_order_when_shuffling=self.block_sources_order_when_shuffling,
                split_when_sharding=self.split_when_sharding,
            )
        else:
            return self

    @property
    def num_shards(self) -> int:
        return self.ex_iterable.num_shards

class RepeatExamplesIterable(_BaseExamplesIterable):
    def __init__(self, ex_iterable: _BaseExamplesIterable, num_times: Optional[int],):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.num_times = num_times

    def _init_state_dict(self) -> dict:
        self._state_dict = {
            "repeat_index": 0,
            "examples_iterable": self.ex_iterable._init_state_dict(),
            "type": self.__class__.__name__,
        }
        return self._state_dict

    def __iter__(self):
        repeat_index = self._state_dict["repeat_index"] if self._state_dict else 0
        while True:
            if self.num_times is not None and repeat_index >= max(self.num_times, 0):
                break
            yield from self.ex_iterable
            repeat_index += 1
            if self._state_dict:
                self._state_dict["repeat_index"] = repeat_index
                self._state_dict["examples_iterable"] = self.ex_iterable._init_state_dict()

    def shuffle_data_sources(self, generator: np.random.Generator) -> "RepeatExamplesIterable":
        return RepeatExamplesIterable(self.ex_iterable.shuffle_data_sources(generator), num_times=self.num_times)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> "RepeatExamplesIterable":
        return RepeatExamplesIterable(self.ex_iterable.shard_data_sources(worker_id, num_workers), num_times=self.num_times)

    @property
    def n_shards(self) -> int:
        return self.ex_iterable.n_shards

class TakeExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        ex_iterable: _BaseExamplesIterable,
        n: int,
        block_sources_order_when_shuffling: bool = True,
        split_when_sharding: bool = True,
    ):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.n = n
        self.block_sources_order_when_shuffling = block_sources_order_when_shuffling
        self.split_when_sharding = split_when_sharding
        # TODO(QL): implement iter_arrow

    @property
    def is_typed(self):
        return self.ex_iterable.is_typed

    @property
    def features(self):
        return self.ex_iterable.features

    def _init_state_dict(self) -> dict:
        self._state_dict = {
            "num_taken": 0,
            "examples_iterable": self.ex_iterable._init_state_dict(),
            "type": self.__class__.__name__,
        }
        return self._state_dict

    def __iter__(self):
        ex_iterable_num_taken = self._state_dict["num_taken"] if self._state_dict else 0
        for key_example in islice(self.ex_iterable, self.n - ex_iterable_num_taken):
            if self._state_dict:
                self._state_dict["num_taken"] += 1
            yield key_example

    @staticmethod
    def split_number(num, n):
        quotient = num // n
        remainder = num % n
        result = [quotient] * n
        for i in range(remainder):
            result[i] += 1
        return result

    def shuffle_data_sources(self, generator: np.random.Generator) -> "TakeExamplesIterable":
        if self.block_sources_order_when_shuffling:
            return self
        else:
            return TakeExamplesIterable(
                self.ex_iterable.shuffle_data_sources(generator),
                n=self.n,
                block_sources_order_when_shuffling=self.block_sources_order_when_shuffling,
                split_when_sharding=self.split_when_sharding,
            )

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "TakeExamplesIterable":
        if self.split_when_sharding:
            return TakeExamplesIterable(
                self.ex_iterable.shard_data_sources(num_shards, index, contiguous=contiguous),
                n=self.split_number(self.n, num_shards)[index],
                block_sources_order_when_shuffling=self.block_sources_order_when_shuffling,
                split_when_sharding=self.split_when_sharding,
            )
        else:
            return TakeExamplesIterable(
                self.ex_iterable.shard_data_sources(num_shards, index, contiguous=contiguous),
                n=self.n,
                block_sources_order_when_shuffling=self.block_sources_order_when_shuffling,
                split_when_sharding=self.split_when_sharding,
            )

    @property
    def num_shards(self) -> int:
        return self.ex_iterable.num_shards

class ExamplesIterable(_BaseExamplesIterable):
    def __init__(self, generate_examples_fn: Callable[..., tuple[Key, dict]], kwargs: dict):
        super().__init__()
        self.generate_examples_fn = generate_examples_fn
        self.kwargs = kwargs

    def _init_state_dict(self) -> dict:
        self._state_dict = {"shard_idx": 0, "shard_example_idx": 0, "type": self.__class__.__name__}
        return self._state_dict

    def __iter__(self):
        shard_idx_start = self._state_dict["shard_idx"] if self._state_dict else 0
        for gen_kwags in islice(_split_gen_kwargs(self.kwargs, max_num_jobs=self.num_shards), shard_idx_start, None):
            shard_example_idx_start = self._state_dict["shard_example_idx"] if self._state_dict else 0
            for key_example in islice(self.generate_examples_fn(**gen_kwags), shard_example_idx_start, None):
                if self._state_dict:
                    self._state_dict["shard_example_idx"] += 1
                yield key_example
            if self._state_dict:
                self._state_dict["shard_idx"] += 1
                self._state_dict["shard_example_idx"] = 0

    def shuffle_data_sources(self, generator: np.random.Generator) -> "ExamplesIterable":
        return ShuffledDataSourcesExamplesIterable(self.generate_examples_fn, self.kwargs, generator)

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "ExamplesIterable":
        gen_kwargs_list = _split_gen_kwargs(self.kwargs, max_num_jobs=self.num_shards)
        shard_indices = self.split_shard_indices_by_worker(num_shards, index, contiguous=contiguous)
        requested_gen_kwargs = _merge_gen_kwargs([gen_kwargs_list[i] for i in shard_indices])
        return ExamplesIterable(self.generate_examples_fn, requested_gen_kwargs)

    @property
    def num_shards(self) -> int:
        return _number_of_shards_in_gen_kwargs(self.kwargs)

class ShuffledDataSourcesExamplesIterable(ExamplesIterable):
    def __init__(self, generate_examples_fn: Callable[..., tuple[Key, dict]], kwargs: dict, generator: np.random.Generator):
        super().__init__(generate_examples_fn, kwargs)
        self.generator = copy.deepcopy(generator)

    def _init_state_dict(self) -> dict:
        self._state_dict = {"shard_idx": 0, "shard_example_idx": 0, "type": self.__class__.__name__}
        return self._state_dict

    def __iter__(self):
        rng = copy.deepcopy(self.generator)
        kwargs_with_shuffled_shards = _shuffle_gen_kwargs(rng, self.kwargs)
        shard_idx_start = self._state_dict["shard_idx"] if self._state_dict else 0
        for gen_kwags in islice(_split_gen_kwargs(kwargs_with_shuffled_shards, max_num_jobs=self.num_shards), shard_idx_start, None):
            shard_example_idx_start = self._state_dict["shard_example_idx"] if self._state_dict else 0
            for key_example in islice(self.generate_examples_fn(**gen_kwags), shard_example_idx_start, None):
                if self._state_dict:
                    self._state_dict["shard_example_idx"] += 1
                yield key_example
            if self._state_dict:
                self._state_dict["shard_idx"] += 1
                self._state_dict["shard_example_idx"] = 0

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "ExamplesIterable":
        rng = copy.deepcopy(self.generator)
        kwargs_with_shuffled_shards = _shuffle_gen_kwargs(rng, self.kwargs)
        return ExamplesIterable(self.generate_examples_fn, kwargs_with_shuffled_shards).shard_data_sources(num_shards, index, contiguous=contiguous)

def add_column_fn(example: dict, idx: int, name: str, column: list[dict]):
    if name in example:
        raise ValueError(f"Error when adding {name}: column {name} is already in the dataset.")
    return {name: column[idx]}

def _rename_columns_fn(example: dict, column_mapping: dict[str, str]):
    if any(col not in example for col in column_mapping):
        raise ValueError(f"Error when renaming {list(column_mapping)} to {list(column_mapping.values())}: columns {set(column_mapping) - set(example)} are not in the dataset.")
    if any(col in example for col in column_mapping.values()):
        raise ValueError(f"Error when renaming {list(column_mapping)} to {list(column_mapping.values())}: columns {set(example) - set(column_mapping.values())} are already in the dataset.")
    return {new_column_name: example[original_column_name] for original_column_name, new_column_name in column_mapping.items()}

class SelectColumnsIterable(_BaseExamplesIterable):
    def __init__(self, ex_iterable: _BaseExamplesIterable, column_names: list[str]):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.column_names = column_names

    @property
    def iter_arrow(self):
        if self.ex_iterable.iter_arrow:
            return self._iter_arrow

    @property
    def is_typed(self):
        return self.ex_iterable.is_typed

    @property
    def features(self):
        return self.ex_iterable.features

    def _init_state_dict(self) -> dict:
        self._state_dict = self.ex_iterable._init_state_dict()
        return self._state_dict

    def __iter__(self):
        for idx, row in self.ex_iterable:
            yield idx, {c: row[c] for c in self.column_names}

    def _iter_arrow(self) -> Iterator[tuple[Key, pa.Table]]:
        for idx, pa_table in self.ex_iterable.iter_arrow():
            if len(pa_table) > 0:  # empty tables have no schema
                yield idx, pa_table.select(self.column_names)

    def shuffle_data_sources(self, generator: np.random.Generator) -> "SelectColumnsIterable":
        return SelectColumnsIterable(self.ex_iterable.shuffle_data_sources(generator), self.column_names)

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "SelectColumnsIterable":
        return SelectColumnsIterable(self.ex_iterable.shard_data_sources(num_shards, index, contiguous=contiguous), self.column_names)

    @property
    def num_shards(self) -> int:
        return self.ex_iterable.num_shards

async def _apply_async(pool, func, x):
    future = pool.apply_async(func, (x,))
    while True:
        if future.ready():
            return future.get()
        else:
            await asyncio.sleep(0)

def _infer_features_from_batch(batch: dict[str, list], try_features: Optional[Features] = None) -> Features:
    pa_table = pa.Table.from_pydict(batch)
    if try_features is not None:
        try:
            pa_table = table_cast(pa_table, pa.schema(try_features.type))
        except (TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
            pass
    return Features.from_arrow_schema(pa_table.schema)

SANITIZED_DEFAULT_SPLIT = str(Split.TRAIN)

def sanitize_patterns(patterns: Union[dict, list, str]) -> dict[str, Union[list[str], "DataFilesList"]]:
    if isinstance(patterns, dict):
        return {str(key): value if isinstance(value, list) else [value] for key, value in patterns.items()}
    elif isinstance(patterns, str):
        return {SANITIZED_DEFAULT_SPLIT: [patterns]}
    elif isinstance(patterns, list):
        if any(isinstance(pattern, dict) for pattern in patterns):
            for pattern in patterns:
                if not (
                    isinstance(pattern, dict)
                    and len(pattern) == 2
                    and "split" in pattern
                    and isinstance(pattern.get("path"), (str, list))
                ):
                    raise ValueError(f"Expected each split to have a 'path' key which can be a string or a list of strings, but got {pattern}")
            splits = [pattern["split"] for pattern in patterns]
            if len(set(splits)) != len(splits):
                raise ValueError(f"Some splits are duplicated in data_files: {splits}")
            return {str(pattern["split"]): pattern["path"] if isinstance(pattern["path"], list) else [pattern["path"]] for pattern in patterns}
        else:
            return {SANITIZED_DEFAULT_SPLIT: patterns}
    else:
        return sanitize_patterns(list(patterns))
    
def extend_module_for_streaming(module_path, download_config: Optional[DownloadConfig] = None):
    module = importlib.import_module(module_path)

    if hasattr(module, "_patched_for_streaming") and module._patched_for_streaming:
        if isinstance(module._patched_for_streaming, DownloadConfig):
            module._patched_for_streaming.token = download_config.token
            module._patched_for_streaming.storage_options = download_config.storage_options
        return

    def wrap_auth(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, download_config=download_config, **kwargs)

        wrapper._decorator_name_ = "wrap_auth"
        return wrapper

    patch_submodule(module, "open", wrap_auth(xopen)).start()
    patch_submodule(module, "os.listdir", wrap_auth(xlistdir)).start()
    patch_submodule(module, "os.walk", wrap_auth(xwalk)).start()
    patch_submodule(module, "glob.glob", wrap_auth(xglob)).start()
    patch_submodule(module, "os.path.join", xjoin).start()
    patch_submodule(module, "os.path.dirname", xdirname).start()
    patch_submodule(module, "os.path.basename", xbasename).start()
    patch_submodule(module, "os.path.relpath", xrelpath).start()
    patch_submodule(module, "os.path.split", xsplit).start()
    patch_submodule(module, "os.path.splitext", xsplitext).start()
    patch_submodule(module, "os.path.exists", wrap_auth(xexists)).start()
    patch_submodule(module, "os.path.isdir", wrap_auth(xisdir)).start()
    patch_submodule(module, "os.path.isfile", wrap_auth(xisfile)).start()
    patch_submodule(module, "os.path.getsize", wrap_auth(xgetsize)).start()
    patch_submodule(module, "pathlib.Path", xPath).start()
    patch_submodule(module, "gzip.open", wrap_auth(xgzip_open)).start()
    patch_submodule(module, "numpy.load", wrap_auth(xnumpy_load)).start()
    patch_submodule(module, "pandas.read_csv", wrap_auth(xpandas_read_csv), attrs=["__version__"]).start()
    patch_submodule(module, "pandas.read_excel", wrap_auth(xpandas_read_excel), attrs=["__version__"]).start()
    patch_submodule(module, "scipy.io.loadmat", wrap_auth(xsio_loadmat), attrs=["__version__"]).start()
    patch_submodule(module, "xml.etree.ElementTree.parse", wrap_auth(xet_parse)).start()
    patch_submodule(module, "xml.dom.minidom.parse", wrap_auth(xxml_dom_minidom_parse)).start()
    if not module.__name__.startswith("datasets.packaged_modules."):
        patch_submodule(module, "pyarrow.parquet.read_table", wrap_auth(xpyarrow_parquet_read_table)).start()
    module._patched_for_streaming = download_config

def lock_importable_file(importable_local_file: str) -> FileLock:
    importable_directory_path = str(Path(importable_local_file).resolve().parent.parent)
    lock_path = importable_directory_path + ".lock"
    return FileLock(lock_path)

def _convert_github_url(url_path: str) -> tuple[str, Optional[str]]:
    parsed = urlparse(url_path)
    sub_directory = None
    if parsed.scheme in ("http", "https", "s3") and parsed.netloc == "github.com":
        if "blob" in url_path:
            if not url_path.endswith(".py"):
                raise ValueError(f"External import from github at {url_path} should point to a file ending with '.py'")
            url_path = url_path.replace("blob", "raw")  # Point to the raw file
        else:
            github_path = parsed.path[1:]
            repo_info, branch = github_path.split("/tree/") if "/tree/" in github_path else (github_path, "master")
            repo_owner, repo_name = repo_info.split("/")
            url_path = f"https://github.com/{repo_owner}/{repo_name}/archive/{branch}.zip"
            sub_directory = f"{repo_name}-{branch}"
    return url_path, sub_directory

def get_imports(file_path: str) -> tuple[str, str, str, str]:
    lines = []
    with open(file_path, encoding="utf-8") as f:
        lines.extend(f.readlines())

    imports: list[tuple[str, str, str, Optional[str]]] = []
    is_in_docstring = False
    for line in lines:
        docstr_start_match = re.findall(r'[\s\S]*?"""[\s\S]*?', line)

        if len(docstr_start_match) == 1:
            is_in_docstring = not is_in_docstring

        if is_in_docstring:
            continue

        match = re.match(r"^import\s+(\.?)([^\s\.]+)[^#\r\n]*(?:#\s+From:\s+)?([^\r\n]*)", line, flags=re.MULTILINE)
        if match is None:
            match = re.match(r"^from\s+(\.?)([^\s\.]+)(?:[^\s]*)\s+import\s+[^#\r\n]*(?:#\s+From:\s+)?([^\r\n]*)", line, flags=re.MULTILINE)
            if match is None:
                continue
        if match.group(1):
            if any(imp[1] == match.group(2) for imp in imports):
                continue
            if match.group(3):
                url_path = match.group(3)
                url_path, sub_directory = _convert_github_url(url_path)
                imports.append(("external", match.group(2), url_path, sub_directory))
            elif match.group(2):
                imports.append(("internal", match.group(2), match.group(2), None))
        else:
            if match.group(3):
                url_path = match.group(3)
                imports.append(("library", match.group(2), url_path, None))
            else:
                imports.append(("library", match.group(2), match.group(2), None))

    return imports

class StreamingDownloadManager:
    is_streaming = True

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        download_config: Optional[DownloadConfig] = None,
        base_path: Optional[str] = None,
    ):
        self._dataset_name = dataset_name
        self._data_dir = data_dir
        self._base_path = base_path or os.path.abspath(".")
        self.download_config = download_config or DownloadConfig()
        self.downloaded_size = None
        self.record_checksums = False

    @property
    def manual_dir(self):
        return self._data_dir

    def download(self, url_or_urls):
        url_or_urls = map_nested(self._download_single, url_or_urls, map_tuple=True)
        return url_or_urls

    def _download_single(self, urlpath: str) -> str:
        urlpath = str(urlpath)
        if is_relative_path(urlpath):
            urlpath = url_or_path_join(self._base_path, urlpath)
        return urlpath

    def extract(self, url_or_urls):
        urlpaths = map_nested(self._extract, url_or_urls, map_tuple=True)
        return urlpaths

    def _extract(self, urlpath: str) -> str:
        urlpath = str(urlpath)
        protocol = _get_extraction_protocol(urlpath, download_config=self.download_config)
        path = urlpath.split("::")[0]
        extension = _get_path_extension(path)
        if extension in ["tgz", "tar"] or path.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
            raise NotImplementedError(
                f"Extraction protocol for TAR archives like '{urlpath}' is not implemented in streaming mode. "
                f"Please use `dl_manager.iter_archive` instead.\n\n Example usage:\n\n"
                f"\turl = dl_manager.download(url)\n"
                f"\ttar_archive_iterator = dl_manager.iter_archive(url)\n\n"
                f"\tfor filename, file in tar_archive_iterator:\n"
                f"\t\t..."
            )
        if protocol is None:
            return urlpath
        elif protocol in SINGLE_FILE_COMPRESSION_PROTOCOLS:
            inner_file = os.path.basename(urlpath.split("::")[0])
            inner_file = inner_file[: inner_file.rindex(".")] if "." in inner_file else inner_file
            return f"{protocol}://{inner_file}::{urlpath}"
        else:
            return f"{protocol}://::{urlpath}"

    def download_and_extract(self, url_or_urls):
        return self.extract(self.download(url_or_urls))

    def iter_archive(self, urlpath_or_buf: Union[str, io.BufferedReader]) -> Iterable[tuple]:
        if hasattr(urlpath_or_buf, "read"):
            return ArchiveIterable.from_buf(urlpath_or_buf)
        else:
            return ArchiveIterable.from_urlpath(urlpath_or_buf, download_config=self.download_config)

    def iter_files(self, urlpaths: Union[str, list[str]]) -> Iterable[str]:
        return FilesIterable.from_urlpaths(urlpaths, download_config=self.download_config)

    def manage_extracted_files(self):
        pass

    def get_recorded_sizes_checksums(self):
        pass

@dataclass
class SplitGenerator:
    name: str
    gen_kwargs: dict = dataclasses.field(default_factory=dict)
    split_info: SplitInfo = dataclasses.field(init=False)

    def __post_init__(self):
        self.name = str(self.name)  # Make sure we convert NamedSplits in strings
        NamedSplit(self.name)  # check that it's a valid split name
        self.split_info = SplitInfo(name=self.name)

class bind(partial):
    def __call__(self, *fn_args, **fn_kwargs):
        return self.func(*fn_args, *self.args, **fn_kwargs)

class IterableDatasetDict(dict):
    def __repr__(self):
        repr = "\n".join([f"{k}: {v}" for k, v in self.items()])
        repr = re.sub(r"^", " " * 4, repr, 0, re.M)
        return f"IterableDatasetDict({{\n{repr}\n}})"

    def with_format(self, type: Optional[str] = None) -> "IterableDatasetDict":
        return IterableDatasetDict({k: dataset.with_format(type=type) for k, dataset in self.items()})

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_split: bool = False,
        input_columns: Optional[Union[str, list[str]]] = None,
        batched: bool = False,
        batch_size: int = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, list[str]]] = None,
        fn_kwargs: Optional[dict] = None,
    ) -> "IterableDatasetDict":
        
        dataset_dict = {}
        for split, dataset in self.items():
            if with_split:
                function = bind(function, split)

            dataset_dict[split] = dataset.map(
                function=function,
                with_indices=with_indices,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                remove_columns=remove_columns,
                fn_kwargs=fn_kwargs,
            )

            if with_split:
                function = function.func

        return IterableDatasetDict(dataset_dict)

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, list[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
    ) -> "IterableDatasetDict":
        return IterableDatasetDict(
            {
                k: dataset.filter(
                    function=function,
                    with_indices=with_indices,
                    input_columns=input_columns,
                    batched=batched,
                    batch_size=batch_size,
                    fn_kwargs=fn_kwargs,
                )
                for k, dataset in self.items()
            }
        )

    def shuffle(
        self,
        seed=None,
        generator: Optional[np.random.Generator] = None,
        buffer_size: int = 1000,
    ) -> "IterableDatasetDict":
        return IterableDatasetDict(
            {
                k: dataset.shuffle(seed=seed, generator=generator, buffer_size=buffer_size)
                for k, dataset in self.items()
            }
        )

    def rename_column(self, original_column_name: str, new_column_name: str) -> "IterableDatasetDict":
        return IterableDatasetDict(
            {
                k: dataset.rename_column(
                    original_column_name=original_column_name,
                    new_column_name=new_column_name,
                )
                for k, dataset in self.items()
            }
        )

    def rename_columns(self, column_mapping: dict[str, str]) -> "IterableDatasetDict":
        return IterableDatasetDict({k: dataset.rename_columns(column_mapping=column_mapping) for k, dataset in self.items()})

    def remove_columns(self, column_names: Union[str, list[str]]) -> "IterableDatasetDict":
        return IterableDatasetDict({k: dataset.remove_columns(column_names) for k, dataset in self.items()})

    def select_columns(self, column_names: Union[str, list[str]]) -> "IterableDatasetDict":

        return IterableDatasetDict({k: dataset.select_columns(column_names) for k, dataset in self.items()})

    def cast_column(self, column: str, feature: FeatureType) -> "IterableDatasetDict":
        return IterableDatasetDict({k: dataset.cast_column(column=column, feature=feature) for k, dataset in self.items()})

    def cast(self, features: Features) -> "IterableDatasetDict":
        return IterableDatasetDict({k: dataset.cast(features=features) for k, dataset in self.items()})

class tracked_list(list):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_item = None

    def __iter__(self) -> Iterator:
        for x in super().__iter__():
            self.last_item = x
            yield x
        self.last_item = None

    def __repr__(self) -> str:
        if self.last_item is None:
            return super().__repr__()
        else:
            return f"{self.__class__.__name__}(current={self.last_item})"

class DatasetGenerationCastError(DatasetGenerationError):
    @classmethod
    def from_cast_error(
        cls,
        cast_error: CastError,
        builder_name: str,
        gen_kwargs: dict[str, Any],
        token: Optional[Union[bool, str]],
    ) -> "DatasetGenerationCastError":
        explanation_message = (f"\n\nAll the data files must have the same columns, but at some point {cast_error.details()}")
        formatted_tracked_gen_kwargs: list[str] = []
        for gen_kwarg in gen_kwargs.values():
            if not isinstance(gen_kwarg, (tracked_str, tracked_list, TrackedIterableFromGenerator)):
                continue
            while (isinstance(gen_kwarg, (tracked_list, TrackedIterableFromGenerator)) and gen_kwarg.last_item is not None):
                gen_kwarg = gen_kwarg.last_item
            if isinstance(gen_kwarg, tracked_str):
                gen_kwarg = gen_kwarg.get_origin()
            if isinstance(gen_kwarg, str) and gen_kwarg.startswith("hf://"):
                resolved_path = HfFileSystem(endpoint=data_config.HF_ENDPOINT, token=token).resolve_path(gen_kwarg)
                gen_kwarg = "hf://" + resolved_path.unresolve()
                if "@" + resolved_path.revision in gen_kwarg:
                    gen_kwarg = (gen_kwarg.replace("@" + resolved_path.revision, "", 1) + f" (at revision {resolved_path.revision})")
            formatted_tracked_gen_kwargs.append(str(gen_kwarg))
        if formatted_tracked_gen_kwargs:
            explanation_message += f"\n\nThis happened while the {builder_name} dataset builder was generating data using\n\n{', '.join(formatted_tracked_gen_kwargs)}"
        help_message = "\n\nPlease either edit the data files to have matching columns, or separate them into different configurations (see docs at https://hf.co/docs/hub/datasets-manual-configuration#multiple-configurations)"
        return cls("An error occurred while generating the dataset" + explanation_message + help_message)

def count_path_segments(path):
    return path.replace("\\", "/").count("/")

class FileNotFoundDatasetsError(DatasetsError, FileNotFoundError):
    """FileNotFoundError raised by this library."""

class DatasetNotFoundError(FileNotFoundDatasetsError):
    """Dataset not found.

    Raised when trying to access:
    - a missing dataset, or
    - a private/gated dataset and the user is not authenticated.
    """

class DataFilesNotFoundError(FileNotFoundDatasetsError):
    """No (supported) data files found."""

class DatasetViewerError(DatasetsError):
    """Dataset viewer error.

    Raised when trying to use the dataset viewer HTTP API and when trying to access:
    - a missing dataset, or
    - a private/gated dataset and the user is not authenticated.
    - unavailable /parquet or /info responses
    """
