import abc
import asyncio
import copy
import collections
import contextlib
import dataclasses
import filecmp
import fsspec
import functools
import glob
import os
import io
import inspect
import importlib
import itertools
import json
import posixpath
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.json as paj
import random
import re
import shutil
import signal
import sys
import tempfile
import textwrap
import time
import typing
import urllib
import weakref
import yaml
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence, Iterable, Mapping, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial, wraps
from itertools import islice, groupby
from operator import itemgetter
from typing import Optional, Union, Callable, TypeVar, Literal, Any, overload, BinaryIO, ClassVar
from pathlib import Path
from math import floor, ceil
from unittest.mock import patch

import pandas as pd
import numpy as np
import torch
import torch.utils.data
import multiprocessing
import huggingface_hub
from huggingface_hub import DatasetCard, DatasetCardData, get_session, hf_hub_url, HfApi
from huggingface_hub.utils import insecure_hashlib

from data_utils import (
    Table, 
    NamedSplit,
    Features,
    FeatureType,
    InMemoryTable,
    Split,
    hf_tqdm,
    tracked_list,
    MemoryMappedTable,
    KeyHasher,
    DuplicatedKeysError,
    OptimizedTypedSequence,
    SchemaInferenceError,
    Hasher,
    LazyDict,
    Value,
    NonExistentDatasetError,
    DatasetTransformationNotAllowedError,
    Image,
    _BaseExamplesIterable,
    TensorFormatter,
    FormattingConfig,
    FormattedExamplesIterable,
    RebatchedArrowExamplesIterable,
    StepExamplesIterable,
    ArrowExamplesIterable,
    MappedExamplesIterable,
    FilteredExamplesIterable,
    BufferShuffledExamplesIterable,
    SkipExamplesIterable,
    RepeatExamplesIterable,
    TakeExamplesIterable,
    SelectColumnsIterable,
    ExamplesIterable,
    VerificationMode,
    DownloadConfig,
    ReadInstruction,
    FileFormatError,
    FileLock,
    ManualDownloadError,
    SplitDict,
    PostProcessedInfo,
    DownloadMode,
    DownloadManager,
    Version,
    Pickler,
    InvalidConfigName,
    DataFilesDict,
    DataFilesPatternsDict,
    StreamingDownloadManager,
    SplitGenerator,
    SplitInfo,
    IterableDatasetDict,
    DatasetGenerationError,
    CastError,
    DatasetGenerationCastError,
    SupervisedKeysData,
    DataFilesList,
    DataFilesNotFoundError,
    EntryNotFoundError,
    DatasetViewerError,
)

from data_utils import (
    __version__,
    url_to_fs,
    table_cast,
    asdict,
    xgetsize,
    xjoin,
    xglob,
    xbasename,
    string_to_dict,
    is_remote_filesystem,
    thread_map,
    Pool,
    concat_tables,
    get_writer_batch_size,
    generate_from_arrow_type,
    cast_array_to_feature,
    embed_table_storage,
    convert_file_size_to_int,
    iflatmap_unordered,
    is_small_dataset,
    estimate_dataset_size,
    list_table_cache_files,
    get_formatter,
    get_format_type_from_alias,
    table_iter,
    format_table,
    query_table,
    _is_range_contiguous,
    _check_if_features_can_be_aligned,
    _align_features,
    get_indices_from_mask_function,
    async_get_indices_from_mask_function,
    _check_valid_indices_value,
    stratified_shuffle_split_generate_indices,
    require_decoding,
    pandas_types_mapper,
    _memory_mapped_record_batch_reader_from_file,
    table_visitor,
    _visit,
    _maybe_add_torch_iterable_dataset_parent_class,
    _maybe_share_with_torch_persistent_workers,
    _convert_to_arrow,
    identity_func,
    _examples_to_batch,
    read_schema_from_file,
    add_column_fn,
    _rename_columns_fn,
    _apply_async,
    _infer_features_from_batch,
    _split_gen_kwargs,
    _number_of_shards_in_gen_kwargs,
    map_nested,
    verify_checksums,
    get_size_checksum_dict,
    verify_splits,
    temporary_assignment,
    size_str,
    has_sufficient_disk_space,
    rename,
    is_remote_url,
    camelcase_to_snakecase,
    sanitize_patterns,
    extend_module_for_streaming,
    get_imports,
    lock_importable_file,
    _split_re,
    require_storage_cast,
    count_path_segments,
    get_datasets_user_agent,
    resolve_pattern,
    cached_path,
    url_or_path_join,
    unique_values,
    make_file_instructions,
)
import xml
import pandas
import csv
from files import arrow
from files import audiofolder
from files import imagefolder
from files import parquet
from files import pdffolder
from files import text
from files import webdataset

import data_config

if typing.TYPE_CHECKING:
    from data import ImageFolder

def _hash_python_lines(lines: list[str]) -> str:
    filtered_lines = []
    for line in lines:
        line = re.sub(r"#.*", "", line)  # remove comments
        if line:
            filtered_lines.append(line)
    full_str = "\n".join(filtered_lines)

    full_bytes = full_str.encode("utf-8")
    return insecure_hashlib.sha256(full_bytes).hexdigest()

hf_dataset_url = partial(hf_hub_url, repo_type="dataset")
fingerprint_warnings: dict[str, bool] = {}
SPLIT_PATTERN_SHARDED = "data/{split}-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*"
ALL_SPLIT_PATTERNS = [SPLIT_PATTERN_SHARDED]
DEFAULT_SPLITS = [Split.TRAIN, Split.VALIDATION, Split.TEST]
NON_WORDS_CHARS = "-._ 0-9"
KEYWORDS_IN_FILENAME_BASE_PATTERNS = ["**/{keyword}[{sep}]*", "**/*[{sep}]{keyword}[{sep}]*"]
KEYWORDS_IN_DIR_NAME_BASE_PATTERNS = [
        "**/{keyword}/**",
        "**/{keyword}[{sep}]*/**",
        "**/*[{sep}]{keyword}/**",
        "**/*[{sep}]{keyword}[{sep}]*/**",
]

SPLIT_KEYWORDS = {
    Split.TRAIN: ["train", "training"],
    Split.VALIDATION: ["validation", "valid", "dev", "val"],
    Split.TEST: ["test", "testing", "eval", "evaluation"],
}
DEFAULT_PATTERNS_ALL = {Split.TRAIN: ["**"]}
DEFAULT_PATTERNS_SPLIT_IN_FILENAME = {
    split: [
        pattern.format(keyword=keyword, sep=NON_WORDS_CHARS)
        for keyword in SPLIT_KEYWORDS[split]
        for pattern in KEYWORDS_IN_FILENAME_BASE_PATTERNS
    ]
    for split in DEFAULT_SPLITS
}
DEFAULT_PATTERNS_SPLIT_IN_DIR_NAME = {
    split: [
        pattern.format(keyword=keyword, sep=NON_WORDS_CHARS)
        for keyword in SPLIT_KEYWORDS[split]
        for pattern in KEYWORDS_IN_DIR_NAME_BASE_PATTERNS
    ]
    for split in DEFAULT_SPLITS
}
ALL_DEFAULT_PATTERNS = [
    DEFAULT_PATTERNS_SPLIT_IN_DIR_NAME,
    DEFAULT_PATTERNS_SPLIT_IN_FILENAME,
    DEFAULT_PATTERNS_ALL,
]

PathLike = Union[str, bytes, os.PathLike]
T = TypeVar("T")
_VisitPath = list[Union[str, Literal[0]]]
_CACHING_ENABLED = True
ListLike = Union[list[T], tuple[T, ...]]
NestedDataStructureLike = Union[T, list[T], dict[str, T]]
INVALID_WINDOWS_CHARACTERS_IN_PATH = r"<>:/\|?*"
_TEMP_DIR_FOR_TEMP_CACHE_FILES: Optional["_TempCacheDir"] = None
fingerprint_rng = random.Random()
memoize = functools.lru_cache
_PACKAGED_DATASETS_MODULES = {
    "csv": (csv.__name__, _hash_python_lines(inspect.getsource(csv).splitlines())),
    "json": (json.__name__, _hash_python_lines(inspect.getsource(json).splitlines())),
    "pandas": (pandas.__name__, _hash_python_lines(inspect.getsource(pandas).splitlines())),
    "parquet": (parquet.__name__, _hash_python_lines(inspect.getsource(parquet).splitlines())),
    "arrow": (arrow.__name__, _hash_python_lines(inspect.getsource(arrow).splitlines())),
    "text": (text.__name__, _hash_python_lines(inspect.getsource(text).splitlines())),
    "imagefolder": (imagefolder.__name__, _hash_python_lines(inspect.getsource(imagefolder).splitlines())),
    "audiofolder": (audiofolder.__name__, _hash_python_lines(inspect.getsource(audiofolder).splitlines())),
    "pdffolder": (pdffolder.__name__, _hash_python_lines(inspect.getsource(pdffolder).splitlines())),
    "webdataset": (webdataset.__name__, _hash_python_lines(inspect.getsource(webdataset).splitlines())),
    "xml": (xml.__name__, _hash_python_lines(inspect.getsource(xml).splitlines())),
}

_PACKAGED_DATASETS_MODULES_2_15_HASHES = {
    "csv": "eea64c71ca8b46dd3f537ed218fc9bf495d5707789152eb2764f5c78fa66d59d",
    "json": "8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96",
    "pandas": "3ac4ffc4563c796122ef66899b9485a3f1a977553e2d2a8a318c72b8cc6f2202",
    "parquet": "ca31c69184d9832faed373922c2acccec0b13a0bb5bbbe19371385c3ff26f1d1",
    "arrow": "74f69db2c14c2860059d39860b1f400a03d11bf7fb5a8258ca38c501c878c137",
    "text": "c4a140d10f020282918b5dd1b8a49f0104729c6177f60a6b49ec2a365ec69f34",
    "imagefolder": "7b7ce5247a942be131d49ad4f3de5866083399a0f250901bd8dc202f8c5f7ce5",
    "audiofolder": "d3c1655c66c8f72e4efb5c79e952975fa6e2ce538473a6890241ddbddee9071c",
}

_EXTENSION_TO_MODULE: dict[str, tuple[str, dict]] = {
    ".csv": ("csv", {}),
    ".tsv": ("csv", {"sep": "\t"}),
    ".json": ("json", {}),
    ".jsonl": ("json", {}),
    ".ndjson": ("json", {}),
    ".parquet": ("parquet", {}),
    ".geoparquet": ("parquet", {}),
    ".gpq": ("parquet", {}),
    ".arrow": ("arrow", {}),
    ".txt": ("text", {}),
    ".tar": ("webdataset", {}),
    ".xml": ("xml", {}),
}
_EXTENSION_TO_MODULE.update({ext: ("imagefolder", {}) for ext in ImageFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext.upper(): ("imagefolder", {}) for ext in ImageFolder.EXTENSIONS})
ALL_ALLOWED_EXTENSIONS = list(_EXTENSION_TO_MODULE.keys()) + [".zip"]

_MODULE_TO_EXTENSIONS: dict[str, list[str]] = {}
for _ext, (_module, _) in _EXTENSION_TO_MODULE.items():
    _MODULE_TO_EXTENSIONS.setdefault(_module, []).append(_ext)

for _module in _MODULE_TO_EXTENSIONS:
    _MODULE_TO_EXTENSIONS[_module].append(".zip")

_MODULE_TO_METADATA_FILE_NAMES: dict[str, list[str]] = {}
for _module in _MODULE_TO_EXTENSIONS:
    _MODULE_TO_METADATA_FILE_NAMES[_module] = []
_MODULE_TO_METADATA_FILE_NAMES["imagefolder"] = imagefolder.ImageFolder.METADATA_FILENAMES
_MODULE_TO_METADATA_FILE_NAMES["audiofolder"] = imagefolder.ImageFolder.METADATA_FILENAMES
_MODULE_TO_METADATA_FILE_NAMES["videofolder"] = imagefolder.ImageFolder.METADATA_FILENAMES
_MODULE_TO_METADATA_FILE_NAMES["pdffolder"] = imagefolder.ImageFolder.METADATA_FILENAMES

@dataclass
class DatasetInfo:
    description: str = dataclasses.field(default_factory=str)
    citation: str = dataclasses.field(default_factory=str)
    homepage: str = dataclasses.field(default_factory=str)
    license: str = dataclasses.field(default_factory=str)
    features: Optional[Features] = None
    post_processed: Optional[PostProcessedInfo] = None
    supervised_keys: Optional[SupervisedKeysData] = None

    builder_name: Optional[str] = None
    dataset_name: Optional[str] = None  # for packaged builders, to be different from builder_name
    config_name: Optional[str] = None
    version: Optional[Union[str, Version]] = None
    splits: Optional[dict] = None
    download_checksums: Optional[dict] = None
    download_size: Optional[int] = None
    post_processing_size: Optional[int] = None
    dataset_size: Optional[int] = None
    size_in_bytes: Optional[int] = None

    _INCLUDED_INFO_IN_YAML: ClassVar[list[str]] = [
        "config_name",
        "download_size",
        "dataset_size",
        "features",
        "splits",
    ]

    def __post_init__(self):
        if self.features is not None and not isinstance(self.features, Features):
            self.features = Features.from_dict(self.features)
        if self.post_processed is not None and not isinstance(self.post_processed, PostProcessedInfo):
            self.post_processed = PostProcessedInfo.from_dict(self.post_processed)
        if self.version is not None and not isinstance(self.version, Version):
            if isinstance(self.version, str):
                self.version = Version(self.version)
            else:
                self.version = Version.from_dict(self.version)
        if self.splits is not None and not isinstance(self.splits, SplitDict):
            self.splits = SplitDict.from_split_dict(self.splits)
        if self.supervised_keys is not None and not isinstance(self.supervised_keys, SupervisedKeysData):
            if isinstance(self.supervised_keys, (tuple, list)):
                self.supervised_keys = SupervisedKeysData(*self.supervised_keys)
            else:
                self.supervised_keys = SupervisedKeysData(**self.supervised_keys)

    def write_to_directory(self, dataset_info_dir, pretty_print=False, storage_options: Optional[dict] = None):
        fs: fsspec.AbstractFileSystem
        fs, *_ = url_to_fs(dataset_info_dir, **(storage_options or {}))
        with fs.open(posixpath.join(dataset_info_dir, data_config.DATASET_INFO_FILENAME), "wb") as f:
            self._dump_info(f, pretty_print=pretty_print)
        if self.license:
            with fs.open(posixpath.join(dataset_info_dir, data_config.LICENSE_FILENAME), "wb") as f:
                self._dump_license(f)

    def _dump_info(self, file, pretty_print=False):
        file.write(json.dumps(asdict(self), indent=4 if pretty_print else None).encode("utf-8"))

    def _dump_license(self, file):
        file.write(self.license.encode("utf-8"))

    @classmethod
    def from_merge(cls, dataset_infos: list["DatasetInfo"]):
        dataset_infos = [dset_info.copy() for dset_info in dataset_infos if dset_info is not None]

        if len(dataset_infos) > 0 and all(dataset_infos[0] == dset_info for dset_info in dataset_infos):
            return dataset_infos[0]

        description = "\n\n".join(unique_values(info.description for info in dataset_infos)).strip()
        citation = "\n\n".join(unique_values(info.citation for info in dataset_infos)).strip()
        homepage = "\n\n".join(unique_values(info.homepage for info in dataset_infos)).strip()
        license = "\n\n".join(unique_values(info.license for info in dataset_infos)).strip()
        features = None
        supervised_keys = None

        return cls(
            description=description,
            citation=citation,
            homepage=homepage,
            license=license,
            features=features,
            supervised_keys=supervised_keys,
        )

    @classmethod
    def from_directory(cls, dataset_info_dir: str, storage_options: Optional[dict] = None) -> "DatasetInfo":
        fs: fsspec.AbstractFileSystem
        fs, *_ = url_to_fs(dataset_info_dir, **(storage_options or {}))
        if not dataset_info_dir:
            raise ValueError("Calling DatasetInfo.from_directory() with undefined dataset_info_dir.")
        with fs.open(posixpath.join(dataset_info_dir, data_config.DATASET_INFO_FILENAME), "r", encoding="utf-8") as f:
            dataset_info_dict = json.load(f)
        return cls.from_dict(dataset_info_dict)

    @classmethod
    def from_dict(cls, dataset_info_dict: dict) -> "DatasetInfo":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dataset_info_dict.items() if k in field_names})

    def update(self, other_dataset_info: "DatasetInfo", ignore_none=True):
        self_dict = self.__dict__
        self_dict.update(
            **{
                k: copy.deepcopy(v)
                for k, v in other_dataset_info.__dict__.items()
                if (v is not None or not ignore_none)
            }
        )

    def copy(self) -> "DatasetInfo":
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})

    def _to_yaml_dict(self) -> dict:
        yaml_dict = {}
        dataset_info_dict = asdict(self)
        for key in dataset_info_dict:
            if key in self._INCLUDED_INFO_IN_YAML:
                value = getattr(self, key)
                if hasattr(value, "_to_yaml_list"):  # Features, SplitDict
                    yaml_dict[key] = value._to_yaml_list()
                elif hasattr(value, "_to_yaml_string"):  # Version
                    yaml_dict[key] = value._to_yaml_string()
                else:
                    yaml_dict[key] = value
        return yaml_dict

    @classmethod
    def _from_yaml_dict(cls, yaml_data: dict) -> "DatasetInfo":
        yaml_data = copy.deepcopy(yaml_data)
        if yaml_data.get("features") is not None:
            yaml_data["features"] = Features._from_yaml_list(yaml_data["features"])
        if yaml_data.get("splits") is not None:
            yaml_data["splits"] = SplitDict._from_yaml_list(yaml_data["splits"])
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in yaml_data.items() if k in field_names})

class BaseReader:
    def __init__(self, path: str, info: Optional["DatasetInfo"]):
        self._path: str = path
        self._info: Optional["DatasetInfo"] = info
        self._filetype_suffix: Optional[str] = None

    def _get_table_from_filename(self, filename_skip_take, in_memory=False) -> Table:
        raise NotImplementedError

    def _read_files(self, files, in_memory=False) -> Table:
        if len(files) == 0 or not all(isinstance(f, dict) for f in files):
            raise ValueError("please provide valid file informations")
        files = copy.deepcopy(files)
        for f in files:
            f["filename"] = os.path.join(self._path, f["filename"])

        pa_tables = thread_map(
            partial(self._get_table_from_filename, in_memory=in_memory),
            files,
            tqdm_class=hf_tqdm,
            desc="Loading dataset shards",
            disable=len(files) <= 16 or None,
        )
        pa_tables = [t for t in pa_tables if len(t) > 0]
        if not pa_tables and (self._info is None or self._info.features is None):
            raise ValueError("Tried to read an empty table. Please specify at least info.features to create an empty table with the right type.")
        pa_tables = pa_tables or [InMemoryTable.from_batches([], schema=pa.schema(self._info.features.type))]
        pa_table = concat_tables(pa_tables) if len(pa_tables) != 1 else pa_tables[0]
        return pa_table

    def get_file_instructions(self, name, instruction, split_infos):
        file_instructions = make_file_instructions(name, split_infos, instruction, filetype_suffix=self._filetype_suffix, prefix_path=self._path)
        files = file_instructions.file_instructions
        return files

    def read(
        self,
        name,
        instructions,
        split_infos,
        in_memory=False,
    ):
        files = self.get_file_instructions(name, instructions, split_infos)
        if not files:
            msg = f'Instruction "{instructions}" corresponds to no data!'
            raise ValueError(msg)
        return self.read_files(files=files, original_instructions=instructions, in_memory=in_memory)

    def read_files(
        self,
        files: list[dict],
        original_instructions: Union[None, ReadInstruction, Split] = None,
        in_memory=False,
    ):
        pa_table = self._read_files(files, in_memory=in_memory)
        if original_instructions is not None:
            split = Split(str(original_instructions))
        else:
            split = None
        dataset_kwargs = {"arrow_table": pa_table, "info": self._info, "split": split}
        return dataset_kwargs

class ArrowReader(BaseReader):
    def __init__(self, path: str, info: Optional["DatasetInfo"]):
        super().__init__(path, info)
        self._filetype_suffix = "arrow"

    def _get_table_from_filename(self, filename_skip_take, in_memory=False) -> Table:
        filename, skip, take = (
            filename_skip_take["filename"],
            filename_skip_take["skip"] if "skip" in filename_skip_take else None,
            filename_skip_take["take"] if "take" in filename_skip_take else None,
        )
        table = ArrowReader.read_table(filename, in_memory=in_memory)
        if take == -1:
            take = len(table) - skip
        if skip is not None and take is not None and not (skip == 0 and take == len(table)):
            table = table.slice(skip, take)
        return table

    @staticmethod
    def read_table(filename, in_memory=False) -> Table:
        table_cls = InMemoryTable if in_memory else MemoryMappedTable
        return table_cls.from_file(filename)

class ArrowWriter:
    _WRITER_CLASS = pa.RecordBatchStreamWriter

    def __init__(
        self,
        schema: Optional[pa.Schema] = None,
        features: Optional[Features] = None,
        path: Optional[str] = None,
        stream: Optional[pa.NativeFile] = None,
        fingerprint: Optional[str] = None,
        writer_batch_size: Optional[int] = None,
        hash_salt: Optional[str] = None,
        check_duplicates: Optional[bool] = False,
        disable_nullable: bool = False,
        update_features: bool = False,
        with_metadata: bool = True,
        unit: str = "examples",
        embed_local_files: bool = False,
        storage_options: Optional[dict] = None,
    ):
        if path is None and stream is None:
            raise ValueError("At least one of path and stream must be provided.")
        if features is not None:
            self._features = features
            self._schema = None
        elif schema is not None:
            self._schema: pa.Schema = schema
            self._features = Features.from_arrow_schema(self._schema)
        else:
            self._features = None
            self._schema = None

        if hash_salt is not None:
            self._hasher = KeyHasher(hash_salt)
        else:
            self._hasher = KeyHasher("")

        self._check_duplicates = check_duplicates
        self._disable_nullable = disable_nullable

        if stream is None:
            fs, path = url_to_fs(path, **(storage_options or {}))
            self._fs: fsspec.AbstractFileSystem = fs
            self._path = path if not is_remote_filesystem(self._fs) else self._fs.unstrip_protocol(path)
            self.stream = self._fs.open(path, "wb")
            self._closable_stream = True
        else:
            self._fs = None
            self._path = None
            self.stream = stream
            self._closable_stream = False

        self.fingerprint = fingerprint
        self.disable_nullable = disable_nullable
        self.writer_batch_size = (writer_batch_size or get_writer_batch_size(self._features) or data_config.DEFAULT_MAX_BATCH_SIZE)
        self.update_features = update_features
        self.with_metadata = with_metadata
        self.unit = unit
        self.embed_local_files = embed_local_files

        self._num_examples = 0
        self._num_bytes = 0
        self.current_examples: list[tuple[dict[str, Any], str]] = []
        self.current_rows: list[pa.Table] = []
        self.pa_writer: Optional[pa.RecordBatchStreamWriter] = None
        self.hkey_record = []

    def __len__(self):
        return self._num_examples + len(self.current_examples) + len(self.current_rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.pa_writer:  # it might be None
            try:
                self.pa_writer.close()
            except Exception:  # pyarrow.lib.ArrowInvalid, OSError
                pass
        if self._closable_stream and not self.stream.closed:
            self.stream.close()  # This also closes self.pa_writer if it is opened

    def _build_writer(self, inferred_schema: pa.Schema):
        schema = self.schema
        inferred_features = Features.from_arrow_schema(inferred_schema)
        if self._features is not None:
            if self.update_features:  # keep original features it they match, or update them
                fields = {field.name: field for field in self._features.type}
                for inferred_field in inferred_features.type:
                    name = inferred_field.name
                    if name in fields:
                        if inferred_field == fields[name]:
                            inferred_features[name] = self._features[name]
                self._features = inferred_features
                schema: pa.Schema = inferred_schema
        else:
            self._features = inferred_features
            schema: pa.Schema = inferred_features.arrow_schema
        if self.disable_nullable:
            schema = pa.schema(pa.field(field.name, field.type, nullable=False) for field in schema)
        if self.with_metadata:
            schema = schema.with_metadata(self._build_metadata(DatasetInfo(features=self._features), self.fingerprint))
        else:
            schema = schema.with_metadata({})
        self._schema = schema
        self.pa_writer = self._WRITER_CLASS(self.stream, schema)

    @property
    def schema(self):
        _schema = (self._schema if self._schema is not None else (pa.schema(self._features.type) if self._features is not None else None))
        if self._disable_nullable and _schema is not None:
            _schema = pa.schema(pa.field(field.name, field.type, nullable=False) for field in _schema)
        return _schema if _schema is not None else []

    @staticmethod
    def _build_metadata(info: DatasetInfo, fingerprint: Optional[str] = None) -> dict[str, str]:
        info_keys = ["features"]  # we can add support for more DatasetInfo keys in the future
        info_as_dict = asdict(info)
        metadata = {}
        metadata["info"] = {key: info_as_dict[key] for key in info_keys}
        if fingerprint is not None:
            metadata["fingerprint"] = fingerprint
        return {"huggingface": json.dumps(metadata)}

    def write_examples_on_file(self):
        if not self.current_examples:
            return
        if self.schema:
            schema_cols = set(self.schema.names)
            examples_cols = self.current_examples[0][0].keys()  # .keys() preserves the order (unlike set)
            common_cols = [col for col in self.schema.names if col in examples_cols]
            extra_cols = [col for col in examples_cols if col not in schema_cols]
            cols = common_cols + extra_cols
        else:
            cols = list(self.current_examples[0][0])
        batch_examples = {}
        for col in cols:
            if all(isinstance(row[0][col], (pa.Array, pa.ChunkedArray)) for row in self.current_examples):
                arrays = [row[0][col] for row in self.current_examples]
                arrays = [chunk for array in arrays for chunk in (array.chunks if isinstance(array, pa.ChunkedArray) else [array])]
                batch_examples[col] = pa.concat_arrays(arrays)
            else:
                batch_examples[col] = [row[0][col].to_pylist()[0] if isinstance(row[0][col], (pa.Array, pa.ChunkedArray)) else row[0][col] for row in self.current_examples]
        self.write_batch(batch_examples=batch_examples)
        self.current_examples = []

    def write_rows_on_file(self):
        if not self.current_rows:
            return
        table = pa.concat_tables(self.current_rows)
        self.write_table(table)
        self.current_rows = []

    def write(
        self,
        example: dict[str, Any],
        key: Optional[Union[str, int, bytes]] = None,
        writer_batch_size: Optional[int] = None,
    ):
        if self._check_duplicates:
            hash = self._hasher.hash(key)
            self.current_examples.append((example, hash))
            self.hkey_record.append((hash, key))
        else:
            self.current_examples.append((example, ""))

        if writer_batch_size is None:
            writer_batch_size = self.writer_batch_size
        if writer_batch_size is not None and len(self.current_examples) >= writer_batch_size:
            if self._check_duplicates:
                self.check_duplicate_keys()
                self.hkey_record = []

            self.write_examples_on_file()

    def check_duplicate_keys(self):
        tmp_record = set()
        for hash, key in self.hkey_record:
            if hash in tmp_record:
                duplicate_key_indices = [str(self._num_examples + index) for index, (duplicate_hash, _) in enumerate(self.hkey_record) if duplicate_hash == hash]
                raise DuplicatedKeysError(key, duplicate_key_indices)
            else:
                tmp_record.add(hash)

    def write_row(self, row: pa.Table, writer_batch_size: Optional[int] = None):
        if len(row) != 1:
            raise ValueError(f"Only single-row pyarrow tables are allowed but got table with {len(row)} rows.")
        self.current_rows.append(row)
        if writer_batch_size is None:
            writer_batch_size = self.writer_batch_size
        if writer_batch_size is not None and len(self.current_rows) >= writer_batch_size:
            self.write_rows_on_file()

    def write_batch(self, batch_examples: dict[str, list], writer_batch_size: Optional[int] = None):
        if batch_examples and len(next(iter(batch_examples.values()))) == 0:
            return
        features = None if self.pa_writer is None and self.update_features else self._features
        try_features = self._features if self.pa_writer is None and self.update_features else None
        arrays = []
        inferred_features = Features()
        if self.schema:
            schema_cols = set(self.schema.names)
            batch_cols = batch_examples.keys()  # .keys() preserves the order (unlike set)
            common_cols = [col for col in self.schema.names if col in batch_cols]
            extra_cols = [col for col in batch_cols if col not in schema_cols]
            cols = common_cols + extra_cols
        else:
            cols = list(batch_examples)
        for col in cols:
            col_values = batch_examples[col]
            col_type = features[col] if features else None
            if isinstance(col_values, (pa.Array, pa.ChunkedArray)):
                array = cast_array_to_feature(col_values, col_type) if col_type is not None else col_values
                arrays.append(array)
                inferred_features[col] = generate_from_arrow_type(col_values.type)
            else:
                col_try_type = try_features[col] if try_features is not None and col in try_features else None
                typed_sequence = OptimizedTypedSequence(col_values, type=col_type, try_type=col_try_type, col=col)
                arrays.append(pa.array(typed_sequence))
                inferred_features[col] = typed_sequence.get_inferred_type()
        schema = inferred_features.arrow_schema if self.pa_writer is None else self.schema
        pa_table = pa.Table.from_arrays(arrays, schema=schema)
        self.write_table(pa_table, writer_batch_size)

    def write_table(self, pa_table: pa.Table, writer_batch_size: Optional[int] = None):
        if writer_batch_size is None:
            writer_batch_size = self.writer_batch_size
        if self.pa_writer is None:
            self._build_writer(inferred_schema=pa_table.schema)
        pa_table = pa_table.combine_chunks()
        pa_table = table_cast(pa_table, self._schema)
        if self.embed_local_files:
            pa_table = embed_table_storage(pa_table)
        self._num_bytes += pa_table.nbytes
        self._num_examples += pa_table.num_rows
        self.pa_writer.write_table(pa_table, writer_batch_size)

    def finalize(self, close_stream=True):
        self.write_rows_on_file()
        if self._check_duplicates:
            self.check_duplicate_keys()
            self.hkey_record = []
        self.write_examples_on_file()
        if self.pa_writer is None and self.schema:
            self._build_writer(self.schema)
        if self.pa_writer is not None:
            self.pa_writer.close()
            self.pa_writer = None
            if close_stream:
                self.stream.close()
        else:
            if close_stream:
                self.stream.close()
            raise SchemaInferenceError("Please pass `features` or at least one example when writing data")
        return self._num_examples, self._num_bytes

class _TempCacheDir:
    def __init__(self):
        self.name = tempfile.mkdtemp(prefix=data_config.TEMP_CACHE_DIR_PREFIX)
        self._finalizer = weakref.finalize(self, self._cleanup)

    def _cleanup(self):
        for dset in get_datasets_with_cache_file_in_temp_dir():
            dset.__del__()
        if os.path.exists(self.name):
            try:
                shutil.rmtree(self.name)
            except Exception as e:
                raise OSError(f"An error occured while trying to delete temporary cache directory {self.name}. Please delete it manually.") from e

    def cleanup(self):
        if self._finalizer.detach():
            self._cleanup()

def get_datasets_with_cache_file_in_temp_dir():
    return list(_DATASETS_WITH_TABLE_IN_TEMP_DIR) if _DATASETS_WITH_TABLE_IN_TEMP_DIR is not None else []

def get_temporary_cache_files_directory() -> str:
    global _TEMP_DIR_FOR_TEMP_CACHE_FILES
    if _TEMP_DIR_FOR_TEMP_CACHE_FILES is None:
        _TEMP_DIR_FOR_TEMP_CACHE_FILES = _TempCacheDir()
    return _TEMP_DIR_FOR_TEMP_CACHE_FILES.name

def is_documented_by(function_with_docstring: Callable):
    def wrapper(target_function):
        target_function.__doc__ = function_with_docstring.__doc__
        return target_function

    return wrapper

def update_metadata_with_features(table: Table, features: Features):
    features = Features({col_name: features[col_name] for col_name in table.column_names})
    if table.schema.metadata is None or b"huggingface" not in table.schema.metadata:
        pa_metadata = ArrowWriter._build_metadata(DatasetInfo(features=features))
    else:
        metadata = json.loads(table.schema.metadata[b"huggingface"].decode())
        if "info" not in metadata:
            metadata["info"] = asdict(DatasetInfo(features=features))
        else:
            metadata["info"]["features"] = asdict(DatasetInfo(features=features))["features"]
        pa_metadata = {"huggingface": json.dumps(metadata)}
    table = table.replace_schema_metadata(pa_metadata)
    return table

def _check_table(table) -> Table:
    if isinstance(table, pa.Table):
        return InMemoryTable(table)
    elif isinstance(table, Table):
        return table
    else:
        raise TypeError(f"Expected a pyarrow.Table or a datasets.table.Table object, but got {table}.")

def _check_column_names(column_names: list[str]):
    counter = Counter(column_names)
    if not all(count == 1 for count in counter.values()):
        duplicated_columns = [col for col in counter if counter[col] > 1]
        raise ValueError(f"The table can't have duplicated columns but columns {duplicated_columns} are duplicated.")

def generate_fingerprint(dataset: "Dataset") -> str:
    state = dataset.__dict__
    hasher = Hasher()
    for key in sorted(state):
        if key == "_fingerprint":
            continue
        hasher.update(key)
        hasher.update(state[key])
    for cache_file in dataset.cache_files:
        hasher.update(os.path.getmtime(cache_file["filename"]))
    return hasher.hexdigest()

def maybe_register_dataset_for_temp_dir_deletion(dataset):
    if _TEMP_DIR_FOR_TEMP_CACHE_FILES is None:
        return

    global _DATASETS_WITH_TABLE_IN_TEMP_DIR
    if _DATASETS_WITH_TABLE_IN_TEMP_DIR is None:
        _DATASETS_WITH_TABLE_IN_TEMP_DIR = weakref.WeakSet()
    if any(Path(_TEMP_DIR_FOR_TEMP_CACHE_FILES.name) in Path(cache_file["filename"]).parents for cache_file in dataset.cache_files):
        _DATASETS_WITH_TABLE_IN_TEMP_DIR.add(dataset)

def generate_random_fingerprint(nbits: int = 64) -> str:
    return f"{fingerprint_rng.getrandbits(nbits):0{nbits // 4}x}"

def update_fingerprint(fingerprint, transform, transform_args):
    global fingerprint_warnings
    hasher = Hasher()
    hasher.update(fingerprint)
    try:
        hasher.update(transform)
    except:  # noqa various errors might raise here from pickle or dill
        if _CACHING_ENABLED:
            if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
        
        return generate_random_fingerprint()
    for key in sorted(transform_args):
        hasher.update(key)
        try:
            hasher.update(transform_args[key])
        except:  # noqa various errors might raise here from pickle or dill
            if _CACHING_ENABLED:
                if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                    fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
            return generate_random_fingerprint()
    return hasher.hexdigest()

def validate_fingerprint(fingerprint: str, max_length=64):
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"Invalid fingerprint '{fingerprint}': it should be a non-empty string.")
    for invalid_char in INVALID_WINDOWS_CHARACTERS_IN_PATH:
        if invalid_char in fingerprint:
            raise ValueError(f"Invalid fingerprint. Bad characters from black list '{INVALID_WINDOWS_CHARACTERS_IN_PATH}' found in '{fingerprint}'. They could create issues when creating cache files.")
    if len(fingerprint) > max_length:
        raise ValueError(f"Invalid fingerprint. Maximum lenth is {max_length} but '{fingerprint}' has length {len(fingerprint)}. It could create issues when creating cache files.")

def format_transform_for_fingerprint(func: Callable, version: Optional[str] = None) -> str:
    transform = f"{func.__module__}.{func.__qualname__}"
    if version is not None:
        transform += f"@{version}"
    return transform

def format_kwargs_for_fingerprint(
    func: Callable,
    args: tuple,
    kwargs: dict[str, Any],
    use_kwargs: Optional[list[str]] = None,
    ignore_kwargs: Optional[list[str]] = None,
    randomized_function: bool = False,
) -> dict[str, Any]:
    kwargs_for_fingerprint = kwargs.copy()
    if args:
        params = [p.name for p in inspect.signature(func).parameters.values() if p != p.VAR_KEYWORD]
        args = args[1:]  # assume the first argument is the dataset
        params = params[1:]
        kwargs_for_fingerprint.update(zip(params, args))
    else:
        del kwargs_for_fingerprint[next(iter(inspect.signature(func).parameters))]

    if use_kwargs:
        kwargs_for_fingerprint = {k: v for k, v in kwargs_for_fingerprint.items() if k in use_kwargs}
    if ignore_kwargs:
        kwargs_for_fingerprint = {k: v for k, v in kwargs_for_fingerprint.items() if k not in ignore_kwargs}
    if randomized_function:  # randomized functions have `seed` and `generator` parameters
        if kwargs_for_fingerprint.get("seed") is None and kwargs_for_fingerprint.get("generator") is None:
            _, seed, pos, *_ = np.random.get_state()
            seed = seed[pos] if pos < 624 else seed[0]
            kwargs_for_fingerprint["generator"] = np.random.default_rng(seed)

    default_values = {p.name: p.default for p in inspect.signature(func).parameters.values() if p.default != inspect._empty}
    for default_varname, default_value in default_values.items():
        if default_varname in kwargs_for_fingerprint and kwargs_for_fingerprint[default_varname] == default_value:
            kwargs_for_fingerprint.pop(default_varname)
    return kwargs_for_fingerprint

def fingerprint_transform(
    inplace: bool,
    use_kwargs: Optional[list[str]] = None,
    ignore_kwargs: Optional[list[str]] = None,
    fingerprint_names: Optional[list[str]] = None,
    randomized_function: bool = False,
    version: Optional[str] = None,
):
    
    if use_kwargs is not None and not isinstance(use_kwargs, list):
        raise ValueError(f"use_kwargs is supposed to be a list, not {type(use_kwargs)}")

    if ignore_kwargs is not None and not isinstance(ignore_kwargs, list):
        raise ValueError(f"ignore_kwargs is supposed to be a list, not {type(use_kwargs)}")

    if inplace and fingerprint_names:
        raise ValueError("fingerprint_names are only used when inplace is False")

    fingerprint_names = fingerprint_names if fingerprint_names is not None else ["new_fingerprint"]

    def _fingerprint(func):
        if not inplace and not all(name in func.__code__.co_varnames for name in fingerprint_names):
            raise ValueError(f"function {func} is missing parameters {fingerprint_names} in signature")

        if randomized_function:  # randomized function have seed and generator parameters
            if "seed" not in func.__code__.co_varnames:
                raise ValueError(f"'seed' must be in {func}'s signature")
            if "generator" not in func.__code__.co_varnames:
                raise ValueError(f"'generator' must be in {func}'s signature")
        transform = format_transform_for_fingerprint(func, version=version)

        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(
                func,
                args,
                kwargs,
                use_kwargs=use_kwargs,
                ignore_kwargs=ignore_kwargs,
                randomized_function=randomized_function,
            )

            if args:
                dataset: Dataset = args[0]
                args = args[1:]
            else:
                dataset: Dataset = kwargs.pop(next(iter(inspect.signature(func).parameters)))

            if inplace:
                new_fingerprint = update_fingerprint(dataset._fingerprint, transform, kwargs_for_fingerprint)
            else:
                for fingerprint_name in fingerprint_names:  # transforms like `train_test_split` have several hashes
                    if kwargs.get(fingerprint_name) is None:
                        kwargs_for_fingerprint["fingerprint_name"] = fingerprint_name
                        kwargs[fingerprint_name] = update_fingerprint(dataset._fingerprint, transform, kwargs_for_fingerprint)
                    else:
                        validate_fingerprint(kwargs[fingerprint_name])

            out = func(dataset, *args, **kwargs)

            if inplace:  # update after calling func so that the fingerprint doesn't change if the function fails
                dataset._fingerprint = new_fingerprint
            return out

        wrapper._decorator_name_ = "fingerprint"
        return wrapper

    return _fingerprint

def transmit_format(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            self: "Dataset" = args[0]
            args = args[1:]
        else:
            self: "Dataset" = kwargs.pop("self")
        unformatted_columns = set(self.column_names) - set(self._format_columns or [])
        self_format = {
            "type": self._format_type,
            "format_kwargs": self._format_kwargs,
            "columns": self._format_columns,
            "output_all_columns": self._output_all_columns,
        }
        out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
        datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
        for dataset in datasets:
            new_format = self_format.copy()
            if new_format["columns"] is not None:  # new formatted columns = (columns - previously unformatted columns)
                new_format["columns"] = sorted(set(dataset.column_names) - unformatted_columns)
            out_format = {
                "type": dataset._format_type,
                "format_kwargs": dataset._format_kwargs,
                "columns": sorted(dataset._format_columns) if dataset._format_columns is not None else None,
                "output_all_columns": dataset._output_all_columns,
            }
            if out_format != new_format:
                fingerprint = dataset._fingerprint
                dataset.set_format(**new_format)
                dataset._fingerprint = fingerprint
        return out

    wrapper._decorator_name_ = "transmit_format"
    return wrapper

def is_caching_enabled() -> bool:
    global _CACHING_ENABLED
    return bool(_CACHING_ENABLED)

@dataclass
class BuilderConfig:
    name: str = "default"
    version: Optional[Union[Version, str]] = Version("0.0.0")
    data_dir: Optional[str] = None
    data_files: Optional[Union[DataFilesDict, DataFilesPatternsDict]] = None
    description: Optional[str] = None

    def __post_init__(self):
        # The config name is used to name the cache directory.
        for invalid_char in INVALID_WINDOWS_CHARACTERS_IN_PATH:
            if invalid_char in self.name:
                raise InvalidConfigName(f"Bad characters from black list '{INVALID_WINDOWS_CHARACTERS_IN_PATH}' found in '{self.name}'. They could create issues when creating a directory for this config on Windows filesystem.")
        if self.data_files is not None and not isinstance(self.data_files, (DataFilesDict, DataFilesPatternsDict)):
            raise ValueError(f"Expected a DataFilesDict in data_files but got {self.data_files}")

    def __eq__(self, o):
        if set(self.__dict__.keys()) != set(o.__dict__.keys()):
            return False
        return all((k, getattr(self, k)) == (k, getattr(o, k)) for k in self.__dict__.keys())

    def create_config_id(self, config_kwargs: dict, custom_features: Optional[Features] = None) -> str:
        suffix: Optional[str] = None
        config_kwargs_to_add_to_suffix = config_kwargs.copy()
        config_kwargs_to_add_to_suffix.pop("name", None)
        config_kwargs_to_add_to_suffix.pop("version", None)
        if "data_dir" in config_kwargs_to_add_to_suffix:
            if config_kwargs_to_add_to_suffix["data_dir"] is None:
                config_kwargs_to_add_to_suffix.pop("data_dir", None)
            else:
                data_dir = config_kwargs_to_add_to_suffix["data_dir"]
                data_dir = os.path.normpath(data_dir)
                config_kwargs_to_add_to_suffix["data_dir"] = data_dir
        if config_kwargs_to_add_to_suffix:
            config_kwargs_to_add_to_suffix = {k: config_kwargs_to_add_to_suffix[k] for k in sorted(config_kwargs_to_add_to_suffix)}
            if all(isinstance(v, (str, bool, int, float)) for v in config_kwargs_to_add_to_suffix.values()):
                suffix = ",".join(str(k) + "=" + urllib.parse.quote_plus(str(v)) for k, v in config_kwargs_to_add_to_suffix.items())
                if len(suffix) > 32:  # hash if too long
                    suffix = Hasher.hash(config_kwargs_to_add_to_suffix)
            else:
                suffix = Hasher.hash(config_kwargs_to_add_to_suffix)

        if custom_features is not None:
            m = Hasher()
            if suffix:
                m.update(suffix)
            m.update(custom_features)
            suffix = m.hexdigest()

        if suffix:
            config_id = self.name + "-" + suffix
            if len(config_id) > data_config.MAX_DATASET_CONFIG_ID_READABLE_LENGTH:
                config_id = self.name + "-" + Hasher.hash(suffix)
            return config_id
        else:
            return self.name

    def _resolve_data_files(self, base_path: str, download_config: DownloadConfig) -> None:
        if isinstance(self.data_files, DataFilesPatternsDict):
            base_path = xjoin(base_path, self.data_dir) if self.data_dir else base_path
            self.data_files = self.data_files.resolve(base_path, download_config)

class DatasetInfoMixin:
    def __init__(self, info: DatasetInfo, split: Optional[NamedSplit]):
        self._info = info
        self._split = split

    @property
    def info(self):
        return self._info

    @property
    def split(self):
        return self._split

    @property
    def builder_name(self) -> str:
        return self._info.builder_name

    @property
    def citation(self) -> str:
        return self._info.citation

    @property
    def config_name(self) -> str:
        return self._info.config_name

    @property
    def dataset_size(self) -> Optional[int]:
        return self._info.dataset_size

    @property
    def description(self) -> str:
        return self._info.description

    @property
    def download_checksums(self) -> Optional[dict]:
        return self._info.download_checksums

    @property
    def download_size(self) -> Optional[int]:
        return self._info.download_size

    @property
    def features(self) -> Optional[Features]:
        return self._info.features.copy() if self._info.features is not None else None

    @property
    def homepage(self) -> Optional[str]:
        return self._info.homepage

    @property
    def license(self) -> Optional[str]:
        return self._info.license

    @property
    def size_in_bytes(self) -> Optional[int]:
        return self._info.size_in_bytes

    @property
    def supervised_keys(self):
        return self._info.supervised_keys

    @property
    def version(self):
        return self._info.version

class DatasetInfosDict(dict[str, DatasetInfo]):
    def write_to_directory(self, dataset_infos_dir, overwrite=False, pretty_print=False) -> None:
        total_dataset_infos = {}
        dataset_infos_path = os.path.join(dataset_infos_dir, data_config.DATASETDICT_INFOS_FILENAME)
        dataset_readme_path = os.path.join(dataset_infos_dir, data_config.REPOCARD_FILENAME)
        if not overwrite:
            total_dataset_infos = self.from_directory(dataset_infos_dir)
        total_dataset_infos.update(self)
        if os.path.exists(dataset_infos_path):
            with open(dataset_infos_path, "w", encoding="utf-8") as f:
                dataset_infos_dict = {config_name: asdict(dset_info) for config_name, dset_info in total_dataset_infos.items()}
                json.dump(dataset_infos_dict, f, indent=4 if pretty_print else None)
        if os.path.exists(dataset_readme_path):
            dataset_card = DatasetCard.load(dataset_readme_path)
            dataset_card_data = dataset_card.data
        else:
            dataset_card = None
            dataset_card_data = DatasetCardData()
        if total_dataset_infos:
            total_dataset_infos.to_dataset_card_data(dataset_card_data)
            dataset_card = (DatasetCard("---\n" + str(dataset_card_data) + "\n---\n") if dataset_card is None else dataset_card)
            dataset_card.save(Path(dataset_readme_path))

    @classmethod
    def from_directory(cls, dataset_infos_dir) -> "DatasetInfosDict":
        if os.path.exists(os.path.join(dataset_infos_dir, data_config.REPOCARD_FILENAME)):
            dataset_card_data = DatasetCard.load(Path(dataset_infos_dir) / data_config.REPOCARD_FILENAME).data
            if "dataset_info" in dataset_card_data:
                return cls.from_dataset_card_data(dataset_card_data)
        if os.path.exists(os.path.join(dataset_infos_dir, data_config.DATASETDICT_INFOS_FILENAME)):
            with open(os.path.join(dataset_infos_dir, data_config.DATASETDICT_INFOS_FILENAME), encoding="utf-8") as f:
                return cls({config_name: DatasetInfo.from_dict(dataset_info_dict) for config_name, dataset_info_dict in json.load(f).items()})
        else:
            return cls()

    @classmethod
    def from_dataset_card_data(cls, dataset_card_data: DatasetCardData) -> "DatasetInfosDict":
        if isinstance(dataset_card_data.get("dataset_info"), (list, dict)):
            if isinstance(dataset_card_data["dataset_info"], list):
                return cls(
                    {
                        dataset_info_yaml_dict.get("config_name", "default"): DatasetInfo._from_yaml_dict(dataset_info_yaml_dict)
                        for dataset_info_yaml_dict in dataset_card_data["dataset_info"]
                    }
                )
            else:
                dataset_info = DatasetInfo._from_yaml_dict(dataset_card_data["dataset_info"])
                dataset_info.config_name = dataset_card_data["dataset_info"].get("config_name", "default")
                return cls({dataset_info.config_name: dataset_info})
        else:
            return cls()

    def to_dataset_card_data(self, dataset_card_data: DatasetCardData) -> None:
        if self:
            if "dataset_info" in dataset_card_data and isinstance(dataset_card_data["dataset_info"], dict):
                dataset_metadata_infos = {dataset_card_data["dataset_info"].get("config_name", "default"): dataset_card_data["dataset_info"]}
            elif "dataset_info" in dataset_card_data and isinstance(dataset_card_data["dataset_info"], list):
                dataset_metadata_infos = {config_metadata["config_name"]: config_metadata for config_metadata in dataset_card_data["dataset_info"]}
            else:
                dataset_metadata_infos = {}
            total_dataset_infos = {
                **dataset_metadata_infos,
                **{config_name: dset_info._to_yaml_dict() for config_name, dset_info in self.items()},
            }
            for config_name, dset_info_yaml_dict in total_dataset_infos.items():
                dset_info_yaml_dict["config_name"] = config_name
            if len(total_dataset_infos) == 1:
                dataset_card_data["dataset_info"] = next(iter(total_dataset_infos.values()))
                config_name = dataset_card_data["dataset_info"].pop("config_name", None)
                if config_name != "default":
                    dataset_card_data["dataset_info"] = {
                        "config_name": config_name,
                        **dataset_card_data["dataset_info"],
                    }
            else:
                dataset_card_data["dataset_info"] = []
                for config_name, dataset_info_yaml_dict in sorted(total_dataset_infos.items()):
                    dataset_info_yaml_dict.pop("config_name", None)
                    dataset_info_yaml_dict = {"config_name": config_name, **dataset_info_yaml_dict}
                    dataset_card_data["dataset_info"].append(dataset_info_yaml_dict)

class MetadataConfigs(dict[str, dict[str, Any]]):
    FIELD_NAME: ClassVar[str] = data_config.METADATA_CONFIGS_FIELD

    @staticmethod
    def _raise_if_data_files_field_not_valid(metadata_config: dict):
        yaml_data_files = metadata_config.get("data_files")
        if yaml_data_files is not None:
            yaml_error_message = textwrap.dedent(
                f"""
                Expected data_files in YAML to be either a string or a list of strings
                or a list of dicts with two keys: 'split' and 'path', but got {yaml_data_files}
                Examples of data_files in YAML:

                   data_files: data.csv

                   data_files: data/*.png

                   data_files:
                    - part0/*
                    - part1/*

                   data_files:
                    - split: train
                      path: train/*
                    - split: test
                      path: test/*

                   data_files:
                    - split: train
                      path:
                      - train/part1/*
                      - train/part2/*
                    - split: test
                      path: test/*

                PS: some symbols like dashes '-' are not allowed in split names
                """
            )
            if not isinstance(yaml_data_files, (list, str)):
                raise ValueError(yaml_error_message)
            if isinstance(yaml_data_files, list):
                for yaml_data_files_item in yaml_data_files:
                    if (
                        not isinstance(yaml_data_files_item, (str, dict))
                        or isinstance(yaml_data_files_item, dict)
                        and not (
                            len(yaml_data_files_item) == 2
                            and "split" in yaml_data_files_item
                            and re.match(_split_re, yaml_data_files_item["split"])
                            and isinstance(yaml_data_files_item.get("path"), (str, list))
                        )
                    ):
                        raise ValueError(yaml_error_message)

    @classmethod
    def _from_exported_parquet_files_and_dataset_infos(
        cls,
        parquet_commit_hash: str,
        exported_parquet_files: list[dict[str, Any]],
        dataset_infos: DatasetInfosDict,
    ) -> "MetadataConfigs":
        metadata_configs = {
            config_name: {
                "data_files": [
                    {
                        "split": split_name,
                        "path": [
                            parquet_file["url"].replace("refs%2Fconvert%2Fparquet", parquet_commit_hash)
                            for parquet_file in parquet_files_for_split
                        ],
                    }
                    for split_name, parquet_files_for_split in groupby(parquet_files_for_config, itemgetter("split"))
                ],
                "version": str(dataset_infos.get(config_name, DatasetInfo()).version or "0.0.0"),
            }
            for config_name, parquet_files_for_config in groupby(exported_parquet_files, itemgetter("config"))
        }
        if dataset_infos:
            metadata_configs = {
                config_name: {
                    "data_files": [
                        data_file
                        for split_name in dataset_info.splits
                        for data_file in metadata_configs[config_name]["data_files"]
                        if data_file["split"] == split_name
                    ],
                    "version": metadata_configs[config_name]["version"],
                }
                for config_name, dataset_info in dataset_infos.items()
            }
        return cls(metadata_configs)

    @classmethod
    def from_dataset_card_data(cls, dataset_card_data: DatasetCardData) -> "MetadataConfigs":
        if dataset_card_data.get(cls.FIELD_NAME):
            metadata_configs = dataset_card_data[cls.FIELD_NAME]
            if not isinstance(metadata_configs, list):
                raise ValueError(f"Expected {cls.FIELD_NAME} to be a list, but got '{metadata_configs}'")
            for metadata_config in metadata_configs:
                if "config_name" not in metadata_config:
                    raise ValueError(f"Each config must include `config_name` field with a string name of a config, but got {metadata_config}. ")
                cls._raise_if_data_files_field_not_valid(metadata_config)
            return cls(
                {
                    config.pop("config_name"): {param: value if param != "features" else Features._from_yaml_list(value) for param, value in config.items()}
                    for metadata_config in metadata_configs
                    if (config := metadata_config.copy())
                }
            )
        return cls()

    def to_dataset_card_data(self, dataset_card_data: DatasetCardData) -> None:
        if self:
            for metadata_config in self.values():
                self._raise_if_data_files_field_not_valid(metadata_config)
            current_metadata_configs = self.from_dataset_card_data(dataset_card_data)
            total_metadata_configs = dict(sorted({**current_metadata_configs, **self}.items()))
            for config_name, config_metadata in total_metadata_configs.items():
                config_metadata.pop("config_name", None)
            dataset_card_data[self.FIELD_NAME] = [{"config_name": config_name, **config_metadata} for config_name, config_metadata in total_metadata_configs.items()]

    def get_default_config_name(self) -> Optional[str]:
        default_config_name = None
        for config_name, metadata_config in self.items():
            if len(self) == 1 or config_name == "default" or metadata_config.get("default"):
                if default_config_name is None:
                    default_config_name = config_name
                else:
                    raise ValueError(f"Dataset has several default configs: '{default_config_name}' and '{config_name}'.")
        return default_config_name

@dataclass
class BuilderConfigsParameters:
    metadata_configs: Optional[MetadataConfigs] = None
    builder_configs: Optional[list[BuilderConfig]] = None
    default_config_name: Optional[str] = None

@dataclass
class DatasetModule:
    module_path: str
    hash: str
    builder_kwargs: dict
    builder_configs_parameters: BuilderConfigsParameters = field(default_factory=BuilderConfigsParameters)
    dataset_infos: Optional[DatasetInfosDict] = None
    importable_file_path: Optional[str] = None

class Dataset(DatasetInfoMixin):
    def __init__(
        self,
        arrow_table: Table,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        indices_table: Optional[Table] = None,
        fingerprint: Optional[str] = None,
    ):
        info = info.copy() if info is not None else DatasetInfo()
        DatasetInfoMixin.__init__(self, info=info, split=split)

        self._data: Table = _check_table(arrow_table)
        self._indices: Optional[Table] = _check_table(indices_table) if indices_table is not None else None
        maybe_register_dataset_for_temp_dir_deletion(self)

        self._format_type: Optional[str] = None
        self._format_kwargs: dict = {}
        self._format_columns: Optional[list] = None
        self._output_all_columns: bool = False
        self._fingerprint: str = fingerprint


        if self._data.schema.metadata is not None and b"huggingface" in self._data.schema.metadata:
            metadata = json.loads(self._data.schema.metadata[b"huggingface"].decode())
            if ("fingerprint" in metadata and self._fingerprint is None):
                self._fingerprint = metadata["fingerprint"]

        inferred_features = Features.from_arrow_schema(arrow_table.schema)
        if self.info.features is None:
            self.info.features = inferred_features
        else:  # make sure the nested columns are in the right order
            try:
                self.info.features = self.info.features.reorder_fields_as(inferred_features)
            except ValueError as e:
                raise ValueError(f"{e}\nThe 'source' features come from dataset_info.json, and the 'target' ones are those of the dataset arrow file.")

        if self.data.schema != self.info.features.arrow_schema:
            self._data = self.data.cast(self.info.features.arrow_schema)

        if self._fingerprint is None:
            self._fingerprint = generate_fingerprint(self)

        if self._info.features is None:
            raise ValueError("Features can't be None in a Dataset object")
        if self._fingerprint is None:
            raise ValueError("Fingerprint can't be None in a Dataset object")
        if self.info.features.type != inferred_features.type:
            raise ValueError(f"External features info don't match the dataset:\nGot\n{self.info.features}\nwith type\n{self.info.features.type}\n\nbut expected something like\n{inferred_features}\nwith type\n{inferred_features.type}")

        if self._indices is not None:
            if not pa.types.is_unsigned_integer(self._indices.column(0).type):
                raise ValueError(f"indices must be an Arrow table of unsigned integers, current type is {self._indices.column(0).type}")
        _check_column_names(self._data.column_names)

        self._data = update_metadata_with_features(self._data, self._info.features)

    @property
    def features(self) -> Features:
        features = super().features
        if features is None:  # this is already checked in __init__
            raise ValueError("Features can't be None in a Dataset object")
        return features

    @classmethod
    def from_file(
        cls,
        filename: str,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        indices_filename: Optional[str] = None,
        in_memory: bool = False,
    ) -> "Dataset":
        table = ArrowReader.read_table(filename, in_memory=in_memory)

        if indices_filename is not None:
            indices_pa_table = ArrowReader.read_table(indices_filename, in_memory=in_memory)
        else:
            indices_pa_table = None

        return cls(
            arrow_table=table,
            info=info,
            split=split,
            indices_table=indices_pa_table,
        )

    @classmethod
    def from_buffer(
        cls,
        buffer: pa.Buffer,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        indices_buffer: Optional[pa.Buffer] = None,
    ) -> "Dataset":
        table = InMemoryTable.from_buffer(buffer)

        if indices_buffer is not None:
            indices_table = InMemoryTable.from_buffer(buffer)
        else:
            indices_table = None

        return cls(table, info=info, split=split, indices_table=indices_table)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        features: Optional[Features] = None,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        preserve_index: Optional[bool] = None,
    ) -> "Dataset":

        if info is not None and features is not None and info.features != features:
            raise ValueError(f"Features specified in `features` and `info.features` can't be different:\n{features}\n{info.features}")
        features = features if features is not None else info.features if info is not None else None
        if info is None:
            info = DatasetInfo()
        info.features = features
        table = InMemoryTable.from_pandas(df=df, preserve_index=preserve_index)
        if features is not None:
            table = table.cast(features.arrow_schema)
        return cls(table, info=info, split=split)

    @classmethod
    def from_dict(
        cls,
        mapping: dict,
        features: Optional[Features] = None,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
    ) -> "Dataset":
        if info is not None and features is not None and info.features != features:
            raise ValueError(f"Features specified in `features` and `info.features` can't be different:\n{features}\n{info.features}")
        features = features if features is not None else info.features if info is not None else None
        arrow_typed_mapping = {}
        for col, data in mapping.items():
            if isinstance(data, (pa.Array, pa.ChunkedArray)):
                data = cast_array_to_feature(data, features[col]) if features is not None else data
            else:
                data = OptimizedTypedSequence(features.encode_column(data, col) if features is not None else data, type=features[col] if features is not None else None, col=col)
            arrow_typed_mapping[col] = data
        mapping = arrow_typed_mapping
        pa_table = InMemoryTable.from_pydict(mapping=mapping)
        if info is None:
            info = DatasetInfo()
        info.features = features
        if info.features is None:
            info.features = Features(
                {
                    col: generate_from_arrow_type(data.type)
                    if isinstance(data, (pa.Array, pa.ChunkedArray))
                    else data.get_inferred_type()
                    for col, data in mapping.items()
                }
            )
        return cls(pa_table, info=info, split=split)

    @classmethod
    def from_list(
        cls,
        mapping: list[dict],
        features: Optional[Features] = None,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
    ) -> "Dataset":
        mapping = {k: [r.get(k) for r in mapping] for k in mapping[0]} if mapping else {}
        return cls.from_dict(mapping, features, info, split)

    @staticmethod
    def from_generator(
        generator: Callable,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        gen_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        split: NamedSplit = Split.TRAIN,
        **kwargs,
    ):

        return GeneratorDatasetInputStream(
            generator=generator,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            gen_kwargs=gen_kwargs,
            num_proc=num_proc,
            split=split,
            **kwargs,
        ).read()

    @staticmethod
    def from_json(
        path_or_paths: Union[PathLike, list[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        field: Optional[str] = None,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        return JsonDatasetReader(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            field=field,
            num_proc=num_proc,
            **kwargs,
        ).read()

    def __setstate__(self, state):
        self.__dict__.update(state)
        maybe_register_dataset_for_temp_dir_deletion(self)
        return self

    def __del__(self):
        if hasattr(self, "_data"):
            del self._data
        if hasattr(self, "_indices"):
            del self._indices

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def save_to_disk(
        self,
        dataset_path: PathLike,
        max_shard_size: Optional[Union[str, int]] = None,
        num_shards: Optional[int] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
    ):
        if max_shard_size is not None and num_shards is not None:
            raise ValueError("Failed to push_to_hub: please specify either max_shard_size or num_shards, but not both.")
        if self.list_indexes():
            raise ValueError("please remove all the indexes using `dataset.drop_index` before saving a dataset")

        if num_shards is None:
            dataset_nbytes = self._estimate_nbytes()
            max_shard_size = convert_file_size_to_int(max_shard_size or data_config.MAX_SHARD_SIZE)
            num_shards = int(dataset_nbytes / max_shard_size) + 1
            num_shards = max(num_shards, num_proc or 1)

        num_proc = num_proc if num_proc is not None else 1
        num_shards = num_shards if num_shards is not None else num_proc

        fs: fsspec.AbstractFileSystem
        fs, _ = url_to_fs(dataset_path, **(storage_options or {}))

        if not is_remote_filesystem(fs):
            parent_cache_files_paths = {Path(cache_filename["filename"]).resolve().parent for cache_filename in self.cache_files}
            if Path(dataset_path).expanduser().resolve() in parent_cache_files_paths:
                raise PermissionError(f"Tried to overwrite {Path(dataset_path).expanduser().resolve()} but a dataset can't overwrite itself.")

        fs.makedirs(dataset_path, exist_ok=True)

        state = {
            key: self.__dict__[key]
            for key in [
                "_fingerprint",
                "_format_columns",
                "_format_kwargs",
                "_format_type",
                "_output_all_columns",
            ]
        }
        state["_split"] = str(self.split) if self.split is not None else self.split
        state["_data_files"] = [{"filename": f"data-{shard_idx:05d}-of-{num_shards:05d}.arrow"} for shard_idx in range(num_shards)]
        for k in state["_format_kwargs"].keys():
            try:
                json.dumps(state["_format_kwargs"][k])
            except TypeError as e:
                raise TypeError(str(e) + f"\nThe format kwargs must be JSON serializable, but key '{k}' isn't.") from None
        dataset_info = asdict(self._info)

        shards_done = 0
        pbar = hf_tqdm(
            unit=" examples",
            total=len(self),
            desc=f"Saving the dataset ({shards_done}/{num_shards} shards)",
        )
        kwargs_per_job = (
            {
                "job_id": shard_idx,
                "shard": self.shard(num_shards=num_shards, index=shard_idx, contiguous=True),
                "fpath": posixpath.join(dataset_path, f"data-{shard_idx:05d}-of-{num_shards:05d}.arrow"),
                "storage_options": storage_options,
            }
            for shard_idx in range(num_shards)
        )
        shard_lengths = [None] * num_shards
        shard_sizes = [None] * num_shards
        if num_proc > 1:
            with Pool(num_proc) as pool:
                with pbar:
                    for job_id, done, content in iflatmap_unordered(pool, Dataset._save_to_disk_single, kwargs_iterable=kwargs_per_job):
                        if done:
                            shards_done += 1
                            pbar.set_description(f"Saving the dataset ({shards_done}/{num_shards} shards)")
                            shard_lengths[job_id], shard_sizes[job_id] = content
                        else:
                            pbar.update(content)
        else:
            with pbar:
                for kwargs in kwargs_per_job:
                    for job_id, done, content in Dataset._save_to_disk_single(**kwargs):
                        if done:
                            shards_done += 1
                            pbar.set_description(f"Saving the dataset ({shards_done}/{num_shards} shards)")
                            shard_lengths[job_id], shard_sizes[job_id] = content
                        else:
                            pbar.update(content)
        with fs.open(posixpath.join(dataset_path, data_config.DATASET_STATE_JSON_FILENAME), "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)
        with fs.open(posixpath.join(dataset_path, data_config.DATASET_INFO_FILENAME), "w", encoding="utf-8") as dataset_info_file:
            sorted_keys_dataset_info = {key: dataset_info[key] for key in sorted(dataset_info)}
            json.dump(sorted_keys_dataset_info, dataset_info_file, indent=2)

    @staticmethod
    def _save_to_disk_single(job_id: int, shard: "Dataset", fpath: str, storage_options: Optional[dict]):
        batch_size = data_config.DEFAULT_MAX_BATCH_SIZE

        num_examples_progress_update = 0
        writer = ArrowWriter(
            features=shard.features,
            path=fpath,
            storage_options=storage_options,
            embed_local_files=True,
        )
        try:
            _time = time.time()
            for pa_table in shard.with_format("arrow").iter(batch_size):
                writer.write_table(pa_table)
                num_examples_progress_update += len(pa_table)
                if time.time() > _time + data_config.PBAR_REFRESH_TIME_INTERVAL:
                    _time = time.time()
                    yield job_id, False, num_examples_progress_update
                    num_examples_progress_update = 0
        finally:
            yield job_id, False, num_examples_progress_update
            num_examples, num_bytes = writer.finalize()
            writer.close()

        yield job_id, True, (num_examples, num_bytes)

    @staticmethod
    def _build_local_temp_path(uri_or_path: str) -> Path:
        src_dataset_path = Path(uri_or_path)
        tmp_dir = get_temporary_cache_files_directory()
        return Path(tmp_dir, src_dataset_path.relative_to(src_dataset_path.anchor))

    @staticmethod
    def load_from_disk(
        dataset_path: PathLike,
        keep_in_memory: Optional[bool] = None,
        storage_options: Optional[dict] = None,
    ) -> "Dataset":
        fs: fsspec.AbstractFileSystem
        fs, dataset_path = url_to_fs(dataset_path, **(storage_options or {}))

        dest_dataset_path = dataset_path
        dataset_dict_json_path = posixpath.join(dest_dataset_path, data_config.DATASETDICT_JSON_FILENAME)
        dataset_state_json_path = posixpath.join(dest_dataset_path, data_config.DATASET_STATE_JSON_FILENAME)
        dataset_info_path = posixpath.join(dest_dataset_path, data_config.DATASET_INFO_FILENAME)

        dataset_dict_is_file = fs.isfile(dataset_dict_json_path)
        dataset_info_is_file = fs.isfile(dataset_info_path)
        dataset_state_is_file = fs.isfile(dataset_state_json_path)
        if not dataset_info_is_file and not dataset_state_is_file:
            if dataset_dict_is_file:
                raise FileNotFoundError(f"No such files: '{dataset_info_path}', nor '{dataset_state_json_path}' found. Expected to load a `Dataset` object, but got a `DatasetDict`. Please use either `datasets.load_from_disk` or `DatasetDict.load_from_disk` instead.")
            raise FileNotFoundError(f"No such files: '{dataset_info_path}', nor '{dataset_state_json_path}' found. Expected to load a `Dataset` object but provided path is not a `Dataset`.")
        if not dataset_info_is_file:
            if dataset_dict_is_file:
                raise FileNotFoundError(f"No such file: '{dataset_info_path}' found. Expected to load a `Dataset` object, but got a `DatasetDict`. Please use either `datasets.load_from_disk` or `DatasetDict.load_from_disk` instead.")
            raise FileNotFoundError(f"No such file: '{dataset_info_path}'. Expected to load a `Dataset` object but provided path is not a `Dataset`.")
        if not dataset_state_is_file:
            if dataset_dict_is_file:
                raise FileNotFoundError(f"No such file: '{dataset_state_json_path}' found. Expected to load a `Dataset` object, but got a `DatasetDict`. Please use either `datasets.load_from_disk` or `DatasetDict.load_from_disk` instead.")
            raise FileNotFoundError(f"No such file: '{dataset_state_json_path}'. Expected to load a `Dataset` object but provided path is not a `Dataset`.")

        if is_remote_filesystem(fs):
            src_dataset_path = dest_dataset_path
            dest_dataset_path = Dataset._build_local_temp_path(src_dataset_path)
            fs.download(src_dataset_path, dest_dataset_path.as_posix(), recursive=True)
            dataset_state_json_path = posixpath.join(dest_dataset_path, data_config.DATASET_STATE_JSON_FILENAME)
            dataset_info_path = posixpath.join(dest_dataset_path, data_config.DATASET_INFO_FILENAME)

        with open(dataset_state_json_path, encoding="utf-8") as state_file:
            state = json.load(state_file)
        with open(dataset_info_path, encoding="utf-8") as dataset_info_file:
            dataset_info = DatasetInfo.from_dict(json.load(dataset_info_file))

        dataset_size = estimate_dataset_size(Path(dest_dataset_path, data_file["filename"]) for data_file in state["_data_files"])
        keep_in_memory = keep_in_memory if keep_in_memory is not None else is_small_dataset(dataset_size)
        table_cls = InMemoryTable if keep_in_memory else MemoryMappedTable

        arrow_table = concat_tables(
            thread_map(
                table_cls.from_file,
                [posixpath.join(dest_dataset_path, data_file["filename"]) for data_file in state["_data_files"]],
                tqdm_class=hf_tqdm,
                desc="Loading dataset from disk",
                disable=len(state["_data_files"]) <= 16 or None,
            )
        )

        split = state["_split"]
        split = Split(split) if split is not None else split

        dataset = Dataset(
            arrow_table=arrow_table,
            info=dataset_info,
            split=split,
            fingerprint=state["_fingerprint"],
        )

        format = {
            "type": state["_format_type"],
            "format_kwargs": state["_format_kwargs"],
            "columns": state["_format_columns"],
            "output_all_columns": state["_output_all_columns"],
        }
        dataset = dataset.with_format(**format)

        return dataset

    @property
    def data(self) -> Table:
        return self._data

    @property
    def cache_files(self) -> list[dict]:
        cache_files = list_table_cache_files(self._data)
        if self._indices is not None:
            cache_files += list_table_cache_files(self._indices)
        return [{"filename": cache_filename} for cache_filename in cache_files]

    @property
    def num_columns(self) -> int:
        return self._data.num_columns

    @property
    def num_rows(self) -> int:
        if self._indices is not None:
            return self._indices.num_rows
        return self._data.num_rows

    @property
    def column_names(self) -> list[str]:
        return self._data.column_names

    @property
    def shape(self) -> tuple[int, int]:
        if self._indices is not None:
            return (self._indices.num_rows, self._data.num_columns)
        return self._data.shape

    def unique(self, column: str) -> list:
        if column not in self._data.column_names:
            raise ValueError(f"Column ({column}) not in table columns ({self._data.column_names}).")

        if self._indices is not None and self._indices.num_rows != self._data.num_rows:
            dataset = self.flatten_indices()
        else:
            dataset = self

        return dataset._data.column(column).unique().to_pylist()

    @fingerprint_transform(inplace=False)
    def flatten(self, new_fingerprint: Optional[str] = None, max_depth=16) -> "Dataset":
        dataset = copy.deepcopy(self)
        for depth in range(1, max_depth):
            if any(isinstance(field.type, pa.StructType) for field in dataset._data.schema):
                dataset._data = dataset._data.flatten()
            else:
                break
        dataset.info.features = self._info.features.flatten(max_depth=max_depth)
        dataset.info.features = Features({col: dataset.info.features[col] for col in dataset.data.column_names})
        dataset._data = update_metadata_with_features(dataset._data, dataset.features)
        dataset._fingerprint = new_fingerprint
        return dataset

    def cast(
        self,
        features: Features,
        batch_size: Optional[int] = 1000,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        num_proc: Optional[int] = None,
    ) -> "Dataset":
        if sorted(features) != sorted(self._data.column_names):
            raise ValueError(f"The columns in features ({list(features)}) must be identical as the columns in the dataset: {self._data.column_names}")

        schema = features.arrow_schema
        format = self.format
        dataset = self.with_format("arrow")
        dataset = dataset.map(
            partial(table_cast, schema=schema),
            batched=True,
            batch_size=batch_size,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            num_proc=num_proc,
            features=features,
            desc="Casting the dataset",
        )
        dataset = dataset.with_format(**format)
        return dataset

    @fingerprint_transform(inplace=False)
    def cast_column(self, column: str, feature: FeatureType, new_fingerprint: Optional[str] = None) -> "Dataset":
        if hasattr(feature, "decode_example"):
            dataset = copy.deepcopy(self)
            dataset._info.features[column] = feature
            dataset._fingerprint = new_fingerprint
            dataset._data = dataset._data.cast(dataset.features.arrow_schema)
            dataset._data = update_metadata_with_features(dataset._data, dataset.features)
            return dataset
        else:
            features = self.features
            features[column] = feature
            return self.cast(features)

    @transmit_format
    @fingerprint_transform(inplace=False)
    def remove_columns(self, column_names: Union[str, list[str]], new_fingerprint: Optional[str] = None) -> "Dataset":
        dataset = copy.deepcopy(self)
        if isinstance(column_names, str):
            column_names = [column_names]

        missing_columns = set(column_names) - set(self._data.column_names)
        if missing_columns:
            raise ValueError(f"Column name {list(missing_columns)} not in the dataset. Current columns in the dataset: {dataset._data.column_names}")

        for column_name in column_names:
            del dataset._info.features[column_name]

        dataset._data = dataset._data.drop(column_names)
        dataset._data = update_metadata_with_features(dataset._data, dataset.features)
        dataset._fingerprint = new_fingerprint
        return dataset

    @fingerprint_transform(inplace=False)
    def rename_column(self, original_column_name: str, new_column_name: str, new_fingerprint: Optional[str] = None) -> "Dataset":
        dataset = copy.deepcopy(self)
        if original_column_name not in dataset._data.column_names:
            raise ValueError(f"Original column name {original_column_name} not in the dataset. Current columns in the dataset: {dataset._data.column_names}")
        if new_column_name in dataset._data.column_names:
            raise ValueError(f"New column name {new_column_name} already in the dataset. Please choose a column name which is not already in the dataset. Current columns in the dataset: {dataset._data.column_names}")
        if not new_column_name:
            raise ValueError("New column name is empty.")

        def rename(columns):
            return [new_column_name if col == original_column_name else col for col in columns]

        new_column_names = rename(self._data.column_names)
        if self._format_columns is not None:
            dataset._format_columns = rename(self._format_columns)

        dataset._info.features = Features({new_column_name if col == original_column_name else col: feature for col, feature in self._info.features.items()})

        dataset._data = dataset._data.rename_columns(new_column_names)
        dataset._data = update_metadata_with_features(dataset._data, dataset.features)
        dataset._fingerprint = new_fingerprint
        return dataset

    @fingerprint_transform(inplace=False)
    def rename_columns(self, column_mapping: dict[str, str], new_fingerprint: Optional[str] = None) -> "Dataset":
        dataset = copy.deepcopy(self)

        extra_columns = set(column_mapping.keys()) - set(dataset.column_names)
        if extra_columns:
            raise ValueError(f"Original column names {extra_columns} not in the dataset. Current columns in the dataset: {dataset._data.column_names}")

        number_of_duplicates_in_new_columns = len(column_mapping.values()) - len(set(column_mapping.values()))
        if number_of_duplicates_in_new_columns != 0:
            raise ValueError("New column names must all be different, but this column mapping has {number_of_duplicates_in_new_columns} duplicates")

        empty_new_columns = [new_col for new_col in column_mapping.values() if not new_col]
        if empty_new_columns:
            raise ValueError(f"New column names {empty_new_columns} are empty.")

        def rename(columns):
            return [column_mapping[col] if col in column_mapping else col for col in columns]

        new_column_names = rename(self._data.column_names)
        if self._format_columns is not None:
            dataset._format_columns = rename(self._format_columns)

        dataset._info.features = Features({column_mapping[col] if col in column_mapping else col: feature for col, feature in (self._info.features or {}).items()})
        dataset._data = dataset._data.rename_columns(new_column_names)
        dataset._data = update_metadata_with_features(dataset._data, dataset.features)
        dataset._fingerprint = new_fingerprint
        return dataset

    @transmit_format
    @fingerprint_transform(inplace=False)
    def select_columns(self, column_names: Union[str, list[str]], new_fingerprint: Optional[str] = None) -> "Dataset":
        if isinstance(column_names, str):
            column_names = [column_names]

        missing_columns = set(column_names) - set(self._data.column_names)
        if missing_columns:
            raise ValueError(f"Column name {list(missing_columns)} not in the dataset. Current columns in the dataset: {self._data.column_names}.")

        dataset = copy.deepcopy(self)
        dataset._data = dataset._data.select(column_names)
        dataset._info.features = Features({col: self._info.features[col] for col in dataset._data.column_names})
        dataset._data = update_metadata_with_features(dataset._data, dataset.features)
        dataset._fingerprint = new_fingerprint
        return dataset

    def __len__(self):
        return self.num_rows

    def __iter__(self):
        if self._indices is None:
            format_kwargs = self._format_kwargs if self._format_kwargs is not None else {}
            formatter = get_formatter(self._format_type, features=self._info.features, **format_kwargs)
            batch_size = data_config.ARROW_READER_BATCH_SIZE_IN_DATASET_ITER
            for pa_subtable in table_iter(self.data, batch_size=batch_size):
                for i in range(pa_subtable.num_rows):
                    pa_subtable_ex = pa_subtable.slice(i, 1)
                    formatted_output = format_table(
                        pa_subtable_ex,
                        0,
                        formatter=formatter,
                        format_columns=self._format_columns,
                        output_all_columns=self._output_all_columns,
                    )
                    yield formatted_output
        else:
            for i in range(self.num_rows):
                yield self._getitem(i,)

    def iter(self, batch_size: int, drop_last_batch: bool = False):
        if self._indices is None:
            format_kwargs = self._format_kwargs if self._format_kwargs is not None else {}
            formatter = get_formatter(self._format_type, features=self._info.features, **format_kwargs)
            for pa_subtable in table_iter(self.data, batch_size=batch_size, drop_last_batch=drop_last_batch):
                formatted_batch = format_table(
                    pa_subtable,
                    range(pa_subtable.num_rows),
                    formatter=formatter,
                    format_columns=self._format_columns,
                    output_all_columns=self._output_all_columns,
                )
                yield formatted_batch
        else:
            num_rows = self.num_rows if not drop_last_batch else self.num_rows // batch_size * batch_size
            for i in range(0, num_rows, batch_size):
                yield self._getitem(
                    slice(i, i + batch_size),
                )

    def __repr__(self):
        return f"Dataset({{\n    features: {list(self._info.features.keys())},\n    num_rows: {self.num_rows}\n}})"

    @property
    def format(self):
        return {
            "type": self._format_type,
            "format_kwargs": self._format_kwargs,
            "columns": self.column_names if self._format_columns is None else self._format_columns,
            "output_all_columns": self._output_all_columns,
        }

    @contextlib.contextmanager
    def formatted_as(
        self,
        type: Optional[str] = None,
        columns: Optional[list] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ):
        old_format_type = self._format_type
        old_format_kwargs = self._format_kwargs
        old_format_columns = self._format_columns
        old_output_all_columns = self._output_all_columns
        try:
            self.set_format(type, columns, output_all_columns, **format_kwargs)
            yield
        finally:
            self.set_format(old_format_type, old_format_columns, old_output_all_columns, **old_format_kwargs)

    @fingerprint_transform(inplace=True)
    def set_format(
        self,
        type: Optional[str] = None,
        columns: Optional[list] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ):
        format_kwargs.update(format_kwargs.pop("format_kwargs", {}))  # allow to use self.set_format(**self.format)

        type = get_format_type_from_alias(type)
        get_formatter(type, features=self._info.features, **format_kwargs)

        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, tuple):
            columns = list(columns)
        if columns is not None:
            missing_columns = set(columns) - set(self._data.column_names)
            if missing_columns:
                raise ValueError(f"Columns {list(missing_columns)} not in the dataset. Current columns in the dataset: {self._data.column_names}")
        if columns is not None:
            columns = columns.copy()  # Ensures modifications made to the list after this call don't cause bugs

        self._format_type = type
        self._format_kwargs = format_kwargs
        self._format_columns = columns
        self._output_all_columns = output_all_columns

    def reset_format(self):
        self.set_format()

    def set_transform(self, transform: Optional[Callable], columns: Optional[list] = None, output_all_columns: bool = False):
        self.set_format("custom", columns=columns, output_all_columns=output_all_columns, transform=transform)

    def with_format(
        self,
        type: Optional[str] = None,
        columns: Optional[list] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ):
        dataset = copy.deepcopy(self)
        dataset.set_format(type=type, columns=columns, output_all_columns=output_all_columns, **format_kwargs)
        return dataset

    def with_transform(
        self,
        transform: Optional[Callable],
        columns: Optional[list] = None,
        output_all_columns: bool = False,
    ):
        dataset = copy.deepcopy(self)
        dataset.set_transform(transform=transform, columns=columns, output_all_columns=output_all_columns)
        return dataset

    def _getitem(self, key: Union[int, slice, str, ListLike[int]], **kwargs) -> Union[dict, list]:
        if isinstance(key, bool):
            raise TypeError("dataset index must be int, str, slice or collection of int, not bool")
        format_type = kwargs["format_type"] if "format_type" in kwargs else self._format_type
        format_columns = kwargs["format_columns"] if "format_columns" in kwargs else self._format_columns
        output_all_columns = (kwargs["output_all_columns"] if "output_all_columns" in kwargs else self._output_all_columns)
        format_kwargs = kwargs["format_kwargs"] if "format_kwargs" in kwargs else self._format_kwargs
        format_kwargs = format_kwargs if format_kwargs is not None else {}
        formatter = get_formatter(format_type, features=self._info.features, **format_kwargs)
        pa_subtable = query_table(self._data, key, indices=self._indices)
        formatted_output = format_table(pa_subtable, key, formatter=formatter, format_columns=format_columns, output_all_columns=output_all_columns)
        return formatted_output

    @overload
    def __getitem__(self, key: Union[int, slice, Iterable[int]]) -> dict:  # noqa: F811
        ...

    @overload
    def __getitem__(self, key: str) -> list:  # noqa: F811
        ...

    def __getitem__(self, key):  # noqa: F811
        return self._getitem(key)

    def __getitems__(self, keys: list) -> list:
        batch = self.__getitem__(keys)
        n_examples = len(batch[next(iter(batch))])
        return [{col: array[i] for col, array in batch.items()} for i in range(n_examples)]

    def cleanup_cache_files(self) -> int:
        current_cache_files = [os.path.abspath(cache_file["filename"]) for cache_file in self.cache_files]
        if not current_cache_files:
            return 0
        cache_directory = os.path.dirname(current_cache_files[0])
        files: list[str] = os.listdir(cache_directory)
        files_to_remove = []
        for f_name in files:
            full_name = os.path.abspath(os.path.join(cache_directory, f_name))
            if f_name.startswith("cache-") and f_name.endswith(".arrow"):
                if full_name in current_cache_files:
                    continue
                files_to_remove.append(full_name)
        for file_path in files_to_remove:
            os.remove(file_path)
        return len(files_to_remove)

    def _get_cache_file_path(self, fingerprint):
        if is_caching_enabled() and self.cache_files:
            cache_file_name = "cache-" + fingerprint + ".arrow"
            cache_directory = os.path.dirname(self.cache_files[0]["filename"])
        else:
            cache_file_name = "cache-" + generate_random_fingerprint() + ".arrow"
            cache_directory = get_temporary_cache_files_directory()
        cache_file_path = os.path.join(cache_directory, cache_file_name)
        return cache_file_path

    @transmit_format
    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, list[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, list[str]]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> "Dataset":
        if keep_in_memory and cache_file_name is not None:
            raise ValueError("Please use either `keep_in_memory` or `cache_file_name` but not both.")

        if num_proc is not None and num_proc <= 0:
            raise ValueError("num_proc must be an integer > 0.")

        if len(self) == 0:
            if self._indices is not None:  # empty indices mapping
                self = Dataset(
                    self.data.slice(0, 0),
                    info=self.info.copy(),
                    split=self.split,
                    fingerprint=new_fingerprint,
                )
            if remove_columns:
                return self.remove_columns(remove_columns)
            else:
                return self

        if function is None:
            function = lambda x: x  # noqa: E731

        if isinstance(input_columns, str):
            input_columns = [input_columns]

        if input_columns is not None:
            missing_columns = set(input_columns) - set(self._data.column_names)
            if missing_columns:
                raise ValueError(f"Input column {list(missing_columns)} not in the dataset. Current columns in the dataset: {self._data.column_names}")

        if isinstance(remove_columns, str):
            remove_columns = [remove_columns]

        if remove_columns is not None:
            missing_columns = set(remove_columns) - set(self._data.column_names)
            if missing_columns:
                raise ValueError(f"Column to remove {list(missing_columns)} not in the dataset. Current columns in the dataset: {self._data.column_names}")

        load_from_cache_file = load_from_cache_file if load_from_cache_file is not None else is_caching_enabled()

        if fn_kwargs is None:
            fn_kwargs = {}

        if num_proc is not None and num_proc > len(self):
            num_proc = len(self)
            
        dataset_kwargs = {
            "shard": self,
            "function": function,
            "with_indices": with_indices,
            "with_rank": with_rank,
            "input_columns": input_columns,
            "batched": batched,
            "batch_size": batch_size,
            "drop_last_batch": drop_last_batch,
            "remove_columns": remove_columns,
            "keep_in_memory": keep_in_memory,
            "writer_batch_size": writer_batch_size,
            "features": features,
            "disable_nullable": disable_nullable,
            "fn_kwargs": fn_kwargs,
        }

        if new_fingerprint is None:
            transform = format_transform_for_fingerprint(Dataset._map_single)
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(Dataset._map_single, (), dataset_kwargs)
            kwargs_for_fingerprint["fingerprint_name"] = "new_fingerprint"
            new_fingerprint = update_fingerprint(self._fingerprint, transform, kwargs_for_fingerprint)
        else:
            validate_fingerprint(new_fingerprint)
        dataset_kwargs["new_fingerprint"] = new_fingerprint

        if self.cache_files:
            if cache_file_name is None:
                cache_file_name = self._get_cache_file_path(new_fingerprint)
        dataset_kwargs["cache_file_name"] = cache_file_name

        def load_processed_shard_from_cache(shard_kwargs):
            shard = shard_kwargs["shard"]
            if shard_kwargs["cache_file_name"] is not None:
                if os.path.exists(shard_kwargs["cache_file_name"]) and load_from_cache_file:
                    info = shard.info.copy()
                    info.features = features
                    return Dataset.from_file(shard_kwargs["cache_file_name"], info=info, split=shard.split)
            raise NonExistentDatasetError

        num_shards = num_proc if num_proc is not None else 1
        if batched and drop_last_batch:
            pbar_total = len(self) // num_shards // batch_size * num_shards * batch_size
        else:
            pbar_total = len(self)

        shards_done = 0
        if num_proc is None or num_proc == 1:
            transformed_dataset = None
            try:
                transformed_dataset = load_processed_shard_from_cache(dataset_kwargs)
            except NonExistentDatasetError:
                pass
            if transformed_dataset is None:
                with hf_tqdm(unit=" examples", total=pbar_total, desc=desc or "Map",) as pbar:
                    for rank, done, content in Dataset._map_single(**dataset_kwargs):
                        if done:
                            shards_done += 1
                            transformed_dataset = content
                        else:
                            pbar.update(content)
            assert transformed_dataset is not None, "Failed to retrieve the result from map"
            if transformed_dataset._fingerprint != self._fingerprint:
                transformed_dataset._fingerprint = new_fingerprint
            return transformed_dataset
        else:

            def format_cache_file_name(cache_file_name: Optional[str], rank: Union[int, Literal["*"]]) -> Optional[str]:
                if not cache_file_name:
                    return cache_file_name
                sep = cache_file_name.rindex(".")
                base_name, extension = cache_file_name[:sep], cache_file_name[sep:]
                if isinstance(rank, int):
                    cache_file_name = base_name + suffix_template.format(rank=rank, num_proc=num_proc) + extension
                else:
                    cache_file_name = (base_name + suffix_template.replace("{rank:05d}", "{rank}").format(rank=rank, num_proc=num_proc) + extension)
                return cache_file_name

            def format_new_fingerprint(new_fingerprint: str, rank: int) -> str:
                new_fingerprint = new_fingerprint + suffix_template.format(rank=rank, num_proc=num_proc)
                validate_fingerprint(new_fingerprint)
                return new_fingerprint

            prev_env = deepcopy(os.environ)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            shards = [self.shard(num_shards=num_proc, index=rank, contiguous=True, keep_in_memory=keep_in_memory) for rank in range(num_proc)]
            kwargs_per_job = [
                {
                    **dataset_kwargs,
                    "shard": shards[rank],
                    "cache_file_name": format_cache_file_name(cache_file_name, rank),
                    "rank": rank,
                    "offset": sum(len(s) for s in shards[:rank]),
                    "new_fingerprint": format_new_fingerprint(new_fingerprint, rank),
                }
                for rank in range(num_shards)
            ]

            transformed_shards = [None] * num_shards
            for rank in range(num_shards):
                try:
                    transformed_shards[rank] = load_processed_shard_from_cache(kwargs_per_job[rank])
                    kwargs_per_job[rank] = None
                except NonExistentDatasetError:
                    pass

            kwargs_per_job = [kwargs for kwargs in kwargs_per_job if kwargs is not None]

            if kwargs_per_job:
                with Pool(len(kwargs_per_job)) as pool:
                    os.environ = prev_env
                    with hf_tqdm(unit="examples", total=pbar_total, desc=(desc or "Map") + f" (num_proc={num_proc})") as pbar:
                        for rank, done, content in iflatmap_unordered(pool, Dataset._map_single, kwargs_iterable=kwargs_per_job):
                            if done:
                                shards_done += 1
                                transformed_shards[rank] = content
                            else:
                                pbar.update(content)
                    pool.close()
                    pool.join()
                for kwargs in kwargs_per_job:
                    del kwargs["shard"]
            if None in transformed_shards:
                raise ValueError(f"Failed to retrieve results from map: result list {transformed_shards} still contains None - at least one worker failed to return its results")
            result = _concatenate_map_style_datasets(transformed_shards)
            if any(transformed_shard._fingerprint != shard._fingerprint for transformed_shard, shard in zip(transformed_shards, shards)):
                result._fingerprint = new_fingerprint
            else:
                result._fingerprint = self._fingerprint
            return result

    @staticmethod
    def _map_single(
        shard: "Dataset",
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[list[str]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[list[str]] = None,
        keep_in_memory: bool = False,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        new_fingerprint: Optional[str] = None,
        rank: Optional[int] = None,
        offset: int = 0,
    ) -> Iterable[tuple[int, bool, Union[int, "Dataset"]]]:
        if fn_kwargs is None:
            fn_kwargs = {}

        if batched and (batch_size is None or batch_size <= 0):
            batch_size = shard.num_rows

        update_data = None

        format_kwargs = shard._format_kwargs.copy()
        if not input_columns and shard._format_type is None:
            format_kwargs["lazy"] = True
        input_formatter = get_formatter(shard._format_type, features=shard.features, **format_kwargs,)

        check_same_num_examples = batched and len(shard.list_indexes()) > 0

        def validate_function_output(processed_inputs):
            allowed_processed_inputs_types = (Mapping, pa.Table, pd.DataFrame)
            
            if processed_inputs is not None and not isinstance(processed_inputs, allowed_processed_inputs_types):
                raise TypeError(f"Provided `function` which is applied to all elements of table returns a variable of type {type(processed_inputs)}. Make sure provided `function` returns a variable of type `dict` (or a pyarrow table) to update the dataset or `None` if you are only interested in side effects.")
            if batched and isinstance(processed_inputs, Mapping):
                allowed_batch_return_types = (list, np.ndarray, pd.Series)
                allowed_batch_return_types += (torch.Tensor,)
                all_dict_values_are_lists = all(isinstance(value, allowed_batch_return_types) for value in processed_inputs.values())
                if all_dict_values_are_lists is False:
                    raise TypeError(f"Provided `function` which is applied to all elements of table returns a `dict` of types {[type(x) for x in processed_inputs.values()]}. When using `batched=True`, make sure provided `function` returns a `dict` of types like `{allowed_batch_return_types}`.")

        def prepare_inputs(pa_inputs, indices, offset=0):
            inputs = format_table(
                pa_inputs,
                0 if not batched else range(pa_inputs.num_rows),
                format_columns=input_columns,
                formatter=input_formatter,
            )
            fn_args = [inputs] if input_columns is None else [inputs[col] for col in input_columns]
            if offset == 0:
                effective_indices = indices
            else:
                effective_indices = [i + offset for i in indices] if isinstance(indices, list) else indices + offset
            additional_args = ()
            if with_indices:
                additional_args += (effective_indices,)
            if with_rank:
                additional_args += (rank,)
            return inputs, fn_args, additional_args, fn_kwargs

        def prepare_outputs(pa_inputs, inputs, processed_inputs):
            nonlocal update_data
            if not (update_data := (processed_inputs is not None)):
                return None
            if isinstance(processed_inputs, LazyDict):
                processed_inputs = {k: v for k, v in processed_inputs.data.items() if k not in processed_inputs.keys_to_format}
                returned_lazy_dict = True
            else:
                returned_lazy_dict = False
            validate_function_output(processed_inputs)
            if shard._format_type or input_columns:
                inputs_to_merge = dict(zip(pa_inputs.column_names, pa_inputs.itercolumns()))
            elif isinstance(inputs, LazyDict):
                inputs_to_merge = {k: (v if k not in inputs.keys_to_format else pa_inputs[k]) for k, v in inputs.data.items()}
            else:
                inputs_to_merge = inputs
            if remove_columns is not None:
                for column in remove_columns:
                    if column in inputs_to_merge:
                        inputs_to_merge.pop(column)
                    if returned_lazy_dict and column in processed_inputs:
                        processed_inputs.pop(column)
            if check_same_num_examples:
                input_num_examples = len(pa_inputs)
                processed_inputs_num_examples = len(processed_inputs[next(iter(processed_inputs.keys()))])
                if input_num_examples != processed_inputs_num_examples:
                    raise DatasetTransformationNotAllowedError("Using `.map` in batched mode on a dataset with attached indexes is allowed only if it doesn't create or remove existing examples. You can first run `.drop_index() to remove your index and then re-add it.") from None
            if isinstance(inputs, Mapping) and isinstance(processed_inputs, Mapping):
                return {**inputs_to_merge, **processed_inputs}
            else:
                return processed_inputs

        def apply_function(pa_inputs, indices, offset=0):
            inputs, fn_args, additional_args, fn_kwargs = prepare_inputs(pa_inputs, indices, offset=offset)
            processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
            return prepare_outputs(pa_inputs, inputs, processed_inputs)

        async def async_apply_function(pa_inputs, indices, offset=0):
            inputs, fn_args, additional_args, fn_kwargs = prepare_inputs(pa_inputs, indices, offset=offset)
            processed_inputs = await function(*fn_args, *additional_args, **fn_kwargs)
            return prepare_outputs(pa_inputs, inputs, processed_inputs)

        def init_buffer_and_writer():
            writer_features = features
            if writer_features is None:
                writer_features = shard.features
                update_features = True
            else:
                update_features = False
            if keep_in_memory or cache_file_name is None:
                buf_writer = pa.BufferOutputStream()
                tmp_file = None
                writer = ArrowWriter(
                    features=writer_features,
                    stream=buf_writer,
                    writer_batch_size=writer_batch_size,
                    update_features=update_features,
                    fingerprint=new_fingerprint,
                    disable_nullable=disable_nullable,
                )
            else:
                buf_writer = None
                cache_dir = os.path.dirname(cache_file_name)
                os.makedirs(cache_dir, exist_ok=True)
                tmp_file = tempfile.NamedTemporaryFile("wb", dir=cache_dir, delete=False)
                writer = ArrowWriter(
                    features=writer_features,
                    path=tmp_file.name,
                    writer_batch_size=writer_batch_size,
                    update_features=update_features,
                    fingerprint=new_fingerprint,
                    disable_nullable=disable_nullable,
                )
            return buf_writer, writer, tmp_file

        tasks: list[asyncio.Task] = []
        if inspect.iscoroutinefunction(function):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
        else:
            loop = None

        def iter_outputs(shard_iterable):
            nonlocal tasks, loop
            if inspect.iscoroutinefunction(function):
                indices: Union[list[int], list[list[int]]] = []
                for i, example in shard_iterable:
                    indices.append(i)
                    tasks.append(loop.create_task(async_apply_function(example, i, offset=offset)))
                    if len(tasks) >= data_config.MAX_NUM_RUNNING_ASYNC_MAP_FUNCTIONS_IN_PARALLEL:
                        _, pending = loop.run_until_complete(asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED))
                        while tasks and len(pending) >= data_config.MAX_NUM_RUNNING_ASYNC_MAP_FUNCTIONS_IN_PARALLEL:
                            _, pending = loop.run_until_complete(asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED))
                    while tasks and tasks[0].done():
                        yield indices.pop(0), tasks.pop(0).result()
                while tasks:
                    yield indices[0], loop.run_until_complete(tasks[0])
                    indices.pop(0), tasks.pop(0)
            else:
                for i, example in shard_iterable:
                    yield i, apply_function(example, i, offset=offset)

        num_examples_progress_update = 0
        buf_writer, writer, tmp_file = None, None, None

        with contextlib.ExitStack() as stack:
            try:
                arrow_formatted_shard = shard.with_format("arrow")
                if not batched:
                    shard_iterable = enumerate(arrow_formatted_shard)
                else:
                    num_rows = len(shard) if not drop_last_batch else len(shard) // batch_size * batch_size
                    shard_iterable = zip(
                        (list(range(i, min(i + batch_size, num_rows))) for i in range(0, num_rows, batch_size)),
                        arrow_formatted_shard.iter(batch_size, drop_last_batch=drop_last_batch),
                    )
                if not batched:
                    _time = time.time()
                    for i, example in iter_outputs(shard_iterable):
                        if update_data:
                            if i == 0:
                                buf_writer, writer, tmp_file = init_buffer_and_writer()
                                stack.enter_context(writer)
                            if isinstance(example, pa.Table):
                                writer.write_row(example)
                            elif isinstance(example, pd.DataFrame):
                                writer.write_row(pa.Table.from_pandas(example))
                            else:
                                writer.write(example)
                        num_examples_progress_update += 1
                        if time.time() > _time + data_config.PBAR_REFRESH_TIME_INTERVAL:
                            _time = time.time()
                            yield rank, False, num_examples_progress_update
                            num_examples_progress_update = 0
                else:
                    _time = time.time()
                    for i, batch in iter_outputs(shard_iterable):
                        num_examples_in_batch = len(i)
                        if update_data:
                            if i and i[0] == 0:
                                buf_writer, writer, tmp_file = init_buffer_and_writer()
                                stack.enter_context(writer)
                            if isinstance(batch, pa.Table):
                                writer.write_table(batch)
                            elif isinstance(batch, pd.DataFrame):
                                writer.write_table(pa.Table.from_pandas(batch))
                           
                            else:
                                writer.write_batch(batch)
                        num_examples_progress_update += num_examples_in_batch
                        if time.time() > _time + data_config.PBAR_REFRESH_TIME_INTERVAL:
                            _time = time.time()
                            yield rank, False, num_examples_progress_update
                            num_examples_progress_update = 0
                if update_data and writer is not None:
                    writer.finalize()  # close_stream=bool(buf_writer is None))  # We only close if we are writing in a file
            except (Exception, KeyboardInterrupt):
                yield rank, False, num_examples_progress_update
                if update_data:
                    if writer is not None:
                        writer.finalize()
                    if tmp_file is not None:
                        tmp_file.close()
                        if os.path.exists(tmp_file.name):
                            os.remove(tmp_file.name)
                if loop:
                    for task in tasks:
                        task.cancel(msg="KeyboardInterrupt")
                    try:
                        loop.run_until_complete(asyncio.gather(*tasks))
                    except (asyncio.CancelledError, ValueError):
                        print("Tasks canceled.")
                raise

        yield rank, False, num_examples_progress_update
        if update_data and tmp_file is not None:
            tmp_file.close()
            shutil.move(tmp_file.name, cache_file_name)
            umask = os.umask(0o666)
            os.umask(umask)
            os.chmod(cache_file_name, 0o666 & ~umask)

        if update_data:
            # Create new Dataset from buffer or file
            info = shard.info.copy()
            info.features = writer._features
            if buf_writer is None:
                yield rank, True, Dataset.from_file(cache_file_name, info=info, split=shard.split)
            else:
                yield rank, True, Dataset.from_buffer(buf_writer.getvalue(), info=info, split=shard.split)
        else:
            yield rank, True, shard

    @transmit_format
    @fingerprint_transform(inplace=False)
    def batch(
        self,
        batch_size: int,
        drop_last_batch: bool = False,
        num_proc: Optional[int] = None,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":

        def batch_fn(example):
            return {k: [v] for k, v in example.items()}

        return self.map(
            batch_fn,
            batched=True,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            num_proc=num_proc,
            new_fingerprint=new_fingerprint,
            desc="Batching examples",
        )

    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["load_from_cache_file", "cache_file_name", "desc"], version="2.0.1")
    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, list[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> "Dataset":
        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError("Using `.filter` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it.`")

        if function is None:
            function = lambda x: True  # noqa: E731

        if len(self) == 0:
            return self

        if inspect.iscoroutinefunction(function) and not batched:
            batch_size = 1

        indices = self.map(
            function=partial(
                async_get_indices_from_mask_function
                if inspect.iscoroutinefunction(function)
                else get_indices_from_mask_function,
                function,
                batched,
                with_indices,
                with_rank,
                input_columns,
                self._indices,
            ),
            with_indices=True,
            with_rank=True,
            features=Features({"indices": Value("uint64")}),
            batched=True,
            batch_size=batch_size,
            remove_columns=self.column_names,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
            suffix_template=suffix_template,
            new_fingerprint=new_fingerprint,
            input_columns=input_columns,
            desc=desc or "Filter",
        )
        new_dataset = copy.deepcopy(self)
        new_dataset._indices = indices.data
        new_dataset._fingerprint = new_fingerprint
        return new_dataset

    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["cache_file_name"])
    def flatten_indices(
        self,
        keep_in_memory: bool = False,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        num_proc: Optional[int] = None,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":
        return self.map(
            batched=True,  # for speed
            keep_in_memory=keep_in_memory,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            features=features,
            disable_nullable=disable_nullable,
            new_fingerprint=new_fingerprint,
            desc="Flattening the indices",
            num_proc=num_proc,
        )

    def _new_dataset_with_indices(self, indices_cache_file_name: Optional[str] = None, indices_buffer: Optional[pa.Buffer] = None, fingerprint: Optional[str] = None) -> "Dataset":
        if indices_cache_file_name is None and indices_buffer is None:
            raise ValueError("At least one of indices_cache_file_name or indices_buffer must be provided.")

        if fingerprint is None:
            raise ValueError("please specify a fingerprint for the dataset with indices")

        if indices_cache_file_name is not None:
            indices_table = MemoryMappedTable.from_file(indices_cache_file_name)
        else:
            indices_table = InMemoryTable.from_buffer(indices_buffer)

        return Dataset(
            self._data,
            info=self.info.copy(),
            split=self.split,
            indices_table=indices_table,
            fingerprint=fingerprint,
        )

    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["indices_cache_file_name"])
    def select(
        self,
        indices: Iterable,
        keep_in_memory: bool = False,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":
        if keep_in_memory and indices_cache_file_name is not None:
            raise ValueError("Please use either `keep_in_memory` or `indices_cache_file_name` but not both.")

        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError("Using `.select` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it.")

        if len(self) == 0:
            return self

        if isinstance(indices, (pa.Array, pa.ChunkedArray)):
            indices = indices.to_numpy().astype(np.int64)

        if isinstance(indices, Iterator):
            indices = list(indices)

        if isinstance(indices, range):
            if _is_range_contiguous(indices) and indices.start >= 0:
                start, length = indices.start, indices.stop - indices.start
                return self._select_contiguous(start, length, new_fingerprint=new_fingerprint)
        else:
            try:
                start = next(iter(indices))
            except StopIteration:
                return self._select_contiguous(0, 0, new_fingerprint=new_fingerprint)
            if start >= 0:
                counter_from_start = itertools.count(start=start)
                if all(i == j for i, j in zip(indices, counter_from_start)):
                    length = next(counter_from_start) - start
                    return self._select_contiguous(start, length, new_fingerprint=new_fingerprint)

        return self._select_with_indices_mapping(
            indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
        )

    @transmit_format
    @fingerprint_transform(inplace=False)
    def _select_contiguous(self, start: int, length: int, new_fingerprint: Optional[str] = None) -> "Dataset":
        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError("Using `.select` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it.")

        if len(self) == 0:
            return self

        _check_valid_indices_value(start, len(self))
        _check_valid_indices_value(start + length - 1, len(self))
        if self._indices is None or length == 0:
            return Dataset(
                self.data.slice(start, length),
                info=self.info.copy(),
                split=self.split,
                fingerprint=new_fingerprint,
            )
        else:
            return Dataset(
                self.data,
                info=self.info.copy(),
                split=self.split,
                indices_table=self._indices.slice(start, length),
                fingerprint=new_fingerprint,
            )

    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["indices_cache_file_name"])
    def _select_with_indices_mapping(
        self,
        indices: Iterable,
        keep_in_memory: bool = False,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":
        if keep_in_memory and indices_cache_file_name is not None:
            raise ValueError("Please use either `keep_in_memory` or `indices_cache_file_name` but not both.")

        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError("Using `.select` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it.")

        if len(self) == 0:
            return self

        if keep_in_memory or indices_cache_file_name is None:
            buf_writer = pa.BufferOutputStream()
            tmp_file = None
            writer = ArrowWriter(stream=buf_writer, writer_batch_size=writer_batch_size, fingerprint=new_fingerprint, unit="indices")
        else:
            buf_writer = None
            cache_dir = os.path.dirname(indices_cache_file_name)
            os.makedirs(cache_dir, exist_ok=True)
            tmp_file = tempfile.NamedTemporaryFile("wb", dir=cache_dir, delete=False)
            writer = ArrowWriter(path=tmp_file.name, writer_batch_size=writer_batch_size, fingerprint=new_fingerprint, unit="indices")

        indices = indices if isinstance(indices, list) else list(indices)

        size = len(self)
        if indices:
            _check_valid_indices_value(int(max(indices)), size=size)
            _check_valid_indices_value(int(min(indices)), size=size)
        else:
            return self._select_contiguous(0, 0, new_fingerprint=new_fingerprint)

        indices_array = pa.array(indices, type=pa.uint64())
        if self._indices is not None:
            indices_array = self._indices.column(0).take(indices_array)

        indices_table = pa.Table.from_arrays([indices_array], names=["indices"])

        with writer:
            try:
                writer.write_table(indices_table)
                writer.finalize()  # close_stream=bool(buf_writer is None))  We only close if we are writing in a file
            except (Exception, KeyboardInterrupt):
                if tmp_file is not None:
                    tmp_file.close()
                    if os.path.exists(tmp_file.name):
                        os.remove(tmp_file.name)
                raise

        if tmp_file is not None:
            tmp_file.close()
            shutil.move(tmp_file.name, indices_cache_file_name)
            umask = os.umask(0o666)
            os.umask(umask)
            os.chmod(indices_cache_file_name, 0o666 & ~umask)

        if buf_writer is None:
            return self._new_dataset_with_indices(indices_cache_file_name=indices_cache_file_name, fingerprint=new_fingerprint)
        else:
            return self._new_dataset_with_indices(indices_buffer=buf_writer.getvalue(), fingerprint=new_fingerprint)

    def skip(self, n: int) -> "Dataset":
        return self.select(range(n, len(self)))

    def repeat(self, num_times: int) -> "Dataset":
        if num_times is None:
            raise ValueError("Map style datasets do not support indefinite repetition.")
        return _concatenate_map_style_datasets([self] * num_times) if num_times > 0 else self.select([])

    def take(self, n: int) -> "Dataset":
        return self.select(range(n))

    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["load_from_cache_file", "indices_cache_file_name"])
    def sort(
        self,
        column_names: Union[str, Sequence[str]],
        reverse: Union[bool, Sequence[bool]] = False,
        null_placement: str = "at_end",
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":
        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError("Using `.sort` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it.")
        if len(self) == 0:
            return self
        if isinstance(column_names, str):
            column_names = [column_names]

        if not isinstance(reverse, bool):
            if len(reverse) != len(column_names):
                raise ValueError("Parameter 'reverse' should be either a boolean or a list of booleans with the same length as 'column_names'.")
        else:
            reverse = [reverse] * len(column_names)

        for column in column_names:
            if not isinstance(column, str) or column not in self._data.column_names:
                raise ValueError(f"Column '{column}' not found in the dataset. Please provide a column selected in: {self._data.column_names}")

        if null_placement not in ["at_start", "at_end"]:
            if null_placement == "first":
                null_placement = "at_start"
            elif null_placement == "last":
                null_placement = "at_end"
            else:
                raise ValueError(f"null_placement '{null_placement}' is an invalid parameter value. Must be either 'last', 'at_end', 'first' or 'at_start'.")

        load_from_cache_file = load_from_cache_file if load_from_cache_file is not None else is_caching_enabled()
        if self.cache_files:
            if indices_cache_file_name is None:
                indices_cache_file_name = self._get_cache_file_path(new_fingerprint)
            if os.path.exists(indices_cache_file_name) and load_from_cache_file:
                return self._new_dataset_with_indices(fingerprint=new_fingerprint, indices_cache_file_name=indices_cache_file_name)

        sort_table = query_table(table=self._data, key=slice(0, len(self)), indices=self._indices,)
        sort_keys = [(col, "ascending" if not col_reverse else "descending") for col, col_reverse in zip(column_names, reverse)]
        indices = pc.sort_indices(sort_table, sort_keys=sort_keys, null_placement=null_placement)

        return self.select(
            indices=indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
        )

    @transmit_format
    @fingerprint_transform(inplace=False, randomized_function=True, ignore_kwargs=["load_from_cache_file", "indices_cache_file_name"])
    def shuffle(
        self,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":
        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError("Using `.shuffle` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it.")
        if len(self) == 0:
            return self

        if keep_in_memory and indices_cache_file_name is not None:
            raise ValueError("Please use either `keep_in_memory` or `indices_cache_file_name` but not both.")

        if seed is not None and generator is not None:
            raise ValueError("Both `seed` and `generator` were provided. Please specify just one of them.")

        if generator is not None and not isinstance(generator, np.random.Generator):
            raise ValueError("The provided generator must be an instance of numpy.random.Generator")

        load_from_cache_file = load_from_cache_file if load_from_cache_file is not None else is_caching_enabled()

        if generator is None:
            if seed is None:
                _, seed, pos, *_ = np.random.get_state()
                seed = seed[pos] if pos < 624 else seed[0]
                _ = np.random.random()  # do 1 step of rng
            generator = np.random.default_rng(seed)

        if self.cache_files:
            if indices_cache_file_name is None:
                indices_cache_file_name = self._get_cache_file_path(new_fingerprint)
            if os.path.exists(indices_cache_file_name) and load_from_cache_file:
                return self._new_dataset_with_indices(fingerprint=new_fingerprint, indices_cache_file_name=indices_cache_file_name)

        permutation = generator.permutation(len(self))

        return self.select(
            indices=permutation,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name if not keep_in_memory else None,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
        )

    @transmit_format
    @fingerprint_transform(
        inplace=False,
        randomized_function=True,
        fingerprint_names=["train_new_fingerprint", "test_new_fingerprint"],
        ignore_kwargs=["load_from_cache_file", "train_indices_cache_file_name", "test_indices_cache_file_name"],
    )
    def train_test_split(
        self,
        test_size: Union[float, int, None] = None,
        train_size: Union[float, int, None] = None,
        shuffle: bool = True,
        stratify_by_column: Optional[str] = None,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        train_indices_cache_file_name: Optional[str] = None,
        test_indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        train_new_fingerprint: Optional[str] = None,
        test_new_fingerprint: Optional[str] = None,
    ) -> "DatasetDict":
        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError("Using `.train_test_split` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it.")
        if len(self) == 0:
            return DatasetDict({"train": self, "test": self})

        if test_size is None and train_size is None:
            test_size = 0.25

        n_samples = len(self)
        if (isinstance(test_size, int) and (test_size >= n_samples or test_size <= 0) or isinstance(test_size, float) and (test_size <= 0 or test_size >= 1)):
            raise ValueError(f"test_size={test_size} should be either positive and smaller than the number of samples {n_samples} or a float in the (0, 1) range")

        if (isinstance(train_size, int) and (train_size >= n_samples or train_size <= 0) or isinstance(train_size, float) and (train_size <= 0 or train_size >= 1)):
            raise ValueError(f"train_size={train_size} should be either positive and smaller than the number of samples {n_samples} or a float in the (0, 1) range")

        if train_size is not None and not isinstance(train_size, (int, float)):
            raise ValueError(f"Invalid value for train_size: {train_size} of type {type(train_size)}")
        if test_size is not None and not isinstance(test_size, (int, float)):
            raise ValueError(f"Invalid value for test_size: {test_size} of type {type(test_size)}")

        if isinstance(train_size, float) and isinstance(test_size, float) and train_size + test_size > 1:
            raise ValueError(f"The sum of test_size and train_size = {train_size + test_size}, should be in the (0, 1) range. Reduce test_size and/or train_size.")

        if isinstance(test_size, float):
            n_test = ceil(test_size * n_samples)
        elif isinstance(test_size, int):
            n_test = float(test_size)

        if isinstance(train_size, float):
            n_train = floor(train_size * n_samples)
        elif isinstance(train_size, int):
            n_train = float(train_size)

        if train_size is None:
            n_train = n_samples - n_test
        elif test_size is None:
            n_test = n_samples - n_train

        if n_train + n_test > n_samples:
            raise ValueError(f"The sum of train_size and test_size = {n_train + n_test}, should be smaller than the number of samples {n_samples}. Reduce test_size and/or train_size")

        n_train, n_test = int(n_train), int(n_test)

        if n_train == 0:
            raise ValueError(f"With n_samples={n_samples}, test_size={test_size} and train_size={train_size}, the resulting train set will be empty. Adjust any of the aforementioned parameters.")

        load_from_cache_file = load_from_cache_file if load_from_cache_file is not None else is_caching_enabled()

        if generator is None and shuffle is True:
            if seed is None:
                _, seed, pos, *_ = np.random.get_state()
                seed = seed[pos] if pos < 624 else seed[0]
                _ = np.random.random()  # do 1 step of rng
            generator = np.random.default_rng(seed)

        if self.cache_files:
            if train_indices_cache_file_name is None or test_indices_cache_file_name is None:

                if train_indices_cache_file_name is None:
                    train_indices_cache_file_name = self._get_cache_file_path(train_new_fingerprint)
                if test_indices_cache_file_name is None:
                    test_indices_cache_file_name = self._get_cache_file_path(test_new_fingerprint)
            if (os.path.exists(train_indices_cache_file_name) and os.path.exists(test_indices_cache_file_name) and load_from_cache_file):
                return DatasetDict(
                    {
                        "train": self._new_dataset_with_indices(fingerprint=train_new_fingerprint, indices_cache_file_name=train_indices_cache_file_name),
                        "test": self._new_dataset_with_indices(fingerprint=test_new_fingerprint, indices_cache_file_name=test_indices_cache_file_name),
                    }
                )
        if not shuffle:
            if stratify_by_column is not None:
                raise ValueError("Stratified train/test split is not implemented for `shuffle=False`")
            train_indices = np.arange(n_train)
            test_indices = np.arange(n_train, n_train + n_test)
        else:
            # stratified partition
            if stratify_by_column is not None:
                if stratify_by_column not in self._info.features.keys():
                    raise ValueError(f"Key {stratify_by_column} not found in {self._info.features.keys()}")
                try:
                    train_indices, test_indices = next(stratified_shuffle_split_generate_indices(self.with_format("numpy")[stratify_by_column], n_train, n_test, rng=generator))
                except Exception as error:
                    if str(error) == "Minimum class count error":
                        raise ValueError(f"The least populated class in {stratify_by_column} column has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.")
                    else:
                        raise error

            else:
                permutation = generator.permutation(len(self))
                test_indices = permutation[:n_test]
                train_indices = permutation[n_test : (n_test + n_train)]

        train_split = self.select(
            indices=train_indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=train_indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=train_new_fingerprint,
        )
        test_split = self.select(
            indices=test_indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=test_indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=test_new_fingerprint,
        )

        return DatasetDict({"train": train_split, "test": test_split})

    def shard(
        self,
        num_shards: int,
        index: int,
        contiguous: bool = True,
        keep_in_memory: bool = False,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
    ) -> "Dataset":
        if not 0 <= index < num_shards:
            raise ValueError("index should be in [0, num_shards-1]")
        if contiguous:
            div = len(self) // num_shards
            mod = len(self) % num_shards
            start = div * index + min(index, mod)
            end = start + div + (1 if index < mod else 0)
            indices = range(start, end)
        else:
            indices = np.arange(index, len(self), num_shards)

        return self.select(
            indices=indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
        )

    def to_json(
        self,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **to_json_kwargs,
    ) -> int:
        return JsonDatasetWriter(
            self,
            path_or_buf,
            batch_size=batch_size,
            num_proc=num_proc,
            storage_options=storage_options,
            **to_json_kwargs,
        ).write()

    def to_pandas(self, batch_size: Optional[int] = None, batched: bool = False) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        if not batched:
            return query_table(
                table=self._data,
                key=slice(0, len(self)),
                indices=self._indices,
            ).to_pandas(types_mapper=pandas_types_mapper)
        else:
            batch_size = batch_size if batch_size else data_config.DEFAULT_MAX_BATCH_SIZE
            return (
                query_table(
                    table=self._data,
                    key=slice(offset, offset + batch_size),
                    indices=self._indices,
                ).to_pandas(types_mapper=pandas_types_mapper)
                for offset in range(0, len(self), batch_size)
            )

    def _estimate_nbytes(self) -> int:
        dataset_nbytes = self.data.nbytes
        decodable_columns = [k for k, v in self._info.features.items() if require_decoding(v, ignore_decode_attribute=True)]

        if decodable_columns:
            extra_nbytes = 0

            def extra_nbytes_visitor(array, feature):
                nonlocal extra_nbytes
                if isinstance(feature, (Image)):
                    for x in array.to_pylist():
                        if x is not None and x["bytes"] is None and x["path"] is not None:
                            size = xgetsize(x["path"])
                            extra_nbytes += size
                    extra_nbytes -= array.field("path").nbytes

            table = self.with_format("arrow")[:1000]
            table_visitor(table, extra_nbytes_visitor)

            extra_nbytes = extra_nbytes * len(self.data) / len(table)
            dataset_nbytes = dataset_nbytes + extra_nbytes

        if self._indices is not None:
            dataset_nbytes = dataset_nbytes * len(self._indices) / len(self.data)
        return dataset_nbytes

    @staticmethod
    def _generate_tables_from_shards(shards: list["Dataset"], batch_size: int):
        for shard_idx, shard in enumerate(shards):
            for pa_table in shard.with_format("arrow").iter(batch_size):
                yield shard_idx, pa_table

    @staticmethod
    def _generate_tables_from_cache_file(filename: str):
        for batch_idx, batch in enumerate(_memory_mapped_record_batch_reader_from_file(filename)):
            yield batch_idx, pa.Table.from_batches([batch])

    def to_iterable_dataset(self, num_shards: Optional[int] = 1) -> "IterableDataset":
        if self._format_type is not None:
            if self._format_kwargs or (self._format_columns is not None and set(self._format_columns) != set(self.column_names)):
                raise NotImplementedError("Converting a formatted dataset with kwargs or selected columns to a formatted iterable dataset is not implemented yet. Please run `my_dataset = my_dataset.with_format(None)` before calling to_iterable_dataset")
        if num_shards > len(self):
            raise ValueError(f"Unable to shard a dataset of size {len(self)} into {num_shards} shards (the number of shards exceeds the number of samples).")
        shards = (
            [copy.deepcopy(self)]
            if num_shards == 1
            else [self.shard(num_shards=num_shards, index=shard_idx, contiguous=True) for shard_idx in range(num_shards)]
        )
        ex_iterable = ArrowExamplesIterable(Dataset._generate_tables_from_shards, kwargs={"shards": shards, "batch_size": data_config.DEFAULT_MAX_BATCH_SIZE},)
        ds = IterableDataset(ex_iterable, info=DatasetInfo(features=self.features))
        if self._format_type:
            ds = ds.with_format(self._format_type)
        return ds

    @transmit_format
    @fingerprint_transform(inplace=False)
    def add_column(self, name: str, column: Union[list, np.array], new_fingerprint: str, feature: Optional[FeatureType] = None):
        if feature:
            pyarrow_schema = Features({name: feature}).arrow_schema
        else:
            pyarrow_schema = None

        column_table = InMemoryTable.from_pydict({name: column}, schema=pyarrow_schema)
        _check_column_names(self._data.column_names + column_table.column_names)
        dataset = self.flatten_indices() if self._indices is not None else self
        table = concat_tables([dataset._data, column_table], axis=1)
        info = dataset.info.copy()
        info.features.update(Features.from_arrow_schema(column_table.schema))
        table = update_metadata_with_features(table, info.features)
        return Dataset(table, info=info, split=self.split, indices_table=None, fingerprint=new_fingerprint)

    @transmit_format
    @fingerprint_transform(inplace=False)
    def add_item(self, item: dict, new_fingerprint: str):
        item_table = InMemoryTable.from_pydict({k: [v] for k, v in item.items()})
        dset_features, item_features = _align_features([self._info.features, Features.from_arrow_schema(item_table.schema)])
        table = concat_tables(
            [
                self._data.cast(dset_features.arrow_schema) if self._info.features != dset_features else self._data,
                item_table.cast(item_features.arrow_schema),
            ]
        )
        if self._indices is None:
            indices_table = None
        else:
            item_indices_array = pa.array([len(self._data)], type=pa.uint64())
            item_indices_table = InMemoryTable.from_arrays([item_indices_array], names=["indices"])
            indices_table = concat_tables([self._indices, item_indices_table])
        info = self.info.copy()
        info.features.update(item_features)
        table = update_metadata_with_features(table, info.features)
        return Dataset(
            table,
            info=info,
            split=self.split,
            indices_table=indices_table,
            fingerprint=new_fingerprint,
        )

class classproperty(property):  # pylint: disable=invalid-name
    def __get__(self, obj, objtype=None):
        return self.fget.__get__(None, objtype)()

class bind(partial):
    def __call__(self, *fn_args, **fn_kwargs):
        return self.func(*fn_args, *self.args, **fn_kwargs)

class DatasetDict(dict):
    def _check_values_type(self):
        for dataset in self.values():
            if not isinstance(dataset, Dataset):
                raise TypeError(f"Values in `DatasetDict` should be of type `Dataset` but got type '{type(dataset)}'")

    def _check_values_features(self):
        items = list(self.items())
        for item_a, item_b in zip(items[:-1], items[1:]):
            if item_a[1].features != item_b[1].features:
                raise ValueError(f"All datasets in `DatasetDict` should have the same features but features for '{item_a[0]}' and '{item_b[0]}' don't match: {item_a[1].features} != {item_b[1].features}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for dataset in self.values():
            if hasattr(dataset, "_data"):
                del dataset._data
            if hasattr(dataset, "_indices"):
                del dataset._indices

    def __getitem__(self, k) -> Dataset:
        if isinstance(k, (str, NamedSplit)) or len(self) == 0:
            return super().__getitem__(k)
        else:
            available_suggested_splits = [split for split in (Split.TRAIN, Split.TEST, Split.VALIDATION) if split in self]
            suggested_split = available_suggested_splits[0] if available_suggested_splits else list(self)[0]
            raise KeyError(f"Invalid key: {k}. Please first select a split. For example: `my_dataset_dictionary['{suggested_split}'][{k}]`. Available splits: {sorted(self)}")

    @property
    def data(self) -> dict[str, Table]:
        self._check_values_type()
        return {k: dataset.data for k, dataset in self.items()}

    @property
    def cache_files(self) -> dict[str, dict]:
        self._check_values_type()
        return {k: dataset.cache_files for k, dataset in self.items()}

    @property
    def num_columns(self) -> dict[str, int]:
        self._check_values_type()
        return {k: dataset.num_columns for k, dataset in self.items()}

    @property
    def num_rows(self) -> dict[str, int]:
        self._check_values_type()
        return {k: dataset.num_rows for k, dataset in self.items()}

    @property
    def column_names(self) -> dict[str, list[str]]:
        self._check_values_type()
        return {k: dataset.column_names for k, dataset in self.items()}

    @property
    def shape(self) -> dict[str, tuple[int]]:
        self._check_values_type()
        return {k: dataset.shape for k, dataset in self.items()}

    def flatten(self, max_depth=16) -> "DatasetDict":
        self._check_values_type()
        return DatasetDict({k: dataset.flatten(max_depth=max_depth) for k, dataset in self.items()})

    def unique(self, column: str) -> dict[str, list]:
        self._check_values_type()
        return {k: dataset.unique(column) for k, dataset in self.items()}

    def cleanup_cache_files(self) -> dict[str, int]:
        self._check_values_type()
        return {k: dataset.cleanup_cache_files() for k, dataset in self.items()}

    def __repr__(self):
        repr = "\n".join([f"{k}: {v}" for k, v in self.items()])
        repr = re.sub(r"^", " " * 4, repr, 0, re.M)
        return f"DatasetDict({{\n{repr}\n}})"

    def cast(self, features: Features) -> "DatasetDict":
        self._check_values_type()
        return DatasetDict({k: dataset.cast(features=features) for k, dataset in self.items()})

    def cast_column(self, column: str, feature) -> "DatasetDict":
        self._check_values_type()
        return DatasetDict({k: dataset.cast_column(column=column, feature=feature) for k, dataset in self.items()})

    def remove_columns(self, column_names: Union[str, list[str]]) -> "DatasetDict":
        self._check_values_type()
        return DatasetDict({k: dataset.remove_columns(column_names=column_names) for k, dataset in self.items()})

    def rename_column(self, original_column_name: str, new_column_name: str) -> "DatasetDict":
        self._check_values_type()
        return DatasetDict(
            {
                k: dataset.rename_column(
                    original_column_name=original_column_name,
                    new_column_name=new_column_name,
                )
                for k, dataset in self.items()
            }
        )

    def rename_columns(self, column_mapping: dict[str, str]) -> "DatasetDict":
        self._check_values_type()
        return DatasetDict({k: dataset.rename_columns(column_mapping=column_mapping) for k, dataset in self.items()})

    def select_columns(self, column_names: Union[str, list[str]]) -> "DatasetDict":
        self._check_values_type()
        return DatasetDict({k: dataset.select_columns(column_names=column_names) for k, dataset in self.items()})

    def class_encode_column(self, column: str, include_nulls: bool = False) -> "DatasetDict":
        self._check_values_type()
        return DatasetDict(
            {k: dataset.class_encode_column(column=column, include_nulls=include_nulls) for k, dataset in self.items()}
        )

    @contextlib.contextmanager
    def formatted_as(
        self,
        type: Optional[str] = None,
        columns: Optional[list] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ):
        self._check_values_type()
        old_format_type = {k: dataset._format_type for k, dataset in self.items()}
        old_format_kwargs = {k: dataset._format_kwargs for k, dataset in self.items()}
        old_format_columns = {k: dataset._format_columns for k, dataset in self.items()}
        old_output_all_columns = {k: dataset._output_all_columns for k, dataset in self.items()}
        try:
            self.set_format(type, columns, output_all_columns, **format_kwargs)
            yield
        finally:
            for k, dataset in self.items():
                dataset.set_format(
                    old_format_type[k],
                    old_format_columns[k],
                    old_output_all_columns[k],
                    **old_format_kwargs[k],
                )

    def set_format(self, type: Optional[str] = None, columns: Optional[list] = None, output_all_columns: bool = False, **format_kwargs):
        self._check_values_type()
        for dataset in self.values():
            dataset.set_format(
                type=type,
                columns=columns,
                output_all_columns=output_all_columns,
                **format_kwargs,
            )

    def reset_format(self):
        self._check_values_type()
        for dataset in self.values():
            dataset.set_format()

    def set_transform(
        self,
        transform: Optional[Callable],
        columns: Optional[list] = None,
        output_all_columns: bool = False,
    ):
        self._check_values_type()
        for dataset in self.values():
            dataset.set_format(
                "custom",
                columns=columns,
                output_all_columns=output_all_columns,
                transform=transform,
            )

    def with_format(
        self,
        type: Optional[str] = None,
        columns: Optional[list] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ) -> "DatasetDict":
        dataset = copy.deepcopy(self)
        dataset.set_format(
            type=type,
            columns=columns,
            output_all_columns=output_all_columns,
            **format_kwargs,
        )
        return dataset

    def with_transform(
        self,
        transform: Optional[Callable],
        columns: Optional[list] = None,
        output_all_columns: bool = False,
    ) -> "DatasetDict":
        dataset = copy.deepcopy(self)
        dataset.set_transform(transform=transform, columns=columns, output_all_columns=output_all_columns)
        return dataset

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        with_split: bool = False,
        input_columns: Optional[Union[str, list[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, list[str]]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_names: Optional[dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        desc: Optional[str] = None,
    ) -> "DatasetDict":
        self._check_values_type()
        if cache_file_names is None:
            cache_file_names = dict.fromkeys(self)

        dataset_dict = {}
        for split, dataset in self.items():
            if with_split:
                function = bind(function, split)

            dataset_dict[split] = dataset.map(
                function=function,
                with_indices=with_indices,
                with_rank=with_rank,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                remove_columns=remove_columns,
                keep_in_memory=keep_in_memory,
                load_from_cache_file=load_from_cache_file,
                cache_file_name=cache_file_names[split],
                writer_batch_size=writer_batch_size,
                features=features,
                disable_nullable=disable_nullable,
                fn_kwargs=fn_kwargs,
                num_proc=num_proc,
                desc=desc,
            )

            if with_split:
                function = function.func

        return DatasetDict(dataset_dict)

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, list[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_names: Optional[dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        desc: Optional[str] = None,
    ) -> "DatasetDict":
        self._check_values_type()
        if cache_file_names is None:
            cache_file_names = dict.fromkeys(self)
        return DatasetDict(
            {
                k: dataset.filter(
                    function=function,
                    with_indices=with_indices,
                    with_rank=with_rank,
                    input_columns=input_columns,
                    batched=batched,
                    batch_size=batch_size,
                    keep_in_memory=keep_in_memory,
                    load_from_cache_file=load_from_cache_file,
                    cache_file_name=cache_file_names[k],
                    writer_batch_size=writer_batch_size,
                    fn_kwargs=fn_kwargs,
                    num_proc=num_proc,
                    desc=desc,
                )
                for k, dataset in self.items()
            }
        )

    def flatten_indices(
        self,
        keep_in_memory: bool = False,
        cache_file_names: Optional[dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        num_proc: Optional[int] = None,
        new_fingerprint: Optional[str] = None,
    ) -> "DatasetDict":
        self._check_values_type()
        if cache_file_names is None:
            cache_file_names = dict.fromkeys(self)
        return DatasetDict(
            {
                k: dataset.flatten_indices(
                    keep_in_memory=keep_in_memory,
                    cache_file_name=cache_file_names[k],
                    writer_batch_size=writer_batch_size,
                    features=features,
                    disable_nullable=disable_nullable,
                    num_proc=num_proc,
                    new_fingerprint=new_fingerprint,
                )
                for k, dataset in self.items()
            }
        )

    def sort(
        self,
        column_names: Union[str, Sequence[str]],
        reverse: Union[bool, Sequence[bool]] = False,
        null_placement: str = "at_end",
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        indices_cache_file_names: Optional[dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
    ) -> "DatasetDict":
        self._check_values_type()
        if indices_cache_file_names is None:
            indices_cache_file_names = dict.fromkeys(self)
        return DatasetDict(
            {
                k: dataset.sort(
                    column_names=column_names,
                    reverse=reverse,
                    null_placement=null_placement,
                    keep_in_memory=keep_in_memory,
                    load_from_cache_file=load_from_cache_file,
                    indices_cache_file_name=indices_cache_file_names[k],
                    writer_batch_size=writer_batch_size,
                )
                for k, dataset in self.items()
            }
        )

    def shuffle(
        self,
        seeds: Optional[Union[int, dict[str, Optional[int]]]] = None,
        seed: Optional[int] = None,
        generators: Optional[dict[str, np.random.Generator]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        indices_cache_file_names: Optional[dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
    ) -> "DatasetDict":
        self._check_values_type()
        if seed is not None and seeds is not None:
            raise ValueError("Please specify seed or seeds, but not both")
        seeds = seed if seed is not None else seeds
        if seeds is None:
            seeds = dict.fromkeys(self)
        elif not isinstance(seeds, dict):
            seeds = dict.fromkeys(self, seeds)
        if generators is None:
            generators = dict.fromkeys(self)
        if indices_cache_file_names is None:
            indices_cache_file_names = dict.fromkeys(self)
        return DatasetDict(
            {
                k: dataset.shuffle(
                    seed=seeds[k],
                    generator=generators[k],
                    keep_in_memory=keep_in_memory,
                    load_from_cache_file=load_from_cache_file,
                    indices_cache_file_name=indices_cache_file_names[k],
                    writer_batch_size=writer_batch_size,
                )
                for k, dataset in self.items()
            }
        )

    def save_to_disk(
        self,
        dataset_dict_path: PathLike,
        max_shard_size: Optional[Union[str, int]] = None,
        num_shards: Optional[dict[str, int]] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
    ):
        fs: fsspec.AbstractFileSystem
        fs, _ = url_to_fs(dataset_dict_path, **(storage_options or {}))

        if num_shards is None:
            num_shards = dict.fromkeys(self)
        elif not isinstance(num_shards, dict):
            raise ValueError("Please provide one `num_shards` per dataset in the dataset dictionary, e.g. {{'train': 128, 'test': 4}}")

        fs.makedirs(dataset_dict_path, exist_ok=True)

        with fs.open(posixpath.join(dataset_dict_path, data_config.DATASETDICT_JSON_FILENAME), "w", encoding="utf-8") as f:
            json.dump({"splits": list(self)}, f)
        for k, dataset in self.items():
            dataset.save_to_disk(
                posixpath.join(dataset_dict_path, k),
                num_shards=num_shards.get(k),
                max_shard_size=max_shard_size,
                num_proc=num_proc,
                storage_options=storage_options,
            )

    @staticmethod
    def load_from_disk(dataset_dict_path: PathLike, keep_in_memory: Optional[bool] = None, storage_options: Optional[dict] = None,) -> "DatasetDict":
        fs: fsspec.AbstractFileSystem
        fs, dataset_dict_path = url_to_fs(dataset_dict_path, **(storage_options or {}))

        dataset_dict_json_path = posixpath.join(dataset_dict_path, data_config.DATASETDICT_JSON_FILENAME)
        dataset_state_json_path = posixpath.join(dataset_dict_path, data_config.DATASET_STATE_JSON_FILENAME)
        dataset_info_path = posixpath.join(dataset_dict_path, data_config.DATASET_INFO_FILENAME)
        if not fs.isfile(dataset_dict_json_path):
            if fs.isfile(dataset_info_path) and fs.isfile(dataset_state_json_path):
                raise FileNotFoundError(f"No such file: '{dataset_dict_json_path}'. Expected to load a `DatasetDict` object, but got a `Dataset`. Please use either `datasets.load_from_disk` or `Dataset.load_from_disk` instead.")
            raise FileNotFoundError(f"No such file: '{dataset_dict_json_path}'. Expected to load a `DatasetDict` object, but provided path is not a `DatasetDict`.")

        with fs.open(dataset_dict_json_path, "r", encoding="utf-8") as f:
            splits = json.load(f)["splits"]

        dataset_dict = DatasetDict()
        for k in splits:
            dataset_dict_split_path = posixpath.join(fs.unstrip_protocol(dataset_dict_path), k)
            dataset_dict[k] = Dataset.load_from_disk(
                dataset_dict_split_path,
                keep_in_memory=keep_in_memory,
                storage_options=storage_options,
            )
        return dataset_dict

    @is_documented_by(Dataset.align_labels_with_mapping)
    def align_labels_with_mapping(self, label2id: dict, label_column: str) -> "DatasetDict":
        self._check_values_type()
        return DatasetDict(
            {
                k: dataset.align_labels_with_mapping(label2id=label2id, label_column=label_column)
                for k, dataset in self.items()
            }
        )

def _concatenate_map_style_datasets(dsets: list[Dataset], info: Optional[DatasetInfo] = None, split: Optional[NamedSplit] = None, axis: int = 0):
    if any(dset.num_rows > 0 for dset in dsets):
        dsets = [dset for dset in dsets if dset.num_rows > 0]
    else:
        return dsets[0]

    if axis == 0:
        _check_if_features_can_be_aligned([dset.features for dset in dsets])
    else:
        if not all(dset.num_rows == dsets[0].num_rows for dset in dsets):
            raise ValueError("Number of rows must match for all datasets")
        _check_column_names([col_name for dset in dsets for col_name in dset._data.column_names])

    format = dsets[0].format
    if any(dset.format != format for dset in dsets):
        format = {}

    def apply_offset_to_indices_table(table, offset):
        if offset == 0:
            return table
        else:
            array = table["indices"]
            new_array = pc.add(array, pa.scalar(offset, type=pa.uint64()))
            return InMemoryTable.from_arrays([new_array], names=["indices"])

    if any(dset._indices is not None for dset in dsets):
        if axis == 0:
            indices_tables = []
            for i in range(len(dsets)):
                if dsets[i]._indices is None:
                    dsets[i] = dsets[i]._select_with_indices_mapping(range(len(dsets[i])))
                indices_tables.append(dsets[i]._indices)

            offset = 0
            for i in range(len(dsets)):
                indices_tables[i] = apply_offset_to_indices_table(indices_tables[i], offset)
                offset += len(dsets[i]._data)

            indices_tables = [t for t in indices_tables if len(t) > 0]
            if indices_tables:
                indices_table = concat_tables(indices_tables)
            else:
                indices_table = InMemoryTable.from_batches([], schema=pa.schema({"indices": pa.int64()}))
        else:
            if len(dsets) == 1:
                indices_table = dsets[0]._indices
            else:
                for i in range(len(dsets)):
                    dsets[i] = dsets[i].flatten_indices()
                indices_table = None
    else:
        indices_table = None

    table = concat_tables([dset._data for dset in dsets], axis=axis)
    if axis == 0:
        features_list = _align_features([dset.features for dset in dsets])
    else:
        features_list = [dset.features for dset in dsets]
    table = update_metadata_with_features(table, {k: v for features in features_list for k, v in features.items()})

    if info is None:
        info = DatasetInfo.from_merge([dset.info for dset in dsets])
    fingerprint = update_fingerprint("".join(dset._fingerprint for dset in dsets), _concatenate_map_style_datasets, {"info": info, "split": split})

    concatenated_dataset = Dataset(
        table,
        info=info,
        split=split,
        indices_table=indices_table,
        fingerprint=fingerprint,
    )
    concatenated_dataset.set_format(**format)
    return concatenated_dataset

@dataclass
class ShufflingConfig:
    generator: np.random.Generator
    _original_seed: Optional[int] = None

@dataclass
class DistributedConfig:
    rank: int
    world_size: int

class IterableDataset(DatasetInfoMixin):
    def __init__(
        self,
        ex_iterable: _BaseExamplesIterable,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        formatting: Optional[FormattingConfig] = None,
        shuffling: Optional[ShufflingConfig] = None,
        distributed: Optional[DistributedConfig] = None,
        token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None,
    ):
        if distributed and distributed.world_size > 1 and shuffling and shuffling._original_seed is None:
            raise RuntimeError("The dataset doesn't have a fixed random seed across nodes to shuffle and split the list of dataset shards by node. Please pass e.g. `seed=42` in `.shuffle()` to make all the nodes use the same seed.")

        info = info.copy() if info is not None else DatasetInfo()
        DatasetInfoMixin.__init__(self, info=info, split=split)

        self._ex_iterable = copy.copy(ex_iterable)
        self._formatting = formatting
        self._shuffling = shuffling
        self._distributed = distributed
        self._token_per_repo_id: dict[str, Union[str, bool, None]] = token_per_repo_id or {}
        self._epoch: Union[int, "torch.Tensor"] = _maybe_share_with_torch_persistent_workers(0)
        self._starting_state_dict: Optional[dict] = None
        self._prepare_ex_iterable_for_iteration()  # set state_dict
        _maybe_add_torch_iterable_dataset_parent_class(self.__class__)  # subclass of torch IterableDataset

    def state_dict(self) -> dict:
        return copy.deepcopy(self._state_dict)

    def load_state_dict(self, state_dict: dict) -> None:
        self._starting_state_dict = state_dict

    def __repr__(self):
        return f"IterableDataset({{\n    features: {list(self._info.features.keys()) if self._info.features is not None else 'Unknown'},\n    num_shards: {self.num_shards}\n}})"

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        self._epoch = _maybe_share_with_torch_persistent_workers(self._epoch)
        _maybe_add_torch_iterable_dataset_parent_class(self.__class__)

    def _head(self, n=5):
        return next(iter(self.iter(batch_size=n)))

    @property
    def epoch(self) -> int:
        return int(self._epoch)

    def _effective_generator(self):
        if self._shuffling and self.epoch == 0:
            return self._shuffling.generator
        elif self._shuffling:
            effective_seed = deepcopy(self._shuffling.generator).integers(0, 1 << 63) - self.epoch
            effective_seed = (1 << 63) + effective_seed if effective_seed < 0 else effective_seed
            return np.random.default_rng(effective_seed)
        else:
            raise ValueError("This dataset is not shuffled")

    @property
    def num_shards(self) -> int:
        if self._distributed and self._ex_iterable.num_shards % self._distributed.world_size == 0:
            return self._ex_iterable.num_shards // self._distributed.world_size
        return self._ex_iterable.num_shards

    @property
    def n_shards(self) -> int:  # backward compatibility
        return self.num_shards

    def _iter_pytorch(self):
        ex_iterable = self._prepare_ex_iterable_for_iteration()
        fsspec.asyn.reset_lock()

        worker_info = torch.utils.data.get_worker_info()
        shards_indices = ex_iterable.split_shard_indices_by_worker(num_shards=worker_info.num_workers, index=worker_info.id, contiguous=False)
        if shards_indices:
            ex_iterable = ex_iterable.shard_data_sources(num_shards=worker_info.num_workers, index=worker_info.id, contiguous=False)
            self._state_dict = {
                "examples_iterable": ex_iterable._init_state_dict(),
                "epoch": self.epoch,
            }
            if self._starting_state_dict and self.epoch == self._starting_state_dict["epoch"]:
                ex_iterable.load_state_dict(self._starting_state_dict["examples_iterable"])

            if self._formatting and (ex_iterable.iter_arrow or self._formatting.is_table):
                formatter = get_formatter(self._formatting.format_type, features=self.features)
                if ex_iterable.iter_arrow:
                    iterator = ex_iterable.iter_arrow()
                else:
                    iterator = _convert_to_arrow(ex_iterable, batch_size=1)
                for key, pa_table in iterator:
                    yield formatter.format_row(pa_table)
                return
            else:
                for key, example in ex_iterable:
                    yield example

    def _is_main_process(self):
        if self._distributed and self._distributed.rank > 0:
            return False

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.id > 0:
            return False
        return True

    def _prepare_ex_iterable_for_iteration(self, batch_size: int = 1, drop_last_batch: bool = False) -> _BaseExamplesIterable:
        ex_iterable = self._ex_iterable
        if self._formatting and (ex_iterable.iter_arrow or self._formatting.is_table):
            ex_iterable = RebatchedArrowExamplesIterable(ex_iterable, batch_size=batch_size, drop_last_batch=drop_last_batch)
        if self._shuffling:
            ex_iterable = ex_iterable.shuffle_data_sources(self._effective_generator())
        else:
            ex_iterable = ex_iterable

        if self._distributed:
            rank = self._distributed.rank
            world_size = self._distributed.world_size
            if ex_iterable.num_shards % world_size == 0:
                ex_iterable = ex_iterable.shard_data_sources(num_shards=world_size, index=rank, contiguous=False)
            else:
                ex_iterable = StepExamplesIterable(ex_iterable, step=world_size, offset=rank)

        if self._formatting or (self.features and ex_iterable.features != self.features):
            ex_iterable = FormattedExamplesIterable(
                ex_iterable,
                formatting=self._formatting,
                features=self.features,
                token_per_repo_id=self._token_per_repo_id,
            )

        self._state_dict = {
            "examples_iterable": ex_iterable._init_state_dict(),
            "epoch": self.epoch,
        }
        if self._starting_state_dict and self.epoch == self._starting_state_dict["epoch"]:
            ex_iterable.load_state_dict(self._starting_state_dict["examples_iterable"])
        return ex_iterable

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if isinstance(self, torch.utils.data.IterableDataset) and worker_info is not None:
            yield from self._iter_pytorch()
            return

        ex_iterable = self._prepare_ex_iterable_for_iteration()
        if self._formatting and (ex_iterable.iter_arrow or self._formatting.is_table):
            formatter = get_formatter(self._formatting.format_type, features=self.features)
            if ex_iterable.iter_arrow:
                iterator = ex_iterable.iter_arrow()
            else:
                iterator = _convert_to_arrow(ex_iterable, batch_size=1)
            for key, pa_table in iterator:
                yield formatter.format_row(pa_table)
            return

        for key, example in ex_iterable:
            yield example

    def iter(self, batch_size: int, drop_last_batch: bool = False):
        if self._formatting:
            formatter = get_formatter(self._formatting.format_type, features=self.features)
            format_dict = formatter.recursive_tensorize if isinstance(formatter, TensorFormatter) else None
        else:
            format_dict = None

        ex_iterable = self._prepare_ex_iterable_for_iteration(batch_size=batch_size, drop_last_batch=drop_last_batch)
        if self._formatting and (ex_iterable.iter_arrow or self._formatting.is_table):
            if ex_iterable.iter_arrow:
                iterator = ex_iterable.iter_arrow()
            else:
                iterator = _convert_to_arrow(ex_iterable, batch_size=batch_size, drop_last_batch=drop_last_batch)
            for key, pa_table in iterator:
                yield formatter.format_batch(pa_table)
            return

        iterator = iter(ex_iterable)
        for key, example in iterator:
            examples = [example] + [example for key, example in islice(iterator, batch_size - 1)]
            if drop_last_batch and len(examples) < batch_size:  # ignore last batch
                return
            batch = _examples_to_batch(examples)
            yield format_dict(batch) if format_dict else batch

    @staticmethod
    def from_generator(
        generator: Callable,
        features: Optional[Features] = None,
        gen_kwargs: Optional[dict] = None,
        split: NamedSplit = Split.TRAIN,
    ) -> "IterableDataset":

        return GeneratorDatasetInputStream(generator=generator, features=features, gen_kwargs=gen_kwargs, streaming=True, split=split).read()

    @staticmethod
    def from_file(filename: str) -> "IterableDataset":
        pa_table_schema = read_schema_from_file(filename)
        inferred_features = Features.from_arrow_schema(pa_table_schema)
        ex_iterable = ArrowExamplesIterable(Dataset._generate_tables_from_cache_file, kwargs={"filename": filename})
        return IterableDataset(ex_iterable=ex_iterable, info=DatasetInfo(features=inferred_features))

    def with_format(self, type: Optional[str] = None,) -> "IterableDataset":
        type = get_format_type_from_alias(type)
        return IterableDataset(
            ex_iterable=self._ex_iterable,
            info=self._info.copy(),
            split=self._split,
            formatting=FormattingConfig(format_type=type),
            shuffling=copy.deepcopy(self._shuffling),
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, list[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, list[str]]] = None,
        features: Optional[Features] = None,
        fn_kwargs: Optional[dict] = None,
    ) -> "IterableDataset":
        if isinstance(input_columns, str):
            input_columns = [input_columns]
        if isinstance(remove_columns, str):
            remove_columns = [remove_columns]
        if function is None:
            function = identity_func
        if fn_kwargs is None:
            fn_kwargs = {}

        ex_iterable = self._ex_iterable
        input_features = (None if (ex_iterable.is_typed and (self._info.features is None or self._info.features == ex_iterable.features)) else self._info.features)

        if self._formatting and self._formatting.is_table:
            ex_iterable = FormattedExamplesIterable(
                ex_iterable,
                formatting=copy.deepcopy(self._formatting),
                features=input_features,
                token_per_repo_id=self._token_per_repo_id,
            )
            ex_iterable = RebatchedArrowExamplesIterable(ex_iterable, batch_size=batch_size if batched else 1, drop_last_batch=drop_last_batch)
        else:
            if self._formatting and self._ex_iterable.iter_arrow:
                ex_iterable = RebatchedArrowExamplesIterable(self._ex_iterable, batch_size=batch_size if batched else 1, drop_last_batch=drop_last_batch)
            if self._formatting or input_features:
                ex_iterable = FormattedExamplesIterable(
                    ex_iterable,
                    formatting=copy.deepcopy(self._formatting),
                    features=input_features,
                    token_per_repo_id=self._token_per_repo_id,
                )

        ex_iterable = MappedExamplesIterable(
            ex_iterable,
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            remove_columns=remove_columns,
            fn_kwargs=fn_kwargs,
            formatting=self._formatting,
            features=features,
        )
        info = self.info.copy()
        info.features = features
        return IterableDataset(
            ex_iterable=ex_iterable,
            info=info,
            split=self._split,
            formatting=self._formatting,
            shuffling=copy.deepcopy(self._shuffling),
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, list[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
    ) -> "IterableDataset":
        if isinstance(input_columns, str):
            input_columns = [input_columns]

        ex_iterable = self._ex_iterable
        if self._info.features or self._formatting:
            ex_iterable = FormattedExamplesIterable(
                ex_iterable,
                formatting=self._formatting,
                features=None if ex_iterable.is_typed else self._info.features,
                token_per_repo_id=self._token_per_repo_id,
            )

        ex_iterable = FilteredExamplesIterable(
            ex_iterable,
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            fn_kwargs=fn_kwargs,
            formatting=self._formatting,
        )
        return IterableDataset(
            ex_iterable=ex_iterable,
            info=self._info,
            split=self._split,
            formatting=self._formatting,
            shuffling=copy.deepcopy(self._shuffling),
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def shuffle(self, seed=None, generator: Optional[np.random.Generator] = None, buffer_size: int = 1000) -> "IterableDataset":
        if generator is None:
            generator = np.random.default_rng(seed)
        else:
            generator = deepcopy(generator)
        shuffling = ShufflingConfig(generator=generator, _original_seed=seed)
        return IterableDataset(
            ex_iterable=BufferShuffledExamplesIterable(self._ex_iterable, buffer_size=buffer_size, generator=generator),
            info=self._info.copy(),
            split=self._split,
            formatting=self._formatting,
            shuffling=shuffling,
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def set_epoch(self, epoch: int):
        self._epoch += epoch - self._epoch  # update torch value in shared memory in-place

    def skip(self, n: int) -> "IterableDataset":
        ex_iterable = SkipExamplesIterable(
            self._ex_iterable,
            n,
            block_sources_order_when_shuffling=self._shuffling is None,
            split_when_sharding=self._distributed is None,
        )
        return IterableDataset(
            ex_iterable=ex_iterable,
            info=self._info.copy(),
            split=self._split,
            formatting=self._formatting,
            shuffling=copy.deepcopy(self._shuffling),
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def repeat(self, num_times: Optional[int]) -> "IterableDataset":
        return IterableDataset(
            ex_iterable=RepeatExamplesIterable(self._ex_iterable, num_times=num_times),
            info=self._info,
            split=self._split,
            formatting=self._formatting,
            shuffling=copy.deepcopy(self._shuffling),
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def take(self, n: int) -> "IterableDataset":
        ex_iterable = TakeExamplesIterable(
            self._ex_iterable,
            n,
            block_sources_order_when_shuffling=self._shuffling is None,
            split_when_sharding=self._distributed is None,
        )
        return IterableDataset(
            ex_iterable=ex_iterable,
            info=self._info.copy(),
            split=self._split,
            formatting=self._formatting,
            shuffling=copy.deepcopy(self._shuffling),
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def shard(self, num_shards: int, index: int, contiguous: bool = True,) -> "IterableDataset":
        ex_iterable = self._ex_iterable.shard_data_sources(num_shards=num_shards, index=index, contiguous=contiguous)
        return IterableDataset(
            ex_iterable=ex_iterable,
            info=self._info.copy(),
            split=self._split,
            formatting=self._formatting,
            shuffling=copy.deepcopy(self._shuffling),
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    @property
    def column_names(self) -> Optional[list[str]]:
        return list(self._info.features.keys()) if self._info.features is not None else None

    def add_column(self, name: str, column: Union[list, np.array]) -> "IterableDataset":
        return self.map(partial(add_column_fn, name=name, column=column), with_indices=True)

    def rename_column(self, original_column_name: str, new_column_name: str) -> "IterableDataset":
       return self.rename_columns({original_column_name: new_column_name})

    def rename_columns(self, column_mapping: dict[str, str]) -> "IterableDataset":
        original_features = self._info.features.copy() if self._info.features else None
        ds_iterable = self.map(partial(_rename_columns_fn, column_mapping=column_mapping), remove_columns=list(column_mapping))
        if original_features is not None:
            ds_iterable._info.features = Features(
                {
                    column_mapping[col] if col in column_mapping.keys() else col: feature
                    for col, feature in original_features.items()
                }
            )
        return ds_iterable

    def remove_columns(self, column_names: Union[str, list[str]]) -> "IterableDataset":
        original_features = self._info.features.copy() if self._info.features else None
        ds_iterable = self.map(remove_columns=column_names)
        if original_features is not None:
            ds_iterable._info.features = original_features.copy()
            for col, _ in original_features.items():
                if col in column_names:
                    del ds_iterable._info.features[col]

        return ds_iterable

    def select_columns(self, column_names: Union[str, list[str]]) -> "IterableDataset":
        if isinstance(column_names, str):
            column_names = [column_names]

        if self._info:
            info = copy.deepcopy(self._info)
            if self._info.features is not None:
                missing_columns = set(column_names) - set(self._info.features.keys())
                if missing_columns:
                    raise ValueError(f"Column name {list(missing_columns)} not in the dataset. Columns in the dataset: {list(self._info.features.keys())}.")
                info.features = Features({c: info.features[c] for c in column_names})

        ex_iterable = SelectColumnsIterable(self._ex_iterable, column_names)
        return IterableDataset(
            ex_iterable=ex_iterable,
            info=info,
            split=self._split,
            formatting=self._formatting,
            shuffling=self._shuffling,
            distributed=self._distributed,
            token_per_repo_id=self._token_per_repo_id,
        )

    def cast_column(self, column: str, feature: FeatureType) -> "IterableDataset":
        info = self._info.copy()
        info.features[column] = feature
        return IterableDataset(
            ex_iterable=self._ex_iterable,
            info=info,
            split=self._split,
            formatting=self._formatting,
            shuffling=copy.deepcopy(self._shuffling),
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def cast(self, features: Features) -> "IterableDataset":
        info = self._info.copy()
        info.features = features
        return IterableDataset(
            ex_iterable=self._ex_iterable,
            info=info,
            split=self._split,
            formatting=self._formatting,
            shuffling=copy.deepcopy(self._shuffling),
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def decode(self, enable: bool = True, num_threads: int = 0) -> "IterableDataset":
        if not self.features:
            raise ValueError("Features decoding is only available for datasets with known features, but features are Unknown. Please set the datasets features with `ds = ds.cast(features)`.")
        ds = self

        def set_decoding(decode: bool, feature):
            if hasattr(feature, "decode"):
                feature.decode = decode

        if enable and num_threads > 0:
            disabled_decoding_features = self.features.copy()
            enabled_decoding_features = self.features.copy()

            _visit(disabled_decoding_features, partial(set_decoding, False))
            _visit(enabled_decoding_features, partial(set_decoding, True))
            ds = ds.cast(disabled_decoding_features)
            pool = multiprocessing.pool.ThreadPool(num_threads)
            func = partial(_apply_async, pool, enabled_decoding_features.decode_example)
            ds = ds.map(func, features=enabled_decoding_features)
            assert isinstance(ds._ex_iterable, MappedExamplesIterable)
            ds._ex_iterable.max_num_running_async_map_functions_in_parallel = 2 * num_threads
        else:
            features = ds.features.copy()
            _visit(features, partial(set_decoding, enable))
            ds = ds.cast(features)
        return ds

    def _step(self, step: int, offset: int) -> "IterableDataset":
        ex_iterable = StepExamplesIterable(self._ex_iterable, step=step, offset=offset)
        return IterableDataset(
            ex_iterable=ex_iterable,
            info=self._info.copy(),
            split=self._split,
            formatting=self._formatting,
            shuffling=copy.deepcopy(self._shuffling),
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def _resolve_features(self):
        if self.features is not None:
            return self
        elif self._ex_iterable.is_typed:
            features = self._ex_iterable.features
        else:
            features = _infer_features_from_batch(self.with_format(None)._head())
        info = self.info.copy()
        info.features = features
        return IterableDataset(
            ex_iterable=self._ex_iterable,
            info=info,
            split=self._split,
            formatting=self._formatting,
            shuffling=copy.deepcopy(self._shuffling),
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def batch(self, batch_size: int, drop_last_batch: bool = False) -> "IterableDataset":
        def batch_fn(unbatched):
            return {k: [v] for k, v in unbatched.items()}

        if self.features:
            features = Features({col: [feature] for col, feature in self.features.items()})
        else:
            features = None
        return self.map(batch_fn, batched=True, batch_size=batch_size, drop_last_batch=drop_last_batch, features=features)

class AbstractDatasetReader(ABC):
    def __init__(
        self,
        path_or_paths: Optional[NestedDataStructureLike[PathLike]] = None,
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        self.path_or_paths = path_or_paths
        self.split = split if split or isinstance(path_or_paths, dict) else "train"
        self.features = features
        self.cache_dir = cache_dir
        self.keep_in_memory = keep_in_memory
        self.streaming = streaming
        self.num_proc = num_proc
        self.kwargs = kwargs

    @abstractmethod
    def read(self) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
        pass

class AbstractDatasetInputStream(ABC):
    def __init__(
        self,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        self.features = features
        self.cache_dir = cache_dir
        self.keep_in_memory = keep_in_memory
        self.streaming = streaming
        self.num_proc = num_proc
        self.kwargs = kwargs

    @abstractmethod
    def read(self) -> Union[Dataset, IterableDataset]:
        pass

class DatasetBuilder:
    VERSION = None  
    BUILDER_CONFIG_CLASS = BuilderConfig
    BUILDER_CONFIGS = []
    DEFAULT_CONFIG_NAME = None
    DEFAULT_WRITER_BATCH_SIZE = None

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config_name: Optional[str] = None,
        hash: Optional[str] = None,
        base_path: Optional[str] = None,
        info: Optional[DatasetInfo] = None,
        features: Optional[Features] = None,
        token: Optional[Union[bool, str]] = None,
        repo_id: Optional[str] = None,
        data_files: Optional[Union[str, list, dict, DataFilesDict]] = None,
        data_dir: Optional[str] = None,
        storage_options: Optional[dict] = None,
        writer_batch_size: Optional[int] = None,
        **config_kwargs,
    ):
        self.name: str = camelcase_to_snakecase(self.__module__.split(".")[-1])
        self.hash: Optional[str] = hash
        self.base_path = base_path
        self.token = token
        self.repo_id = repo_id
        self.storage_options = storage_options or {}
        self.dataset_name = camelcase_to_snakecase(dataset_name) if dataset_name else self.name
        self._writer_batch_size = writer_batch_size or self.DEFAULT_WRITER_BATCH_SIZE

        if data_files is not None and not isinstance(data_files, DataFilesDict):
            data_files = DataFilesDict.from_patterns(
                sanitize_patterns(data_files),
                base_path=base_path,
                download_config=DownloadConfig(token=token, storage_options=self.storage_options),
            )

        if "features" in inspect.signature(self.BUILDER_CONFIG_CLASS.__init__).parameters and features is not None:
            config_kwargs["features"] = features
        if data_files is not None:
            config_kwargs["data_files"] = data_files
        if data_dir is not None:
            config_kwargs["data_dir"] = data_dir
        self.config_kwargs = config_kwargs
        self.config, self.config_id = self._create_builder_config(
            config_name=config_name,
            custom_features=features,
            **config_kwargs,
        )

        if info is None:
            info = self.get_exported_dataset_info()
            info.update(self._info())
        info.builder_name = self.name
        info.dataset_name = self.dataset_name
        info.config_name = self.config.name
        info.version = self.config.version
        self.info = info
        if features is not None:
            self.info.features = features

        self._cache_dir_root = str(cache_dir or data_config.HF_DATASETS_CACHE)
        self._cache_dir_root = (self._cache_dir_root if is_remote_url(self._cache_dir_root) else os.path.expanduser(self._cache_dir_root))
        self._cache_downloaded_dir = (posixpath.join(self._cache_dir_root, data_config.DOWNLOADED_DATASETS_DIR) if cache_dir else str(data_config.DOWNLOADED_DATASETS_PATH))
        self._cache_downloaded_dir = (self._cache_downloaded_dir if is_remote_url(self._cache_downloaded_dir) else os.path.expanduser(self._cache_downloaded_dir))

        self._legacy_relative_data_dir = None
        self._cache_dir = self._build_cache_dir()
        if not is_remote_url(self._cache_dir_root):
            os.makedirs(self._cache_dir_root, exist_ok=True)
            lock_path = os.path.join(self._cache_dir_root, Path(self._cache_dir).as_posix().replace("/", "_") + ".lock")
            with FileLock(lock_path):
                if os.path.exists(self._cache_dir):  # check if data exist
                    if len(os.listdir(self._cache_dir)) > 0:
                        if os.path.exists(os.path.join(self._cache_dir, data_config.DATASET_INFO_FILENAME)):
                            self.info = DatasetInfo.from_directory(self._cache_dir)
                    else:  # dir exists but no data, remove the empty dir as data aren't available anymore
                        os.rmdir(self._cache_dir)

        self._output_dir = self._cache_dir
        self._fs: fsspec.AbstractFileSystem = fsspec.filesystem("file")
        self.dl_manager = None
        self._record_infos = False
        self._file_format = None

        extend_dataset_builder_for_streaming(self)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        extend_dataset_builder_for_streaming(self)

    @property
    def manual_download_instructions(self) -> Optional[str]:
        return None

    def _check_legacy_cache(self) -> Optional[str]:
        if (self.__module__.startswith("datasets.") and not is_remote_url(self._cache_dir_root) and self.config.name == "default"):

            namespace = self.repo_id.split("/")[0] if self.repo_id and self.repo_id.count("/") > 0 else None
            config_name = self.repo_id.replace("/", "--") if self.repo_id is not None else self.dataset_name
            config_id = config_name + self.config_id[len(self.config.name) :]
            hash = _PACKAGED_DATASETS_MODULES.get(self.name, "missing")[1]
            legacy_relative_data_dir = posixpath.join(
                self.dataset_name if namespace is None else f"{namespace}___{self.dataset_name}",
                config_id,
                "0.0.0",
                hash,
            )
            legacy_cache_dir = posixpath.join(self._cache_dir_root, legacy_relative_data_dir)
            if os.path.isdir(legacy_cache_dir):
                return legacy_relative_data_dir

    def _check_legacy_cache2(self, dataset_module: "DatasetModule") -> Optional[str]:
        if (self.__module__.startswith("datasets.")and not is_remote_url(self._cache_dir_root) and not (set(self.config_kwargs) - {"data_files", "data_dir"})):

            def update_hash_with_config_parameters(hash: str, config_parameters: dict) -> str:
                params_to_exclude = {"config_name", "version", "description"}
                params_to_add_to_hash = {
                    param: value
                    for param, value in sorted(config_parameters.items())
                    if param not in params_to_exclude
                }
                m = Hasher()
                m.update(hash)
                m.update(params_to_add_to_hash)
                return m.hexdigest()

            namespace = self.repo_id.split("/")[0] if self.repo_id and self.repo_id.count("/") > 0 else None
            with patch.object(Pickler, "_legacy_no_dict_keys_sorting", True):
                config_id = self.config.name + "-" + Hasher.hash({"data_files": self.config.data_files})
            hash = _PACKAGED_DATASETS_MODULES_2_15_HASHES.get(self.name, "missing")
            if (dataset_module.builder_configs_parameters.metadata_configs and self.config.name in dataset_module.builder_configs_parameters.metadata_configs):
                hash = update_hash_with_config_parameters(hash, dataset_module.builder_configs_parameters.metadata_configs[self.config.name])
            legacy_relative_data_dir = posixpath.join(
                self.dataset_name if namespace is None else f"{namespace}___{self.dataset_name}",
                config_id,
                "0.0.0",
                hash,
            )
            legacy_cache_dir = posixpath.join(self._cache_dir_root, legacy_relative_data_dir)
            if os.path.isdir(legacy_cache_dir):
                return legacy_relative_data_dir

    @classmethod
    def get_all_exported_dataset_infos(cls) -> DatasetInfosDict:
        return DatasetInfosDict.from_directory(cls.get_imported_module_dir())

    def get_exported_dataset_info(self) -> DatasetInfo:
        return self.get_all_exported_dataset_infos().get(self.config.name, DatasetInfo())

    def _create_builder_config(self, config_name=None, custom_features=None, **config_kwargs) -> tuple[BuilderConfig, str]:
        builder_config = None

        if config_name is None and self.BUILDER_CONFIGS:
            if self.DEFAULT_CONFIG_NAME is not None:
                builder_config = self.builder_configs.get(self.DEFAULT_CONFIG_NAME)
            else:
                if len(self.BUILDER_CONFIGS) > 1:
                    if not config_kwargs:
                        raise ValueError("Config name is missing.\nPlease pick one among the available configs: {list(self.builder_configs.keys())}\nExample of usage:\n\t`{example_of_usage}`")
                else:
                    builder_config = self.BUILDER_CONFIGS[0]

        if isinstance(config_name, str):
            builder_config = self.builder_configs.get(config_name)
            if builder_config is None and self.BUILDER_CONFIGS:
                raise ValueError(f"BuilderConfig '{config_name}' not found. Available: {list(self.builder_configs.keys())}")

        if not builder_config:
            if config_name is not None:
                config_kwargs["name"] = config_name
            elif self.DEFAULT_CONFIG_NAME and not config_kwargs:
                config_kwargs["name"] = self.DEFAULT_CONFIG_NAME
            if "version" not in config_kwargs and hasattr(self, "VERSION") and self.VERSION:
                config_kwargs["version"] = self.VERSION
            builder_config = self.BUILDER_CONFIG_CLASS(**config_kwargs)

        else:
            builder_config = copy.deepcopy(builder_config) if config_kwargs else builder_config
            for key, value in config_kwargs.items():
                if value is not None:
                    if not hasattr(builder_config, key):
                        raise ValueError(f"BuilderConfig {builder_config} doesn't have a '{key}' key.")
                    setattr(builder_config, key, value)

        if not builder_config.name:
            raise ValueError(f"BuilderConfig must have a name, got {builder_config.name}")

        builder_config._resolve_data_files(base_path=self.base_path, download_config=DownloadConfig(token=self.token, storage_options=self.storage_options),)

        config_id = builder_config.create_config_id(config_kwargs, custom_features=custom_features,)
        if (builder_config.name in self.builder_configs and builder_config != self.builder_configs[builder_config.name]):
            raise ValueError("Cannot name a custom BuilderConfig the same as an available BuilderConfig. Change the name. Available BuilderConfigs: {list(self.builder_configs.keys())}")
        if not builder_config.version:
            raise ValueError(f"BuilderConfig {builder_config.name} must have a version")

        return builder_config, config_id

    @classproperty
    @classmethod
    @memoize()
    def builder_configs(cls) -> dict[str, BuilderConfig]:
        configs = {config.name: config for config in cls.BUILDER_CONFIGS}
        if len(configs) != len(cls.BUILDER_CONFIGS):
            names = [config.name for config in cls.BUILDER_CONFIGS]
            raise ValueError(f"Names in BUILDER_CONFIGS must not be duplicated. Got {names}")
        return configs

    @property
    def cache_dir(self):
        return self._cache_dir

    def _use_legacy_cache_dir_if_possible(self, dataset_module: "DatasetModule"):
        self._legacy_relative_data_dir = (self._check_legacy_cache2(dataset_module) or self._check_legacy_cache() or None)
        self._cache_dir = self._build_cache_dir()
        self._output_dir = self._cache_dir

    def _relative_data_dir(self, with_version=True, with_hash=True) -> str:
        if self._legacy_relative_data_dir is not None and with_version and with_hash:
            return self._legacy_relative_data_dir

        namespace = self.repo_id.split("/")[0] if self.repo_id and self.repo_id.count("/") > 0 else None
        builder_data_dir = self.dataset_name if namespace is None else f"{namespace}___{self.dataset_name}"
        builder_data_dir = posixpath.join(builder_data_dir, self.config_id)
        if with_version:
            builder_data_dir = posixpath.join(builder_data_dir, str(self.config.version))
        if with_hash and self.hash and isinstance(self.hash, str):
            builder_data_dir = posixpath.join(builder_data_dir, self.hash)
        return builder_data_dir

    def _build_cache_dir(self):
        return posixpath.join(self._cache_dir_root, self._relative_data_dir(with_version=True))

    @abc.abstractmethod
    def _info(self) -> DatasetInfo:
        raise NotImplementedError

    @classmethod
    def get_imported_module_dir(cls):
        return os.path.dirname(inspect.getfile(inspect.getmodule(cls)))

    def _rename(self, src: str, dst: str):
        rename(self._fs, src, dst)

    def download_and_prepare(
        self,
        output_dir: Optional[str] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        verification_mode: Optional[Union[VerificationMode, str]] = None,
        dl_manager: Optional[DownloadManager] = None,
        base_path: Optional[str] = None,
        file_format: str = "arrow",
        max_shard_size: Optional[Union[int, str]] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **download_and_prepare_kwargs,
    ):
        output_dir = output_dir if output_dir is not None else self._cache_dir
        fs, output_dir = url_to_fs(output_dir, **(storage_options or {}))
        self._fs = fs
        self._output_dir = output_dir if not is_remote_filesystem(self._fs) else self._fs.unstrip_protocol(output_dir)

        download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
        verification_mode = VerificationMode(verification_mode or VerificationMode.BASIC_CHECKS)
        base_path = base_path if base_path is not None else self.base_path

        if file_format is not None and file_format not in ["arrow", "parquet"]:
            raise ValueError(f"Unsupported file_format: {file_format}. Expected 'arrow' or 'parquet'")
        self._file_format = file_format

        if self._fs._strip_protocol(self._output_dir) == "":
            raise RuntimeError(f"Unable to download and prepare the dataset at the root {self._output_dir}. Please specify a subdirectory, e.g. '{self._output_dir + self.dataset_name}'")

        if dl_manager is None:
            if download_config is None:
                download_config = DownloadConfig(
                    cache_dir=self._cache_downloaded_dir,
                    force_download=download_mode == DownloadMode.FORCE_REDOWNLOAD,
                    force_extract=download_mode == DownloadMode.FORCE_REDOWNLOAD,
                    use_etag=False,
                    num_proc=num_proc,
                    token=self.token,
                    storage_options=self.storage_options,
                )  # We don't use etag for data files to speed up the process

            dl_manager = DownloadManager(
                dataset_name=self.dataset_name,
                download_config=download_config,
                data_dir=self.config.data_dir,
                base_path=base_path,
                record_checksums=(self._record_infos or verification_mode == VerificationMode.ALL_CHECKS),
            )

        is_local = not is_remote_filesystem(self._fs)
        self.dl_manager = dl_manager

        if is_local:
            Path(self._output_dir).parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._output_dir + "_builder.lock"

        with FileLock(lock_path) if is_local else contextlib.nullcontext():
            data_exists = self._fs.exists(posixpath.join(self._output_dir, data_config.DATASET_INFO_FILENAME))
            if data_exists and download_mode == DownloadMode.REUSE_DATASET_IF_EXISTS:
                self.info = self._load_info()
                self.download_post_processing_resources(dl_manager)
                return

            if is_local:  # if cache dir is local, check for available space
                if not has_sufficient_disk_space(self.info.size_in_bytes or 0, directory=Path(self._output_dir).parent):
                    raise OSError(f"Not enough disk space. Needed: {size_str(self.info.size_in_bytes or 0)} (download: {size_str(self.info.download_size or 0)}, generated: {size_str(self.info.dataset_size or 0)}, post-processed: {size_str(self.info.post_processing_size or 0)})")

            @contextlib.contextmanager
            def incomplete_dir(dirname):
                if not is_local:
                    self._fs.makedirs(dirname, exist_ok=True)
                    yield dirname
                else:
                    tmp_dir = dirname + ".incomplete"
                    os.makedirs(tmp_dir, exist_ok=True)
                    try:
                        yield tmp_dir
                        if os.path.isdir(dirname):
                            shutil.rmtree(dirname)
                        shutil.move(tmp_dir, dirname)
                    finally:
                        if os.path.exists(tmp_dir):
                            shutil.rmtree(tmp_dir)

            _ = self._fs._strip_protocol(self._output_dir) if is_local else self._output_dir

            self._check_manual_download(dl_manager)

            with incomplete_dir(self._output_dir) as tmp_output_dir:
                with temporary_assignment(self, "_output_dir", tmp_output_dir):
                    prepare_split_kwargs = {"file_format": file_format}
                    if max_shard_size is not None:
                        prepare_split_kwargs["max_shard_size"] = max_shard_size
                    if num_proc is not None:
                        prepare_split_kwargs["num_proc"] = num_proc
                    self._download_and_prepare(
                        dl_manager=dl_manager,
                        verification_mode=verification_mode,
                        **prepare_split_kwargs,
                        **download_and_prepare_kwargs,
                    )
                    self.info.dataset_size = sum(split.num_bytes for split in self.info.splits.values())
                    self.info.download_checksums = dl_manager.get_recorded_sizes_checksums()
                    if self.info.download_size is not None:
                        self.info.size_in_bytes = self.info.dataset_size + self.info.download_size
                    self._save_info()

            self.download_post_processing_resources(dl_manager)

    def _check_manual_download(self, dl_manager):
        if self.manual_download_instructions is not None and dl_manager.manual_dir is None:
            raise ManualDownloadError(
                textwrap.dedent(
                    f"""\
                    The dataset {self.dataset_name} with config {self.config.name} requires manual data.
                    Please follow the manual download instructions: {self.manual_download_instructions}
                    Manual data can be loaded with: datasets.load_dataset("{self.repo_id or self.dataset_name}", data_dir="<path/to/manual/data>")"""
                )
            )

    def _download_and_prepare(self, dl_manager, verification_mode, **prepare_split_kwargs):
        split_dict = SplitDict(dataset_name=self.dataset_name)
        split_generators_kwargs = self._make_split_generators_kwargs(prepare_split_kwargs)
        split_generators = self._split_generators(dl_manager, **split_generators_kwargs)

        if verification_mode == VerificationMode.ALL_CHECKS and dl_manager.record_checksums:
            verify_checksums(self.info.download_checksums, dl_manager.get_recorded_sizes_checksums(), "dataset source files")

        for split_generator in split_generators:
            if str(split_generator.split_info.name).lower() == "all":
                raise ValueError("`all` is a special split keyword corresponding to the union of all splits, so cannot be used as key in ._split_generator().")

            split_dict.add(split_generator.split_info)

            try:
                self._prepare_split(split_generator, **prepare_split_kwargs)
            except OSError as e:
                raise OSError(
                    "Cannot find data file. "
                    + (self.manual_download_instructions or "")
                    + "\nOriginal error:\n"
                    + str(e)
                ) from None
            except DuplicatedKeysError as e:
                raise DuplicatedKeysError(e.key, e.duplicate_key_indices, fix_msg=f"To avoid duplicate keys, please fix the dataset script {self.name}.py",) from None
            dl_manager.manage_extracted_files()

        if verification_mode == VerificationMode.BASIC_CHECKS or verification_mode == VerificationMode.ALL_CHECKS:
            verify_splits(self.info.splits, split_dict)

        self.info.splits = split_dict
        self.info.download_size = dl_manager.downloaded_size

    def download_post_processing_resources(self, dl_manager):
        for split in self.info.splits or []:
            for resource_name, resource_file_name in self._post_processing_resources(split).items():
                if not not is_remote_filesystem(self._fs):
                    raise NotImplementedError(f"Post processing is not supported on filesystem {self._fs}")
                if os.sep in resource_file_name:
                    raise ValueError(f"Resources shouldn't be in a sub-directory: {resource_file_name}")
                resource_path = os.path.join(self._output_dir, resource_file_name)
                if not os.path.exists(resource_path):
                    downloaded_resource_path = self._download_post_processing_resources(split, resource_name, dl_manager)
                    if downloaded_resource_path:
                        shutil.move(downloaded_resource_path, resource_path)

    def _load_info(self) -> DatasetInfo:
        return DatasetInfo.from_directory(self._output_dir, storage_options=self._fs.storage_options)

    def _save_info(self):
        file_lock = (FileLock(self._output_dir + "_info.lock") if not is_remote_filesystem(self._fs) else contextlib.nullcontext())
        with file_lock:
            self.info.write_to_directory(self._output_dir, storage_options=self._fs.storage_options)

    def _save_infos(self):
        file_lock = (FileLock(self._output_dir + "_infos.lock") if not is_remote_filesystem(self._fs) else contextlib.nullcontext())
        with file_lock:
            DatasetInfosDict(**{self.config.name: self.info}).write_to_directory(self.get_imported_module_dir())

    def _make_split_generators_kwargs(self, prepare_split_kwargs):
        del prepare_split_kwargs
        return {}

    def as_dataset(
        self,
        split: Optional[Split] = None,
        run_post_process=True,
        verification_mode: Optional[Union[VerificationMode, str]] = None,
        in_memory=False,
    ) -> Union[Dataset, DatasetDict]:
        if self._file_format is not None and self._file_format != "arrow":
            raise FileFormatError('Loading a dataset not written in the "arrow" format is not supported.')
        if is_remote_filesystem(self._fs):
            raise NotImplementedError(f"Loading a dataset cached in a {type(self._fs).__name__} is not supported.")
        if not os.path.exists(self._output_dir):
            raise FileNotFoundError(f"Dataset {self.dataset_name}: could not find data in {self._output_dir}. Please make sure to call builder.download_and_prepare(), or use datasets.load_dataset() before trying to access the Dataset object.")

        if split is None:
            split = {s: s for s in self.info.splits}

        verification_mode = VerificationMode(verification_mode or VerificationMode.BASIC_CHECKS)

        datasets = map_nested(
            partial(
                self._build_single_dataset,
                run_post_process=run_post_process,
                verification_mode=verification_mode,
                in_memory=in_memory,
            ),
            split,
            map_tuple=True,
            disable_tqdm=True,
        )
        if isinstance(datasets, dict):
            datasets = DatasetDict(datasets)
        return datasets

    def _build_single_dataset(
        self,
        split: Union[str, ReadInstruction, Split],
        run_post_process: bool,
        verification_mode: VerificationMode,
        in_memory: bool = False,
    ):
        if not isinstance(split, ReadInstruction):
            split = str(split)
            if split == "all":
                split = "+".join(self.info.splits.keys())
            split = Split(split)

        ds = self._as_dataset(split=split, in_memory=in_memory,)
        if run_post_process:
            for resource_file_name in self._post_processing_resources(split).values():
                if os.sep in resource_file_name:
                    raise ValueError(f"Resources shouldn't be in a sub-directory: {resource_file_name}")
            resources_paths = {
                resource_name: os.path.join(self._output_dir, resource_file_name)
                for resource_name, resource_file_name in self._post_processing_resources(split).items()
            }
            post_processed = self._post_process(ds, resources_paths)
            if post_processed is not None:
                ds = post_processed
                recorded_checksums = {}
                record_checksums = False
                for resource_name, resource_path in resources_paths.items():
                    size_checksum = get_size_checksum_dict(resource_path)
                    recorded_checksums[resource_name] = size_checksum
                if verification_mode == VerificationMode.ALL_CHECKS and record_checksums:
                    if self.info.post_processed is None or self.info.post_processed.resources_checksums is None:
                        expected_checksums = None
                    else:
                        expected_checksums = self.info.post_processed.resources_checksums.get(split)
                    verify_checksums(expected_checksums, recorded_checksums, "post processing resources")
                if self.info.post_processed is None:
                    self.info.post_processed = PostProcessedInfo()
                if self.info.post_processed.resources_checksums is None:
                    self.info.post_processed.resources_checksums = {}
                self.info.post_processed.resources_checksums[str(split)] = recorded_checksums
                self.info.post_processing_size = sum(
                    checksums_dict["num_bytes"]
                    for split_checksums_dicts in self.info.post_processed.resources_checksums.values()
                    for checksums_dict in split_checksums_dicts.values()
                )
                if self.info.dataset_size is not None and self.info.download_size is not None:
                    self.info.size_in_bytes = (self.info.dataset_size + self.info.download_size + self.info.post_processing_size)
                self._save_info()
                ds._info.post_processed = self.info.post_processed
                ds._info.post_processing_size = self.info.post_processing_size
                ds._info.size_in_bytes = self.info.size_in_bytes
                if self.info.post_processed.features is not None:
                    if self.info.post_processed.features.type != ds.features.type:
                        raise ValueError(f"Post-processed features info don't match the dataset:\nGot\n{self.info.post_processed.features}\nbut expected something like\n{ds.features}")
                    else:
                        ds.info.features = self.info.post_processed.features

        return ds

    def _as_dataset(self, split: Union[ReadInstruction, Split] = Split.TRAIN, in_memory: bool = False) -> Dataset:
        cache_dir = self._fs._strip_protocol(self._output_dir)
        dataset_name = self.dataset_name
        if self._check_legacy_cache():
            dataset_name = self.name
        dataset_kwargs = ArrowReader(cache_dir, self.info).read(
            name=dataset_name,
            instructions=split,
            split_infos=self.info.splits.values(),
            in_memory=in_memory,
        )
        fingerprint = self._get_dataset_fingerprint(split)
        return Dataset(fingerprint=fingerprint, **dataset_kwargs)

    def _get_dataset_fingerprint(self, split: Union[ReadInstruction, Split]) -> str:
        hasher = Hasher()
        hasher.update(Path(self._relative_data_dir()).as_posix())
        hasher.update(str(split))  # for example: train, train+test, train[:10%], test[:33%](pct1_dropremainder)
        fingerprint = hasher.hexdigest()
        return fingerprint

    def as_streaming_dataset(self, split: Optional[str] = None, base_path: Optional[str] = None) -> Union[dict[str, IterableDataset], IterableDataset]:
        if is_remote_filesystem(self._fs):
            raise NotImplementedError(f"Loading a streaming dataset cached in a {type(self._fs).__name__} is not supported yet.")

        dl_manager = StreamingDownloadManager(
            base_path=base_path or self.base_path,
            download_config=DownloadConfig(token=self.token, storage_options=self.storage_options),
            dataset_name=self.dataset_name,
            data_dir=self.config.data_dir,
        )
        self._check_manual_download(dl_manager)
        splits_generators = {sg.name: sg for sg in self._split_generators(dl_manager)}
        if split is None:
            splits_generator = splits_generators
        elif split in splits_generators:
            splits_generator = splits_generators[split]
        else:
            raise ValueError(f"Bad split: {split}. Available splits: {list(splits_generators)}")

        datasets = map_nested(self._as_streaming_dataset_single, splits_generator, map_tuple=True)
        if isinstance(datasets, dict):
            datasets = IterableDatasetDict(datasets)
        return datasets

    def _as_streaming_dataset_single(self, splits_generator,) -> IterableDataset:
        ex_iterable = self._get_examples_iterable_for_split(splits_generator)
        token_per_repo_id = {self.repo_id: self.token} if self.repo_id else {}
        return IterableDataset(ex_iterable, info=self.info, split=splits_generator.name, token_per_repo_id=token_per_repo_id)

    def _post_process(self, dataset: Dataset, resources_paths: Mapping[str, str]) -> Optional[Dataset]:
        return None

    def _post_processing_resources(self, split: str) -> dict[str, str]:
        return {}

    def _download_post_processing_resources(self, split: str, resource_name: str, dl_manager: DownloadManager) -> Optional[str]:
        return None

    @abc.abstractmethod
    def _split_generators(self, dl_manager: Union[DownloadManager, StreamingDownloadManager]):
        raise NotImplementedError()

    @abc.abstractmethod
    def _prepare_split(
        self,
        split_generator: SplitGenerator,
        file_format: str = "arrow",
        max_shard_size: Optional[Union[str, int]] = None,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        raise NotImplementedError()

    def _get_examples_iterable_for_split(self, split_generator: SplitGenerator) -> ExamplesIterable:
        raise NotImplementedError()

class GeneratorBasedBuilder(DatasetBuilder):
    @abc.abstractmethod
    def _generate_examples(self, **kwargs):
        raise NotImplementedError()

    def _prepare_split(
        self,
        split_generator: SplitGenerator,
        check_duplicate_keys: bool,
        file_format="arrow",
        num_proc: Optional[int] = None,
        max_shard_size: Optional[Union[int, str]] = None,
    ):
        max_shard_size = convert_file_size_to_int(max_shard_size or data_config.MAX_SHARD_SIZE)

        if self.info.splits is not None:
            split_info = self.info.splits[split_generator.name]
        else:
            split_info = split_generator.split_info

        SUFFIX = "-JJJJJ-SSSSS-of-NNNNN"
        fname = f"{self.dataset_name}-{split_generator.name}{SUFFIX}.{file_format}"
        fpath = posixpath.join(self._output_dir, fname)

        if num_proc and num_proc > 1:
            num_input_shards = _number_of_shards_in_gen_kwargs(split_generator.gen_kwargs)
            if num_input_shards <= 1:
                num_proc = 1
            elif num_input_shards < num_proc:
                num_proc = num_input_shards

        pbar = hf_tqdm(
            unit=" examples",
            total=split_info.num_examples,
            desc=f"Generating {split_info.name} split",
        )

        _prepare_split_args = {
            "fpath": fpath,
            "file_format": file_format,
            "max_shard_size": max_shard_size,
            "split_info": split_info,
            "check_duplicate_keys": check_duplicate_keys,
        }

        if num_proc is None or num_proc == 1:
            result = None
            gen_kwargs = split_generator.gen_kwargs
            job_id = 0
            with pbar:
                for job_id, done, content in self._prepare_split_single(gen_kwargs=gen_kwargs, job_id=job_id, **_prepare_split_args):
                    if done:
                        result = content
                    else:
                        pbar.update(content)
            assert result is not None, "Failed to retrieve results from prepare_split"
            examples_per_job, bytes_per_job, features_per_job, shards_per_job, shard_lengths_per_job = ([item] for item in result)
        else:
            kwargs_per_job = [
                {"gen_kwargs": gen_kwargs, "job_id": job_id, **_prepare_split_args}
                for job_id, gen_kwargs in enumerate(_split_gen_kwargs(split_generator.gen_kwargs, max_num_jobs=num_proc))
            ]
            num_jobs = len(kwargs_per_job)

            examples_per_job = [None] * num_jobs
            bytes_per_job = [None] * num_jobs
            features_per_job = [None] * num_jobs
            shards_per_job = [None] * num_jobs
            shard_lengths_per_job = [None] * num_jobs

            with Pool(num_proc) as pool:
                with pbar:
                    for job_id, done, content in iflatmap_unordered(pool, self._prepare_split_single, kwargs_iterable=kwargs_per_job):
                        if done:
                            (
                                examples_per_job[job_id],
                                bytes_per_job[job_id],
                                features_per_job[job_id],
                                shards_per_job[job_id],
                                shard_lengths_per_job[job_id],
                            ) = content
                        else:
                            pbar.update(content)

            assert None not in examples_per_job, (f"Failed to retrieve results from prepare_split: result list {examples_per_job} still contains None - at least one worker failed to return its results")

        total_shards = sum(shards_per_job)
        total_num_examples = sum(examples_per_job)
        total_num_bytes = sum(bytes_per_job)
        features = features_per_job[0]

        split_generator.split_info.num_examples = total_num_examples
        split_generator.split_info.num_bytes = total_num_bytes

        if total_shards > 1:
            def _rename_shard(shard_and_job: tuple[int]):
                shard_id, job_id = shard_and_job
                global_shard_id = sum(shards_per_job[:job_id]) + shard_id
                self._rename(
                    fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                    fpath.replace("JJJJJ-SSSSS", f"{global_shard_id:05d}").replace("NNNNN", f"{total_shards:05d}"),
                )

            shards_and_jobs = [
                (shard_id, job_id)
                for job_id, num_shards in enumerate(shards_per_job)
                for shard_id in range(num_shards)
            ]
            thread_map(_rename_shard, shards_and_jobs, disable=True, max_workers=64)

            split_generator.split_info.shard_lengths = [shard_length for shard_lengths in shard_lengths_per_job for shard_length in shard_lengths]
        else:
            shard_id, job_id = 0, 0
            self._rename(fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"), fpath.replace(SUFFIX, ""),)

        if self.info.features is None:
            self.info.features = features

    def _prepare_split_single(
        self,
        gen_kwargs: dict,
        fpath: str,
        file_format: str,
        max_shard_size: int,
        split_info: SplitInfo,
        check_duplicate_keys: bool,
        job_id: int,
    ) -> Iterable[tuple[int, bool, Union[int, tuple]]]:
        generator = self._generate_examples(**gen_kwargs)
        writer_class = ArrowWriter
        embed_local_files = file_format == "parquet"
        shard_lengths = []
        total_num_examples, total_num_bytes = 0, 0

        shard_id = 0
        num_examples_progress_update = 0
        try:
            writer = writer_class(
                features=self.info.features,
                path=fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                writer_batch_size=self._writer_batch_size,
                hash_salt=split_info.name,
                check_duplicates=check_duplicate_keys,
                storage_options=self._fs.storage_options,
                embed_local_files=embed_local_files,
            )
            try:
                _time = time.time()
                for key, record in generator:
                    if max_shard_size is not None and writer._num_bytes > max_shard_size:
                        num_examples, num_bytes = writer.finalize()
                        writer.close()
                        shard_lengths.append(num_examples)
                        total_num_examples += num_examples
                        total_num_bytes += num_bytes
                        shard_id += 1
                        writer = writer_class(
                            features=writer._features,
                            path=fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                            writer_batch_size=self._writer_batch_size,
                            hash_salt=split_info.name,
                            check_duplicates=check_duplicate_keys,
                            storage_options=self._fs.storage_options,
                            embed_local_files=embed_local_files,
                        )
                    example = self.info.features.encode_example(record) if self.info.features is not None else record
                    writer.write(example, key)
                    num_examples_progress_update += 1
                    if time.time() > _time + data_config.PBAR_REFRESH_TIME_INTERVAL:
                        _time = time.time()
                        yield job_id, False, num_examples_progress_update
                        num_examples_progress_update = 0
            finally:
                yield job_id, False, num_examples_progress_update
                num_shards = shard_id + 1
                num_examples, num_bytes = writer.finalize()
                writer.close()
                shard_lengths.append(num_examples)
                total_num_examples += num_examples
                total_num_bytes += num_bytes
        except Exception as e:
            if isinstance(e, SchemaInferenceError) and e.__context__ is not None:
                e = e.__context__
            raise DatasetGenerationError("An error occurred while generating the dataset") from e

        yield job_id, True, (total_num_examples, total_num_bytes, writer._features, num_shards, shard_lengths)

    def _download_and_prepare(self, dl_manager, verification_mode, **prepare_splits_kwargs):
        super()._download_and_prepare(
            dl_manager,
            verification_mode,
            check_duplicate_keys=verification_mode == VerificationMode.BASIC_CHECKS
            or verification_mode == VerificationMode.ALL_CHECKS,
            **prepare_splits_kwargs,
        )

    def _get_examples_iterable_for_split(self, split_generator: SplitGenerator) -> ExamplesIterable:
        return ExamplesIterable(self._generate_examples, split_generator.gen_kwargs)

@dataclass
class GeneratorConfig(BuilderConfig):
    generator: Optional[Callable] = None
    gen_kwargs: Optional[dict] = None
    features: Optional[Features] = None
    split: NamedSplit = Split.TRAIN

    def __post_init__(self):
        super().__post_init__()
        if self.generator is None:
            raise ValueError("generator must be specified")

        if self.gen_kwargs is None:
            self.gen_kwargs = {}

class Generator(GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = GeneratorConfig

    def _info(self):
        return DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        return [SplitGenerator(name=self.config.split, gen_kwargs=self.config.gen_kwargs)]

    def _generate_examples(self, **gen_kwargs):
        yield from enumerate(self.config.generator(**gen_kwargs))

class GeneratorDatasetInputStream(AbstractDatasetInputStream):
    def __init__(
        self,
        generator: Callable,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        gen_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        split: NamedSplit = Split.TRAIN,
        **kwargs,
    ):
        super().__init__(
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=streaming,
            num_proc=num_proc,
            **kwargs,
        )
        self.builder = Generator(
            cache_dir=cache_dir,
            features=features,
            generator=generator,
            gen_kwargs=gen_kwargs,
            split=split,
            **kwargs,
        )

    def read(self):
        if self.streaming:
            dataset = self.builder.as_streaming_dataset(split=self.builder.config.split)
        else:
            download_config = None
            download_mode = None
            verification_mode = None
            base_path = None

            self.builder.download_and_prepare(
                download_config=download_config,
                download_mode=download_mode,
                verification_mode=verification_mode,
                base_path=base_path,
                num_proc=self.num_proc,
            )
            dataset = self.builder.as_dataset(split=self.builder.config.split, verification_mode=verification_mode, in_memory=self.keep_in_memory)
        return dataset

def extend_dataset_builder_for_streaming(builder: "DatasetBuilder"):
    download_config = DownloadConfig(storage_options=builder.storage_options, token=builder.token)
    extend_module_for_streaming(builder.__module__, download_config=download_config)
    if not builder.__module__.startswith("datasets."):  # check that it's not a packaged builder like csv
        importable_file = inspect.getfile(builder.__class__)
        with lock_importable_file(importable_file):
            for imports in get_imports(importable_file):
                if imports[0] == "internal":
                    internal_import_name = imports[1]
                    internal_module_name = ".".join(builder.__module__.split(".")[:-1] + [internal_import_name])
                    extend_module_for_streaming(internal_module_name, download_config=download_config)

    parent_builder_modules = [
        cls.__module__
        for cls in type(builder).__mro__[1:]  # make sure it's not the same module we've already patched
        if issubclass(cls, DatasetBuilder) and cls.__module__ != DatasetBuilder.__module__
    ]  # check it's not a standard builder from datasets.builder
    for module in parent_builder_modules:
        extend_module_for_streaming(module, download_config=download_config)

def ujson_dumps(*args, **kwargs):
    try:
        return pd.io.json.ujson_dumps(*args, **kwargs)
    except AttributeError:
        return pd.io.json.dumps(*args, **kwargs)

def ujson_loads(*args, **kwargs):
    try:
        return pd.io.json.ujson_loads(*args, **kwargs)
    except AttributeError:
        return pd.io.json.loads(*args, **kwargs)

def pandas_read_json(path_or_buf, **kwargs):
    if data_config.PANDAS_VERSION.major >= 2:
        kwargs["dtype_backend"] = "pyarrow"
    return pd.read_json(path_or_buf, **kwargs)

@dataclass
class JsonConfig(BuilderConfig):
    features: Optional[Features] = None
    encoding: str = "utf-8"
    encoding_errors: Optional[str] = None
    field: Optional[str] = None
    use_threads: bool = True  # deprecated
    block_size: Optional[int] = None  # deprecated
    chunksize: int = 10 << 20  # 10MB
    newlines_in_values: Optional[bool] = None

    def __post_init__(self):
        super().__post_init__()

def readline(f: io.RawIOBase):
    res = bytearray()
    while True:
        b = f.read(1)
        if not b:
            break
        res += b
        if res.endswith(b"\n"):
            break
    return bytes(res)

class ArrowBasedBuilder(DatasetBuilder):
    @abc.abstractmethod
    def _generate_tables(self, **kwargs):
        raise NotImplementedError()

    def _prepare_split(
        self,
        split_generator: SplitGenerator,
        file_format: str = "arrow",
        num_proc: Optional[int] = None,
        max_shard_size: Optional[Union[str, int]] = None,
    ):
        max_shard_size = convert_file_size_to_int(max_shard_size or data_config.MAX_SHARD_SIZE)

        try:
            split_info = self.info.splits[split_generator.name]
        except Exception:
            split_info = split_generator.split_info

        SUFFIX = "-JJJJJ-SSSSS-of-NNNNN"
        fname = f"{self.dataset_name}-{split_generator.name}{SUFFIX}.{file_format}"
        fpath = posixpath.join(self._output_dir, fname)

        if num_proc and num_proc > 1:
            num_input_shards = _number_of_shards_in_gen_kwargs(split_generator.gen_kwargs)
            if num_input_shards <= 1:
                num_proc = 1
            elif num_input_shards < num_proc:
                num_proc = num_input_shards

        pbar = hf_tqdm(
            unit=" examples",
            total=split_info.num_examples,
            desc=f"Generating {split_info.name} split",
        )

        _prepare_split_args = {
            "fpath": fpath,
            "file_format": file_format,
            "max_shard_size": max_shard_size,
        }

        if num_proc is None or num_proc == 1:
            result = None
            gen_kwargs = split_generator.gen_kwargs
            job_id = 0
            with pbar:
                for job_id, done, content in self._prepare_split_single(gen_kwargs=gen_kwargs, job_id=job_id, **_prepare_split_args):
                    if done:
                        result = content
                    else:
                        pbar.update(content)
            assert result is not None, "Failed to retrieve results from prepare_split"
            examples_per_job, bytes_per_job, features_per_job, shards_per_job, shard_lengths_per_job = ([item] for item in result)
        else:
            kwargs_per_job = [
                {"gen_kwargs": gen_kwargs, "job_id": job_id, **_prepare_split_args}
                for job_id, gen_kwargs in enumerate(
                    _split_gen_kwargs(split_generator.gen_kwargs, max_num_jobs=num_proc)
                )
            ]
            num_jobs = len(kwargs_per_job)

            examples_per_job = [None] * num_jobs
            bytes_per_job = [None] * num_jobs
            features_per_job = [None] * num_jobs
            shards_per_job = [None] * num_jobs
            shard_lengths_per_job = [None] * num_jobs

            with Pool(num_proc) as pool:
                with pbar:
                    for job_id, done, content in iflatmap_unordered(pool, self._prepare_split_single, kwargs_iterable=kwargs_per_job):
                        if done:
                            (
                                examples_per_job[job_id],
                                bytes_per_job[job_id],
                                features_per_job[job_id],
                                shards_per_job[job_id],
                                shard_lengths_per_job[job_id],
                            ) = content
                        else:
                            pbar.update(content)

            assert None not in examples_per_job, (f"Failed to retrieve results from prepare_split: result list {examples_per_job} still contains None - at least one worker failed to return its results")

        total_shards = sum(shards_per_job)
        total_num_examples = sum(examples_per_job)
        total_num_bytes = sum(bytes_per_job)
        features = features_per_job[0]

        split_generator.split_info.num_examples = total_num_examples
        split_generator.split_info.num_bytes = total_num_bytes

        if total_shards > 1:

            def _rename_shard(shard_id_and_job: tuple[int]):
                shard_id, job_id = shard_id_and_job
                global_shard_id = sum(shards_per_job[:job_id]) + shard_id
                self._rename(fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"), fpath.replace("JJJJJ-SSSSS", f"{global_shard_id:05d}").replace("NNNNN", f"{total_shards:05d}"))

            shard_ids_and_jobs = [
                (shard_id, job_id)
                for job_id, num_shards in enumerate(shards_per_job)
                for shard_id in range(num_shards)
            ]
            thread_map(_rename_shard, shard_ids_and_jobs, disable=True, max_workers=64)

            split_generator.split_info.shard_lengths = [shard_length for shard_lengths in shard_lengths_per_job for shard_length in shard_lengths]
        else:
            shard_id, job_id = 0, 0
            self._rename(fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"), fpath.replace(SUFFIX, ""))

        if self.info.features is None:
            self.info.features = features

    def _prepare_split_single(self, gen_kwargs: dict, fpath: str, file_format: str, max_shard_size: int, job_id: int) -> Iterable[tuple[int, bool, Union[int, tuple]]]:
        gen_kwargs = {k: tracked_list(v) if isinstance(v, list) else v for k, v in gen_kwargs.items()}
        generator = self._generate_tables(**gen_kwargs)
        writer_class = ArrowWriter
        embed_local_files = file_format == "parquet"
        shard_lengths = []
        total_num_examples, total_num_bytes = 0, 0

        shard_id = 0
        num_examples_progress_update = 0
        try:
            writer = writer_class(
                features=self.info.features,
                path=fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                writer_batch_size=self._writer_batch_size,
                storage_options=self._fs.storage_options,
                embed_local_files=embed_local_files,
            )
            try:
                _time = time.time()
                for _, table in generator:
                    if max_shard_size is not None and writer._num_bytes > max_shard_size:
                        num_examples, num_bytes = writer.finalize()
                        writer.close()
                        shard_lengths.append(num_examples)
                        total_num_examples += num_examples
                        total_num_bytes += num_bytes
                        shard_id += 1
                        writer = writer_class(
                            features=writer._features,
                            path=fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                            writer_batch_size=self._writer_batch_size,
                            storage_options=self._fs.storage_options,
                            embed_local_files=embed_local_files,
                        )
                    try:
                        writer.write_table(table)
                    except CastError as cast_error:
                        raise DatasetGenerationCastError.from_cast_error(
                            cast_error=cast_error,
                            builder_name=self.info.builder_name,
                            gen_kwargs=gen_kwargs,
                            token=self.token,
                        )
                    num_examples_progress_update += len(table)
                    if time.time() > _time + data_config.PBAR_REFRESH_TIME_INTERVAL:
                        _time = time.time()
                        yield job_id, False, num_examples_progress_update
                        num_examples_progress_update = 0
            finally:
                yield job_id, False, num_examples_progress_update
                num_shards = shard_id + 1
                num_examples, num_bytes = writer.finalize()
                writer.close()
                shard_lengths.append(num_examples)
                total_num_examples += num_examples
                total_num_bytes += num_bytes
        except Exception as e:
            if isinstance(e, SchemaInferenceError) and e.__context__ is not None:
                e = e.__context__
            if isinstance(e, DatasetGenerationError):
                raise
            raise DatasetGenerationError("An error occurred while generating the dataset") from e

        yield job_id, True, (total_num_examples, total_num_bytes, writer._features, num_shards, shard_lengths)

    def _get_examples_iterable_for_split(self, split_generator: SplitGenerator) -> ExamplesIterable:
        return ArrowExamplesIterable(self._generate_tables, kwargs=split_generator.gen_kwargs)

class Json(ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = JsonConfig

    def _info(self):
        if self.config.block_size is not None:
            self.config.chunksize = self.config.block_size
        if self.config.newlines_in_values is not None:
            raise ValueError("The JSON loader parameter `newlines_in_values` is no longer supported")
        return DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download_and_extract(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files]
            splits.append(SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.config.features is not None:
            for column_name in set(self.config.features) - set(pa_table.column_names):
                type = self.config.features.arrow_schema.field(column_name).type
                pa_table = pa_table.append_column(column_name, pa.array([None] * len(pa_table), type=type))
            pa_table = table_cast(pa_table, self.config.features.arrow_schema)
        return pa_table

    def _generate_tables(self, files):
        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            if self.config.field is not None:
                with open(file, encoding=self.config.encoding, errors=self.config.encoding_errors) as f:
                    dataset = ujson_loads(f.read())
                dataset = dataset[self.config.field]
                df = pandas_read_json(io.StringIO(ujson_dumps(dataset)))
                if df.columns.tolist() == [0]:
                    df.columns = list(self.config.features) if self.config.features else ["text"]
                pa_table = pa.Table.from_pandas(df, preserve_index=False)
                yield file_idx, self._cast_table(pa_table)

            else:
                with open(file, "rb") as f:
                    batch_idx = 0
                    block_size = max(self.config.chunksize // 32, 16 << 10)
                    encoding_errors = (self.config.encoding_errors if self.config.encoding_errors is not None else "strict")
                    while True:
                        batch = f.read(self.config.chunksize)
                        if not batch:
                            break
                        try:
                            batch += f.readline()
                        except (AttributeError, io.UnsupportedOperation):
                            batch += readline(f)
                        if self.config.encoding != "utf-8":
                            batch = batch.decode(self.config.encoding, errors=encoding_errors).encode("utf-8")
                        try:
                            while True:
                                try:
                                    pa_table = paj.read_json(io.BytesIO(batch), read_options=paj.ReadOptions(block_size=block_size))
                                    break
                                except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
                                    if (isinstance(e, pa.ArrowInvalid) and "straddling" not in str(e) or block_size > len(batch)):
                                        raise
                                    else:
                                        block_size *= 2
                        except pa.ArrowInvalid as e:
                            try:
                                with open(file, encoding=self.config.encoding, errors=self.config.encoding_errors) as f:
                                    df = pandas_read_json(f)
                            except ValueError:
                                raise e
                            if df.columns.tolist() == [0]:
                                df.columns = list(self.config.features) if self.config.features else ["text"]
                            try:
                                pa_table = pa.Table.from_pandas(df, preserve_index=False)
                            except pa.ArrowInvalid as e:
                                raise ValueError(f"Failed to convert pandas DataFrame to Arrow Table from file {file}.") from None
                            yield file_idx, self._cast_table(pa_table)
                            break
                        yield (file_idx, batch_idx), self._cast_table(pa_table)
                        batch_idx += 1

class JsonDatasetReader(AbstractDatasetReader):
    def __init__(
        self,
        path_or_paths: NestedDataStructureLike[PathLike],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        field: Optional[str] = None,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=streaming,
            num_proc=num_proc,
            **kwargs,
        )
        self.field = field
        path_or_paths = path_or_paths if isinstance(path_or_paths, dict) else {self.split: path_or_paths}
        self.builder = Json(
            cache_dir=cache_dir,
            data_files=path_or_paths,
            features=features,
            field=field,
            **kwargs,
        )

    def read(self):
        if self.streaming:
            dataset = self.builder.as_streaming_dataset(split=self.split)
        else:
            download_config = None
            download_mode = None
            verification_mode = None
            base_path = None

            self.builder.download_and_prepare(
                download_config=download_config,
                download_mode=download_mode,
                verification_mode=verification_mode,
                base_path=base_path,
                num_proc=self.num_proc,
            )
            dataset = self.builder.as_dataset(split=self.split, verification_mode=verification_mode, in_memory=self.keep_in_memory)
        return dataset

class JsonDatasetWriter:
    def __init__(
        self,
        dataset: Dataset,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **to_json_kwargs,
    ):
        if num_proc is not None and num_proc <= 0:
            raise ValueError(f"num_proc {num_proc} must be an integer > 0.")

        self.dataset = dataset
        self.path_or_buf = path_or_buf
        self.batch_size = batch_size if batch_size else data_config.DEFAULT_MAX_BATCH_SIZE
        self.num_proc = num_proc
        self.encoding = "utf-8"
        self.storage_options = storage_options or {}
        self.to_json_kwargs = to_json_kwargs

    def write(self) -> int:
        _ = self.to_json_kwargs.pop("path_or_buf", None)
        orient = self.to_json_kwargs.pop("orient", "records")
        lines = self.to_json_kwargs.pop("lines", True if orient == "records" else False)
        if "index" not in self.to_json_kwargs and orient in ["split", "table"]:
            self.to_json_kwargs["index"] = False

        default_compression = "infer" if isinstance(self.path_or_buf, (str, bytes, os.PathLike)) else None
        compression = self.to_json_kwargs.pop("compression", default_compression)

        if compression not in [None, "infer", "gzip", "bz2", "xz"]:
            raise NotImplementedError(f"`datasets` currently does not support {compression} compression")

        if not lines and self.batch_size < self.dataset.num_rows:
            raise NotImplementedError("Output JSON will not be formatted correctly when lines = False and batch_size < number of rows in the dataset. Use pandas.DataFrame.to_json() instead.")

        if isinstance(self.path_or_buf, (str, bytes, os.PathLike)):
            with fsspec.open(self.path_or_buf, "wb", compression=compression, **(self.storage_options or {})) as buffer:
                written = self._write(file_obj=buffer, orient=orient, lines=lines, **self.to_json_kwargs)
        else:
            if compression:
                raise NotImplementedError(f"The compression parameter is not supported when writing to a buffer, but compression={compression} was passed. Please provide a local path instead.")
            written = self._write(file_obj=self.path_or_buf, orient=orient, lines=lines, **self.to_json_kwargs)
        return written

    def _batch_json(self, args):
        offset, orient, lines, to_json_kwargs = args

        batch = query_table(table=self.dataset.data, key=slice(offset, offset + self.batch_size), indices=self.dataset._indices)
        json_str = batch.to_pandas().to_json(path_or_buf=None, orient=orient, lines=lines, **to_json_kwargs)
        if not json_str.endswith("\n"):
            json_str += "\n"
        return json_str.encode(self.encoding)

    def _write(self, file_obj: BinaryIO, orient, lines, **to_json_kwargs) -> int:
        written = 0

        if self.num_proc is None or self.num_proc == 1:
            for offset in hf_tqdm(range(0, len(self.dataset), self.batch_size), unit="ba", desc="Creating json from Arrow format",):
                json_str = self._batch_json((offset, orient, lines, to_json_kwargs))
                written += file_obj.write(json_str)
        else:
            num_rows, batch_size = len(self.dataset), self.batch_size
            with multiprocessing.Pool(self.num_proc) as pool:
                for json_str in hf_tqdm(
                    pool.imap(self._batch_json, [(offset, orient, lines, to_json_kwargs) for offset in range(0, num_rows, batch_size)]),
                    total=(num_rows // batch_size) + 1 if num_rows % batch_size else num_rows // batch_size,
                    unit="ba",
                    desc="Creating json from Arrow format"
                ):
                    written += file_obj.write(json_str)
        return written

@dataclass
class FolderBasedBuilderConfig(BuilderConfig):
    features: Optional[Features] = None
    drop_labels: bool = None
    drop_metadata: bool = None
    filters: Optional[Union[ds.Expression, list[tuple], list[list[tuple]]]] = None

    def __post_init__(self):
        super().__post_init__()

class ImageFolderConfig(FolderBasedBuilderConfig):
    drop_labels: bool = None
    drop_metadata: bool = None

    def __post_init__(self):
        super().__post_init__()

@dataclass
class LargeList:
    feature: Any
    id: Optional[str] = None
    pa_type: ClassVar[Any] = None
    _type: str = field(default="LargeList", init=False, repr=False)

def _visit_with_path(feature: FeatureType, func: Callable[[FeatureType, _VisitPath], Optional[FeatureType]], visit_path: _VisitPath = []) -> FeatureType:
    if isinstance(feature, Sequence) and isinstance(feature.feature, dict):
        feature = {k: [f] for k, f in feature.feature.items()}
    if isinstance(feature, Features):
        out = func(Features({k: _visit_with_path(f, func, visit_path + [k]) for k, f in feature.items()}), visit_path)
    elif isinstance(feature, dict):
        out = func({k: _visit_with_path(f, func, visit_path + [k]) for k, f in feature.items()}, visit_path)
    elif isinstance(feature, (list, tuple)):
        out = func([_visit_with_path(feature[0], func, visit_path + [0])], visit_path)
    elif isinstance(feature, Sequence):
        out = func(Sequence(_visit_with_path(feature.feature, func, visit_path + [0]), length=feature.length), visit_path)
    elif isinstance(feature, LargeList):
        out = func(LargeList(_visit_with_path(feature.feature, func, visit_path + [0])), visit_path)
    else:
        out = func(feature, visit_path)
    return feature if out is None else out

def _nested_apply(item: Any, feature_path: _VisitPath, func: Callable[[Any, _VisitPath], Any]):
    item = func(item, feature_path)
    if feature_path:
        key = feature_path[0]
        if key == 0:
            for i in range(len(item)):
                item[i] = _nested_apply(item[i], feature_path[1:], func)
        else:
            item[key] = _nested_apply(item[key], feature_path[1:], func)
    return item

class FolderBasedBuilder(GeneratorBasedBuilder):
    BASE_FEATURE: type[FeatureType]
    BASE_COLUMN_NAME: str
    BUILDER_CONFIG_CLASS: FolderBasedBuilderConfig
    EXTENSIONS: list[str]

    METADATA_FILENAMES: list[str] = ["metadata.csv", "metadata.jsonl", "metadata.parquet"]

    def _info(self):
        return DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        do_analyze = not self.config.drop_labels or not self.config.drop_metadata
        labels, path_depths = set(), set()
        metadata_files = collections.defaultdict(set)

        def analyze(files_or_archives, downloaded_files_or_dirs, split):
            if len(downloaded_files_or_dirs) == 0:
                return
            if os.path.isfile(downloaded_files_or_dirs[0]):
                original_files, downloaded_files = files_or_archives, downloaded_files_or_dirs
                for original_file, downloaded_file in zip(original_files, downloaded_files):
                    original_file, downloaded_file = str(original_file), str(downloaded_file)
                    _, original_file_ext = os.path.splitext(original_file)
                    if original_file_ext.lower() in self.EXTENSIONS:
                        if not self.config.drop_labels:
                            labels.add(os.path.basename(os.path.dirname(original_file)))
                            path_depths.add(count_path_segments(original_file))
                    elif os.path.basename(original_file) in self.METADATA_FILENAMES:
                        metadata_files[split].add((original_file, downloaded_file))
            else:
                archives, downloaded_dirs = files_or_archives, downloaded_files_or_dirs
                for archive, downloaded_dir in zip(archives, downloaded_dirs):
                    archive, downloaded_dir = str(archive), str(downloaded_dir)
                    for downloaded_dir_file in dl_manager.iter_files(downloaded_dir):
                        _, downloaded_dir_file_ext = os.path.splitext(downloaded_dir_file)
                        if downloaded_dir_file_ext in self.EXTENSIONS:
                            if not self.config.drop_labels:
                                labels.add(os.path.basename(os.path.dirname(downloaded_dir_file)))
                                path_depths.add(count_path_segments(downloaded_dir_file))
                        elif os.path.basename(downloaded_dir_file) in self.METADATA_FILENAMES:
                            metadata_files[split].add((None, downloaded_dir_file))
                        
        data_files = self.config.data_files
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            files, archives = self._split_files_and_archives(files)
            downloaded_files = dl_manager.download(files)
            downloaded_dirs = dl_manager.download_and_extract(archives)
            if do_analyze:  # drop_metadata is None or False, drop_labels is None or False
                analyze(files, downloaded_files, split_name)
                analyze(archives, downloaded_dirs, split_name)

                if metadata_files:
                    add_metadata = not self.config.drop_metadata
                    add_labels = False
                else:
                    add_metadata = False
                    add_labels = ((len(labels) > 1 and len(path_depths) == 1) if self.config.drop_labels is None else not self.config.drop_labels)
            else:
                add_labels, add_metadata, metadata_files = False, False, {}

            splits.append(
                SplitGenerator(
                    name=split_name,
                    gen_kwargs={
                        "files": tuple(zip(files, downloaded_files))
                        + tuple((None, dl_manager.iter_files(downloaded_dir)) for downloaded_dir in downloaded_dirs),
                        "metadata_files": metadata_files.get(split_name, []),
                        "add_labels": add_labels,
                        "add_metadata": add_metadata,
                    },
                )
            )

        if add_metadata:
            features_per_metadata_file: list[tuple[str, Features]] = []

            metadata_ext = {os.path.splitext(original_metadata_file or downloaded_metadata_file)[-1] for original_metadata_file, downloaded_metadata_file in itertools.chain.from_iterable(metadata_files.values())}
            if len(metadata_ext) > 1:
                raise ValueError(f"Found metadata files with different extensions: {list(metadata_ext)}")
            metadata_ext = metadata_ext.pop()

            for split_metadata_files in metadata_files.values():
                pa_metadata_table = None
                for _, downloaded_metadata_file in split_metadata_files:
                    for pa_metadata_table in self._read_metadata(downloaded_metadata_file, metadata_ext=metadata_ext):
                        break  # just fetch the first rows
                    if pa_metadata_table is not None:
                        features_per_metadata_file.append((downloaded_metadata_file, Features.from_arrow_schema(pa_metadata_table.schema)))
                        break  # no need to fetch all the files
            for downloaded_metadata_file, metadata_features in features_per_metadata_file:
                if metadata_features != features_per_metadata_file[0][1]:
                    raise ValueError(f"Metadata files {downloaded_metadata_file} and {features_per_metadata_file[0][0]} have different features: {features_per_metadata_file[0]} != {metadata_features}")
            metadata_features = features_per_metadata_file[0][1]
            feature_not_found = True

            def _set_feature(feature):
                nonlocal feature_not_found
                if isinstance(feature, dict):
                    out = type(feature)()
                    for key in feature:
                        if (key == "file_name" or key.endswith("_file_name")) and feature[key] == Value("string"):
                            key = key[: -len("_file_name")] or self.BASE_COLUMN_NAME
                            out[key] = self.BASE_FEATURE()
                            feature_not_found = False
                        elif (key == "file_names" or key.endswith("_file_names")) and feature[key] == Sequence(Value("string")):
                            key = key[: -len("_file_names")] or (self.BASE_COLUMN_NAME + "s")
                            out[key] = Sequence(self.BASE_FEATURE())
                            feature_not_found = False
                        elif (key == "file_names" or key.endswith("_file_names")) and feature[key] == [Value("string")]:
                            key = key[: -len("_file_names")] or (self.BASE_COLUMN_NAME + "s")
                            out[key] = [self.BASE_FEATURE()]
                            feature_not_found = False
                        else:
                            out[key] = feature[key]
                    return out
                return feature

            metadata_features = _visit(metadata_features, _set_feature)

            if feature_not_found:
                raise ValueError("`file_name` or `*_file_name` must be present as dictionary key (with type string) in metadata files")
        else:
            metadata_features = None

        if self.config.features is None:
            if add_metadata:
                self.info.features = metadata_features
            else:
                self.info.features = Features({self.BASE_COLUMN_NAME: self.BASE_FEATURE()})

        return splits

    def _split_files_and_archives(self, data_files):
        files, archives = [], []
        for data_file in data_files:
            _, data_file_ext = os.path.splitext(data_file)
            if data_file_ext.lower() in self.EXTENSIONS:
                files.append(data_file)
            elif os.path.basename(data_file) in self.METADATA_FILENAMES:
                files.append(data_file)
            else:
                archives.append(data_file)
        return files, archives

    def _read_metadata(self, metadata_file: str, metadata_ext: str = "") -> Iterator[pa.Table]:
        if self.config.filters is not None:
            filter_expr = (pq.filters_to_expression(self.config.filters) if isinstance(self.config.filters, list) else self.config.filters)
        else:
            filter_expr = None
        if metadata_ext == ".csv":
            chunksize = 10_000  # 10k lines
            schema = self.config.features.arrow_schema if self.config.features else None
            dtype = (
                {
                    name: dtype.to_pandas_dtype() if not require_storage_cast(feature) else object
                    for name, dtype, feature in zip(schema.names, schema.types, self.config.features.values())
                }
                if schema is not None
                else None
            )
            csv_file_reader = pd.read_csv(metadata_file, iterator=True, dtype=dtype, chunksize=chunksize)
            for df in csv_file_reader:
                pa_table = pa.Table.from_pandas(df)
                if self.config.filters is not None:
                    pa_table = pa_table.filter(filter_expr)
                if len(pa_table) > 0:
                    yield pa_table
        elif metadata_ext == ".jsonl":
            with open(metadata_file, "rb") as f:
                chunksize: int = 10 << 20  # 10MB
                block_size = max(chunksize // 32, 16 << 10)
                while True:
                    batch = f.read(chunksize)
                    if not batch:
                        break
                    try:
                        batch += f.readline()
                    except (AttributeError, io.UnsupportedOperation):
                        batch += readline(f)
                    while True:
                        try:
                            pa_table = paj.read_json(io.BytesIO(batch), read_options=paj.ReadOptions(block_size=block_size))
                            break
                        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
                            if (isinstance(e, pa.ArrowInvalid) and "straddling" not in str(e) or block_size > len(batch)):
                                raise
                            else:
                                block_size *= 2
                    if self.config.filters is not None:
                        pa_table = pa_table.filter(filter_expr)
                    if len(pa_table) > 0:
                        yield pa_table
        else:
            with open(metadata_file, "rb") as f:
                parquet_fragment = ds.ParquetFileFormat().make_fragment(f)
                if parquet_fragment.row_groups:
                    batch_size = parquet_fragment.row_groups[0].num_rows
                else:
                    batch_size = data_config.DEFAULT_MAX_BATCH_SIZE
                for record_batch in parquet_fragment.to_batches(
                    batch_size=batch_size,
                    filter=filter_expr,
                    batch_readahead=0,
                    fragment_readahead=0,
                ):
                    yield pa.Table.from_batches([record_batch])

    def _generate_examples(self, files, metadata_files, add_metadata, add_labels):
        sample_idx = 0
        if add_metadata:
            feature_paths = []

            def find_feature_path(feature, feature_path):
                nonlocal feature_paths
                if feature_path and isinstance(feature, self.BASE_FEATURE):
                    feature_paths.append(feature_path)

            _visit_with_path(self.info.features, find_feature_path)

            for original_metadata_file, downloaded_metadata_file in metadata_files:
                metadata_ext = os.path.splitext(original_metadata_file or downloaded_metadata_file)[-1]
                downloaded_metadata_dir = os.path.dirname(downloaded_metadata_file)

                def set_feature(item, feature_path: _VisitPath):
                    if len(feature_path) == 2 and isinstance(feature_path[0], str) and feature_path[1] == 0:
                        item[feature_path[0]] = item.pop("file_names", None) or item.pop(feature_path[0] + "_file_names", None)
                    elif len(feature_path) == 1 and isinstance(feature_path[0], str):
                        item[feature_path[0]] = item.pop("file_name", None) or item.pop(feature_path[0] + "_file_name", None)
                    elif len(feature_path) == 0:
                        file_relpath = os.path.normpath(item).replace("\\", "/")
                        item = os.path.join(downloaded_metadata_dir, file_relpath)
                    return item

                for pa_metadata_table in self._read_metadata(downloaded_metadata_file, metadata_ext=metadata_ext):
                    for sample in pa_metadata_table.to_pylist():
                        for feature_path in feature_paths:
                            _nested_apply(sample, feature_path, set_feature)
                        yield sample_idx, sample
                        sample_idx += 1
        else:
            if self.config.filters is not None:
                filter_expr = (
                    pq.filters_to_expression(self.config.filters)
                    if isinstance(self.config.filters, list)
                    else self.config.filters
                )
            for original_file, downloaded_file_or_dir in files:
                downloaded_files = [downloaded_file_or_dir] if original_file else downloaded_file_or_dir
                for downloaded_file in downloaded_files:
                    original_file_ext = os.path.splitext(original_file or downloaded_file)[-1]
                    if original_file_ext.lower() not in self.EXTENSIONS:
                        continue
                    sample = {self.BASE_COLUMN_NAME: downloaded_file}
                    if add_labels:
                        sample["label"] = os.path.basename(os.path.dirname(original_file or downloaded_file))
                    if self.config.filters is not None:
                        pa_table = pa.Table.from_pylist([sample]).filter(filter_expr)
                        if len(pa_table) == 0:
                            continue
                    yield sample_idx, sample
                    sample_idx += 1

class ImageFolder(FolderBasedBuilder):
    BASE_FEATURE = Image
    BASE_COLUMN_NAME = "image"
    BUILDER_CONFIG_CLASS = ImageFolderConfig
    EXTENSIONS: list[str]  # definition at the bottom of the script

class _DatasetModuleFactory:
    def get_module(self) -> DatasetModule:
        raise NotImplementedError

def increase_load_count(name: str):
    if not data_config.HF_HUB_OFFLINE and data_config.HF_UPDATE_DOWNLOAD_COUNTS:
        try:
            get_session().head(
                "/".join((data_config.S3_DATASETS_BUCKET_PREFIX, name, name + ".py")),
                headers={"User-Agent": get_datasets_user_agent()},
                timeout=3,
            )
        except Exception:
            pass

def glob_pattern_to_regex(pattern: str):
    return (
        pattern.replace("\\", r"\\")
        .replace(".", r"\.")
        .replace("*", ".*")
        .replace("+", r"\+")
        .replace("//", "/")
        .replace("(", r"\(")
        .replace(")", r"\)")
        .replace("|", r"\|")
        .replace("^", r"\^")
        .replace("$", r"\$")
        .rstrip("/")
        .replace("?", ".")
    )

def _get_data_files_patterns(pattern_resolver: Callable[[str], list[str]]) -> dict[str, list[str]]:
    for split_pattern in ALL_SPLIT_PATTERNS:
        pattern = split_pattern.replace("{split}", "*")
        try:
            data_files = pattern_resolver(pattern)
        except FileNotFoundError:
            continue
        if len(data_files) > 0:
            splits: set[str] = set()
            for p in data_files:
                p_parts = string_to_dict(xbasename(p), glob_pattern_to_regex(xbasename(split_pattern)))
                assert p_parts is not None
                splits.add(p_parts["split"])

            if any(not re.match(_split_re, split) for split in splits):
                raise ValueError(f"Split name should match '{_split_re}'' but got '{splits}'.")
            sorted_splits = [str(split) for split in DEFAULT_SPLITS if split in splits] + sorted(splits - {str(split) for split in DEFAULT_SPLITS})
            return {split: [split_pattern.format(split=split)] for split in sorted_splits}
    for patterns_dict in ALL_DEFAULT_PATTERNS:
        non_empty_splits = []
        for split, patterns in patterns_dict.items():
            for pattern in patterns:
                try:
                    data_files = pattern_resolver(pattern)
                except FileNotFoundError:
                    continue
                if len(data_files) > 0:
                    non_empty_splits.append(split)
                    break
        if non_empty_splits:
            return {split: patterns_dict[split] for split in non_empty_splits}
    raise FileNotFoundError(f"Couldn't resolve pattern {pattern} with resolver {pattern_resolver}")

def get_data_patterns(base_path: str, download_config: Optional[DownloadConfig] = None) -> dict[str, list[str]]:
    resolver = partial(resolve_pattern, base_path=base_path, download_config=download_config)
    try:
        return _get_data_files_patterns(resolver)
    except FileNotFoundError:
        raise EmptyDatasetError(f"The directory at {base_path} doesn't contain any data files") from None

class PackagedDatasetModuleFactory(_DatasetModuleFactory):
    def __init__(
        self,
        name: str,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, list, dict]] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
    ):
        self.name = name
        self.data_files = data_files
        self.data_dir = data_dir
        self.download_config = download_config
        self.download_mode = download_mode
        increase_load_count(name)

    def get_module(self) -> DatasetModule:
        base_path = Path(self.data_dir or "").expanduser().resolve().as_posix()
        patterns = (
            sanitize_patterns(self.data_files)
            if self.data_files is not None
            else get_data_patterns(base_path, download_config=self.download_config)
        )
        data_files = DataFilesDict.from_patterns(
            patterns,
            download_config=self.download_config,
            base_path=base_path,
        )

        module_path, hash = _PACKAGED_DATASETS_MODULES[self.name]

        builder_kwargs = {
            "data_files": data_files,
            "dataset_name": self.name,
        }

        return DatasetModule(module_path, hash, builder_kwargs)

class EmptyDatasetError(FileNotFoundError):
    pass

def _copy_script_and_other_resources_in_importable_dir(
    name: str,
    importable_directory_path: str,
    subdirectory_name: str,
    original_local_path: str,
    local_imports: list[tuple[str, str]],
    additional_files: list[tuple[str, str]],
    download_mode: Optional[Union[DownloadMode, str]],
) -> str:
    importable_subdirectory = os.path.join(importable_directory_path, subdirectory_name)
    importable_file = os.path.join(importable_subdirectory, name + ".py")
    with lock_importable_file(importable_file):
        if download_mode == DownloadMode.FORCE_REDOWNLOAD and os.path.exists(importable_directory_path):
            shutil.rmtree(importable_directory_path)
        os.makedirs(importable_directory_path, exist_ok=True)

        init_file_path = os.path.join(importable_directory_path, "__init__.py")
        if not os.path.exists(init_file_path):
            with open(init_file_path, "w"):
                pass

        os.makedirs(importable_subdirectory, exist_ok=True)
        init_file_path = os.path.join(importable_subdirectory, "__init__.py")
        if not os.path.exists(init_file_path):
            with open(init_file_path, "w"):
                pass

        if not os.path.exists(importable_file):
            shutil.copyfile(original_local_path, importable_file)
        meta_path = os.path.splitext(importable_file)[0] + ".json"
        if not os.path.exists(meta_path):
            meta = {"original file path": original_local_path, "local file path": importable_file}
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

        for import_name, import_path in local_imports:
            if os.path.isfile(import_path):
                full_path_local_import = os.path.join(importable_subdirectory, import_name + ".py")
                if not os.path.exists(full_path_local_import):
                    shutil.copyfile(import_path, full_path_local_import)
            elif os.path.isdir(import_path):
                full_path_local_import = os.path.join(importable_subdirectory, import_name)
                if not os.path.exists(full_path_local_import):
                    shutil.copytree(import_path, full_path_local_import)
            else:
                raise ImportError(f"Error with local import at {import_path}")

        for file_name, original_path in additional_files:
            destination_additional_path = os.path.join(importable_subdirectory, file_name)
            if not os.path.exists(destination_additional_path) or not filecmp.cmp(original_path, destination_additional_path):
                shutil.copyfile(original_path, destination_additional_path)
        return importable_file

def _create_importable_file(
    local_path: str,
    local_imports: list[tuple[str, str]],
    additional_files: list[tuple[str, str]],
    dynamic_modules_path: str,
    module_namespace: str,
    subdirectory_name: str,
    name: str,
    download_mode: DownloadMode,
) -> None:
    importable_directory_path = os.path.join(dynamic_modules_path, module_namespace, name.replace("/", "--"))
    Path(importable_directory_path).mkdir(parents=True, exist_ok=True)
    (Path(importable_directory_path).parent / "__init__.py").touch(exist_ok=True)
    _ = _copy_script_and_other_resources_in_importable_dir(
        name=name.split("/")[-1],
        importable_directory_path=importable_directory_path,
        subdirectory_name=subdirectory_name,
        original_local_path=local_path,
        local_imports=local_imports,
        additional_files=additional_files,
        download_mode=download_mode,
    )

def _load_importable_file(
    dynamic_modules_path: str,
    module_namespace: str,
    subdirectory_name: str,
    name: str,
) -> tuple[str, str]:
    module_path = ".".join(
        [
            os.path.basename(dynamic_modules_path),
            module_namespace,
            name.replace("/", "--"),
            subdirectory_name,
            name.split("/")[-1],
        ]
    )
    return module_path, subdirectory_name

def files_to_hash(file_paths: list[str]) -> str:
    to_use_files: list[Union[Path, str]] = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            to_use_files.extend(list(Path(file_path).rglob("*.[pP][yY]")))
        else:
            to_use_files.append(file_path)

    lines = []
    for file_path in to_use_files:
        with open(file_path, encoding="utf-8") as f:
            lines.extend(f.readlines())
    return _hash_python_lines(lines)

def _download_additional_modules(name: str, base_path: str, imports: tuple[str, str, str, str], download_config: Optional[DownloadConfig]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    local_imports = []
    library_imports = []
    download_config = download_config.copy()
    if download_config.download_desc is None:
        download_config.download_desc = "Downloading extra modules"
    for import_type, import_name, import_path, sub_directory in imports:
        if import_type == "library":
            library_imports.append((import_name, import_path))  # Import from a library
            continue

        if import_name == name:
            raise ValueError(f"Error in the {name} script, importing relative {import_name} module but {import_name} is the name of the script. Please change relative import {import_name} to another name and add a '# From: URL_OR_PATH' comment pointing to the original relative import file path.")
        if import_type == "internal":
            url_or_filename = url_or_path_join(base_path, import_path + ".py")
        elif import_type == "external":
            url_or_filename = import_path
        else:
            raise ValueError("Wrong import_type")

        local_import_path = cached_path(url_or_filename, download_config=download_config)
        if sub_directory is not None:
            local_import_path = os.path.join(local_import_path, sub_directory)
        local_imports.append((import_name, local_import_path))

    return local_imports, library_imports

def _check_library_imports(name: str, library_imports: list[tuple[str, str]]) -> None:
    needs_to_be_installed = {}
    for library_import_name, library_import_path in library_imports:
        try:
            lib = importlib.import_module(library_import_name)  # noqa F841
        except ImportError:
            if library_import_name not in needs_to_be_installed or library_import_path != library_import_name:
                needs_to_be_installed[library_import_name] = library_import_path
    if needs_to_be_installed:
        _dependencies_str = "dependencies" if len(needs_to_be_installed) > 1 else "dependency"
        _them_str = "them" if len(needs_to_be_installed) > 1 else "it"
        if "sklearn" in needs_to_be_installed.keys():
            needs_to_be_installed["sklearn"] = "scikit-learn"
        if "Bio" in needs_to_be_installed.keys():
            needs_to_be_installed["Bio"] = "biopython"
        raise ImportError(f"To be able to use {name}, you need to install the following {_dependencies_str}: {', '.join(needs_to_be_installed)}.\nPlease install {_them_str} using 'pip install {' '.join(needs_to_be_installed.values())}' for instance.")

def init_hf_modules(hf_modules_cache: Optional[Union[Path, str]] = None) -> str:
    hf_modules_cache = hf_modules_cache if hf_modules_cache is not None else data_config.HF_MODULES_CACHE
    hf_modules_cache = str(hf_modules_cache)
    if hf_modules_cache not in sys.path:
        sys.path.append(hf_modules_cache)

        os.makedirs(hf_modules_cache, exist_ok=True)
        if not os.path.exists(os.path.join(hf_modules_cache, "__init__.py")):
            with open(os.path.join(hf_modules_cache, "__init__.py"), "w"):
                pass
    return hf_modules_cache

def init_dynamic_modules(name: str = data_config.MODULE_NAME_FOR_DYNAMIC_MODULES, hf_modules_cache: Optional[Union[Path, str]] = None):
    hf_modules_cache = init_hf_modules(hf_modules_cache)
    dynamic_modules_path = os.path.join(hf_modules_cache, name)
    os.makedirs(dynamic_modules_path, exist_ok=True)
    if not os.path.exists(os.path.join(dynamic_modules_path, "__init__.py")):
        with open(os.path.join(dynamic_modules_path, "__init__.py"), "w"):
            pass
    return dynamic_modules_path

def _get_importable_file_path(
    dynamic_modules_path: str,
    module_namespace: str,
    subdirectory_name: str,
    name: str,
) -> str:
    importable_directory_path = os.path.join(dynamic_modules_path, module_namespace, name.replace("/", "--"))
    return os.path.join(importable_directory_path, subdirectory_name, name.split("/")[-1] + ".py")

def _raise_timeout_error(signum, frame):
    raise ValueError("Loading this dataset requires you to execute custom code contained in the dataset repository on your local machine. Please set the option `trust_remote_code=True` to permit loading of this dataset.")

def resolve_trust_remote_code(trust_remote_code: Optional[bool], repo_id: str) -> bool:
    trust_remote_code = trust_remote_code if trust_remote_code is not None else data_config.HF_DATASETS_TRUST_REMOTE_CODE
    if trust_remote_code is None:
        if data_config.TIME_OUT_REMOTE_CODE > 0:
            try:
                signal.signal(signal.SIGALRM, _raise_timeout_error)
                signal.alarm(data_config.TIME_OUT_REMOTE_CODE)
                while trust_remote_code is None:
                    answer = input(f"The repository for {repo_id} contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/{repo_id}.\n You can avoid this prompt in future by passing the argument `trust_remote_code=True`.\n\n Do you wish to run the custom code? [y/N] ")
                    if answer.lower() in ["yes", "y", "1"]:
                        trust_remote_code = True
                    elif answer.lower() in ["no", "n", "0", ""]:
                        trust_remote_code = False
                signal.alarm(0)
            except Exception:
                raise ValueError(f"The repository for {repo_id} contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/{repo_id}.\n Please pass the argument `trust_remote_code=True` to allow custom code to be run.")
        else:
            _raise_timeout_error(None, None)
    return trust_remote_code

class LocalDatasetModuleFactoryWithScript(_DatasetModuleFactory):
    def __init__(
        self,
        path: str,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        dynamic_modules_path: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
    ):
        self.path = path
        self.name = Path(path).stem
        self.download_config = download_config or DownloadConfig()
        self.download_mode = download_mode
        self.dynamic_modules_path = dynamic_modules_path
        self.trust_remote_code = trust_remote_code

    def get_module(self) -> DatasetModule:
        dataset_infos_path = Path(self.path).parent / data_config.DATASETDICT_INFOS_FILENAME
        dataset_readme_path = Path(self.path).parent / data_config.REPOCARD_FILENAME
        imports = get_imports(self.path)
        local_imports, library_imports = _download_additional_modules(
            name=self.name,
            base_path=str(Path(self.path).parent),
            imports=imports,
            download_config=self.download_config,
        )
        additional_files = []
        if dataset_infos_path.is_file():
            additional_files.append((data_config.DATASETDICT_INFOS_FILENAME, str(dataset_infos_path)))
        if dataset_readme_path.is_file():
            additional_files.append((data_config.REPOCARD_FILENAME, dataset_readme_path))
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        hash = files_to_hash([self.path] + [loc[1] for loc in local_imports])
        importable_file_path = _get_importable_file_path(
            dynamic_modules_path=dynamic_modules_path,
            module_namespace="datasets",
            subdirectory_name=hash,
            name=self.name,
        )
        if not os.path.exists(importable_file_path):
            trust_remote_code = resolve_trust_remote_code(self.trust_remote_code, self.name)
            if trust_remote_code:
                _create_importable_file(
                    local_path=self.path,
                    local_imports=local_imports,
                    additional_files=additional_files,
                    dynamic_modules_path=dynamic_modules_path,
                    module_namespace="datasets",
                    subdirectory_name=hash,
                    name=self.name,
                    download_mode=self.download_mode,
                )
            else:
                raise ValueError(f"Loading {self.name} requires you to execute the dataset script in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set the option `trust_remote_code=True` to remove this error.")
        _check_library_imports(name=self.name, library_imports=library_imports)
        module_path, hash = _load_importable_file(
            dynamic_modules_path=dynamic_modules_path,
            module_namespace="datasets",
            subdirectory_name=hash,
            name=self.name,
        )

        importlib.invalidate_caches()
        builder_kwargs = {"base_path": str(Path(self.path).parent)}
        return DatasetModule(module_path, hash, builder_kwargs, importable_file_path=importable_file_path)

def infer_module_for_data_files_list(data_files_list: DataFilesList, download_config: Optional[DownloadConfig] = None) -> tuple[Optional[str], dict]:
    extensions_counter = Counter(
        ("." + suffix.lower(), xbasename(filepath) in FolderBasedBuilder.METADATA_FILENAMES)
        for filepath in data_files_list[: data_config.DATA_FILES_MAX_NUMBER_FOR_MODULE_INFERENCE]
        for suffix in xbasename(filepath).split(".")[1:]
    )
    if extensions_counter:

        def sort_key(ext_count: tuple[tuple[str, bool], int]) -> tuple[int, bool]:
            (ext, is_metadata), count = ext_count
            return (not is_metadata, count, ext == ".parquet", ext == ".jsonl", ext == ".json", ext == ".csv", ext)

        for (ext, _), _ in sorted(extensions_counter.items(), key=sort_key, reverse=True):
            if ext in _EXTENSION_TO_MODULE:
                return _EXTENSION_TO_MODULE[ext]
            elif ext == ".zip":
                return infer_module_for_data_files_list_in_archives(data_files_list, download_config=download_config)
    return None, {}

def infer_module_for_data_files_list_in_archives(data_files_list: DataFilesList, download_config: Optional[DownloadConfig] = None) -> tuple[Optional[str], dict]:
    archived_files = []
    archive_files_counter = 0
    for filepath in data_files_list:
        if str(filepath).endswith(".zip"):
            archive_files_counter += 1
            if archive_files_counter > data_config.GLOBBED_DATA_FILES_MAX_NUMBER_FOR_MODULE_INFERENCE:
                break
            extracted = xjoin(StreamingDownloadManager().extract(filepath), "**")
            archived_files += [
                f.split("::")[0]
                for f in xglob(extracted, recursive=True, download_config=download_config)[
                    : data_config.ARCHIVED_DATA_FILES_MAX_NUMBER_FOR_MODULE_INFERENCE
                ]
            ]
    extensions_counter = Counter("." + suffix.lower() for filepath in archived_files for suffix in xbasename(filepath).split(".")[1:])
    if extensions_counter:
        most_common = extensions_counter.most_common(1)[0][0]
        if most_common in _EXTENSION_TO_MODULE:
            return _EXTENSION_TO_MODULE[most_common]
    return None, {}

def infer_module_for_data_files(data_files: DataFilesDict, path: Optional[str] = None, download_config: Optional[DownloadConfig] = None) -> tuple[Optional[str], dict[str, Any]]:
    split_modules = {split: infer_module_for_data_files_list(data_files_list, download_config=download_config) for split, data_files_list in data_files.items()}
    module_name, default_builder_kwargs = next(iter(split_modules.values()))
    if any((module_name, default_builder_kwargs) != split_module for split_module in split_modules.values()):
        raise ValueError(f"Couldn't infer the same data file format for all splits. Got {split_modules}")
    if not module_name:
        raise DataFilesNotFoundError("No (supported) data files found" + (f" in {path}" if path else ""))
    return module_name, default_builder_kwargs

def import_main_class(module_path) -> Optional[type[DatasetBuilder]]:
    module = importlib.import_module(module_path)
    module_main_cls = None
    for name, obj in module.__dict__.items():
        if inspect.isclass(obj) and issubclass(obj, DatasetBuilder):
            if inspect.isabstract(obj):
                continue
            module_main_cls = obj
            obj_module = inspect.getmodule(obj)
            if obj_module is not None and module == obj_module:
                break

    return module_main_cls

def create_builder_configs_from_metadata_configs(
    module_path: str,
    metadata_configs: MetadataConfigs,
    base_path: Optional[str] = None,
    default_builder_kwargs: dict[str, Any] = None,
    download_config: Optional[DownloadConfig] = None,
) -> tuple[list[BuilderConfig], str]:
    builder_cls = import_main_class(module_path)
    builder_config_cls = builder_cls.BUILDER_CONFIG_CLASS
    default_config_name = metadata_configs.get_default_config_name()
    builder_configs = []
    default_builder_kwargs = {} if default_builder_kwargs is None else default_builder_kwargs

    base_path = base_path if base_path is not None else ""
    for config_name, config_params in metadata_configs.items():
        config_data_files = config_params.get("data_files")
        config_data_dir = config_params.get("data_dir")
        config_base_path = xjoin(base_path, config_data_dir) if config_data_dir else base_path
        try:
            config_patterns = (
                sanitize_patterns(config_data_files)
                if config_data_files is not None
                else get_data_patterns(config_base_path, download_config=download_config)
            )
            config_data_files_dict = DataFilesPatternsDict.from_patterns(
                config_patterns,
                allowed_extensions=ALL_ALLOWED_EXTENSIONS,
            )
        except EmptyDatasetError as e:
            raise EmptyDatasetError(f"Dataset at '{base_path}' doesn't contain data files matching the patterns for config '{config_name}', check `data_files` and `data_fir` parameters in the `configs` YAML field in README.md.") from e
        builder_configs.append(
            builder_config_cls(
                name=config_name,
                data_files=config_data_files_dict,
                data_dir=config_data_dir,
                **{
                    param: value
                    for param, value in {**default_builder_kwargs, **config_params}.items()
                    if hasattr(builder_config_cls, param) and param not in ("default", "data_files", "data_dir")
                },
            )
        )
    return builder_configs, default_config_name

class LocalDatasetModuleFactoryWithoutScript(_DatasetModuleFactory):
    def __init__(
        self,
        path: str,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, list, dict]] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
    ):
        if data_dir and os.path.isabs(data_dir):
            raise ValueError(f"`data_dir` must be relative to a dataset directory's root: {path}")

        self.path = Path(path).as_posix()
        self.name = Path(path).stem
        self.data_files = data_files
        self.data_dir = data_dir
        self.download_mode = download_mode

    def get_module(self) -> DatasetModule:
        readme_path = os.path.join(self.path, data_config.REPOCARD_FILENAME)
        standalone_yaml_path = os.path.join(self.path, data_config.REPOYAML_FILENAME)
        dataset_card_data = DatasetCard.load(readme_path).data if os.path.isfile(readme_path) else DatasetCardData()
        if os.path.exists(standalone_yaml_path):
            with open(standalone_yaml_path, encoding="utf-8") as f:
                standalone_yaml_data = yaml.safe_load(f.read())
                if standalone_yaml_data:
                    _dataset_card_data_dict = dataset_card_data.to_dict()
                    _dataset_card_data_dict.update(standalone_yaml_data)
                    dataset_card_data = DatasetCardData(**_dataset_card_data_dict)
        metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card_data)
        dataset_infos = DatasetInfosDict.from_dataset_card_data(dataset_card_data)
        base_path = Path(self.path, self.data_dir or "").expanduser().resolve().as_posix()
        if self.data_files is not None:
            patterns = sanitize_patterns(self.data_files)
        elif metadata_configs and not self.data_dir and "data_files" in next(iter(metadata_configs.values())):
            patterns = sanitize_patterns(next(iter(metadata_configs.values()))["data_files"])
        else:
            patterns = get_data_patterns(base_path)
        data_files = DataFilesDict.from_patterns(patterns, base_path=base_path, allowed_extensions=ALL_ALLOWED_EXTENSIONS)
        module_name, default_builder_kwargs = infer_module_for_data_files(data_files=data_files, path=self.path)
        data_files = data_files.filter(extensions=_MODULE_TO_EXTENSIONS[module_name], file_names=_MODULE_TO_METADATA_FILE_NAMES[module_name])
        module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]
        if metadata_configs:
            builder_configs, default_config_name = create_builder_configs_from_metadata_configs(
                module_path,
                metadata_configs,
                base_path=base_path,
                default_builder_kwargs=default_builder_kwargs,
            )
        else:
            builder_configs: list[BuilderConfig] = [import_main_class(module_path).BUILDER_CONFIG_CLASS(data_files=data_files, **default_builder_kwargs,)]
            default_config_name = None
        builder_kwargs = {
            "base_path": self.path,
            "dataset_name": camelcase_to_snakecase(Path(self.path).name),
        }
        if self.data_dir:
            builder_kwargs["data_files"] = data_files
        if os.path.isfile(os.path.join(self.path, data_config.DATASETDICT_INFOS_FILENAME)):
            with open(os.path.join(self.path, data_config.DATASETDICT_INFOS_FILENAME), encoding="utf-8") as f:
                legacy_dataset_infos = DatasetInfosDict({config_name: DatasetInfo.from_dict(dataset_info_dict) for config_name, dataset_info_dict in json.load(f).items()})
                if len(legacy_dataset_infos) == 1:
                    legacy_config_name = next(iter(legacy_dataset_infos))
                    legacy_dataset_infos["default"] = legacy_dataset_infos.pop(legacy_config_name)
            legacy_dataset_infos.update(dataset_infos)
            dataset_infos = legacy_dataset_infos
        if default_config_name is None and len(dataset_infos) == 1:
            default_config_name = next(iter(dataset_infos))

        hash = Hasher.hash({"dataset_infos": dataset_infos, "builder_configs": builder_configs})
        return DatasetModule(
            module_path,
            hash,
            builder_kwargs,
            dataset_infos=dataset_infos,
            builder_configs_parameters=BuilderConfigsParameters(
                metadata_configs=metadata_configs,
                builder_configs=builder_configs,
                default_config_name=default_config_name,
            ),
        )

class CachedDatasetModuleFactory(_DatasetModuleFactory):
    def __init__(self, name: str, cache_dir: Optional[str] = None, dynamic_modules_path: Optional[str] = None):
        self.name = name
        self.cache_dir = cache_dir
        self.dynamic_modules_path = dynamic_modules_path
        assert self.name.count("/") <= 1

    def get_module(self) -> DatasetModule:
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        importable_directory_path = os.path.join(dynamic_modules_path, "datasets", self.name.replace("/", "--"))
        hashes = (
            [h for h in os.listdir(importable_directory_path) if len(h) == 64]
            if os.path.isdir(importable_directory_path)
            else None
        )
        if hashes:
            def _get_modification_time(module_hash):
                return ((Path(importable_directory_path) / module_hash / (self.name.split("/")[-1] + ".py")).stat().st_mtime)

            hash = sorted(hashes, key=_get_modification_time)[-1]
            warning_msg = (f"Using the latest cached version of the module from {os.path.join(importable_directory_path, hash)} (last modified on {time.ctime(_get_modification_time(hash))}) since it couldn't be found locally at {self.name}")
            if not data_config.HF_HUB_OFFLINE:
                warning_msg += ", or remotely on the Hugging Face Hub."
            importable_file_path = _get_importable_file_path(
                dynamic_modules_path=dynamic_modules_path,
                module_namespace="datasets",
                subdirectory_name=hash,
                name=self.name,
            )
            module_path, hash = _load_importable_file(
                dynamic_modules_path=dynamic_modules_path,
                module_namespace="datasets",
                subdirectory_name=hash,
                name=self.name,
            )
            importlib.invalidate_caches()
            builder_kwargs = {"repo_id": self.name,}
            return DatasetModule(module_path, hash, builder_kwargs, importable_file_path=importable_file_path)
        cache_dir = os.path.expanduser(str(self.cache_dir or data_config.HF_DATASETS_CACHE))
        namespace_and_dataset_name = self.name.split("/")
        namespace_and_dataset_name[-1] = camelcase_to_snakecase(namespace_and_dataset_name[-1])
        cached_relative_path = "___".join(namespace_and_dataset_name)
        cached_datasets_directory_path_root = os.path.join(cache_dir, cached_relative_path)
        cached_directory_paths = [
            cached_directory_path
            for cached_directory_path in glob.glob(os.path.join(cached_datasets_directory_path_root, "*", "*", "*"))
            if os.path.isdir(cached_directory_path)
        ]
        if cached_directory_paths:
            builder_kwargs = {
                "repo_id": self.name,
                "dataset_name": self.name.split("/")[-1],
            }
            warning_msg = f"Using the latest cached version of the dataset since {self.name} couldn't be found on the Hugging Face Hub"
            if data_config.HF_HUB_OFFLINE:
                warning_msg += " (offline mode is enabled)."
            return DatasetModule("datasets.packaged_modules.cache.cache", "auto", {**builder_kwargs, "version": "auto"})
        raise FileNotFoundError(f"Dataset {self.name} is not cached in {self.cache_dir}")

class HubDatasetModuleFactoryWithScript(_DatasetModuleFactory):
    def __init__(
        self,
        name: str,
        commit_hash: str,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        dynamic_modules_path: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
    ):
        self.name = name
        self.commit_hash = commit_hash
        self.download_config = download_config or DownloadConfig()
        self.download_mode = download_mode
        self.dynamic_modules_path = dynamic_modules_path
        self.trust_remote_code = trust_remote_code
        increase_load_count(name)

    def download_loading_script(self) -> str:
        file_path = hf_dataset_url(self.name, self.name.split("/")[-1] + ".py", revision=self.commit_hash)
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading builder script"
        return cached_path(file_path, download_config=download_config)

    def download_dataset_infos_file(self) -> str:
        dataset_infos = hf_dataset_url(self.name, data_config.DATASETDICT_INFOS_FILENAME, revision=self.commit_hash)
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading metadata"
        try:
            return cached_path(dataset_infos, download_config=download_config)
        except (FileNotFoundError, ConnectionError):
            return None

    def download_dataset_readme_file(self) -> str:
        readme_url = hf_dataset_url(self.name, data_config.REPOCARD_FILENAME, revision=self.commit_hash)
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading readme"
        try:
            return cached_path(readme_url, download_config=download_config)
        except (FileNotFoundError, ConnectionError):
            return None

    def get_module(self) -> DatasetModule:
        local_path = self.download_loading_script()
        dataset_infos_path = self.download_dataset_infos_file()
        dataset_readme_path = self.download_dataset_readme_file()
        imports = get_imports(local_path)
        local_imports, library_imports = _download_additional_modules(
            name=self.name,
            base_path=hf_dataset_url(self.name, "", revision=self.commit_hash),
            imports=imports,
            download_config=self.download_config,
        )
        additional_files = []
        if dataset_infos_path:
            additional_files.append((data_config.DATASETDICT_INFOS_FILENAME, dataset_infos_path))
        if dataset_readme_path:
            additional_files.append((data_config.REPOCARD_FILENAME, dataset_readme_path))
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        hash = files_to_hash([local_path] + [loc[1] for loc in local_imports])
        importable_file_path = _get_importable_file_path(
            dynamic_modules_path=dynamic_modules_path,
            module_namespace="datasets",
            subdirectory_name=hash,
            name=self.name,
        )
        if not os.path.exists(importable_file_path):
            trust_remote_code = resolve_trust_remote_code(self.trust_remote_code, self.name)
            if trust_remote_code:
                _create_importable_file(
                    local_path=local_path,
                    local_imports=local_imports,
                    additional_files=additional_files,
                    dynamic_modules_path=dynamic_modules_path,
                    module_namespace="datasets",
                    subdirectory_name=hash,
                    name=self.name,
                    download_mode=self.download_mode,
                )
            else:
                raise ValueError(f"Loading {self.name} requires you to execute the dataset script in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set the option `trust_remote_code=True` to remove this error.")
        _check_library_imports(name=self.name, library_imports=library_imports)
        module_path, hash = _load_importable_file(
            dynamic_modules_path=dynamic_modules_path,
            module_namespace="datasets",
            subdirectory_name=hash,
            name=self.name,
        )
        importlib.invalidate_caches()
        builder_kwargs = {
            "base_path": hf_dataset_url(self.name, "", revision=self.commit_hash).rstrip("/"),
            "repo_id": self.name,
        }
        return DatasetModule(module_path, hash, builder_kwargs, importable_file_path=importable_file_path)

def get_authentication_headers_for_url(url: str, token: Optional[Union[str, bool]] = None) -> dict:
    if url.startswith(data_config.HF_ENDPOINT):
        return huggingface_hub.utils.build_hf_headers(token=token, library_name="datasets", library_version=__version__)
    else:
        return {}

def get_exported_dataset_infos(dataset: str, commit_hash: str, token: Optional[Union[str, bool]]) -> dict[str, dict[str, Any]]:
    dataset_viewer_info_url = data_config.HF_ENDPOINT.replace("://", "://datasets-server.") + "/info?dataset="
    try:
        info_response = get_session().get(
            url=dataset_viewer_info_url + dataset,
            headers=get_authentication_headers_for_url(data_config.HF_ENDPOINT + f"datasets/{dataset}", token=token),
            timeout=100.0,
        )
        info_response.raise_for_status()
        if "X-Revision" in info_response.headers:
            if info_response.headers["X-Revision"] == commit_hash or commit_hash is None:
                info_response = info_response.json()
                if (
                    info_response.get("partial") is False
                    and not info_response.get("pending", True)
                    and not info_response.get("failed", True)
                    and "dataset_info" in info_response
                ):
                    return info_response["dataset_info"]
    except Exception as e:  # noqa catch any exception of the dataset viewer API and consider the dataset info doesn't exist
        pass
    raise DatasetViewerError("No exported dataset infos available.")

class HubDatasetModuleFactoryWithoutScript(_DatasetModuleFactory):
    def __init__(
        self,
        name: str,
        commit_hash: str,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, list, dict]] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        use_exported_dataset_infos: bool = False,
    ):
        self.name = name
        self.commit_hash = commit_hash
        self.data_files = data_files
        self.data_dir = data_dir
        self.download_config = download_config or DownloadConfig()
        self.download_mode = download_mode
        self.use_exported_dataset_infos = use_exported_dataset_infos
        increase_load_count(name)

    def get_module(self) -> DatasetModule:
        api = HfApi(
            endpoint=data_config.HF_ENDPOINT,
            token=self.download_config.token,
            library_name="datasets",
            library_version=__version__,
            user_agent=get_datasets_user_agent(self.download_config.user_agent),
        )
        try:
            dataset_readme_path = api.hf_hub_download(
                repo_id=self.name,
                filename=data_config.REPOCARD_FILENAME,
                repo_type="dataset",
                revision=self.commit_hash,
                proxies=self.download_config.proxies,
            )
            dataset_card_data = DatasetCard.load(dataset_readme_path).data
        except EntryNotFoundError:
            dataset_card_data = DatasetCardData()
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading standalone yaml"
        try:
            standalone_yaml_path = cached_path(
                hf_dataset_url(self.name, data_config.REPOYAML_FILENAME, revision=self.commit_hash),
                download_config=download_config,
            )
            with open(standalone_yaml_path, encoding="utf-8") as f:
                standalone_yaml_data = yaml.safe_load(f.read())
                if standalone_yaml_data:
                    _dataset_card_data_dict = dataset_card_data.to_dict()
                    _dataset_card_data_dict.update(standalone_yaml_data)
                    dataset_card_data = DatasetCardData(**_dataset_card_data_dict)
        except FileNotFoundError:
            pass
        base_path = f"hf://datasets/{self.name}@{self.commit_hash}/{self.data_dir or ''}".rstrip("/")
        metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card_data)
        dataset_infos = DatasetInfosDict.from_dataset_card_data(dataset_card_data)
        if data_config.USE_PARQUET_EXPORT and self.use_exported_dataset_infos:
            try:
                exported_dataset_infos = get_exported_dataset_infos(
                    dataset=self.name, commit_hash=self.commit_hash, token=self.download_config.token
                )
                exported_dataset_infos = DatasetInfosDict(
                    {
                        config_name: DatasetInfo.from_dict(exported_dataset_infos[config_name])
                        for config_name in exported_dataset_infos
                    }
                )
            except DatasetViewerError:
                exported_dataset_infos = None
        else:
            exported_dataset_infos = None
        if exported_dataset_infos:
            exported_dataset_infos.update(dataset_infos)
            dataset_infos = exported_dataset_infos
        if self.data_files is not None:
            patterns = sanitize_patterns(self.data_files)
        elif metadata_configs and not self.data_dir and "data_files" in next(iter(metadata_configs.values())):
            patterns = sanitize_patterns(next(iter(metadata_configs.values()))["data_files"])
        else:
            patterns = get_data_patterns(base_path, download_config=self.download_config)
        data_files = DataFilesDict.from_patterns(
            patterns,
            base_path=base_path,
            allowed_extensions=ALL_ALLOWED_EXTENSIONS,
            download_config=self.download_config,
        )
        module_name, default_builder_kwargs = infer_module_for_data_files(
            data_files=data_files,
            path=self.name,
            download_config=self.download_config,
        )
        data_files = data_files.filter(file_names=_MODULE_TO_METADATA_FILE_NAMES[module_name])
        module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]
        if metadata_configs:
            builder_configs, default_config_name = create_builder_configs_from_metadata_configs(
                module_path,
                metadata_configs,
                base_path=base_path,
                default_builder_kwargs=default_builder_kwargs,
                download_config=self.download_config,
            )
        else:
            builder_configs: list[BuilderConfig] = [
                import_main_class(module_path).BUILDER_CONFIG_CLASS(
                    data_files=data_files,
                    **default_builder_kwargs,
                )
            ]
            default_config_name = None
        builder_kwargs = {
            "base_path": hf_dataset_url(self.name, "", revision=self.commit_hash).rstrip("/"),
            "repo_id": self.name,
            "dataset_name": camelcase_to_snakecase(Path(self.name).name),
        }
        if self.data_dir:
            builder_kwargs["data_files"] = data_files
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading metadata"
        try:
            dataset_infos_path = cached_path(hf_dataset_url(self.name, data_config.DATASETDICT_INFOS_FILENAME, revision=self.commit_hash), download_config=download_config,)
            with open(dataset_infos_path, encoding="utf-8") as f:
                legacy_dataset_infos = DatasetInfosDict(
                    {
                        config_name: DatasetInfo.from_dict(dataset_info_dict)
                        for config_name, dataset_info_dict in json.load(f).items()
                    }
                )
                if len(legacy_dataset_infos) == 1:
                    legacy_config_name = next(iter(legacy_dataset_infos))
                    legacy_dataset_infos["default"] = legacy_dataset_infos.pop(legacy_config_name)
            legacy_dataset_infos.update(dataset_infos)
            dataset_infos = legacy_dataset_infos
        except FileNotFoundError:
            pass
        if default_config_name is None and len(dataset_infos) == 1:
            default_config_name = next(iter(dataset_infos))

        return DatasetModule(
            module_path,
            self.commit_hash,
            builder_kwargs,
            dataset_infos=dataset_infos,
            builder_configs_parameters=BuilderConfigsParameters(
                metadata_configs=metadata_configs,
                builder_configs=builder_configs,
                default_config_name=default_config_name,
            ),
        )

def get_exported_parquet_files(dataset: str, commit_hash: str, token: Optional[Union[str, bool]]) -> list[dict[str, Any]]:
    dataset_viewer_parquet_url = data_config.HF_ENDPOINT.replace("://", "://datasets-server.") + "/parquet?dataset="
    try:
        parquet_data_files_response = get_session().get(
            url=dataset_viewer_parquet_url + dataset,
            headers=get_authentication_headers_for_url(data_config.HF_ENDPOINT + f"datasets/{dataset}", token=token),
            timeout=100.0,
        )
        parquet_data_files_response.raise_for_status()
        if "X-Revision" in parquet_data_files_response.headers:
            if parquet_data_files_response.headers["X-Revision"] == commit_hash or commit_hash is None:
                parquet_data_files_response_json = parquet_data_files_response.json()
                if (
                    parquet_data_files_response_json.get("partial") is False
                    and not parquet_data_files_response_json.get("pending", True)
                    and not parquet_data_files_response_json.get("failed", True)
                    and "parquet_files" in parquet_data_files_response_json
                ):
                    return parquet_data_files_response_json["parquet_files"]
    except Exception as e:  # noqa catch any exception of the dataset viewer API and consider the parquet export doesn't exist
        pass
    raise DatasetViewerError("No exported Parquet files available.")

class HubDatasetModuleFactoryWithParquetExport(_DatasetModuleFactory):
    def __init__(self, name: str, commit_hash: str, download_config: Optional[DownloadConfig] = None):
        self.name = name
        self.commit_hash = commit_hash
        self.download_config = download_config or DownloadConfig()
        increase_load_count(name)

    def get_module(self) -> DatasetModule:
        exported_parquet_files = get_exported_parquet_files(dataset=self.name, commit_hash=self.commit_hash, token=self.download_config.token)
        exported_dataset_infos = get_exported_dataset_infos(dataset=self.name, commit_hash=self.commit_hash, token=self.download_config.token)
        dataset_infos = DatasetInfosDict(
            {
                config_name: DatasetInfo.from_dict(exported_dataset_infos[config_name])
                for config_name in exported_dataset_infos
            }
        )
        parquet_commit_hash = (
            HfApi(
                endpoint=data_config.HF_ENDPOINT,
                token=self.download_config.token,
                library_name="datasets",
                library_version=__version__,
                user_agent=get_datasets_user_agent(self.download_config.user_agent),
            )
            .dataset_info(
                self.name,
                revision="refs/convert/parquet",
                token=self.download_config.token,
                timeout=100.0,
            )
            .sha
        )  # fix the revision in case there are new commits in the meantime
        metadata_configs = MetadataConfigs._from_exported_parquet_files_and_dataset_infos(
            parquet_commit_hash=parquet_commit_hash,
            exported_parquet_files=exported_parquet_files,
            dataset_infos=dataset_infos,
        )
        module_path, _ = _PACKAGED_DATASETS_MODULES["parquet"]
        builder_configs, default_config_name = create_builder_configs_from_metadata_configs(
            module_path,
            metadata_configs,
            download_config=self.download_config,
        )
        builder_kwargs = {
            "repo_id": self.name,
            "dataset_name": camelcase_to_snakecase(Path(self.name).name),
        }

        return DatasetModule(
            module_path,
            self.commit_hash,
            builder_kwargs,
            dataset_infos=dataset_infos,
            builder_configs_parameters=BuilderConfigsParameters(
                metadata_configs=metadata_configs,
                builder_configs=builder_configs,
                default_config_name=default_config_name,
            ),
        )

