import fsspec
import os
import re
import requests
import posixpath
import yaml

from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Optional, Union, Callable
from collections.abc import Mapping, Sequence

from huggingface_hub import HfApi
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from large_class import (
    Dataset,
    DatasetDict,
    IterableDatasetDict, 
    IterableDataset,
    DatasetBuilder,
    DatasetModule,
    DataFilesDict,
    EmptyDatasetError,
    PathLike,
    _PACKAGED_DATASETS_MODULES,
    ImageFolder,
    PackagedDatasetModuleFactory,
    LocalDatasetModuleFactoryWithScript,
    LocalDatasetModuleFactoryWithoutScript,
    CachedDatasetModuleFactory,
    HubDatasetModuleFactoryWithScript,
    HubDatasetModuleFactoryWithoutScript,
    HubDatasetModuleFactoryWithParquetExport,
    import_main_class,
    lock_importable_file,
    BuilderConfig,
)
from data_utils import (
    Version,
    DownloadConfig,
    Features,
    Split,
    DownloadMode,
    VerificationMode,
    EntryNotFoundError,
    DatasetNotFoundError,
    DataFilesNotFoundError,
    DatasetViewerError,
)
from data_utils import (
    is_small_dataset,
    url_to_fs,
    is_relative_path,
    relative_to_absolute_path,
    __version__,
    get_datasets_user_agent,
    _raise_if_offline_mode_is_enabled,
    snakecase_to_camelcase,
)

import data_config


def load_dataset():
    # "frgfm/imagenette"
    # "shunk031/MSCOCO"
    pass


INVALID_WINDOWS_CHARACTERS_IN_PATH = r"<>:/\|?*"
REGEX_YAML_BLOCK = re.compile(r"^(\s*---[\r\n]+)([\S\s]*?)([\r\n]+---(\r\n|\n|$))")
SANITIZED_DEFAULT_SPLIT = str(Split.TRAIN)
REPOCARD_NAME = "README.md"
METADATA_CONFIGS_FIELD = "configs"
TEMPLATE_MODELCARD_PATH = Path(__file__).parent / "templates" / "modelcard_template.md"
TEMPLATE_DATASETCARD_PATH = Path(__file__).parent / "templates" / "datasetcard_template.md"
yaml_dump: Callable[..., str] = partial(yaml.dump, stream=None, allow_unicode=True)  # type: ignore

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

def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: Optional[Union[str, Split]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    streaming: bool = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[dict] = None,
    trust_remote_code: Optional[bool] = None,
    **config_kwargs,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    if data_files is not None and not data_files:
        raise ValueError(f"Empty 'data_files': '{data_files}'. It should be either non-empty or None (default).")
    if Path(path, data_config.DATASET_STATE_JSON_FILENAME).exists():
        raise ValueError("You are trying to load a dataset that was saved using `save_to_disk`. Please use `load_from_disk` instead.")

    if streaming and num_proc is not None:
        raise NotImplementedError("Loading a streaming dataset in parallel with `num_proc` is not implemented. To parallelize streaming, you can wrap the dataset with a PyTorch DataLoader using `num_workers` > 1 instead.")

    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    verification_mode = VerificationMode((verification_mode or VerificationMode.BASIC_CHECKS) if not save_infos else VerificationMode.ALL_CHECKS)

    builder_instance = load_dataset_builder(
        path=path,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
        features=features,
        download_config=download_config,
        download_mode=download_mode,
        revision=revision,
        token=token,
        storage_options=storage_options,
        trust_remote_code=trust_remote_code,
        _require_default_config_name=name is None,
        **config_kwargs,
    )

    if streaming:
        return builder_instance.as_streaming_dataset(split=split)

    builder_instance.download_and_prepare(
        download_config=download_config,
        download_mode=download_mode,
        verification_mode=verification_mode,
        num_proc=num_proc,
        storage_options=storage_options,
    )

    keep_in_memory = (keep_in_memory if keep_in_memory is not None else is_small_dataset(builder_instance.info.dataset_size))
    ds = builder_instance.as_dataset(split=split, verification_mode=verification_mode, in_memory=keep_in_memory)
    if save_infos:
        builder_instance._save_infos()

    return ds


def load_from_disk(dataset_path: PathLike, keep_in_memory: Optional[bool] = None, storage_options: Optional[dict] = None) -> Union[Dataset, DatasetDict]:
    fs: fsspec.AbstractFileSystem
    fs, *_ = url_to_fs(dataset_path, **(storage_options or {}))
    if not fs.exists(dataset_path):
        raise FileNotFoundError(f"Directory {dataset_path} not found")
    if fs.isfile(posixpath.join(dataset_path, data_config.DATASET_INFO_FILENAME)) and fs.isfile(posixpath.join(dataset_path, data_config.DATASET_STATE_JSON_FILENAME)):
        return Dataset.load_from_disk(dataset_path, keep_in_memory=keep_in_memory, storage_options=storage_options)
    elif fs.isfile(posixpath.join(dataset_path, data_config.DATASETDICT_JSON_FILENAME)):
        return DatasetDict.load_from_disk(dataset_path, keep_in_memory=keep_in_memory, storage_options=storage_options)
    else:
        raise FileNotFoundError(f"Directory {dataset_path} is neither a `Dataset` directory nor a `DatasetDict` directory.")

def load_dataset_builder(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    storage_options: Optional[dict] = None,
    trust_remote_code: Optional[bool] = None,
    _require_default_config_name=True,
    **config_kwargs,
) -> DatasetBuilder:
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    if token is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.token = token
    if storage_options is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.storage_options.update(storage_options)
    dataset_module = dataset_module_factory(
        path,
        revision=revision,
        download_config=download_config,
        download_mode=download_mode,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        _require_default_config_name=_require_default_config_name,
        _require_custom_configs=bool(config_kwargs),
    )
    builder_kwargs = dataset_module.builder_kwargs
    data_dir = builder_kwargs.pop("data_dir", data_dir)
    data_files = builder_kwargs.pop("data_files", data_files)
    config_name = builder_kwargs.pop("config_name", name or dataset_module.builder_configs_parameters.default_config_name)
    dataset_name = builder_kwargs.pop("dataset_name", None)
    info = dataset_module.dataset_infos.get(config_name) if dataset_module.dataset_infos else None

    if (
        path in _PACKAGED_DATASETS_MODULES
        and data_files is None
        and dataset_module.builder_configs_parameters.builder_configs[0].data_files is None
    ):
        error_msg = f"Please specify the data files or data directory to load for the {path} dataset builder."
        example_extensions = [extension for extension in _EXTENSION_TO_MODULE if _EXTENSION_TO_MODULE[extension] == path]
        if example_extensions:
            error_msg += f'\nFor example `data_files={{"train": "path/to/data/train/*.{example_extensions[0]}"}}`'
        raise ValueError(error_msg)

    builder_cls = get_dataset_builder_class(dataset_module, dataset_name=dataset_name)
    builder_instance: DatasetBuilder = builder_cls(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        config_name=config_name,
        data_dir=data_dir,
        data_files=data_files,
        hash=dataset_module.hash,
        info=info,
        features=features,
        token=token,
        storage_options=storage_options,
        **builder_kwargs,
        **config_kwargs,
    )
    builder_instance._use_legacy_cache_dir_if_possible(dataset_module)

    return builder_instance

def dataset_module_factory(
    path: str,
    revision: Optional[Union[str, Version]] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    dynamic_modules_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[dict, list, str, DataFilesDict]] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    _require_default_config_name=True,
    _require_custom_configs=False,
    **download_kwargs,
) -> DatasetModule:
    if download_config is None:
        download_config = DownloadConfig(**download_kwargs)
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    download_config.extract_compressed_file = True
    download_config.force_extract = True
    download_config.force_download = download_mode == DownloadMode.FORCE_REDOWNLOAD

    filename = list(filter(lambda x: x, path.replace(os.sep, "/").split("/")))[-1]
    if not filename.endswith(".py"):
        filename = filename + ".py"
    combined_path = os.path.join(path, filename)

    
    if path in _PACKAGED_DATASETS_MODULES:
        return PackagedDatasetModuleFactory(
            path,
            data_dir=data_dir,
            data_files=data_files,
            download_config=download_config,
            download_mode=download_mode,
        ).get_module()
    elif path.endswith(filename):
        if os.path.isfile(path):
            return LocalDatasetModuleFactoryWithScript(
                path,
                download_mode=download_mode,
                dynamic_modules_path=dynamic_modules_path,
                trust_remote_code=trust_remote_code,
            ).get_module()
        else:
            raise FileNotFoundError(f"Couldn't find a dataset script at {relative_to_absolute_path(path)}")
    elif os.path.isfile(combined_path):
        return LocalDatasetModuleFactoryWithScript(
            combined_path,
            download_mode=download_mode,
            dynamic_modules_path=dynamic_modules_path,
            trust_remote_code=trust_remote_code,
        ).get_module()
    elif os.path.isdir(path):
        return LocalDatasetModuleFactoryWithoutScript(
            path, data_dir=data_dir, data_files=data_files, download_mode=download_mode
        ).get_module()
    elif is_relative_path(path) and path.count("/") <= 1:
        try:
            api = HfApi(
                endpoint=data_config.HF_ENDPOINT,
                token=download_config.token,
                library_name="datasets",
                library_version=__version__,
                user_agent=get_datasets_user_agent(download_config.user_agent),
            )
            try:
                _raise_if_offline_mode_is_enabled()
                dataset_readme_path = api.hf_hub_download(
                    repo_id=path,
                    filename=data_config.REPOCARD_FILENAME,
                    repo_type="dataset",
                    revision=revision,
                    proxies=download_config.proxies,
                )
                commit_hash = os.path.basename(os.path.dirname(dataset_readme_path))
            except LocalEntryNotFoundError as e:
                if isinstance(e.__cause__, (OfflineModeIsEnabled, requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
                    raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({e.__class__.__name__})") from e
                else:
                    raise
            except EntryNotFoundError:
                commit_hash = api.dataset_info(
                    path,
                    revision=revision,
                    timeout=100.0,
                ).sha
            except (OfflineModeIsEnabled, requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({e.__class__.__name__})") from e
            except GatedRepoError as e:
                message = f"Dataset '{path}' is a gated dataset on the Hub."
                if e.response.status_code == 401:
                    message += " You must be authenticated to access it."
                elif e.response.status_code == 403:
                    message += f" Visit the dataset page at https://huggingface.co/datasets/{path} to ask for access."
                raise DatasetNotFoundError(message) from e
            except RevisionNotFoundError as e:
                raise DatasetNotFoundError(f"Revision '{revision}' doesn't exist for dataset '{path}' on the Hub.") from e
            except RepositoryNotFoundError as e:
                raise DatasetNotFoundError(f"Dataset '{path}' doesn't exist on the Hub or cannot be accessed.") from e
            try:
                dataset_script_path = api.hf_hub_download(
                    repo_id=path,
                    filename=filename,
                    repo_type="dataset",
                    revision=commit_hash,
                    proxies=download_config.proxies,
                )
                if _require_custom_configs or (revision and revision != "main"):
                    can_load_config_from_parquet_export = False
                elif _require_default_config_name:
                    with open(dataset_script_path, encoding="utf-8") as f:
                        can_load_config_from_parquet_export = "DEFAULT_CONFIG_NAME" not in f.read()
                else:
                    can_load_config_from_parquet_export = True
                if data_config.USE_PARQUET_EXPORT and can_load_config_from_parquet_export:
                    try:
                        out = HubDatasetModuleFactoryWithParquetExport(path, download_config=download_config, commit_hash=commit_hash).get_module()
                        return out
                    except DatasetViewerError:
                        pass
                return HubDatasetModuleFactoryWithScript(
                    path,
                    commit_hash=commit_hash,
                    download_config=download_config,
                    download_mode=download_mode,
                    dynamic_modules_path=dynamic_modules_path,
                    trust_remote_code=trust_remote_code,
                ).get_module()
            except EntryNotFoundError:
                if data_dir or data_files or (revision and revision != "main"):
                    use_exported_dataset_infos = False
                else:
                    use_exported_dataset_infos = True
                return HubDatasetModuleFactoryWithoutScript(
                    path,
                    commit_hash=commit_hash,
                    data_dir=data_dir,
                    data_files=data_files,
                    download_config=download_config,
                    download_mode=download_mode,
                    use_exported_dataset_infos=use_exported_dataset_infos,
                ).get_module()
            except GatedRepoError as e:
                message = f"Dataset '{path}' is a gated dataset on the Hub."
                if e.response.status_code == 401:
                    message += " You must be authenticated to access it."
                elif e.response.status_code == 403:
                    message += f" Visit the dataset page at https://huggingface.co/datasets/{path} to ask for access."
                raise DatasetNotFoundError(message) from e
            except RevisionNotFoundError as e:
                raise DatasetNotFoundError(f"Revision '{revision}' doesn't exist for dataset '{path}' on the Hub.") from e
        except Exception as e1:
            try:
                return CachedDatasetModuleFactory(path, dynamic_modules_path=dynamic_modules_path, cache_dir=cache_dir).get_module()
            except Exception:
                if isinstance(e1, OfflineModeIsEnabled):
                    raise ConnectionError(f"Couldn't reach the Hugging Face Hub for dataset '{path}': {e1}") from None
                if isinstance(e1, (DataFilesNotFoundError, DatasetNotFoundError, EmptyDatasetError)):
                    raise e1 from None
                if isinstance(e1, FileNotFoundError):
                    if trust_remote_code:
                        raise FileNotFoundError(f"Couldn't find a dataset script at {relative_to_absolute_path(combined_path)} or any data file in the same directory. Couldn't find '{path}' on the Hugging Face Hub either: {type(e1).__name__}: {e1}") from None
                    else:
                        raise FileNotFoundError(f"Couldn't find any data file at {relative_to_absolute_path(path)}. Couldn't find '{path}' on the Hugging Face Hub either: {type(e1).__name__}: {e1}") from None
                raise e1 from None
    elif trust_remote_code:
        raise FileNotFoundError(f"Couldn't find a dataset script at {relative_to_absolute_path(combined_path)} or any data file in the same directory.")
    else:
        raise FileNotFoundError(f"Couldn't find any data file at {relative_to_absolute_path(path)}.")

def get_dataset_builder_class(dataset_module: "DatasetModule", dataset_name: Optional[str] = None) -> type[DatasetBuilder]:
    with (lock_importable_file(dataset_module.importable_file_path) if dataset_module.importable_file_path else nullcontext()):
        builder_cls = import_main_class(dataset_module.module_path)
    if dataset_module.builder_configs_parameters.builder_configs:
        dataset_name = dataset_name or dataset_module.builder_kwargs.get("dataset_name")
        if dataset_name is None:
            raise ValueError("dataset_name should be specified but got None")
        builder_cls = configure_builder_class(
            builder_cls,
            builder_configs=dataset_module.builder_configs_parameters.builder_configs,
            default_config_name=dataset_module.builder_configs_parameters.default_config_name,
            dataset_name=dataset_name,
        )
    return builder_cls

class _InitializeConfiguredDatasetBuilder:
    def __call__(self, builder_cls, metadata_configs, default_config_name, name):
        obj = _InitializeConfiguredDatasetBuilder()
        obj.__class__ = configure_builder_class(builder_cls, metadata_configs, default_config_name=default_config_name, dataset_name=name)
        return obj

def configure_builder_class(
    builder_cls: type[DatasetBuilder],
    builder_configs: list[BuilderConfig],
    default_config_name: Optional[str],
    dataset_name: str,
) -> type[DatasetBuilder]:
    
    class ConfiguredDatasetBuilder(builder_cls):
        BUILDER_CONFIGS = builder_configs
        DEFAULT_CONFIG_NAME = default_config_name

        __module__ = builder_cls.__module__  # so that the actual packaged builder can be imported

        def __reduce__(self):  # to make dynamically created class pickable, see _InitializeParameterizedDatasetBuilder
            parent_builder_cls = self.__class__.__mro__[1]
            return (
                _InitializeConfiguredDatasetBuilder(),
                (
                    parent_builder_cls,
                    self.BUILDER_CONFIGS,
                    self.DEFAULT_CONFIG_NAME,
                    self.dataset_name,
                ),
                self.__dict__.copy(),
            )

    ConfiguredDatasetBuilder.__name__ = (f"{builder_cls.__name__.lower().capitalize()}{snakecase_to_camelcase(dataset_name)}")
    ConfiguredDatasetBuilder.__qualname__ = (f"{builder_cls.__name__.lower().capitalize()}{snakecase_to_camelcase(dataset_name)}")

    return ConfiguredDatasetBuilder
