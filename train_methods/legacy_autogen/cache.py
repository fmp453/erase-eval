from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

import diskcache

class AbstractCache(Protocol):

    def get(self, key: str, default: Any | None = None) -> Any | None:
        ...

    def set(self, key: str, value: Any) -> None:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> Self:
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        ...

class DiskCache(AbstractCache):
    def __init__(self, seed: str | int):
        self.cache = diskcache.Cache(seed)

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return self.cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.cache.set(key, value)

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

class CacheFactory:
    @staticmethod
    def cache_factory(
        seed: str | int,
        cache_path_root: str = ".cache",
    ) -> AbstractCache:
        path = Path(cache_path_root, str(seed))
        return DiskCache(Path(".", path))

class Cache(AbstractCache):
    ALLOWED_CONFIG_KEYS = [
        "cache_seed",
        "cache_path_root",
    ]

    @staticmethod
    def disk(cache_seed: str | int = 42, cache_path_root: str = ".cache") -> "Cache":
        return Cache({"cache_seed": cache_seed, "cache_path_root": cache_path_root})

    def __init__(self, config: dict[str, Any]):
        self.config = config
        # Ensure that the seed is always treated as a string before being passed to any cache factory or stored.
        self.config["cache_seed"] = str(self.config.get("cache_seed", 42))

        # validate config
        for key in self.config.keys():
            if key not in self.ALLOWED_CONFIG_KEYS:
                raise ValueError(f"Invalid config key: {key}")
        # create cache instance
        self.cache = CacheFactory.cache_factory(
            seed=self.config["cache_seed"],
            cache_path_root=self.config.get("cache_path_root", ""),
        )

    def __enter__(self) -> "Cache":
        return self.cache.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return self.cache.__exit__(exc_type, exc_value, traceback)

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return self.cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.cache.set(key, value)

    def close(self) -> None:
        self.cache.close()
