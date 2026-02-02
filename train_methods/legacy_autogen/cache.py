import os
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

import diskcache

class AbstractCache(Protocol):
    """
    This protocol defines the basic interface for cache operations.
    Implementing classes should provide concrete implementations for
    these methods to handle caching mechanisms.
    """

    def get(self, key: str, default: Any | None = None) -> Any | None:
        """
        Retrieve an item from the cache.

        Args:
            key (str): The key identifying the item in the cache.
            default (optional): The default value to return if the key is not found.
                                Defaults to None.

        Returns:
            The value associated with the key if found, else the default value.
        """
        ...

    def set(self, key: str, value: Any) -> None:
        """
        Set an item in the cache.

        Args:
            key (str): The key under which the item is to be stored.
            value: The value to be stored in the cache.
        """
        ...

    def close(self) -> None:
        """
        Close the cache. Perform any necessary cleanup, such as closing network connections or
        releasing resources.
        """
        ...

    def __enter__(self) -> Self:
        """
        Enter the runtime context related to this object.

        The with statement will bind this method's return value to the target(s)
        specified in the as clause of the statement, if any.
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Exit the runtime context and close the cache.

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_value: The exception value if an exception was raised in the context.
            traceback: The traceback if an exception was raised in the context.
        """
        ...

class DiskCache(AbstractCache):
    """
    Implementation of AbstractCache using the DiskCache library.

    This class provides a concrete implementation of the AbstractCache
    interface using the diskcache library for caching data on disk.

    Attributes:
        cache (diskcache.Cache): The DiskCache instance used for caching.

    Methods:
        __init__(self, seed): Initializes the DiskCache with the given seed.
        get(self, key, default=None): Retrieves an item from the cache.
        set(self, key, value): Sets an item in the cache.
        close(self): Closes the cache.
        __enter__(self): Context management entry.
        __exit__(self, exc_type, exc_value, traceback): Context management exit.
    """

    def __init__(self, seed: str | int):
        """
        Initialize the DiskCache instance.

        Args:
            seed (str | int): A seed or namespace for the cache. This is used to create
                        a unique storage location for the cache data.

        """
        self.cache = diskcache.Cache(seed)

    def get(self, key: str, default: Any | None = None) -> Any | None:
        """
        Retrieve an item from the cache.

        Args:
            key (str): The key identifying the item in the cache.
            default (optional): The default value to return if the key is not found.
                                Defaults to None.

        Returns:
            The value associated with the key if found, else the default value.
        """
        return self.cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set an item in the cache.

        Args:
            key (str): The key under which the item is to be stored.
            value: The value to be stored in the cache.
        """
        self.cache.set(key, value)

    def close(self) -> None:
        """
        Close the cache.

        Perform any necessary cleanup, such as closing file handles or
        releasing resources.
        """
        self.cache.close()

    def __enter__(self) -> Self:
        """
        Enter the runtime context related to the object.

        Returns:
            self: The instance itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Exit the runtime context related to the object.

        Perform cleanup actions such as closing the cache.

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_value: The exception value if an exception was raised in the context.
            traceback: The traceback if an exception was raised in the context.
        """
        self.close()

class CacheFactory:
    @staticmethod
    def cache_factory(
        seed: str | int,
        cache_path_root: str = ".cache",
    ) -> AbstractCache:
        """
        Factory function for creating cache instances.

        Args:
            seed (str | int): Used as a seed or namespace for the cache.
            cache_path_root (str): Root path for the disk cache.

        Returns:
            An instance of DiskCache

        """
        # Default to DiskCache if neither Redis nor Cosmos DB configurations are provided
        path = Path(cache_path_root, str(seed))
        return DiskCache(os.path.join(".", path))

class Cache(AbstractCache):
    """
    A wrapper class for managing cache configuration and instances.

    This class provides a unified interface for creating and interacting with
    different types of cache (e.g., Redis, Disk). It abstracts the underlying
    cache implementation details, providing methods for cache operations.

    Attributes:
        config (dict[str, Any]): A dictionary containing cache configuration.
        cache: The cache instance created based on the provided configuration.
    """

    ALLOWED_CONFIG_KEYS = [
        "cache_seed",
        "cache_path_root",
    ]

    @staticmethod
    def disk(cache_seed: str | int = 42, cache_path_root: str = ".cache") -> "Cache":
        """
        Create a Disk cache instance.

        Args:
            cache_seed (str | int, optional): A seed for the cache. Defaults to 42.
            cache_path_root (str, optional): The root path for the disk cache. Defaults to ".cache".

        Returns:
            Cache: A Cache instance configured for Disk caching.
        """
        return Cache({"cache_seed": cache_seed, "cache_path_root": cache_path_root})

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the Cache with the given configuration.

        Validates the configuration keys and creates the cache instance.

        Args:
            config (dict[str, Any]): A dictionary containing the cache configuration.

        Raises:
            ValueError: If an invalid configuration key is provided.
        """
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
        """
        Enter the runtime context related to the cache object.

        Returns:
            The cache instance for use within a context block.
        """
        return self.cache.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Exit the runtime context related to the cache object.

        Cleans up the cache instance and handles any exceptions that occurred
        within the context.

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_value: The exception value if an exception was raised in the context.
            traceback: The traceback if an exception was raised in the context.
        """
        return self.cache.__exit__(exc_type, exc_value, traceback)

    def get(self, key: str, default: Any | None = None) -> Any | None:
        """
        Retrieve an item from the cache.

        Args:
            key (str): The key identifying the item in the cache.
            default (optional): The default value to return if the key is not found.
                                Defaults to None.

        Returns:
            The value associated with the key if found, else the default value.
        """
        return self.cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set an item in the cache.

        Args:
            key (str): The key under which the item is to be stored.
            value: The value to be stored in the cache.
        """
        self.cache.set(key, value)

    def close(self) -> None:
        """
        Close the cache.

        Perform any necessary cleanup, such as closing connections or releasing resources.
        """
        self.cache.close()
