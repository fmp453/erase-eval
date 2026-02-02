import os
import pickle
from types import TracebackType
from typing import Any, Protocol, Self, TypedDict

import diskcache
import redis
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError

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

class RedisCache(AbstractCache):
    """
    Implementation of AbstractCache using the Redis database.

    This class provides a concrete implementation of the AbstractCache
    interface using the Redis database for caching data.

    Attributes:
        seed (str | int): A seed or namespace used as a prefix for cache keys.
        cache (redis.Redis): The Redis client used for caching.

    Methods:
        __init__(self, seed, redis_url): Initializes the RedisCache with the given seed and Redis URL.
        _prefixed_key(self, key): Internal method to get a namespaced cache key.
        get(self, key, default=None): Retrieves an item from the cache.
        set(self, key, value): Sets an item in the cache.
        close(self): Closes the Redis client.
        __enter__(self): Context management entry.
        __exit__(self, exc_type, exc_value, traceback): Context management exit.
    """

    def __init__(self, seed: str | int, redis_url: str):
        """
        Initialize the RedisCache instance.

        Args:
            seed (str | int): A seed or namespace for the cache. This is used as a prefix for all cache keys.
            redis_url (str): The URL for the Redis server.

        """
        self.seed = seed
        self.cache = redis.Redis.from_url(redis_url)

    def _prefixed_key(self, key: str) -> str:
        """
        Get a namespaced key for the cache.

        Args:
            key (str): The original key.

        Returns:
            str: The namespaced key.
        """
        return f"autogen:{self.seed}:{key}"

    def get(self, key: str, default: Any | None = None) -> Any | None:
        """
        Retrieve an item from the Redis cache.

        Args:
            key (str): The key identifying the item in the cache.
            default (optional): The default value to return if the key is not found.
                                Defaults to None.

        Returns:
            The deserialized value associated with the key if found, else the default value.
        """
        result = self.cache.get(self._prefixed_key(key))
        if result is None:
            return default
        return pickle.loads(result)

    def set(self, key: str, value: Any) -> None:
        """
        Set an item in the Redis cache.

        Args:
            key (str): The key under which the item is to be stored.
            value: The value to be stored in the cache.

        Notes:
            The value is serialized using pickle before being stored in Redis.
        """
        serialized_value = pickle.dumps(value)
        self.cache.set(self._prefixed_key(key), serialized_value)

    def close(self) -> None:
        """
        Close the Redis client.

        Perform any necessary cleanup, such as closing network connections.
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
        exc_val: BaseException | None,
        exc_tb: TracebackType | None
    ) -> None:
        """
        Exit the runtime context related to the object.

        Perform cleanup actions such as closing the Redis client.

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_value: The exception value if an exception was raised in the context.
            traceback: The traceback if an exception was raised in the context.
        """
        self.close()

class CosmosDBConfig(TypedDict, total=False):
    connection_string: str
    database_id: str
    container_id: str
    cache_seed: str | int | None
    client: CosmosClient | None

class CosmosDBCache(AbstractCache):
    """
    Synchronous implementation of AbstractCache using Azure Cosmos DB NoSQL API.

    This class provides a concrete implementation of the AbstractCache
    interface using Azure Cosmos DB for caching data, with synchronous operations.

    Attributes:
        seed (str | int): A seed or namespace used as a partition key.
        client (CosmosClient): The Cosmos DB client used for caching.
        container: The container instance used for caching.
    """

    def __init__(self, seed: str | int, cosmosdb_config: CosmosDBConfig):
        """
        Initialize the CosmosDBCache instance.

        Args:
            seed (str | int): A seed or namespace for the cache, used as a partition key.
            connection_string (str): The connection string for the Cosmos DB account.
            container_id (str): The container ID to be used for caching.
            client (Optional[CosmosClient]): An existing CosmosClient instance to be used for caching.
        """
        self.seed = str(seed)
        self.client = cosmosdb_config.get("client") or CosmosClient.from_connection_string(
            cosmosdb_config["connection_string"]
        )
        database_id = cosmosdb_config.get("database_id", "autogen_cache")
        self.database = self.client.get_database_client(database_id)
        container_id = cosmosdb_config.get("container_id")
        self.container = self.database.create_container_if_not_exists(
            id=container_id, partition_key=PartitionKey(path="/partitionKey")
        )

    @classmethod
    def create_cache(cls, seed: str | int, cosmosdb_config: CosmosDBConfig):
        """
        Factory method to create a CosmosDBCache instance based on the provided configuration.
        This method decides whether to use an existing CosmosClient or create a new one.
        """
        if "client" in cosmosdb_config and isinstance(cosmosdb_config["client"], CosmosClient):
            return cls.from_existing_client(seed, **cosmosdb_config)
        else:
            return cls.from_config(seed, cosmosdb_config)

    @classmethod
    def from_config(cls, seed: str | int, cosmosdb_config: CosmosDBConfig):
        return cls(str(seed), cosmosdb_config)

    @classmethod
    def from_connection_string(cls, seed: str | int, connection_string: str, database_id: str, container_id: str):
        config = {"connection_string": connection_string, "database_id": database_id, "container_id": container_id}
        return cls(str(seed), config)

    @classmethod
    def from_existing_client(cls, seed: str | int, client: CosmosClient, database_id: str, container_id: str):
        config = {"client": client, "database_id": database_id, "container_id": container_id}
        return cls(str(seed), config)

    def get(self, key: str, default: Any | None = None) -> Any | None:
        """
        Retrieve an item from the Cosmos DB cache.

        Args:
            key (str): The key identifying the item in the cache.
            default (optional): The default value to return if the key is not found.

        Returns:
            The deserialized value associated with the key if found, else the default value.
        """
        try:
            response = self.container.read_item(item=key, partition_key=str(self.seed))
            return pickle.loads(response["data"])
        except CosmosResourceNotFoundError:
            return default
        except Exception as e:
            # Log the exception or rethrow after logging if needed
            # Consider logging or handling the error appropriately here
            raise e

    def set(self, key: str, value: Any) -> None:
        """
        Set an item in the Cosmos DB cache.

        Args:
            key (str): The key under which the item is to be stored.
            value: The value to be stored in the cache.

        Notes:
            The value is serialized using pickle before being stored.
        """
        try:
            serialized_value = pickle.dumps(value)
            item = {"id": key, "partitionKey": str(self.seed), "data": serialized_value}
            self.container.upsert_item(item)
        except Exception as e:
            # Log or handle exception
            raise e

    def close(self) -> None:
        """
        Close the Cosmos DB client.

        Perform any necessary cleanup, such as closing network connections.
        """
        # CosmosClient doesn"t require explicit close in the current SDK
        # If you created the client inside this class, you should close it if necessary
        pass

    def __enter__(self):
        """
        Context management entry.

        Returns:
            self: The instance itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: Exception | None,
        traceback: Any | None,
    ) -> None:
        """
        Context management exit.

        Perform cleanup actions such as closing the Cosmos DB client.
        """
        self.close()

class CacheFactory:
    @staticmethod
    def cache_factory(
        seed: str | int,
        redis_url: str | None = None,
        cache_path_root: str = ".cache",
        cosmosdb_config: dict[str, Any] | None = None,
    ) -> AbstractCache:
        """
        Factory function for creating cache instances.

        This function decides whether to create a RedisCache, DiskCache, or CosmosDBCache instance
        based on the provided parameters. If RedisCache is available and a redis_url is provided,
        a RedisCache instance is created. If connection_string, database_id, and container_id
        are provided, a CosmosDBCache is created. Otherwise, a DiskCache instance is used.

        Args:
            seed (str | int): Used as a seed or namespace for the cache.
            redis_url (str | None): URL for the Redis server.
            cache_path_root (str): Root path for the disk cache.
            cosmosdb_config (Optional[Dict[str, str]]): Dictionary containing 'connection_string', 'database_id', and 'container_id' for Cosmos DB cache.

        Returns:
            An instance of RedisCache, DiskCache, or CosmosDBCache.

        ```

        """
        if redis_url:
            return RedisCache(seed, redis_url)

        if cosmosdb_config:
            return CosmosDBCache.create_cache(seed, cosmosdb_config)

        # Default to DiskCache if neither Redis nor Cosmos DB configurations are provided
        path = os.path.join(cache_path_root, str(seed))
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
        "redis_url",
        "cache_path_root",
        "cosmos_db_config",
    ]

    @staticmethod
    def redis(cache_seed: str | int = 42, redis_url: str = "redis://localhost:6379/0") -> "Cache":
        """
        Create a Redis cache instance.

        Args:
            cache_seed (str | int, optional): A seed for the cache. Defaults to 42.
            redis_url (str, optional): The URL for the Redis server. Defaults to "redis://localhost:6379/0".

        Returns:
            Cache: A Cache instance configured for Redis.
        """
        return Cache({"cache_seed": cache_seed, "redis_url": redis_url})

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

    @staticmethod
    def cosmos_db(
        connection_string: str | None = None,
        container_id: str | None = None,
        cache_seed: str | int = 42,
        client: Any | None = None,
    ) -> "Cache":
        """
        Create a Cosmos DB cache instance with 'autogen_cache' as database ID.

        Args:
            connection_string (str, optional): Connection string to the Cosmos DB account.
            container_id (str, optional): The container ID for the Cosmos DB account.
            cache_seed (str | int, optional): A seed for the cache.
            client: Optional[CosmosClient]: Pass an existing Cosmos DB client.
        Returns:
            Cache: A Cache instance configured for Cosmos DB.
        """
        cosmos_db_config = {
            "connection_string": connection_string,
            "database_id": "autogen_cache",
            "container_id": container_id,
            "client": client,
        }
        return Cache({"cache_seed": str(cache_seed), "cosmos_db_config": cosmos_db_config})

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
            redis_url=self.config.get("redis_url", ""),
            cache_path_root=self.config.get("cache_path_root", ""),
            cosmosdb_config=self.config.get("cosmos_db_config", ""),
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
