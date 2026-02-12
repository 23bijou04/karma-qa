"""Caching layer for DBpedia queries."""

import hashlib
import json
import time
from typing import Any, Optional
from functools import lru_cache


class InMemoryCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.cache: dict[str, tuple[Any, float]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl

    def _make_key(self, key: str) -> str:
        """Create a hash key for the cache."""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired."""
        hashed_key = self._make_key(key)
        if hashed_key in self.cache:
            value, expiry = self.cache[hashed_key]
            if time.time() < expiry:
                return value
            else:
                del self.cache[hashed_key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        if len(self.cache) >= self.max_size:
            self._evict_expired()
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

        hashed_key = self._make_key(key)
        expiry = time.time() + (ttl or self.default_ttl)
        self.cache[hashed_key] = (value, expiry)

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            k for k, (_, expiry) in self.cache.items() if current_time >= expiry
        ]
        for k in expired_keys:
            del self.cache[k]

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()


class RedisCache:
    """Redis-based cache for distributed caching."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600,
        prefix: str = "dbpedia_qa:",
    ):
        self.default_ttl = default_ttl
        self.prefix = prefix
        self._client = None

        try:
            import redis

            self._client = redis.Redis(
                host=host, port=port, db=db, password=password, decode_responses=True
            )
            self._client.ping()
        except Exception:
            self._client = None

    @property
    def is_available(self) -> bool:
        """Check if Redis is available."""
        return self._client is not None

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{hashlib.md5(key.encode()).hexdigest()}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._client:
            return None
        try:
            prefixed_key = self._make_key(key)
            value = self._client.get(prefixed_key)
            if value:
                return json.loads(value)
        except Exception:
            pass
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache."""
        if not self._client:
            return
        try:
            prefixed_key = self._make_key(key)
            self._client.setex(
                prefixed_key, ttl or self.default_ttl, json.dumps(value)
            )
        except Exception:
            pass

    def clear(self) -> None:
        """Clear all cache entries with prefix."""
        if not self._client:
            return
        try:
            keys = self._client.keys(f"{self.prefix}*")
            if keys:
                self._client.delete(*keys)
        except Exception:
            pass


class CacheManager:
    """Unified cache manager supporting both in-memory and Redis caching."""

    def __init__(
        self,
        use_redis: bool = False,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        default_ttl: int = 3600,
        max_memory_size: int = 10000,
    ):
        self.default_ttl = default_ttl
        self._memory_cache = InMemoryCache(
            max_size=max_memory_size, default_ttl=default_ttl
        )
        self._redis_cache: Optional[RedisCache] = None

        if use_redis:
            self._redis_cache = RedisCache(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                default_ttl=default_ttl,
            )

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then Redis)."""
        value = self._memory_cache.get(key)
        if value is not None:
            return value

        if self._redis_cache and self._redis_cache.is_available:
            value = self._redis_cache.get(key)
            if value is not None:
                self._memory_cache.set(key, value)
                return value

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in both memory and Redis cache."""
        self._memory_cache.set(key, value, ttl)
        if self._redis_cache and self._redis_cache.is_available:
            self._redis_cache.set(key, value, ttl)

    def clear(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        if self._redis_cache:
            self._redis_cache.clear()

    def cached(self, ttl: Optional[int] = None):
        """Decorator for caching function results."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result

            return wrapper

        return decorator
