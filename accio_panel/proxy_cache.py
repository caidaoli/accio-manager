"""账号代理选择缓存层 - 减少热路径重复计算"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Account
    from .store import AccountStore


class ProxyCandidateCache:
    """账号候选池缓存 - 按模型/提供商维度缓存过滤结果"""

    def __init__(self, ttl: float = 2.0):
        """
        Args:
            ttl: 缓存生存时间（秒），默认 2 秒平衡新鲜度与命中率
        """
        self._cache: dict[str, list[Account]] = {}
        self._cache_time: dict[str, float] = {}
        self._ttl = ttl

    def _make_key(
        self,
        model: str | None,
        provider: str | None,
        exclude_ids: frozenset[str] | None,
    ) -> str:
        """生成缓存键"""
        provider_part = provider or "*"
        model_part = model or "*"
        exclude_part = ""
        if exclude_ids:
            # 排序确保键稳定性
            exclude_part = f":exclude={','.join(sorted(exclude_ids))}"
        return f"{provider_part}:{model_part}{exclude_part}"

    def get_candidates(
        self,
        store: AccountStore,
        model: str | None,
        provider: str | None,
        exclude_account_ids: set[str] | None,
        filter_func,
    ) -> list[Account] | None:
        """
        获取缓存的候选账号列表

        Args:
            store: 账号存储
            model: 模型名称
            provider: 提供商名称
            exclude_account_ids: 排除的账号 ID 集合
            filter_func: 过滤函数，签名 (store, model, provider, exclude_ids) -> list[Account]

        Returns:
            缓存命中时返回账号列表，未命中返回 None
        """
        exclude_frozen = frozenset(exclude_account_ids) if exclude_account_ids else None
        cache_key = self._make_key(model, provider, exclude_frozen)

        now = time.time()

        # 检查缓存是否有效
        if cache_key in self._cache:
            cache_age = now - self._cache_time[cache_key]
            if cache_age < self._ttl:
                return self._cache[cache_key]

        # 缓存未命中，调用过滤函数
        candidates = filter_func(store, model, provider, exclude_account_ids)

        # 更新缓存
        self._cache[cache_key] = candidates
        self._cache_time[cache_key] = now

        return candidates

    def invalidate(self, model: str | None = None, provider: str | None = None) -> None:
        """
        使缓存失效

        Args:
            model: 指定模型的缓存失效，None 表示全部
            provider: 指定提供商的缓存失效，None 表示全部
        """
        if model is None and provider is None:
            # 清空全部缓存
            self._cache.clear()
            self._cache_time.clear()
            return

        # 按前缀匹配删除
        prefix = self._make_key(model, provider, None)
        keys_to_delete = [
            key for key in self._cache if key.startswith(prefix.split(":exclude=")[0])
        ]
        for key in keys_to_delete:
            self._cache.pop(key, None)
            self._cache_time.pop(key, None)

    def get_stats(self) -> dict[str, int]:
        """获取缓存统计信息"""
        now = time.time()
        valid_entries = sum(
            1
            for key, timestamp in self._cache_time.items()
            if (now - timestamp) < self._ttl
        )
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
        }

    def cleanup_expired(self) -> int:
        """清理过期缓存条目，返回清理数量"""
        now = time.time()
        expired_keys = [
            key
            for key, timestamp in self._cache_time.items()
            if (now - timestamp) >= self._ttl
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_time.pop(key, None)
        return len(expired_keys)
