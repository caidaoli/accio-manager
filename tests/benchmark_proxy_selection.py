"""性能基准测试 - 账号选择与存储"""
from __future__ import annotations

import statistics
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from accio_panel.app_settings import PanelSettings
from accio_panel.client import AccioClient
from accio_panel.config import Settings
from accio_panel.models import Account
from accio_panel.proxy_selection import _ordered_proxy_candidates, _cached_quota_view
from accio_panel.store import AccountStore


def create_test_account(index: int) -> Account:
    """创建测试账号"""
    return Account(
        id=f"test-{index:04d}",
        name=f"账号{index}",
        access_token=f"access-token-{index}",
        refresh_token=f"refresh-token-{index}",
        utdid=f"utdid-{index}",
        fill_priority=index % 10,
        manual_enabled=True,
        auto_disabled=False,
        last_remaining_quota=1000000 - (index * 1000),
        last_total_quota=1000000,
        added_at="2024-01-01 00:00:00",
        updated_at="2024-01-01 00:00:00",
    )


class BenchmarkProxySelection(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.accounts_dir = self.data_dir / "accounts"
        self.accounts_dir.mkdir(parents=True, exist_ok=True)

        self.store = AccountStore(
            accounts_dir=self.accounts_dir,
            legacy_file_path=None,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_benchmark_account_list(self):
        """基准测试: 账号列表读取"""
        account_counts = [10, 50, 100, 200]

        print("\n=== 账号列表读取基准测试 ===")
        for count in account_counts:
            # 准备数据
            for i in range(count):
                self.store.save(create_test_account(i))

            # 预热
            self.store.list_accounts()

            # 测量
            latencies = []
            for _ in range(100):
                start = time.perf_counter()
                self.store.list_accounts()
                latencies.append((time.perf_counter() - start) * 1000)

            print(f"\n{count} 账号:")
            print(f"  P50: {statistics.median(latencies):.2f}ms")
            print(f"  P95: {statistics.quantiles(latencies, n=20)[18]:.2f}ms")
            print(f"  P99: {statistics.quantiles(latencies, n=100)[98]:.2f}ms")
            print(f"  平均: {statistics.mean(latencies):.2f}ms")

    def test_benchmark_candidate_filtering(self):
        """基准测试: 账号候选过滤"""
        account_count = 100

        # 准备数据
        for i in range(account_count):
            account = create_test_account(i)
            # 30% 禁用
            if i % 3 == 0:
                account.manual_enabled = False
            # 20% 自动禁用
            if i % 5 == 0:
                account.auto_disabled = True
            self.store.save(account)

        print("\n=== 账号候选过滤基准测试 ===")
        print(f"总账号数: {account_count}")

        # 预热
        _ordered_proxy_candidates(self.store)

        # 测量
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            candidates = _ordered_proxy_candidates(self.store)
            latencies.append((time.perf_counter() - start) * 1000)

        print(f"有效候选数: {len(candidates)}")
        print(f"P50: {statistics.median(latencies):.2f}ms")
        print(f"P95: {statistics.quantiles(latencies, n=20)[18]:.2f}ms")
        print(f"P99: {statistics.quantiles(latencies, n=100)[98]:.2f}ms")
        print(f"平均: {statistics.mean(latencies):.2f}ms")

    def test_benchmark_concurrent_read(self):
        """基准测试: 并发读取"""
        account_count = 50
        concurrency = 10

        # 准备数据
        for i in range(account_count):
            self.store.save(create_test_account(i))

        print("\n=== 并发读取基准测试 ===")
        print(f"账号数: {account_count}, 并发数: {concurrency}")

        # 预热
        self.store.list_accounts()

        # 测量
        def read_task():
            start = time.perf_counter()
            self.store.list_accounts()
            return (time.perf_counter() - start) * 1000

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(read_task) for _ in range(100)]
            latencies = [f.result() for f in futures]

        print(f"P50: {statistics.median(latencies):.2f}ms")
        print(f"P95: {statistics.quantiles(latencies, n=20)[18]:.2f}ms")
        print(f"P99: {statistics.quantiles(latencies, n=100)[98]:.2f}ms")
        print(f"平均: {statistics.mean(latencies):.2f}ms")

    def test_benchmark_quota_view_calculation(self):
        """基准测试: 额度视图计算"""
        account_count = 100

        # 准备数据
        accounts = [create_test_account(i) for i in range(account_count)]

        print("\n=== 额度视图计算基准测试 ===")
        print(f"账号数: {account_count}")

        # 测量
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            for account in accounts:
                _cached_quota_view(account)
            latencies.append((time.perf_counter() - start) * 1000)

        print(f"P50: {statistics.median(latencies):.2f}ms")
        print(f"P95: {statistics.quantiles(latencies, n=20)[18]:.2f}ms")
        print(f"P99: {statistics.quantiles(latencies, n=100)[98]:.2f}ms")
        print(f"平均: {statistics.mean(latencies):.2f}ms")

    def test_benchmark_write_operations(self):
        """基准测试: 写操作"""
        account_count = 50

        print("\n=== 写操作基准测试 ===")
        print(f"账号数: {account_count}")

        # 测量创建
        create_latencies = []
        for i in range(account_count):
            start = time.perf_counter()
            self.store.save(create_test_account(i))
            create_latencies.append((time.perf_counter() - start) * 1000)

        print("\n创建操作:")
        print(f"  P50: {statistics.median(create_latencies):.2f}ms")
        print(f"  P95: {statistics.quantiles(create_latencies, n=20)[18]:.2f}ms")
        print(f"  平均: {statistics.mean(create_latencies):.2f}ms")

        # 测量更新
        accounts = self.store.list_accounts()
        update_latencies = []
        for account in accounts[:20]:
            account.last_remaining_quota = 500000
            start = time.perf_counter()
            self.store.save(account)
            update_latencies.append((time.perf_counter() - start) * 1000)

        print("\n更新操作:")
        print(f"  P50: {statistics.median(update_latencies):.2f}ms")
        print(f"  P95: {statistics.quantiles(update_latencies, n=20)[18]:.2f}ms")
        print(f"  平均: {statistics.mean(update_latencies):.2f}ms")


if __name__ == "__main__":
    # 运行基准测试
    suite = unittest.TestLoader().loadTestsFromTestCase(BenchmarkProxySelection)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
