"""Benchmark system adapters."""

from src.benchmark.adapters.factory import BenchmarkAdapterFactory
from src.benchmark.adapters.protocol import BenchmarkSystemProtocol

__all__ = [
    "BenchmarkAdapterFactory",
    "BenchmarkSystemProtocol",
]
