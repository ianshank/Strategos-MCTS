"""
Benchmark Tests for Chess Verification.

Performance tests for verification components to ensure
acceptable latency and throughput.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from src.games.chess.action_space import ChessActionEncoder
from src.games.chess.config import ChessActionSpaceConfig
from src.games.chess.state import ChessGameState
from src.games.chess.verification import (
    ChessGameVerifier,
    ChessVerificationFactory,
    MoveValidator,
    create_game_verifier,
    create_move_validator,
)
from tests.games.chess.builders import (
    ChessGameSequenceBuilder,
    ChessPositionBuilder,
    initial_position,
)


def benchmark(
    func: Callable[[], None],
    iterations: int = 100,
    warmup: int = 10,
) -> dict[str, float]:
    """Run a benchmark and return timing statistics.

    Args:
        func: Function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    times.sort()
    return {
        "min_ms": times[0],
        "max_ms": times[-1],
        "mean_ms": sum(times) / len(times),
        "median_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "p99_ms": times[int(len(times) * 0.99)],
        "iterations": iterations,
    }


class TestMoveValidatorBenchmarks:
    """Benchmark tests for MoveValidator."""

    @pytest.fixture
    def validator(self) -> MoveValidator:
        """Create a move validator."""
        return create_move_validator()

    @pytest.mark.benchmark
    def test_single_move_validation_latency(self, validator: MoveValidator) -> None:
        """Benchmark: Single move validation should be fast."""
        state = initial_position()
        move = "e2e4"

        def validate() -> None:
            validator.validate_move(state, move)

        stats = benchmark(validate, iterations=1000, warmup=100)

        # Assert latency requirements
        assert stats["median_ms"] < 1.0, f"Median latency {stats['median_ms']}ms exceeds 1ms"
        assert stats["p99_ms"] < 5.0, f"P99 latency {stats['p99_ms']}ms exceeds 5ms"

        print(f"\nMove validation latency: median={stats['median_ms']:.3f}ms, p99={stats['p99_ms']:.3f}ms")

    @pytest.mark.benchmark
    def test_all_legal_moves_validation(self, validator: MoveValidator) -> None:
        """Benchmark: Validating all legal moves from a position."""
        state = initial_position()
        legal_moves = state.get_legal_actions()

        def validate_all() -> None:
            for move in legal_moves:
                validator.validate_move(state, move)

        stats = benchmark(validate_all, iterations=100, warmup=10)

        # 20 moves in initial position
        per_move_ms = stats["median_ms"] / len(legal_moves)

        assert per_move_ms < 0.5, f"Per-move latency {per_move_ms}ms exceeds 0.5ms"

        print(f"\nAll moves validation: total={stats['median_ms']:.3f}ms, per_move={per_move_ms:.3f}ms")

    @pytest.mark.benchmark
    def test_complex_position_validation(self, validator: MoveValidator) -> None:
        """Benchmark: Validation in complex middlegame positions."""
        # Complex middlegame with many legal moves
        complex_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        state = ChessGameState.from_fen(complex_fen)
        legal_moves = state.get_legal_actions()

        def validate_all() -> None:
            for move in legal_moves:
                validator.validate_move(state, move)

        stats = benchmark(validate_all, iterations=100, warmup=10)

        per_move_ms = stats["median_ms"] / max(len(legal_moves), 1)

        assert per_move_ms < 1.0, f"Per-move latency {per_move_ms}ms exceeds 1ms"

        print(f"\nComplex position: {len(legal_moves)} moves, per_move={per_move_ms:.3f}ms")


class TestGameVerifierBenchmarks:
    """Benchmark tests for ChessGameVerifier."""

    @pytest.fixture
    def verifier(self) -> ChessGameVerifier:
        """Create a game verifier."""
        return create_game_verifier()

    @pytest.mark.benchmark
    def test_short_sequence_verification(self, verifier: ChessGameVerifier) -> None:
        """Benchmark: Short move sequence verification."""
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]
        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        def verify() -> None:
            verifier.verify_move_sequence(initial_fen, moves)

        stats = benchmark(verify, iterations=100, warmup=10)

        per_move_ms = stats["median_ms"] / len(moves)

        assert stats["median_ms"] < 10.0, f"Total latency {stats['median_ms']}ms exceeds 10ms"

        print(f"\nShort sequence ({len(moves)} moves): total={stats['median_ms']:.3f}ms, per_move={per_move_ms:.3f}ms")

    @pytest.mark.benchmark
    def test_long_sequence_verification(self, verifier: ChessGameVerifier) -> None:
        """Benchmark: Long move sequence verification."""
        # Longer game sequence
        moves = [
            "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6",
            "d2d3", "f8e7", "c2c3", "e8g8", "e1g1", "d7d6",
            "h2h3", "c8e6", "c4e6", "f7e6", "b1d2", "a7a5",
            "d2c4", "d8c8", "c1e3", "b7b6", "d1d2", "c6a7",
        ]
        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        def verify() -> None:
            verifier.verify_move_sequence(initial_fen, moves)

        stats = benchmark(verify, iterations=50, warmup=5)

        per_move_ms = stats["median_ms"] / len(moves)

        # Allow more time for longer sequences
        assert stats["median_ms"] < 100.0, f"Total latency {stats['median_ms']}ms exceeds 100ms"
        assert per_move_ms < 5.0, f"Per-move latency {per_move_ms}ms exceeds 5ms"

        print(f"\nLong sequence ({len(moves)} moves): total={stats['median_ms']:.3f}ms, per_move={per_move_ms:.3f}ms")

    @pytest.mark.benchmark
    def test_fen_parsing_throughput(self, verifier: ChessGameVerifier) -> None:
        """Benchmark: FEN parsing throughput."""
        test_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "8/8/4k3/8/4P3/8/4K3/8 w - - 0 1",
        ]

        def parse_all() -> None:
            for fen in test_fens:
                verifier.verify_position(fen)

        stats = benchmark(parse_all, iterations=100, warmup=10)

        per_fen_ms = stats["median_ms"] / len(test_fens)

        assert per_fen_ms < 1.0, f"Per-FEN latency {per_fen_ms}ms exceeds 1ms"

        print(f"\nFEN parsing: per_fen={per_fen_ms:.3f}ms")


class TestActionEncoderBenchmarks:
    """Benchmark tests for ChessActionEncoder."""

    @pytest.fixture
    def encoder(self) -> ChessActionEncoder:
        """Create an action encoder."""
        return ChessActionEncoder(ChessActionSpaceConfig())

    @pytest.mark.benchmark
    def test_encoding_throughput(self, encoder: ChessActionEncoder) -> None:
        """Benchmark: Move encoding throughput."""
        test_moves = [
            "e2e4", "d2d4", "g1f3", "b1c3", "f1c4",
            "c1f4", "e1g1", "d1d2", "a1d1", "h2h3",
        ]

        def encode_all() -> None:
            for move in test_moves:
                encoder.encode(move)

        stats = benchmark(encode_all, iterations=1000, warmup=100)

        per_move_us = (stats["median_ms"] / len(test_moves)) * 1000  # microseconds

        assert per_move_us < 100, f"Per-move encoding {per_move_us}μs exceeds 100μs"

        print(f"\nMove encoding: per_move={per_move_us:.1f}μs")

    @pytest.mark.benchmark
    def test_decoding_throughput(self, encoder: ChessActionEncoder) -> None:
        """Benchmark: Move decoding throughput."""
        # Decode a range of indices
        indices = list(range(0, 1000, 10))

        def decode_all() -> None:
            for idx in indices:
                encoder.decode(idx)

        stats = benchmark(decode_all, iterations=100, warmup=10)

        per_decode_us = (stats["median_ms"] / len(indices)) * 1000  # microseconds

        assert per_decode_us < 100, f"Per-decode latency {per_decode_us}μs exceeds 100μs"

        print(f"\nMove decoding: per_decode={per_decode_us:.1f}μs")

    @pytest.mark.benchmark
    def test_legal_move_mask_generation(self, encoder: ChessActionEncoder) -> None:
        """Benchmark: Legal move mask generation."""
        state = initial_position()

        def generate_mask() -> None:
            encoder.get_legal_action_mask(state)

        stats = benchmark(generate_mask, iterations=100, warmup=10)

        assert stats["median_ms"] < 5.0, f"Mask generation {stats['median_ms']}ms exceeds 5ms"

        print(f"\nMask generation: median={stats['median_ms']:.3f}ms")


class TestFactoryBenchmarks:
    """Benchmark tests for factory creation."""

    @pytest.mark.benchmark
    def test_factory_creation(self) -> None:
        """Benchmark: Factory instantiation time."""

        def create_factory() -> None:
            ChessVerificationFactory()

        stats = benchmark(create_factory, iterations=100, warmup=10)

        assert stats["median_ms"] < 10.0, f"Factory creation {stats['median_ms']}ms exceeds 10ms"

        print(f"\nFactory creation: median={stats['median_ms']:.3f}ms")

    @pytest.mark.benchmark
    def test_validator_creation(self) -> None:
        """Benchmark: MoveValidator creation time."""
        factory = ChessVerificationFactory()

        def create_validator() -> None:
            factory.create_move_validator(reuse_encoder=False)

        stats = benchmark(create_validator, iterations=100, warmup=10)

        assert stats["median_ms"] < 50.0, f"Validator creation {stats['median_ms']}ms exceeds 50ms"

        print(f"\nValidator creation: median={stats['median_ms']:.3f}ms")

    @pytest.mark.benchmark
    def test_validator_with_cache(self) -> None:
        """Benchmark: MoveValidator creation with cached encoder."""
        factory = ChessVerificationFactory()
        # Prime the cache
        factory.create_move_validator(reuse_encoder=True)

        def create_validator() -> None:
            factory.create_move_validator(reuse_encoder=True)

        stats = benchmark(create_validator, iterations=100, warmup=10)

        # Should be faster with cache
        assert stats["median_ms"] < 10.0, f"Cached validator creation {stats['median_ms']}ms exceeds 10ms"

        print(f"\nCached validator creation: median={stats['median_ms']:.3f}ms")


class TestGameStateOperationsBenchmarks:
    """Benchmark tests for ChessGameState operations."""

    @pytest.mark.benchmark
    def test_state_creation_from_fen(self) -> None:
        """Benchmark: State creation from FEN."""
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

        def create_state() -> None:
            ChessGameState.from_fen(fen)

        stats = benchmark(create_state, iterations=1000, warmup=100)

        assert stats["median_ms"] < 1.0, f"State creation {stats['median_ms']}ms exceeds 1ms"

        print(f"\nState creation: median={stats['median_ms']:.3f}ms")

    @pytest.mark.benchmark
    def test_legal_move_generation(self) -> None:
        """Benchmark: Legal move generation."""
        state = initial_position()

        def generate_moves() -> None:
            state.get_legal_actions()

        stats = benchmark(generate_moves, iterations=1000, warmup=100)

        assert stats["median_ms"] < 0.5, f"Move generation {stats['median_ms']}ms exceeds 0.5ms"

        print(f"\nLegal move generation: median={stats['median_ms']:.3f}ms")

    @pytest.mark.benchmark
    def test_move_application(self) -> None:
        """Benchmark: Move application."""
        state = initial_position()

        def apply_move() -> None:
            state.apply_action("e2e4")

        stats = benchmark(apply_move, iterations=1000, warmup=100)

        assert stats["median_ms"] < 0.5, f"Move application {stats['median_ms']}ms exceeds 0.5ms"

        print(f"\nMove application: median={stats['median_ms']:.3f}ms")

    @pytest.mark.benchmark
    def test_game_phase_detection(self) -> None:
        """Benchmark: Game phase detection."""
        positions = [
            initial_position(),
            ChessGameState.from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
            ChessGameState.from_fen("8/8/4k3/8/4P3/8/4K3/8 w - - 0 1"),
        ]

        def detect_phases() -> None:
            for state in positions:
                state.get_game_phase()

        stats = benchmark(detect_phases, iterations=500, warmup=50)

        per_position_ms = stats["median_ms"] / len(positions)

        assert per_position_ms < 0.5, f"Phase detection {per_position_ms}ms exceeds 0.5ms"

        print(f"\nPhase detection: per_position={per_position_ms:.3f}ms")


class TestBuilderBenchmarks:
    """Benchmark tests for test builders."""

    @pytest.mark.benchmark
    def test_position_builder_throughput(self) -> None:
        """Benchmark: Position builder throughput."""

        def build_positions() -> None:
            ChessPositionBuilder().with_initial_position().build()
            ChessPositionBuilder().with_middlegame().build()
            ChessPositionBuilder().with_endgame().build()

        stats = benchmark(build_positions, iterations=100, warmup=10)

        per_build_ms = stats["median_ms"] / 3

        assert per_build_ms < 2.0, f"Per-build latency {per_build_ms}ms exceeds 2ms"

        print(f"\nPosition builder: per_build={per_build_ms:.3f}ms")

    @pytest.mark.benchmark
    def test_sequence_builder_throughput(self) -> None:
        """Benchmark: Sequence builder throughput."""

        def build_sequences() -> None:
            ChessGameSequenceBuilder().with_opening("scholars_mate").build()
            ChessGameSequenceBuilder().with_opening("fools_mate").build()
            ChessGameSequenceBuilder().with_opening("italian_game").build()

        stats = benchmark(build_sequences, iterations=100, warmup=10)

        per_build_ms = stats["median_ms"] / 3

        assert per_build_ms < 5.0, f"Per-build latency {per_build_ms}ms exceeds 5ms"

        print(f"\nSequence builder: per_build={per_build_ms:.3f}ms")


class TestMemoryBenchmarks:
    """Memory usage benchmarks."""

    @pytest.mark.benchmark
    def test_state_memory_footprint(self) -> None:
        """Benchmark: Memory footprint of ChessGameState."""
        import sys

        state = initial_position()
        base_size = sys.getsizeof(state)

        # Create many states
        [ChessGameState.from_fen(initial_position().fen) for _ in range(100)]

        # Estimate per-state memory
        # This is approximate due to shared references
        print(f"\nState memory: base_size≈{base_size} bytes")

    @pytest.mark.benchmark
    def test_encoder_memory_footprint(self) -> None:
        """Benchmark: Memory footprint of ChessActionEncoder."""

        encoder = ChessActionEncoder(ChessActionSpaceConfig())

        # Estimate memory
        # Note: actual memory usage depends on internal structures
        print(f"\nEncoder action_size: {encoder.action_size}")
