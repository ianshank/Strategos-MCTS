"""
Unit tests for learning dashboard.

Tests:
- Dashboard initialization
- Statistics computation
- Report generation
- Data export

Best Practices 2025:
- Isolated test data
- Temporary file handling
- HTML validation
"""

import json
import os
import tempfile

import pytest

from examples.chess_demo.learning_dashboard import (
    AgentStats,
    DashboardConfig,
    GameStats,
    LearningDashboard,
    create_dashboard,
)


class TestDashboardConfig:
    """Tests for DashboardConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DashboardConfig()

        assert config.max_records_in_memory > 0
        assert config.learning_db_path
        assert config.output_dir

    def test_custom_config(self):
        """Test custom configuration."""
        config = DashboardConfig(
            learning_db_path="/tmp/custom.jsonl",
            output_dir="/tmp/custom_output",
            max_records_in_memory=500,
        )

        assert config.learning_db_path == "/tmp/custom.jsonl"
        assert config.output_dir == "/tmp/custom_output"
        assert config.max_records_in_memory == 500


class TestAgentStats:
    """Tests for AgentStats dataclass."""

    def test_agent_stats_creation(self):
        """Test creating agent stats."""
        stats = AgentStats(name="HRM")

        assert stats.name == "HRM"
        assert stats.total_decisions == 0
        assert stats.times_selected == 0
        assert stats.avg_confidence == 0.0

    def test_agent_stats_with_data(self):
        """Test agent stats with values."""
        stats = AgentStats(
            name="MCTS",
            total_decisions=100,
            times_selected=45,
            avg_confidence=0.85,
            avg_time_ms=150.5,
            agreement_rate=0.45,
        )

        assert stats.total_decisions == 100
        assert stats.agreement_rate == 0.45


class TestGameStats:
    """Tests for GameStats dataclass."""

    def test_game_stats_creation(self):
        """Test creating game stats."""
        stats = GameStats(game_id="test_game")

        assert stats.game_id == "test_game"
        assert stats.total_moves == 0


class TestLearningDashboard:
    """Tests for LearningDashboard."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name
            # Write some test records
            records = [
                {
                    "timestamp": "2025-01-01T00:00:00Z",
                    "game_id": "game_1",
                    "move_number": 1,
                    "fen_before": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "fen_after": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                    "selected_move": "e2e4",
                    "agent_results": [
                        {"agent": "HRM", "move": "e2e4", "confidence": 0.8, "time_ms": 100},
                        {"agent": "TRM", "move": "e2e4", "confidence": 0.75, "time_ms": 80},
                        {"agent": "MCTS", "move": "d2d4", "confidence": 0.7, "time_ms": 200},
                        {"agent": "SYMBOLIC", "move": "e2e4", "confidence": 0.65, "time_ms": 50},
                    ],
                    "ensemble_confidence": 0.85,
                    "consensus_achieved": True,
                    "time_to_decide_ms": 500,
                    "game_phase": "opening",
                    "evaluation_before": 0.0,
                    "evaluation_after": 0.1,
                },
                {
                    "timestamp": "2025-01-01T00:01:00Z",
                    "game_id": "game_1",
                    "move_number": 2,
                    "fen_before": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                    "fen_after": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
                    "selected_move": "e7e5",
                    "agent_results": [
                        {"agent": "HRM", "move": "e7e5", "confidence": 0.82, "time_ms": 90},
                        {"agent": "TRM", "move": "e7e5", "confidence": 0.78, "time_ms": 85},
                        {"agent": "MCTS", "move": "e7e5", "confidence": 0.8, "time_ms": 180},
                        {"agent": "SYMBOLIC", "move": "d7d5", "confidence": 0.6, "time_ms": 45},
                    ],
                    "ensemble_confidence": 0.8,
                    "consensus_achieved": False,
                    "time_to_decide_ms": 450,
                    "game_phase": "opening",
                    "evaluation_before": 0.1,
                    "evaluation_after": 0.05,
                },
            ]
            for record in records:
                f.write(json.dumps(record) + "\n")

        # File is now closed and flushed
        yield temp_path

        # Cleanup
        os.unlink(temp_path)

    @pytest.fixture
    def dashboard(self, temp_db):
        """Create dashboard with test data."""
        config = DashboardConfig(
            learning_db_path=temp_db,
            output_dir=tempfile.mkdtemp(),
        )
        return LearningDashboard(config)

    def test_dashboard_loads_records(self, dashboard):
        """Test that dashboard loads records from file."""
        assert len(dashboard._records) == 2

    def test_get_summary(self, dashboard):
        """Test getting dashboard summary."""
        summary = dashboard.get_summary()

        assert summary["status"] == "ok"
        assert "overall" in summary
        assert "agents" in summary
        assert "phases" in summary
        assert "recent_games" in summary

        overall = summary["overall"]
        assert overall["total_games"] == 1
        assert overall["total_moves"] == 2
        assert 0 < overall["avg_confidence"] < 1

    def test_get_agent_comparison(self, dashboard):
        """Test agent comparison data."""
        comparison = dashboard.get_agent_comparison()

        assert "agents" in comparison
        assert len(comparison["agents"]) == 4  # HRM, TRM, MCTS, SYMBOLIC

        # Check agent data
        hrm_data = next(a for a in comparison["agents"] if a["name"] == "HRM")
        assert hrm_data["total_decisions"] == 2

    def test_get_confidence_trend(self, dashboard):
        """Test confidence trend data."""
        trend = dashboard.get_confidence_trend()

        assert isinstance(trend, list)
        # Should have at least one data point
        if trend:
            assert "avg_confidence" in trend[0]
            assert "avg_time_ms" in trend[0]

    def test_get_game_details(self, dashboard):
        """Test getting details for a specific game."""
        details = dashboard.get_game_details("game_1")

        assert details["game_id"] == "game_1"
        assert details["total_moves"] == 2
        assert "moves" in details
        assert len(details["moves"]) == 2
        assert "evaluation_progression" in details

    def test_get_game_details_not_found(self, dashboard):
        """Test getting details for non-existent game."""
        details = dashboard.get_game_details("nonexistent")

        assert "error" in details

    def test_generate_html_report(self, dashboard):
        """Test HTML report generation."""
        html = dashboard.generate_html_report()

        assert "<!DOCTYPE html>" in html
        assert "Learning Dashboard" in html
        assert "Agent Performance" in html
        assert "game_1" in html  # Game ID should appear

    def test_save_report(self, dashboard):
        """Test saving HTML report to file."""
        path = dashboard.save_report()

        assert os.path.exists(path)
        assert path.endswith(".html")

        with open(path) as f:
            content = f.read()
            assert "<!DOCTYPE html>" in content

    def test_export_json(self, dashboard):
        """Test JSON export."""
        path = dashboard.export_json()

        assert os.path.exists(path)
        assert path.endswith(".json")

        with open(path) as f:
            data = json.load(f)
            assert "summary" in data
            assert "agent_comparison" in data
            assert "games" in data

    def test_add_record(self, dashboard):
        """Test adding a new record."""
        initial_count = len(dashboard._records)

        new_record = {
            "timestamp": "2025-01-01T00:02:00Z",
            "game_id": "game_2",
            "move_number": 1,
            "selected_move": "d2d4",
            "agent_results": [],
            "ensemble_confidence": 0.7,
            "consensus_achieved": False,
            "time_to_decide_ms": 300,
            "game_phase": "opening",
            "evaluation_before": 0.0,
            "evaluation_after": 0.05,
        }

        dashboard.add_record(new_record)

        assert len(dashboard._records) == initial_count + 1
        assert "game_2" in dashboard._game_stats


class TestDashboardEmpty:
    """Tests for empty dashboard."""

    def test_empty_dashboard_summary(self):
        """Test summary with no data."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        try:
            config = DashboardConfig(learning_db_path=temp_path)
            dashboard = LearningDashboard(config)

            summary = dashboard.get_summary()

            assert summary["status"] == "no_data"
        finally:
            os.unlink(temp_path)

    def test_empty_dashboard_html(self):
        """Test HTML generation with no data."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        try:
            config = DashboardConfig(learning_db_path=temp_path)
            dashboard = LearningDashboard(config)

            html = dashboard.generate_html_report()

            # Should still generate valid HTML
            assert "<!DOCTYPE html>" in html
        finally:
            os.unlink(temp_path)


class TestCreateDashboard:
    """Tests for factory function."""

    def test_create_dashboard_default(self):
        """Test creating dashboard with defaults."""
        dashboard = create_dashboard()

        assert isinstance(dashboard, LearningDashboard)

    def test_create_dashboard_custom_config(self):
        """Test creating dashboard with custom config."""
        config = DashboardConfig(max_records_in_memory=100)
        dashboard = create_dashboard(config)

        assert dashboard.config.max_records_in_memory == 100
