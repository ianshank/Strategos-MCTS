"""
Learning Dashboard for Chess Ensemble.

Captures and displays comprehensive analytics from each game run:
- Agent performance metrics
- Decision confidence trends
- Move quality analysis
- Game phase statistics
- Real-time visualization

Best Practices 2025:
- Async data processing
- Efficient aggregation
- JSON-based persistence
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Optional imports
try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class DashboardConfig:
    """Dashboard configuration."""

    learning_db_path: str = field(default_factory=lambda: os.getenv("CHESS_LEARNING_DB", "./chess_learning.jsonl"))
    output_dir: str = field(default_factory=lambda: os.getenv("CHESS_DASHBOARD_OUTPUT", "./chess_dashboard"))
    max_records_in_memory: int = 10000


@dataclass
class AgentStats:
    """Statistics for a single agent."""

    name: str
    total_decisions: int = 0
    times_selected: int = 0
    avg_confidence: float = 0.0
    avg_time_ms: float = 0.0
    agreement_rate: float = 0.0
    confidence_trend: list[float] = field(default_factory=list)
    time_trend: list[float] = field(default_factory=list)


@dataclass
class GameStats:
    """Statistics for a single game."""

    game_id: str
    total_moves: int = 0
    avg_confidence: float = 0.0
    consensus_rate: float = 0.0
    avg_decision_time_ms: float = 0.0
    final_evaluation: float = 0.0
    phase_breakdown: dict[str, int] = field(default_factory=dict)
    winning_side: str = "unknown"


class LearningDashboard:
    """
    Comprehensive learning dashboard for chess ensemble analytics.

    Features:
    - Real-time metrics tracking
    - Historical trend analysis
    - Agent comparison charts
    - Game-by-game breakdown
    - Export to HTML/JSON
    """

    def __init__(self, config: DashboardConfig | None = None):
        self.config = config or DashboardConfig()
        self._records: list[dict[str, Any]] = []
        self._game_stats: dict[str, GameStats] = {}
        self._agent_stats: dict[str, AgentStats] = {}
        self._load_records()

    def _load_records(self) -> None:
        """Load existing learning records from database."""
        if not os.path.exists(self.config.learning_db_path):
            return

        try:
            with open(self.config.learning_db_path) as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        self._records.append(record)

                        # Limit records in memory
                        if len(self._records) > self.config.max_records_in_memory:
                            self._records = self._records[-self.config.max_records_in_memory :]

            self._compute_statistics()
        except Exception as e:
            print(f"Warning: Failed to load learning records: {e}")

    def _compute_statistics(self) -> None:
        """Compute aggregate statistics from records."""
        self._game_stats = {}
        self._agent_stats = {}

        # Initialize agent stats
        agent_names = ["HRM", "TRM", "MCTS", "SYMBOLIC"]
        for name in agent_names:
            self._agent_stats[name] = AgentStats(name=name)

        # Process each record
        for record in self._records:
            game_id = record.get("game_id", "unknown")

            # Initialize game stats if needed
            if game_id not in self._game_stats:
                self._game_stats[game_id] = GameStats(game_id=game_id)

            game = self._game_stats[game_id]
            game.total_moves += 1
            game.avg_confidence = (
                game.avg_confidence * (game.total_moves - 1) + record.get("ensemble_confidence", 0)
            ) / game.total_moves
            if record.get("consensus_achieved"):
                game.consensus_rate = (game.consensus_rate * (game.total_moves - 1) + 1) / game.total_moves

            phase = record.get("game_phase", "unknown")
            game.phase_breakdown[phase] = game.phase_breakdown.get(phase, 0) + 1
            game.final_evaluation = record.get("evaluation_after", 0)

            # Process agent results
            selected_move = record.get("selected_move")
            for agent_result in record.get("agent_results", []):
                agent_name = agent_result.get("agent", "unknown")
                if agent_name not in self._agent_stats:
                    self._agent_stats[agent_name] = AgentStats(name=agent_name)

                agent = self._agent_stats[agent_name]
                agent.total_decisions += 1

                if agent_result.get("move") == selected_move:
                    agent.times_selected += 1

                conf = agent_result.get("confidence", 0)
                time_ms = agent_result.get("time_ms", 0)

                agent.avg_confidence = (
                    agent.avg_confidence * (agent.total_decisions - 1) + conf
                ) / agent.total_decisions
                agent.avg_time_ms = (agent.avg_time_ms * (agent.total_decisions - 1) + time_ms) / agent.total_decisions

                agent.confidence_trend.append(conf)
                agent.time_trend.append(time_ms)

        # Calculate agreement rates
        for agent in self._agent_stats.values():
            if agent.total_decisions > 0:
                agent.agreement_rate = agent.times_selected / agent.total_decisions

    def add_record(self, record: dict[str, Any]) -> None:
        """Add a new learning record."""
        self._records.append(record)
        self._compute_statistics()

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive dashboard summary."""
        if not self._records:
            return {"status": "no_data", "message": "No learning records available"}

        # Overall metrics
        total_games = len(self._game_stats)
        total_moves = len(self._records)

        avg_confidence = statistics.mean(r.get("ensemble_confidence", 0) for r in self._records) if self._records else 0

        consensus_rate = (
            sum(1 for r in self._records if r.get("consensus_achieved")) / total_moves if total_moves > 0 else 0
        )

        avg_time = statistics.mean(r.get("time_to_decide_ms", 0) for r in self._records) if self._records else 0

        # Agent breakdown
        agent_summary = {}
        for name, stats in self._agent_stats.items():
            agent_summary[name] = {
                "total_decisions": stats.total_decisions,
                "times_selected": stats.times_selected,
                "selection_rate": stats.agreement_rate,
                "avg_confidence": round(stats.avg_confidence, 3),
                "avg_time_ms": round(stats.avg_time_ms, 2),
            }

        # Phase breakdown
        phase_totals = {}
        for game in self._game_stats.values():
            for phase, count in game.phase_breakdown.items():
                phase_totals[phase] = phase_totals.get(phase, 0) + count

        # Recent games
        recent_games = []
        for game_id, stats in list(self._game_stats.items())[-5:]:
            recent_games.append(
                {
                    "game_id": game_id,
                    "moves": stats.total_moves,
                    "avg_confidence": round(stats.avg_confidence, 3),
                    "consensus_rate": round(stats.consensus_rate, 3),
                    "final_eval": round(stats.final_evaluation, 3),
                }
            )

        return {
            "status": "ok",
            "generated_at": datetime.now(UTC).isoformat(),
            "overall": {
                "total_games": total_games,
                "total_moves": total_moves,
                "avg_confidence": round(avg_confidence, 3),
                "consensus_rate": round(consensus_rate, 3),
                "avg_decision_time_ms": round(avg_time, 2),
            },
            "agents": agent_summary,
            "phases": phase_totals,
            "recent_games": recent_games,
        }

    def get_agent_comparison(self) -> dict[str, Any]:
        """Get detailed agent comparison data."""
        comparison = {
            "agents": [],
            "metrics": ["selection_rate", "avg_confidence", "avg_time_ms"],
        }

        for name, stats in self._agent_stats.items():
            comparison["agents"].append(
                {
                    "name": name,
                    "selection_rate": round(stats.agreement_rate, 3),
                    "avg_confidence": round(stats.avg_confidence, 3),
                    "avg_time_ms": round(stats.avg_time_ms, 2),
                    "total_decisions": stats.total_decisions,
                }
            )

        return comparison

    def get_confidence_trend(self, window: int = 20) -> list[dict[str, Any]]:
        """Get confidence trend over recent moves."""
        if not self._records:
            return []

        # Use moving average
        recent = self._records[-100:]  # Last 100 moves
        trend = []

        for i in range(0, len(recent), max(1, len(recent) // 20)):
            window_records = recent[i : i + window]
            if window_records:
                avg_conf = statistics.mean(r.get("ensemble_confidence", 0) for r in window_records)
                avg_time = statistics.mean(r.get("time_to_decide_ms", 0) for r in window_records)
                trend.append(
                    {
                        "index": i,
                        "avg_confidence": round(avg_conf, 3),
                        "avg_time_ms": round(avg_time, 2),
                    }
                )

        return trend

    def get_game_details(self, game_id: str) -> dict[str, Any]:
        """Get detailed analysis for a specific game."""
        if game_id not in self._game_stats:
            return {"error": f"Game {game_id} not found"}

        game = self._game_stats[game_id]
        game_records = [r for r in self._records if r.get("game_id") == game_id]

        # Move-by-move breakdown
        moves = []
        for record in game_records:
            agent_votes = {}
            for ar in record.get("agent_results", []):
                agent_votes[ar.get("agent", "unknown")] = {
                    "move": ar.get("move"),
                    "confidence": ar.get("confidence", 0),
                }

            moves.append(
                {
                    "move_number": record.get("move_number"),
                    "selected_move": record.get("selected_move"),
                    "fen": record.get("fen_after"),
                    "confidence": record.get("ensemble_confidence"),
                    "consensus": record.get("consensus_achieved"),
                    "phase": record.get("game_phase"),
                    "evaluation": record.get("evaluation_after"),
                    "agent_votes": agent_votes,
                }
            )

        # Evaluation progression
        eval_progression = [r.get("evaluation_after", 0) for r in game_records]

        return {
            "game_id": game_id,
            "total_moves": game.total_moves,
            "avg_confidence": round(game.avg_confidence, 3),
            "consensus_rate": round(game.consensus_rate, 3),
            "final_evaluation": round(game.final_evaluation, 3),
            "phase_breakdown": game.phase_breakdown,
            "moves": moves,
            "evaluation_progression": eval_progression,
        }

    def generate_html_report(self) -> str:
        """Generate HTML dashboard report."""
        summary = self.get_summary()
        comparison = self.get_agent_comparison()
        trend = self.get_confidence_trend()

        html = (
            """
<!DOCTYPE html>
<html>
<head>
    <title>Chess Ensemble Learning Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card h2 { margin-top: 0; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        .metric { display: inline-block; margin: 10px 20px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #4CAF50; }
        .metric-label { color: #666; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #4CAF50; color: white; }
        tr:hover { background: #f5f5f5; }
        .progress-bar { background: #e0e0e0; border-radius: 4px; height: 20px; overflow: hidden; }
        .progress-fill { background: #4CAF50; height: 100%; transition: width 0.3s; }
        .timestamp { color: #999; font-size: 0.8em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chess Ensemble Learning Dashboard</h1>
        <p class="timestamp">Generated: """
            + summary.get("generated_at", "N/A")
            + """</p>

        <div class="card">
            <h2>Overall Metrics</h2>
            <div class="metric">
                <div class="metric-value">"""
            + str(summary.get("overall", {}).get("total_games", 0))
            + """</div>
                <div class="metric-label">Total Games</div>
            </div>
            <div class="metric">
                <div class="metric-value">"""
            + str(summary.get("overall", {}).get("total_moves", 0))
            + """</div>
                <div class="metric-label">Total Moves</div>
            </div>
            <div class="metric">
                <div class="metric-value">"""
            + str(round(summary.get("overall", {}).get("avg_confidence", 0) * 100, 1))
            + """%</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
            <div class="metric">
                <div class="metric-value">"""
            + str(round(summary.get("overall", {}).get("consensus_rate", 0) * 100, 1))
            + """%</div>
                <div class="metric-label">Consensus Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">"""
            + str(round(summary.get("overall", {}).get("avg_decision_time_ms", 0), 0))
            + """ms</div>
                <div class="metric-label">Avg Decision Time</div>
            </div>
        </div>

        <div class="card">
            <h2>Agent Performance</h2>
            <table>
                <tr>
                    <th>Agent</th>
                    <th>Selection Rate</th>
                    <th>Avg Confidence</th>
                    <th>Avg Time (ms)</th>
                    <th>Total Decisions</th>
                </tr>
"""
        )
        for agent in comparison.get("agents", []):
            selection_pct = round(agent.get("selection_rate", 0) * 100, 1)
            html += f"""
                <tr>
                    <td><strong>{agent.get("name", "Unknown")}</strong></td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {selection_pct}%"></div>
                        </div>
                        {selection_pct}%
                    </td>
                    <td>{round(agent.get("avg_confidence", 0) * 100, 1)}%</td>
                    <td>{round(agent.get("avg_time_ms", 0), 1)}</td>
                    <td>{agent.get("total_decisions", 0)}</td>
                </tr>
"""
        html += """
            </table>
        </div>

        <div class="card">
            <h2>Game Phase Distribution</h2>
            <table>
                <tr>
                    <th>Phase</th>
                    <th>Moves</th>
                    <th>Distribution</th>
                </tr>
"""
        total_phase_moves = sum(summary.get("phases", {}).values()) or 1
        for phase, count in summary.get("phases", {}).items():
            pct = round(count / total_phase_moves * 100, 1)
            html += f"""
                <tr>
                    <td><strong>{phase.title()}</strong></td>
                    <td>{count}</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {pct}%"></div>
                        </div>
                        {pct}%
                    </td>
                </tr>
"""
        html += """
            </table>
        </div>

        <div class="card">
            <h2>Recent Games</h2>
            <table>
                <tr>
                    <th>Game ID</th>
                    <th>Moves</th>
                    <th>Avg Confidence</th>
                    <th>Consensus Rate</th>
                    <th>Final Evaluation</th>
                </tr>
"""
        for game in summary.get("recent_games", []):
            eval_val = game.get("final_eval", 0)
            eval_color = "#4CAF50" if eval_val > 0 else "#f44336" if eval_val < 0 else "#999"
            html += f"""
                <tr>
                    <td>{game.get("game_id", "N/A")}</td>
                    <td>{game.get("moves", 0)}</td>
                    <td>{round(game.get("avg_confidence", 0) * 100, 1)}%</td>
                    <td>{round(game.get("consensus_rate", 0) * 100, 1)}%</td>
                    <td style="color: {eval_color}">{round(eval_val, 3)}</td>
                </tr>
"""
        html += """
            </table>
        </div>

        <div class="card">
            <h2>Confidence Trend</h2>
            <div style="display: flex; align-items: flex-end; height: 100px; gap: 4px;">
"""
        for point in trend[-20:]:  # Last 20 data points
            height = int(point.get("avg_confidence", 0) * 100)
            html += f"""
                <div style="width: 20px; height: {height}px; background: #4CAF50; border-radius: 2px 2px 0 0;" title="Confidence: {point.get("avg_confidence", 0)}"></div>
"""
        html += """
            </div>
            <p style="text-align: center; color: #666;">Recent Move Confidence (left=older, right=newer)</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def save_report(self, output_path: str | None = None) -> str:
        """Save HTML report to file."""
        path = output_path or os.path.join(self.config.output_dir, "dashboard.html")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        html = self.generate_html_report()
        with open(path, "w") as f:
            f.write(html)

        return path

    def export_json(self, output_path: str | None = None) -> str:
        """Export dashboard data as JSON."""
        path = output_path or os.path.join(self.config.output_dir, "dashboard.json")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        data = {
            "summary": self.get_summary(),
            "agent_comparison": self.get_agent_comparison(),
            "confidence_trend": self.get_confidence_trend(),
            "games": {gid: self.get_game_details(gid) for gid in list(self._game_stats.keys())[-10:]},  # Last 10 games
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path

    if MATPLOTLIB_AVAILABLE:

        def generate_charts(self, output_dir: str | None = None) -> list[str]:
            """Generate matplotlib charts for the dashboard."""
            out_dir = output_dir or self.config.output_dir
            os.makedirs(out_dir, exist_ok=True)
            charts = []

            # Agent comparison chart
            comparison = self.get_agent_comparison()
            if comparison.get("agents"):
                fig, ax = plt.subplots(figsize=(10, 6))

                agents = [a["name"] for a in comparison["agents"]]
                selection_rates = [a["selection_rate"] * 100 for a in comparison["agents"]]
                confidences = [a["avg_confidence"] * 100 for a in comparison["agents"]]

                x = range(len(agents))
                width = 0.35

                ax.bar([i - width / 2 for i in x], selection_rates, width, label="Selection Rate %", color="#4CAF50")
                ax.bar([i + width / 2 for i in x], confidences, width, label="Avg Confidence %", color="#2196F3")

                ax.set_ylabel("Percentage")
                ax.set_title("Agent Performance Comparison")
                ax.set_xticks(x)
                ax.set_xticklabels(agents)
                ax.legend()
                ax.set_ylim(0, 100)

                chart_path = os.path.join(out_dir, "agent_comparison.png")
                plt.savefig(chart_path, dpi=150, bbox_inches="tight")
                plt.close()
                charts.append(chart_path)

            # Confidence trend chart
            trend = self.get_confidence_trend()
            if trend:
                fig, ax = plt.subplots(figsize=(12, 4))

                indices = [t["index"] for t in trend]
                confidences = [t["avg_confidence"] * 100 for t in trend]

                ax.plot(indices, confidences, "g-", linewidth=2, marker="o")
                ax.fill_between(indices, confidences, alpha=0.3, color="green")
                ax.set_xlabel("Move Index")
                ax.set_ylabel("Confidence %")
                ax.set_title("Ensemble Confidence Trend")
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)

                chart_path = os.path.join(out_dir, "confidence_trend.png")
                plt.savefig(chart_path, dpi=150, bbox_inches="tight")
                plt.close()
                charts.append(chart_path)

            return charts


def create_dashboard(
    config: DashboardConfig | None = None,
) -> LearningDashboard:
    """Factory function to create dashboard."""
    return LearningDashboard(config)
