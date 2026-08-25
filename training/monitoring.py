"""
Monitoring and Logging Module

Comprehensive observability for training pipeline including:
- Training metrics tracking
- Real-time dashboards
- Alerting system
- Performance profiling
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Training alert."""

    timestamp: str
    severity: str  # "info", "warning", "critical"
    category: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float


@dataclass
class TrainingSnapshot:
    """Snapshot of training state."""

    timestamp: float
    epoch: int
    global_step: int
    metrics: dict[str, float]
    resource_usage: dict[str, float]
    model_info: dict[str, Any]


class TrainingMonitor:
    """Monitor training progress and performance."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize training monitor.

        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.log_dir = Path(config.get("logging", {}).get("log_dir", "./logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Alert thresholds (ensure numeric values are floats)
        self.alert_config = config.get("alerts", {})
        self.loss_spike_threshold = float(self.alert_config.get("loss_spike_threshold", 2.0))
        self.gradient_explosion_threshold = float(self.alert_config.get("gradient_explosion_threshold", 100.0))
        self.oom_warning_threshold = float(self.alert_config.get("oom_warning_threshold", 0.9))

        # Metric history
        self.metric_history = {}
        self.gradient_norms = deque(maxlen=1000)
        self.loss_values = deque(maxlen=1000)
        self.alerts = []
        self.snapshots = []

        # Setup logging
        self._setup_logging()

        logger.info(f"TrainingMonitor initialized, logs at {self.log_dir}")

    def _setup_logging(self) -> None:
        """Setup structured logging."""
        log_format = self.config.get("logging", {}).get("format", "json")
        log_level = self.config.get("logging", {}).get("level", "INFO")

        # File handler for training logs
        training_log = self.log_dir / "training.log"

        if log_format == "json":
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(training_log)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level))

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metric values
            step: Current training step
        """
        timestamp = time.time()

        for name, value in metrics.items():
            if name not in self.metric_history:
                self.metric_history[name] = []
            self.metric_history[name].append((step, timestamp, value))

            # Check for anomalies
            self._check_metric_anomalies(name, value, step)

        # Track specific metrics
        if "loss" in metrics:
            self.loss_values.append(metrics["loss"])
            self._check_loss_spike(metrics["loss"], step)

        logger.debug(f"Step {step}: {metrics}")

    def log_gradient_norm(self, norm: float, step: int) -> None:
        """
        Log gradient norm for monitoring.

        Args:
            norm: Gradient norm value
            step: Current training step
        """
        self.gradient_norms.append((step, norm))

        if norm > self.gradient_explosion_threshold:
            self._create_alert(
                severity="critical",
                category="gradient",
                message=f"Gradient explosion detected at step {step}",
                metric_name="gradient_norm",
                metric_value=norm,
                threshold=self.gradient_explosion_threshold,
            )

    def _check_loss_spike(self, loss: float, step: int) -> None:
        """Check for sudden loss spikes."""
        if len(self.loss_values) < 10:
            return

        recent_mean = np.mean(list(self.loss_values)[-100:])
        if loss > recent_mean * self.loss_spike_threshold:
            self._create_alert(
                severity="warning",
                category="loss",
                message=f"Loss spike detected at step {step}: {loss:.4f} vs mean {recent_mean:.4f}",
                metric_name="loss",
                metric_value=loss,
                threshold=recent_mean * self.loss_spike_threshold,
            )

    def _check_metric_anomalies(self, name: str, value: float, step: int) -> None:
        """Check for metric anomalies."""
        # NaN/Inf check
        if np.isnan(value) or np.isinf(value):
            self._create_alert(
                severity="critical",
                category="metric",
                message=f"Invalid metric value detected: {name}={value} at step {step}",
                metric_name=name,
                metric_value=value,
                threshold=0.0,
            )

    def _create_alert(
        self, severity: str, category: str, message: str, metric_name: str, metric_value: float, threshold: float
    ) -> None:
        """Create and store an alert."""
        alert = Alert(
            timestamp=datetime.now().isoformat(),
            severity=severity,
            category=category,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
        )

        self.alerts.append(alert)

        # Log alert
        if severity == "critical":
            logger.critical(message)
        elif severity == "warning":
            logger.warning(message)
        else:
            logger.info(message)

        # Persist alert
        self._persist_alert(alert)

    def _persist_alert(self, alert: Alert) -> None:
        """Persist alert to disk."""
        alert_file = self.log_dir / "alerts.jsonl"
        with open(alert_file, "a") as f:
            alert_dict = {
                "timestamp": alert.timestamp,
                "severity": alert.severity,
                "category": alert.category,
                "message": alert.message,
                "metric_name": alert.metric_name,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
            }
            f.write(json.dumps(alert_dict) + "\n")

    def check_resource_usage(self) -> dict[str, float]:
        """Check system resource usage."""
        import psutil

        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent / 100.0

        resources = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent * 100,
            "memory_available_gb": memory_info.available / (1024**3),
        }

        # Check GPU if available
        try:
            import torch

            if torch.cuda.is_available():
                # Use total GPU memory, not peak allocated
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                gpu_memory_ratio = allocated_memory / total_memory
                resources["gpu_memory_percent"] = gpu_memory_ratio * 100
                resources["gpu_memory_allocated_gb"] = allocated_memory / (1024**3)
                resources["gpu_memory_total_gb"] = total_memory / (1024**3)

                if gpu_memory_ratio > self.oom_warning_threshold:
                    self._create_alert(
                        severity="warning",
                        category="resource",
                        message=f"GPU memory usage high: {gpu_memory_ratio:.2%}",
                        metric_name="gpu_memory",
                        metric_value=gpu_memory_ratio,
                        threshold=self.oom_warning_threshold,
                    )
        except ImportError:
            pass

        return resources

    def create_snapshot(self, epoch: int, global_step: int, model: Any = None) -> TrainingSnapshot:
        """
        Create training state snapshot.

        Args:
            epoch: Current epoch
            global_step: Global training step
            model: Model (optional)

        Returns:
            Training snapshot
        """
        current_metrics = {}
        for name, history in self.metric_history.items():
            if history:
                current_metrics[name] = history[-1][2]  # Latest value

        resource_usage = self.check_resource_usage()

        model_info = {}
        if model is not None:
            try:
                model_info["num_parameters"] = sum(p.numel() for p in model.parameters())
                model_info["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            except Exception:
                pass

        snapshot = TrainingSnapshot(
            timestamp=time.time(),
            epoch=epoch,
            global_step=global_step,
            metrics=current_metrics,
            resource_usage=resource_usage,
            model_info=model_info,
        )

        self.snapshots.append(snapshot)

        return snapshot

    def get_metric_summary(self, metric_name: str) -> dict[str, float]:
        """
        Get summary statistics for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Summary statistics
        """
        if metric_name not in self.metric_history:
            return {}

        values = [v[2] for v in self.metric_history[metric_name]]

        if not values:
            return {}

        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "latest": float(values[-1]),
            "num_samples": len(values),
        }

    def export_metrics(self, output_path: str) -> None:
        """
        Export all metrics to file.

        Args:
            output_path: Path to export file
        """
        export_data = {
            "metric_history": {
                name: [(step, ts, val) for step, ts, val in history] for name, history in self.metric_history.items()
            },
            "alerts": [
                {"timestamp": a.timestamp, "severity": a.severity, "category": a.category, "message": a.message}
                for a in self.alerts
            ],
            "summaries": {name: self.get_metric_summary(name) for name in self.metric_history},
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Metrics exported to {output_path}")


class MetricsDashboard:
    """Generate training metrics dashboards."""

    def __init__(self, monitor: TrainingMonitor):
        """
        Initialize metrics dashboard.

        Args:
            monitor: Training monitor instance
        """
        self.monitor = monitor
        self.dashboard_dir = Path("training/reports/dashboards")
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self) -> str:
        """
        Generate HTML dashboard report.

        Returns:
            Path to generated HTML file
        """
        html_content = self._create_html_content()
        output_path = self.dashboard_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Dashboard generated at {output_path}")
        return str(output_path)

    def _create_html_content(self) -> str:
        """Create HTML content for dashboard."""
        # Get summaries
        summaries = {name: self.monitor.get_metric_summary(name) for name in self.monitor.metric_history}

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Training Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric-card {{
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px;
            border-radius: 5px;
            display: inline-block;
            min-width: 200px;
        }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .alert-critical {{ background-color: #ffebee; border-color: #ef5350; }}
        .alert-warning {{ background-color: #fff3e0; border-color: #ff9800; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Multi-Agent MCTS Training Dashboard</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>Metric Summaries</h2>
    <div class="metrics-container">
"""

        # Add metric cards
        for name, summary in summaries.items():
            if summary:
                html += f"""
        <div class="metric-card">
            <h3>{name}</h3>
            <div class="metric-value">{summary["latest"]:.4f}</div>
            <p>Min: {summary["min"]:.4f} | Max: {summary["max"]:.4f}</p>
            <p>Mean: {summary["mean"]:.4f} | Std: {summary["std"]:.4f}</p>
            <p>Samples: {summary["num_samples"]}</p>
        </div>
"""

        html += """
    </div>

    <h2>Recent Alerts</h2>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Severity</th>
            <th>Category</th>
            <th>Message</th>
        </tr>
"""

        # Add alerts
        for alert in self.monitor.alerts[-20:]:  # Last 20 alerts
            alert_class = f"alert-{alert.severity}"
            html += f"""
        <tr class="{alert_class}">
            <td>{alert.timestamp}</td>
            <td>{alert.severity}</td>
            <td>{alert.category}</td>
            <td>{alert.message}</td>
        </tr>
"""

        html += """
    </table>

    <h2>Resource Usage</h2>
"""

        # Add resource info
        resources = self.monitor.check_resource_usage()
        for resource, value in resources.items():
            html += f"""
    <div class="metric-card">
        <h3>{resource}</h3>
        <div class="metric-value">{value:.2f}</div>
    </div>
"""

        html += """
</body>
</html>
"""

        return html

    def generate_training_curves(self) -> dict[str, list[tuple[int, float]]]:
        """
        Generate data for training curves.

        Returns:
            Dictionary of metric curves
        """
        curves = {}

        for name, history in self.monitor.metric_history.items():
            # Extract step and value
            curves[name] = [(entry[0], entry[2]) for entry in history]

        return curves

    def get_live_metrics(self) -> dict[str, Any]:
        """Get current live metrics for real-time dashboard."""
        live_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "alerts": [],
            "resources": self.monitor.check_resource_usage(),
        }

        # Latest metrics
        for name, history in self.monitor.metric_history.items():
            if history:
                live_data["metrics"][name] = history[-1][2]

        # Recent alerts
        live_data["alerts"] = [
            {"timestamp": a.timestamp, "severity": a.severity, "message": a.message} for a in self.monitor.alerts[-5:]
        ]

        return live_data


class AlertManager:
    """Manage and route alerts."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize alert manager.

        Args:
            config: Alerting configuration
        """
        self.config = config
        self.alert_handlers = []
        self.alert_rules = self._define_rules()
        self.suppressed_alerts = {}

        logger.info("AlertManager initialized")

    def _define_rules(self) -> list[dict[str, Any]]:
        """Define alerting rules."""
        rules = [
            {
                "name": "loss_divergence",
                "condition": lambda m: m.get("loss", 0) > 10.0,
                "severity": "critical",
                "message": "Training loss diverging",
            },
            {
                "name": "low_accuracy",
                "condition": lambda m: m.get("accuracy", 1.0) < 0.5,
                "severity": "warning",
                "message": "Model accuracy below threshold",
            },
            {
                "name": "slow_convergence",
                "condition": lambda m: m.get("epoch", 0) > 5 and m.get("loss", 0) > 1.0,
                "severity": "info",
                "message": "Slow training convergence detected",
            },
        ]
        return rules

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        Add alert handler.

        Args:
            handler: Function to handle alerts
        """
        self.alert_handlers.append(handler)

    def check_rules(self, metrics: dict[str, float]) -> list[Alert]:
        """
        Check all alerting rules.

        Args:
            metrics: Current metrics

        Returns:
            List of triggered alerts
        """
        triggered = []

        for rule in self.alert_rules:
            if rule["condition"](metrics):
                # Check suppression
                if self._should_suppress(rule["name"]):
                    continue

                alert = Alert(
                    timestamp=datetime.now().isoformat(),
                    severity=rule["severity"],
                    category="rule",
                    message=rule["message"],
                    metric_name=rule["name"],
                    metric_value=0.0,
                    threshold=0.0,
                )

                triggered.append(alert)
                self._route_alert(alert)

        return triggered

    def _should_suppress(self, rule_name: str) -> bool:
        """Check if alert should be suppressed (cooldown period)."""
        if rule_name not in self.suppressed_alerts:
            self.suppressed_alerts[rule_name] = time.time()
            return False

        # 5 minute cooldown
        if time.time() - self.suppressed_alerts[rule_name] > 300:
            self.suppressed_alerts[rule_name] = time.time()
            return False

        return True

    def _route_alert(self, alert: Alert) -> None:
        """Route alert to all handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def email_handler(self, alert: Alert) -> None:
        """Example email alert handler."""
        # In production, would send email
        logger.info(f"[EMAIL] Alert: {alert.message}")

    def slack_handler(self, alert: Alert) -> None:
        """Example Slack alert handler."""
        # In production, would send Slack message
        logger.info(f"[SLACK] Alert: {alert.message}")

    def get_alert_summary(self, alerts: list[Alert]) -> dict[str, int]:
        """Get summary of alerts by severity."""
        summary = {"critical": 0, "warning": 0, "info": 0}

        for alert in alerts:
            summary[alert.severity] = summary.get(alert.severity, 0) + 1

        return summary


if __name__ == "__main__":
    # Test monitoring module
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing Monitoring Module")

    # Load config
    config_path = "training/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    monitoring_config = config.get("monitoring", {})

    # Test TrainingMonitor
    monitor = TrainingMonitor(monitoring_config)

    # Simulate training metrics
    for step in range(100):
        metrics = {
            "loss": 1.0 / (step + 1) + np.random.randn() * 0.1,
            "accuracy": 0.5 + 0.5 * (step / 100) + np.random.randn() * 0.05,
            "learning_rate": 0.001 * (0.95 ** (step // 10)),
        }
        monitor.log_metrics(metrics, step)

        # Log gradient norm
        grad_norm = np.random.randn() * 5 + 10
        monitor.log_gradient_norm(grad_norm, step)

    # Check resource usage
    resources = monitor.check_resource_usage()
    logger.info(f"Resource usage: {resources}")

    # Get metric summary
    loss_summary = monitor.get_metric_summary("loss")
    logger.info(f"Loss summary: {loss_summary}")

    # Create snapshot
    snapshot = monitor.create_snapshot(epoch=5, global_step=100)
    logger.info(f"Training snapshot: {snapshot}")

    # Export metrics
    monitor.export_metrics("training/reports/metrics_export.json")

    # Test Dashboard
    dashboard = MetricsDashboard(monitor)
    html_path = dashboard.generate_html_report()
    logger.info(f"Dashboard HTML: {html_path}")

    # Test AlertManager
    alert_manager = AlertManager(monitoring_config)

    # Add custom handler
    def custom_handler(alert):
        logger.info(f"[CUSTOM] {alert.severity}: {alert.message}")

    alert_manager.add_handler(custom_handler)

    # Check rules
    test_metrics = {"loss": 15.0, "accuracy": 0.3, "epoch": 10}
    triggered = alert_manager.check_rules(test_metrics)
    logger.info(f"Triggered alerts: {len(triggered)}")

    logger.info("Monitoring Module test complete")
