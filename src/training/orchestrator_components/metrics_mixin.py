from typing import Any

import psutil
import torch

from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)

class MetricsMixin:

    def _get_memory_utilization(self) -> dict[str, Any]:
        """Get current memory and GPU utilization metrics."""
        memory_info = {}
        process = psutil.Process()
        memory_info['cpu_memory_mb'] = round(process.memory_info().rss / (1024 * 1024), 2)
        memory_info['cpu_percent'] = process.cpu_percent()
        if self.device != 'cpu' and torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                memory_info['gpu_memory_allocated_mb'] = round(gpu_memory_allocated, 2)
                memory_info['gpu_memory_reserved_mb'] = round(gpu_memory_reserved, 2)
                memory_info['gpu_memory_total_mb'] = round(gpu_memory_total, 2)
                memory_info['gpu_utilization_percent'] = round(gpu_memory_allocated / gpu_memory_total * 100, 2)
            except Exception as e:
                logger.debug('Failed to get GPU memory info', error=str(e))
        return memory_info

    def _log_metrics(self, iteration: int, metrics: dict):
        """Log metrics to console and tracking systems."""
        logger.info('Iteration metrics summary', iteration=iteration, **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()})
        if self.config.use_wandb:
            try:
                import wandb
                wandb_metrics = self.monitor.export_to_wandb(iteration)
                wandb_metrics.update(metrics)
                wandb.log(wandb_metrics, step=iteration)
                logger.debug('Metrics logged to Weights & Biases', iteration=iteration, num_metrics=len(wandb_metrics))
            except Exception as e:
                logger.warning('Failed to log metrics to Weights & Biases', error=str(e), iteration=iteration)
