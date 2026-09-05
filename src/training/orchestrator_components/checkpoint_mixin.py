from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .host_protocol import OrchestratorHost
else:
    OrchestratorHost = object  # type: ignore[misc,assignment]

from pathlib import Path
import time

import torch

from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)


class CheckpointMixin:

    def _save_checkpoint(self: "OrchestratorHost", iteration: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        from src.utils import distributed

        if getattr(self.config, "distributed", False) and (not distributed.is_main_process()):
            return
        save_start = time.perf_counter()
        model_to_save = distributed.unwrap_model(self.policy_value_net)
        checkpoint = {
            "iteration": iteration,
            "policy_value_net": model_to_save.state_dict(),
            "hrm_agent": self.hrm_agent.state_dict(),
            "trm_agent": self.trm_agent.state_dict(),
            "pv_optimizer": self.pv_optimizer.state_dict(),
            "hrm_optimizer": self.hrm_optimizer.state_dict(),
            "trm_optimizer": self.trm_optimizer.state_dict(),
            "config": self.config.to_dict(),
            "metrics": metrics,
            "best_win_rate": self.best_win_rate,
        }
        path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
        try:
            torch.save(checkpoint, path)
            checkpoint_size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(
                "Checkpoint saved",
                checkpoint_path=str(path),
                iteration=iteration,
                checkpoint_size_mb=round(checkpoint_size_mb, 2),
                save_time_ms=round((time.perf_counter() - save_start) * 1000, 2),
                is_best=is_best,
            )
        except Exception as e:
            logger.error("Failed to save checkpoint", error=str(e), checkpoint_path=str(path), iteration=iteration)
            return
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            try:
                torch.save(checkpoint, best_path)
                self.best_model_path = best_path
                logger.info(
                    "Best model checkpoint saved",
                    best_model_path=str(best_path),
                    iteration=iteration,
                    win_rate=self.best_win_rate,
                )
            except Exception as e:
                logger.error("Failed to save best model checkpoint", error=str(e), best_model_path=str(best_path))

    def load_checkpoint(self: "OrchestratorHost", path: str):
        """Load checkpoint from file."""
        logger.info("Loading checkpoint", checkpoint_path=path, device=self.device)
        load_start = time.perf_counter()
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as e:
            logger.error("Failed to load checkpoint file", checkpoint_path=path, error=str(e))
            raise
        try:
            self.policy_value_net.load_state_dict(checkpoint["policy_value_net"])
            self.hrm_agent.load_state_dict(checkpoint["hrm_agent"])
            self.trm_agent.load_state_dict(checkpoint["trm_agent"])
            self.pv_optimizer.load_state_dict(checkpoint["pv_optimizer"])
            self.hrm_optimizer.load_state_dict(checkpoint["hrm_optimizer"])
            self.trm_optimizer.load_state_dict(checkpoint["trm_optimizer"])
            self.current_iteration = checkpoint["iteration"]
            self.best_win_rate = checkpoint.get("best_win_rate", 0.0)
            load_time = time.perf_counter() - load_start
            checkpoint_path = Path(path)
            checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else 0
            logger.info(
                "Checkpoint loaded successfully",
                checkpoint_path=path,
                iteration=self.current_iteration,
                best_win_rate=round(self.best_win_rate, 4),
                checkpoint_size_mb=round(checkpoint_size_mb, 2),
                load_time_ms=round(load_time * 1000, 2),
                checkpoint_metrics=checkpoint.get("metrics", {}),
            )
        except KeyError as e:
            logger.error(
                "Checkpoint is missing required keys",
                checkpoint_path=path,
                missing_key=str(e),
                available_keys=list(checkpoint.keys()),
            )
            raise
        except Exception as e:
            logger.error("Failed to restore model state from checkpoint", checkpoint_path=path, error=str(e))
            raise
