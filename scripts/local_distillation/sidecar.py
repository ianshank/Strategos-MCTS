"""Connect Four checkpoint sidecar: refuse chess-shaped fallbacks."""

from __future__ import annotations

from typing import Any

from scripts.local_distillation.settings import DistillationSettings

_REQUIRED_NETWORK_KEYS = (
    "type",
    "input_channels",
    "board_rows",
    "board_cols",
    "action_size",
    "num_res_blocks",
    "num_channels",
)


class SidecarError(ValueError):
    """Malformed or chess-shaped architecture metadata."""


def c4_network_architecture(settings: DistillationSettings) -> dict[str, Any]:
    """Full C4 ResNet dict. Omitting board_rows would become 8×8 in policy-lift."""
    return {
        "type": "resnet",
        "input_channels": settings.input_channels,
        "board_rows": settings.board_rows,
        "board_cols": settings.board_cols,
        "board_size": max(settings.board_rows, settings.board_cols),
        "action_size": settings.action_size,
        "num_res_blocks": settings.num_res_blocks,
        "num_channels": settings.num_channels,
    }


def validate_c4_sidecar(meta: dict[str, Any], settings: DistillationSettings) -> dict[str, Any]:
    """Require an explicit C4 ``network`` object; never chess_default_architecture."""
    network = meta.get("network") if isinstance(meta, dict) else None
    if not isinstance(network, dict):
        raise SidecarError("sidecar must contain a 'network' object; chess_default_architecture is forbidden")
    missing = [key for key in _REQUIRED_NETWORK_KEYS if key not in network]
    if missing:
        raise SidecarError(f"sidecar network missing keys: {missing}")
    if str(network["type"]) != "resnet":
        raise SidecarError(f"C4 sidecar type must be resnet, got {network['type']!r}")
    if int(network["input_channels"]) != settings.input_channels:
        raise SidecarError("sidecar input_channels does not match Connect Four")
    if int(network["board_rows"]) != settings.board_rows or int(network["board_cols"]) != settings.board_cols:
        raise SidecarError("sidecar board_rows/board_cols must be Connect Four 6×7")
    if int(network["action_size"]) != settings.action_size:
        raise SidecarError("sidecar action_size does not match Connect Four")
    return network
