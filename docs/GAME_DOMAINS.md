# Game Domains Reference for Strategos-MCTS

Strategos-MCTS supports dynamic, modular domain registration for AlphaZero-style neural self-play and MCTS evaluation.

## Registered Game Domains Summary

| Domain | Type | Metric | Action Space Size | State Tensor Shape | Requirements / Extra |
|---|---|---|---|---|---|
| **`chess`** | Two-Player Zero-Sum | `win_rate` | 4672 | `(17, 8, 8)` | `pip install -e ".[chess]"` (`python-chess`) |
| **`connect_four`** | Two-Player Zero-Sum | `win_rate` | 7 | `(3, 6, 7)` | Built-in (Standard dependencies) |
| **`othello`** | Two-Player Zero-Sum | `win_rate` | 65 | `(3, 8, 8)` | Built-in (Standard dependencies) |
| **`reasoning`** | Single-Agent | `mean_reward` | 8 | `(128,)` | Built-in (Synthetic smoke domain) |
| **`planning`** | Single-Agent | `mean_reward` | 5 | `(128,)` | Built-in (Synthetic smoke domain) |

## Registering a Custom Game Domain

To add a new domain to the framework:

1. Implement a `GameState` subclass adhering to `src.framework.mcts.neural_mcts.GameState`:
   - `get_legal_actions() -> list[Any]`
   - `apply_action(action: Any) -> GameState`
   - `is_terminal() -> bool`
   - `get_reward(player: int = 1) -> float`
   - `to_tensor() -> torch.Tensor`
   - `get_hash() -> str`

2. Register the domain with `DomainRegistry`:

```python
from src.framework.domain_registry import DomainRegistry, DomainSpec

DomainRegistry.register(
    DomainSpec(
        name="my_custom_domain",
        metric="win_rate",  # or "mean_reward"
        single_agent=False,
        initial_state_fn=MyCustomState.create_initial_state,
        action_space_size=10,
    )
)
```
