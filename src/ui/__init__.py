"""
Shared UI building blocks.

Holds UI logic that must be unit-testable and covered by the project coverage
gate. The root ``app.py`` sits outside ``[tool.coverage.run] source = ["src"]``,
so anything defined there is invisible to the gate by construction; code placed
here is measured like the rest of ``src/``.
"""
