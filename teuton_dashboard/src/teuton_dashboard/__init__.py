"""Teuton queue-native dashboard.

FastAPI app that publishes a SSE stream of the orchestrator's outstanding-work
queue plus a SQLite-indexed view of receipts/verdicts/audits/heartbeats. A
SvelteKit SPA (built into ``static/``) consumes both surfaces.
"""

__version__ = "0.1.0"
