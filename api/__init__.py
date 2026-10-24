"""The sample API package.

A reference HTTP API over the search surface and the
built graph — search, entity lookup, neighborhood, and
triple inspection with evidence. It
is the backend the hands-on guide and the demo app
 run against.

The implementation lives in :mod:`api.server` (standard
library only); importing this package pulls nothing heavy.
"""
from __future__ import annotations

__all__ = ["server"]
