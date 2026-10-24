"""The sample API package (plan §7 R17).

A reference HTTP API over the R16 search surface and the
built graph — search, entity lookup, neighborhood, and
triple inspection with evidence (the API form of R9). It
is the backend the hands-on guide (R13) and the demo app
(R18) run against.

The implementation lives in :mod:`api.server` (standard
library only); importing this package pulls nothing heavy.
"""
from __future__ import annotations

__all__ = ["server"]
