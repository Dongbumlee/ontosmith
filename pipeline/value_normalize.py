"""Deterministic normalization of declared value attributes.

A pack's schema_front may declare, per entity type, value
attributes (dates, amounts, identifiers — the values a
document states ABOUT an entity) with one of the six closed
value types below. This module is the code side of the
judgment split the design fixes (value-attributes design
note, 2026-11-06): form — parsing, unit/currency handling,
normalization — is deterministic code, never a model. A
surface that parses here is normalized by THIS module's
output, not by anything the proposer wrote; a surface that
does not parse can still ride to the judge when the proposer
supplied a well-formed normalized value of its own (the
conversion's correctness — a relative date against its
reference day — is then the judge's question, exactly as the
design assigns it). A surface with neither is a format
failure: it is returned as a result, never raised, and the
judgment path never sees it.

Governing decisions (design note §11): the type set is
closed at six — no free-string type; money maps symbols and
currency names to ISO 4217 codes deterministically and NEVER
guesses a currency (an amount with no currency stated is
stored without one and flagged incomplete); number splits
value and unit and never converts units; datetime keeps a
stated offset and otherwise stays naive (no timezone is
invented), and a date-only surface under a datetime
declaration is demoted to date; duration is length only —
a start/end pair is an interval (a value-node promotion
case), not a duration, and fails here as one.

A date carries its STATED precision (precision amendment,
2026-11-06): a surface stating only a month or a year
("May 2008", "2008-05", "2008") normalizes to its
reduced-precision ISO form with the precision named in
the result (``precision``: day / month / year). No
component is invented to complete it — the precision IS
the record of what the document stated, and the store
keeps it beside the value so a range query reads the
value as the period it names, not as a fabricated day.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

#: The closed value-type vocabulary a pack may declare.
VALUE_TYPES = ("date", "datetime", "number", "money",
               "duration", "identifier")

#: Attribute names are property names in the graph store and
#: keys in every record, so the declaration vocabulary is a
#: snake_case slug — validated at declaration time.
ATTRIBUTE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_ISO_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_ISO_YEAR_RE = re.compile(r"^(\d{4})$")
_MONTH_YEAR_RE = re.compile(r"^([A-Za-z]+)\.?\s+(\d{4})$")
_MONTH_NAME_RE = re.compile(
    r"^(?:(\d{1,2})(?:st|nd|rd|th)?\s+)?"
    r"([A-Za-z]+)\.?"
    r"(?:\s+(\d{1,2}(?:st|nd|rd|th)?))?,?\s+(\d{4})$")
_NEXT_WEEKDAY_RE = re.compile(
    r"^(?:next|this coming)\s+([a-z]+)$")
_IN_DAYS_RE = re.compile(r"^in\s+(\d+)\s+days?$")

_NUMBER_RE = re.compile(
    r"^([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:\s*(%|[A-Za-z][A-Za-z0-9./^\-]*))?$")
_SCALE_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9,
                "trillion": 1e12}
_NUMBER_SCALED_RE = re.compile(
    r"^([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"\s+(thousand|million|billion|trillion)"
    r"(?:\s+(%|[A-Za-z][A-Za-z0-9./^\-]*))?$")

#: Deterministic currency table: symbol or stated name ->
#: ISO 4217 code. This is a notation table, not a guess about
#: an unstated currency: a surface with no symbol and no
#: currency word maps to NO code (incomplete, §11-4).
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP",
                     "¥": "JPY", "₩": "KRW", "₹": "INR"}
#: "US$400,000.00" — the US prefix on the dollar symbol is
#: a stated notation for USD, as deterministic as the bare
#: symbol (contract documents use it constantly; the gap
#: was found by the 2026-11-06 value-attribute
#: re-measurement, where the bare "$" handling left the
#: "US" residue unparseable).
_US_DOLLAR_PREFIX_RE = re.compile(r"^US\s*\$\s*",
                                  re.IGNORECASE)
_CURRENCY_WORDS = {
    "usd": "USD", "us dollar": "USD", "us dollars": "USD",
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "gbp": "GBP", "pound sterling": "GBP",
    "pounds sterling": "GBP",
    "jpy": "JPY", "yen": "JPY",
    "krw": "KRW", "won": "KRW",
    "chf": "CHF", "swiss franc": "CHF", "swiss francs": "CHF",
    "cad": "CAD", "canadian dollar": "CAD",
    "canadian dollars": "CAD",
    "aud": "AUD", "australian dollar": "AUD",
    "australian dollars": "AUD",
    "cny": "CNY", "yuan": "CNY", "renminbi": "CNY",
    "inr": "INR", "rupee": "INR", "rupees": "INR",
}

_DURATION_ISO_RE = re.compile(
    r"^P(?:(\d+(?:\.\d+)?)Y)?(?:(\d+(?:\.\d+)?)M)?"
    r"(?:(\d+(?:\.\d+)?)W)?(?:(\d+(?:\.\d+)?)D)?"
    r"(?:T(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?"
    r"(?:(\d+(?:\.\d+)?)S)?)?$")
_DURATION_UNITS = [
    (r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|년)", "Y"),
    (r"(\d+(?:\.\d+)?)\s*(?:months?|개월)", "M"),
    (r"(\d+(?:\.\d+)?)\s*(?:weeks?|주)", "W"),
    (r"(\d+(?:\.\d+)?)\s*(?:days?|일)", "D"),
    (r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|시간)", "H"),
    (r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|분)", "Min"),
    (r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|초)", "S"),
]


def _num(text: str):
    """A numeric token (commas allowed) as int when integral,
    else float."""
    value = float(text.replace(",", ""))
    return int(value) if value.is_integer() else value


def _parse_date_parts(surface: str) -> date | None:
    """ISO and month-name date surfaces, or None."""
    text = surface.strip()
    m = _ISO_DATE_RE.match(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)),
                        int(m.group(3)))
        except ValueError:
            return None
    m = _MONTH_NAME_RE.match(text)
    if m:
        day_a, month_name, day_b, year = m.groups()
        month = _MONTHS.get(month_name.lower())
        day_text = day_a or day_b
        if month is None or day_text is None:
            return None
        day = int(re.sub(r"(st|nd|rd|th)$", "", day_text))
        try:
            return date(int(year), month, day)
        except ValueError:
            return None
    return None


def _parse_relative_date(surface: str,
                         reference: date) -> date | None:
    """Relative date surfaces resolved deterministically
    against an explicit reference date (the document date).
    Only called when a reference exists — resolving without
    one would be a guess."""
    text = surface.strip().lower()
    if text == "today":
        return reference
    if text == "tomorrow":
        return reference + timedelta(days=1)
    if text == "yesterday":
        return reference - timedelta(days=1)
    m = _IN_DAYS_RE.match(text)
    if m:
        return reference + timedelta(days=int(m.group(1)))
    m = _NEXT_WEEKDAY_RE.match(text)
    if m and m.group(1) in _WEEKDAYS:
        delta = (_WEEKDAYS[m.group(1)] - reference.weekday()) % 7
        return reference + timedelta(days=delta or 7)
    return None


def _well_formed_date(value) -> bool:
    return isinstance(value, str) and \
        _parse_date_parts(value) is not None and \
        _ISO_DATE_RE.match(value.strip()) is not None


def _parse_date_surface(surface: str):
    """(value, precision) for a stated date surface, or
    None. Full dates give day precision; reduced-precision
    surfaces — ISO "2008-05" / "2008" and month-name
    "May 2008" — give month / year precision and keep
    their reduced form as the value: no component is
    invented to complete them."""
    text = surface.strip()
    parsed = _parse_date_parts(text)
    if parsed is not None:
        return parsed.isoformat(), "day"
    m = _ISO_MONTH_RE.match(text)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}", "month"
        return None
    if _ISO_YEAR_RE.match(text):
        return text, "year"
    m = _MONTH_YEAR_RE.match(text)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month is not None:
            return (f"{int(m.group(2)):04d}-{month:02d}",
                    "month")
    return None


def _proposed_date(value):
    """(value, precision) if a proposer-supplied normalized
    date is well-formed ISO at any stated precision (day,
    month or year), else None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if _ISO_DATE_RE.match(text) or _ISO_MONTH_RE.match(
            text) or _ISO_YEAR_RE.match(text):
        return _parse_date_surface(text)
    return None


def _normalize_date(surface, proposed, reference_date):
    parsed = _parse_date_surface(surface)
    if parsed is not None:
        value, precision = parsed
        return {"value": value, "precision": precision,
                "code_normalized": True}
    if reference_date is not None:
        resolved = _parse_relative_date(surface,
                                        reference_date)
        if resolved is not None:
            return {"value": resolved.isoformat(),
                    "precision": "day",
                    "code_normalized": True}
    if proposed is not None:
        checked = _proposed_date(proposed)
        if checked is not None:
            value, precision = checked
            return {"value": value, "precision": precision,
                    "code_normalized": False}
    return None


_TIME_RE = re.compile(
    r"^(.*?)(?:[T\s]+)(\d{1,2}):(\d{2})"
    r"(?::(\d{2})(?:\.\d+)?)?\s*([AaPp]\.?[Mm]\.?)?"
    r"\s*(Z|[+-]\d{2}:?\d{2})?$")


def _normalize_datetime(surface, proposed, reference_date):
    text = surface.strip()
    parsed_date = _parse_date_parts(text)
    if parsed_date is not None:
        # A date-only surface under a datetime declaration
        # is demoted to date (§11-7): no time is invented.
        return {"value": parsed_date.isoformat(),
                "normalized_type": "date",
                "precision": "day",
                "code_normalized": True}
    m = _TIME_RE.match(text)
    if m:
        head, hh, mm, ss, ampm, offset = m.groups()
        day = _parse_date_parts(head)
        if day is not None:
            hour = int(hh)
            if ampm:
                if ampm.lower().startswith("p") and hour < 12:
                    hour += 12
                elif ampm.lower().startswith("a") and hour == 12:
                    hour = 0
            try:
                moment = datetime(day.year, day.month, day.day,
                                  hour, int(mm), int(ss or 0))
            except ValueError:
                moment = None
            if moment is not None:
                value = moment.isoformat()
                if offset:
                    value += "+00:00" if offset == "Z" \
                        else offset
                return {"value": value, "code_normalized": True}
    if proposed is not None and isinstance(proposed, str):
        candidate = proposed.strip()
        try:
            datetime.fromisoformat(candidate.replace("Z",
                                                     "+00:00"))
            return {"value": candidate,
                    "code_normalized": False}
        except ValueError:
            pass
        if _well_formed_date(candidate):
            return {"value": candidate, "normalized_type":
                    "date", "precision": "day",
                    "code_normalized": False}
    return None


def _parse_number(surface: str):
    """(value, unit) or None. Scale words (million, …) are
    notation and fold into the value; a trailing unit token
    is split off and NEVER converted (§11-5)."""
    text = surface.strip()
    m = _NUMBER_SCALED_RE.match(text)
    if m:
        value = float(m.group(1).replace(",", "")) \
            * _SCALE_WORDS[m.group(2)]
        return (int(value) if value.is_integer() else value,
                m.group(3))
    m = _NUMBER_RE.match(text)
    if m:
        return _num(m.group(1)), m.group(2)
    return None


def _normalize_number(surface, proposed, reference_date):
    parsed = _parse_number(surface)
    if parsed is not None:
        value, unit = parsed
        return {"value": value, "unit": unit,
                "code_normalized": True}
    if isinstance(proposed, dict) and \
            isinstance(proposed.get("value"), (int, float)) and \
            not isinstance(proposed.get("value"), bool):
        return {"value": proposed["value"],
                "unit": proposed.get("unit"),
                "code_normalized": False}
    if isinstance(proposed, (int, float)) and \
            not isinstance(proposed, bool):
        return {"value": proposed, "unit": None,
                "code_normalized": False}
    return None


def _normalize_money(surface, proposed, reference_date):
    text = surface.strip()
    currency = None
    rest = text
    m = _US_DOLLAR_PREFIX_RE.match(rest)
    if m:
        currency = "USD"
        rest = rest[m.end():].strip()
    if currency is None:
        for symbol, code in _CURRENCY_SYMBOLS.items():
            if symbol in rest:
                currency = code
                rest = rest.replace(symbol, " ", 1).strip()
                break
    if currency is None:
        lowered = rest.lower()
        for word in sorted(_CURRENCY_WORDS, key=len,
                           reverse=True):
            m = re.search(r"(?:^|\s)" + re.escape(word)
                          + r"(?:\s|$)", lowered)
            if m:
                currency = _CURRENCY_WORDS[word]
                rest = (rest[:m.start()] + " "
                        + rest[m.end():]).strip()
                break
    parsed = _parse_number(rest) if rest else None
    if parsed is not None:
        value, unit = parsed
        if unit is not None:
            return None  # a unit that is not a currency: not money
        return {"value": value, "currency": currency,
                "complete": currency is not None,
                "code_normalized": True}
    if isinstance(proposed, dict) and \
            isinstance(proposed.get("value"), (int, float)) and \
            not isinstance(proposed.get("value"), bool):
        code = proposed.get("currency")
        if code is not None and code not in \
                set(_CURRENCY_WORDS.values()):
            return None
        return {"value": proposed["value"], "currency": code,
                "complete": code is not None,
                "code_normalized": False}
    return None


def _normalize_duration(surface, proposed, reference_date):
    text = surface.strip()
    m = _DURATION_ISO_RE.match(text)
    if m and any(m.groups()):
        return {"value": text, "code_normalized": True}
    lowered = text.lower()
    if " to " in lowered or "from " in lowered or \
            (lowered.count("-") > 0 and _parse_date_parts(
                text.split("-")[0].strip()) is not None):
        return None  # a start/end pair is an interval (§11-6)
    parts: dict[str, str] = {}
    rest = text
    for pattern, key in _DURATION_UNITS:
        m = re.search(pattern, rest, flags=re.IGNORECASE)
        if m:
            parts[key] = m.group(1)
            rest = rest[:m.start()] + " " + rest[m.end():]
    residue = re.sub(r"[\s,and]+", "", rest.lower())
    if parts and not residue:
        def _fmt(v: str) -> str:
            return str(int(float(v))) if float(v).is_integer() \
                else v
        date_part = "".join(
            _fmt(parts[k]) + k for k in ("Y", "M", "W", "D")
            if k in parts)
        time_part = "".join(
            _fmt(parts[k]) + ("M" if k == "Min" else k)
            for k in ("H", "Min", "S") if k in parts)
        value = "P" + date_part
        if time_part:
            value += "T" + time_part
        return {"value": value, "code_normalized": True}
    if proposed is not None and isinstance(proposed, str):
        m = _DURATION_ISO_RE.match(proposed.strip())
        if m and any(m.groups()):
            return {"value": proposed.strip(),
                    "code_normalized": False}
    return None


def _normalize_identifier(surface, proposed, reference_date):
    value = re.sub(r"\s+", " ", surface.strip()).strip("\"' ")
    if value:
        return {"value": value, "code_normalized": True}
    return None


_NORMALIZERS = {
    "date": _normalize_date,
    "datetime": _normalize_datetime,
    "number": _normalize_number,
    "money": _normalize_money,
    "duration": _normalize_duration,
    "identifier": _normalize_identifier,
}


def normalize_value(value_type: str, surface,
                    proposed=None,
                    reference_date: date | None = None) -> dict:
    """Normalize one proposed attribute value, deterministically.

    ``surface`` is the value as written in the document;
    ``proposed`` is an optional proposer-supplied normalized
    value, used only when the surface itself does not parse
    (its correctness is then the judge's question, flagged by
    ``code_normalized: False`` in the result). ``reference_date``
    (the document's date, when known) is what relative date
    surfaces resolve against; without it they do not parse.

    Returns a result dict and never raises. Success:
    ``{"ok": True, "value_type", "normalized_type", "value",
    "surface", "code_normalized", "complete", ...}`` where the
    value is JSON-shaped and typed (an ISO string for
    date/datetime/duration, a number for number/money, a
    string for identifier) and type extras ride along
    (``unit`` for number, ``currency`` + ``complete`` for
    money, ``precision`` for date — the stated precision,
    day / month / year). Failure: ``{"ok": False, "value_type", "surface",
    "reason"}`` — a format violation the caller records; the
    value never reaches the judge."""
    if value_type not in _NORMALIZERS:
        return {"ok": False, "value_type": value_type,
                "surface": surface,
                "reason": f"unknown value_type {value_type!r} "
                          f"(closed set: {VALUE_TYPES})"}
    if not isinstance(surface, str) or not surface.strip():
        return {"ok": False, "value_type": value_type,
                "surface": surface,
                "reason": "the proposed value surface is empty "
                          "or not a string"}
    try:
        out = _NORMALIZERS[value_type](surface, proposed,
                                       reference_date)
    except Exception as e:  # a parser defect is a format
        # failure for this value, never a crash of the batch
        out = None
        reason = f"normalization failed: {e}"
    else:
        reason = f"the surface does not parse as {value_type}"
    if out is None:
        return {"ok": False, "value_type": value_type,
                "surface": surface, "reason": reason}
    result = {"ok": True, "value_type": value_type,
              "normalized_type": out.get("normalized_type",
                                         value_type),
              "value": out["value"], "surface": surface,
              "code_normalized": out.get("code_normalized",
                                         True),
              "complete": out.get("complete", True)}
    for key in ("unit", "currency", "precision"):
        if key in out:
            result[key] = out[key]
    return result
