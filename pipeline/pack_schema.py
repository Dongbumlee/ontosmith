"""Pack schema validation: the pack formats, made executable.

The pack manifest (``pack.json``, pack_format 1) and the
conformance record (``conformance.json``, record_format 1)
are defined by JSON Schema (draft 2020-12) documents shipped
as data beside the packs themselves:

    packs/pack.schema.json
    packs/conformance.schema.json

Those files are the single field-level definition of the
formats (docs/PACK_DESIGN.md §1/§2 carry the prose contract;
§1.6 carries the field tables generated from them). The
loader (pipeline.profile_config.load_pack_manifest)
validates every manifest against the pack schema before any
field is trusted, and ``ontosmith ontology check``
validates a whole pack directory — manifest, conformance
record, payload pin — for contributors.

The product's core dependencies are numpy + requests only,
so no JSON Schema library is available; this module
implements exactly the subset of draft 2020-12 the two
shipped schemas use:

    type (object / array / string / integer / number /
    boolean), const, enum, pattern, minLength, minimum,
    minItems, required, properties, additionalProperties
    (false or a schema), items, format "date", and local
    $ref (#/$defs/<name>).

A schema using any other keyword is refused at load time —
a schema the validator cannot fully evaluate must never
silently pass. Validation is fail-closed in the loader's
sense: every defect is returned as a string naming the
field path and the violated rule, and the caller refuses.
"""
from __future__ import annotations

import json
import re
import sysconfig
from datetime import date
from pathlib import Path

SCHEMA_FILES = {
    "pack": "pack.schema.json",
    "conformance": "conformance.schema.json",
}

_SUPPORTED_KEYWORDS = {
    "$schema", "$id", "$defs", "$ref", "title", "description",
    "type", "const", "enum", "pattern", "minLength", "minimum",
    "minItems", "required", "properties", "additionalProperties",
    "items", "format",
}

_SCHEMA_CACHE: dict = {}


def schema_candidates(kind: str, hint_dirs=()) -> list:
    """Where the schema file for ``kind`` may live, in
    resolution order: the tree of the pack being validated
    (the caller hints its packs/ directory), the source
    tree beside this package, then the installed data-files
    tree (share/ontosmith/packs). The first existing file
    wins."""
    name = SCHEMA_FILES[kind]
    dirs = [Path(d) for d in hint_dirs]
    dirs.append(Path(__file__).resolve().parent.parent
                / "packs")
    dirs.append(Path(sysconfig.get_path("data")) / "share"
                / "ontosmith" / "packs")
    seen = set()
    out = []
    for d in dirs:
        p = d / name
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _check_schema_keywords(node, where: str) -> None:
    """Refuse a schema that uses a keyword this validator
    does not implement (see the module docstring). Walks
    only subschema positions — property names and $defs
    names are identifiers, not keywords."""
    if not isinstance(node, dict):
        return
    for key in node:
        if key not in _SUPPORTED_KEYWORDS:
            raise ValueError(
                f"pack schema {where}: unsupported "
                f"keyword {key!r} — pipeline.pack_schema "
                f"implements a named subset of JSON "
                f"Schema; extend it before using more")
    for sub in (node.get("properties") or {}).values():
        _check_schema_keywords(sub, where)
    for sub in (node.get("$defs") or {}).values():
        _check_schema_keywords(sub, where)
    for key in ("items", "additionalProperties"):
        sub = node.get(key)
        if isinstance(sub, dict):
            _check_schema_keywords(sub, where)


def load_schema(kind: str, hint_dirs=()) -> dict | None:
    """The parsed schema document for ``kind``, or None when
    no candidate file exists. Raises ValueError when a found
    schema is malformed or uses unsupported keywords."""
    for path in schema_candidates(kind, hint_dirs):
        if not path.is_file():
            continue
        key = str(path)
        if key not in _SCHEMA_CACHE:
            try:
                doc = json.loads(path.read_text())
            except Exception as e:
                raise ValueError(
                    f"pack schema {path} is not valid "
                    f"JSON: {e}") from e
            if not isinstance(doc, dict):
                raise ValueError(
                    f"pack schema {path} is not a JSON object")
            _check_schema_keywords(doc, str(path))
            _SCHEMA_CACHE[key] = doc
        return _SCHEMA_CACHE[key]
    return None


def _type_name(instance) -> str:
    if isinstance(instance, bool):
        return "boolean"
    if isinstance(instance, dict):
        return "object"
    if isinstance(instance, list):
        return "array"
    if isinstance(instance, str):
        return "string"
    if isinstance(instance, int):
        return "integer"
    if isinstance(instance, float):
        return "number"
    if instance is None:
        return "null"
    return type(instance).__name__


def _type_ok(instance, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "integer":
        return isinstance(instance, int) \
            and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) \
            and not isinstance(instance, bool)
    if expected == "boolean":
        return isinstance(instance, bool)
    return False


def _validate(instance, schema: dict, root: dict, path: str,
              defects: list) -> None:
    ref = schema.get("$ref")
    if ref is not None:
        if not ref.startswith("#/$defs/"):
            defects.append(
                f"{path}: unsupported $ref {ref!r} (only "
                f"local #/$defs/ references are implemented)")
            return
        target = (root.get("$defs") or {}).get(
            ref[len("#/$defs/"):])
        if target is None:
            defects.append(
                f"{path}: $ref {ref!r} does not resolve")
            return
        _validate(instance, target, root, path, defects)
        return

    expected = schema.get("type")
    if expected is not None and not _type_ok(instance, expected):
        defects.append(
            f"{path}: expected {expected}, found "
            f"{_type_name(instance)}")
        return

    if "const" in schema and instance != schema["const"]:
        defects.append(
            f"{path}: {instance!r} is not the required "
            f"value {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        defects.append(
            f"{path}: {instance!r} is not one of "
            + ", ".join(repr(v) for v in schema["enum"]))

    if isinstance(instance, str):
        pattern = schema.get("pattern")
        if pattern is not None \
                and re.search(pattern, instance) is None:
            defects.append(
                f"{path}: {instance!r} does not match "
                f"pattern {pattern}")
        min_length = schema.get("minLength")
        if min_length is not None \
                and len(instance) < min_length:
            defects.append(
                f"{path}: shorter than minLength {min_length}")
        if schema.get("format") == "date":
            try:
                parsed = date.fromisoformat(instance)
                if parsed.isoformat() != instance:
                    raise ValueError("not canonical")
            except ValueError:
                defects.append(
                    f"{path}: {instance!r} is not a valid "
                    f"date (YYYY-MM-DD)")

    if isinstance(instance, (int, float)) \
            and not isinstance(instance, bool):
        minimum = schema.get("minimum")
        if minimum is not None and instance < minimum:
            defects.append(
                f"{path}: {instance!r} is below minimum "
                f"{minimum!r}")

    if isinstance(instance, list):
        min_items = schema.get("minItems")
        if min_items is not None \
                and len(instance) < min_items:
            defects.append(
                f"{path}: fewer than minItems {min_items}")
        item_schema = schema.get("items")
        if item_schema is not None:
            for i, item in enumerate(instance):
                _validate(item, item_schema, root,
                          f"{path}[{i}]", defects)

    if isinstance(instance, dict):
        for field in schema.get("required") or ():
            if field not in instance:
                defects.append(
                    f"{path}: missing required field "
                    f"{field!r}")
        properties = schema.get("properties") or {}
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            child = f"{path}.{key}"
            if key in properties:
                _validate(value, properties[key], root,
                          child, defects)
            elif additional is False:
                defects.append(
                    f"{child}: field is not defined by the "
                    f"schema (an undefined field is a format "
                    f"change, not a silent extra)")
            elif isinstance(additional, dict):
                _validate(value, additional, root, child,
                          defects)


def validate(instance, kind: str, hint_dirs=()) -> list:
    """Validate ``instance`` against the ``kind`` schema
    ("pack" or "conformance"). Returns the defect list
    (empty = valid). A missing schema file is itself a
    defect — validation never passes on an absent schema."""
    schema = load_schema(kind, hint_dirs)
    if schema is None:
        searched = ", ".join(
            str(p) for p in schema_candidates(kind, hint_dirs))
        return [f"the {SCHEMA_FILES[kind]} schema file was "
                f"not found (searched: {searched})"]
    defects: list = []
    _validate(instance, schema, schema, "$", defects)
    return defects


def validate_pack_manifest(manifest, hint_dirs=()) -> list:
    """Defects of a pack manifest against
    packs/pack.schema.json (empty = valid)."""
    return validate(manifest, "pack", hint_dirs)


def validate_conformance_record(record, hint_dirs=()) -> list:
    """Defects of a conformance record against
    packs/conformance.schema.json (empty = valid)."""
    return validate(record, "conformance", hint_dirs)


def validate_pack_dir(pack_dir) -> list:
    """Schema-level validation of a whole pack directory:
    pack.json parses + validates, the conformance record it
    names parses + validates, and the record's pack_id
    matches the manifest's. (Pin verification and payload
    validation are the loader's, in
    pipeline.profile_config.pack_setup_config.) Returns the
    defect list (empty = valid)."""
    pack_dir = Path(pack_dir)
    hint = [pack_dir.parent]
    mpath = pack_dir / "pack.json"
    if not mpath.is_file():
        return [f"pack manifest not found: {mpath}"]
    try:
        manifest = json.loads(mpath.read_text())
    except Exception as e:
        return [f"pack manifest {mpath} is not valid "
                f"JSON: {e}"]
    if not isinstance(manifest, dict):
        return [f"pack manifest {mpath} is not a JSON object"]
    defects = validate_pack_manifest(manifest, hint)
    record_name = manifest.get("conformance_record")
    if isinstance(record_name, str) and record_name:
        rpath = pack_dir / record_name
        if not rpath.is_file():
            defects.append(
                f"conformance record not found: {rpath}")
        else:
            try:
                record = json.loads(rpath.read_text())
            except Exception as e:
                defects.append(
                    f"conformance record {rpath} is not "
                    f"valid JSON: {e}")
                return defects
            if not isinstance(record, dict):
                defects.append(
                    f"conformance record {rpath} is not a "
                    f"JSON object")
                return defects
            defects += validate_conformance_record(record,
                                                   hint)
            if record.get("pack_id") is not None \
                    and manifest.get("pack_id") is not None \
                    and record["pack_id"] \
                    != manifest["pack_id"]:
                defects.append(
                    f"conformance record pack_id "
                    f"{record['pack_id']!r} does not match "
                    f"the manifest's "
                    f"{manifest['pack_id']!r}")
    return defects
