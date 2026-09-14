"""Optional, explicit preprocessing for log-like strings."""

from __future__ import annotations

import re
from collections.abc import Iterable

_TIMESTAMP = re.compile(
    r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{2,4})[ T]\d{2}:\d{2}:\d{2}"
    r"(?:[.,]\d{1,6})?(?:Z|[+-]\d{2}:?\d{2})?[,]?"
)
_SEVERITY = re.compile(
    r"(?i)(?<!\S)(?:trace|debug|info|notice|warn(?:ing)?|error|fatal|critical|alert|emerg(?:ency)?)\b"
)
_UUID = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
# Windows logs also use short bare hexadecimal sequence fields such as
# ``0000000e``. Normalize seven- and eight-character variants through the
# *number* filter so decimal and bare-hex forms receive the same marker. The
# bounded width deliberately excludes longer stable identifiers (for example,
# package component hashes) that should retain their structural signal.
_HEX = re.compile(r"(?i)\b0x[0-9a-f]+\b")
_NUMBER = re.compile(
    r"(?i)(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|(?=[0-9a-f]{7,8}\b)(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,8})(?![\w.])"
)
_PATH = re.compile(r"(?:(?:[A-Za-z]:\\|/)[^\s,;:'\"<>|]+(?:[/\\][^\s,;:'\"<>|]+)*)")
_QUOTED = re.compile(r"(?:'[^']*'|\"[^\"]*\")")
_IDENTIFIER = re.compile(
    r"(?i)\b(?:pid|process(?:[_ ]?id)?|tid|thread(?:[_ ]?id)?|session(?:[_ ]?id)?)\s*[:=]\s*[^\s,;]+"
)

_FILTERS = {
    "timestamp": (_TIMESTAMP, "<#T>"),
    "severity": (_SEVERITY, "<#S>"),
    "uuid": (_UUID, "<#U>"),
    "ip": (_IP, "<#I>"),
    "hex": (_HEX, "<#H>"),
    "number": (_NUMBER, "<#N>"),
    "path": (_PATH, "<#P>"),
    "quoted": (_QUOTED, "<#Q>"),
    "identifier": (_IDENTIFIER, "<#D>"),
}
_VARIABLE_FILTERS = frozenset({"uuid", "ip", "hex", "number", "path", "quoted", "identifier"})
_FILTER_ORDER = tuple(_FILTERS)


class LogPreprocessor:
    """Replace selected log fields with fixed-width category masks.

    Every replacement is exactly four characters long. The marker is unique to
    the filter: ``<#T>`` for timestamps, ``<#S>`` for severity, ``<#U>`` for
    UUIDs, ``<#I>`` for IPs, ``<#H>`` for hexadecimal values, ``<#N>`` for
    numbers, ``<#P>`` for paths, ``<#Q>`` for quoted values, and ``<#D>`` for
    identifiers. ``variable`` expands to every value-masking filter and ``all``
    enables every available filter.
    """

    def __init__(self, filters: str | Iterable[str] | None = None):
        if filters is None:
            values: tuple[str, ...] = ()
        elif isinstance(filters, str):
            values = (filters,)
        else:
            values = tuple(filters)
        if any(not isinstance(value, str) for value in values):
            raise TypeError("log_filters must contain only strings")

        selected = set(values)
        if "all" in selected:
            selected = set(_FILTERS)
        else:
            selected.discard("all")
            if "variable" in selected:
                selected.remove("variable")
                selected.update(_VARIABLE_FILTERS)
        unknown = selected.difference(_FILTERS)
        if unknown:
            choices = ", ".join((*_FILTER_ORDER, "variable", "all"))
            raise ValueError(f"Unknown log filter(s) {sorted(unknown)!r}; choose from {choices}")
        self.filters = tuple(name for name in _FILTER_ORDER if name in selected)

    def transform(self, value: str) -> str:
        """Return the fixed-width transformed value, or the original string."""
        if not self.filters:
            return value
        result = value
        for name in self.filters:
            pattern, marker = _FILTERS[name]
            result = pattern.sub(marker, result)
        return result

    def transform_many(self, values: Iterable[str]) -> list[str]:
        return [self.transform(value) for value in values]
