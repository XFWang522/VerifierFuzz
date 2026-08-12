"""Audit artifact serialization and sinks."""

from .html import build_html, write_html
from .sarif import build_sarif, write_sarif
from .serialize import case_from_dict, case_to_dict, finding_to_dict
from .sinks import JsonlSink, render_summary

__all__ = [
    "JsonlSink",
    "build_html",
    "build_sarif",
    "case_from_dict",
    "case_to_dict",
    "finding_to_dict",
    "render_summary",
    "write_html",
    "write_sarif",
]
