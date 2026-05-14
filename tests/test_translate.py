"""Unit tests for ``scdataloader.utils.translate`` and its ``_lookup_name`` helper.

These use a fake bionty registry so they exercise the id-splitting / name-joining
logic without needing a populated ontology database.
"""

import bionty as bt

from scdataloader.utils import _lookup_name, translate


class _Record:
    def __init__(self, name):
        self.name = name


class _Filtered:
    def __init__(self, record):
        self._record = record

    def one_or_none(self):
        return self._record


class _FakeRegistry:
    """Mimics a bionty registry: ``.filter(ontology_id=...).one_or_none()``."""

    _DB = {"HANCESTRO:0005": "European", "HANCESTRO:0008": "African"}

    def filter(self, ontology_id):
        name = self._DB.get(ontology_id)
        return _Filtered(_Record(name) if name is not None else None)


def test_lookup_name_single_id():
    assert _lookup_name(_FakeRegistry(), "HANCESTRO:0005") == "European"


def test_lookup_name_compound_id():
    # CellxGene records multi-ethnic donors as comma-joined ids; this used to
    # raise because the whole string was looked up as a single term.
    assert (
        _lookup_name(_FakeRegistry(), "HANCESTRO:0005,HANCESTRO:0008")
        == "European,African"
    )


def test_lookup_name_unknown_id_returns_none():
    assert _lookup_name(_FakeRegistry(), "HANCESTRO:9999") is None


def test_lookup_name_compound_with_unknown_part_returns_none():
    assert _lookup_name(_FakeRegistry(), "HANCESTRO:0005,HANCESTRO:9999") is None


def test_lookup_name_tolerates_whitespace():
    assert (
        _lookup_name(_FakeRegistry(), "HANCESTRO:0005, HANCESTRO:0008")
        == "European,African"
    )


def test_translate_list_with_compound_id(monkeypatch):
    monkeypatch.setattr(bt, "Ethnicity", _FakeRegistry())
    out = translate(
        ["HANCESTRO:0005", "HANCESTRO:0005,HANCESTRO:0008"],
        "self_reported_ethnicity_ontology_term_id",
    )
    assert out == ["European", "European,African"]
