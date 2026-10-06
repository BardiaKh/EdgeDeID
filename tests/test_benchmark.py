"""The radiology benchmark loader: every gold value must be findable in its own note.

A gold value that is not verbatim in the note can never be matched (the metric
compares strings), so it silently turns a correct prediction into a miss AND a
false positive. That is how revizied_evals.xlsx and ISO-rewritten gold dates
cost EdgeDeID-SFT 33 of its 34 reported misses.
"""
import os
import re

import pytest

from edgedeid.eval.benchmark import DEFAULT_PATH, _iso, load_radiology_benchmark
from edgedeid.eval.metrics import extract_entities

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_PATH), reason="benchmark xlsx not present")


@pytest.fixture(scope="module")
def bench():
    return load_radiology_benchmark(verbose=False)


def test_shape(bench):
    notes, gts, st = bench
    assert len(notes) == len(gts) == 100
    assert st.tag_repairs == 4


def test_every_gold_value_is_verbatim_in_its_note(bench):
    notes, gts, _ = bench
    missing = [(i, t, v) for i, (n, g) in enumerate(zip(notes, gts))
               for t, vs in extract_entities(g).items() for v in vs if v.lower() not in n.lower()]
    assert not missing, missing[:10]


def test_all_gold_dates_reanchored(bench):
    *_, st = bench
    assert st.gold_dates == 246 and not st.dates_unanchored


def test_no_tags_reach_the_model(bench):
    notes, *_ = bench
    assert not [i for i, n in enumerate(notes) if re.search(r"</?(name|date|place|age|id|device|divice)\b", n)]


@pytest.mark.parametrize("span,iso", [("January 15, 2022", "2022-01-15"), ("March15 ,2023", "2023-03-15"),
                                      ("April 24", "????-04-24"), ("March", "????-03-??"),
                                      ("15th of January 2022", "2022-01-15"), ("01/15/2022", "2022-01-15")])
def test_iso(span, iso):
    assert _iso(span) == iso
