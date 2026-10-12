from __future__ import annotations

import pytest

from finqa_search.numbers import (
    cell_number,
    merge,
    normalise_header,
    table_mentions,
    text_mentions,
    unit_of,
    variants,
    years_in,
)


def test_sentence_numbers_carry_percent_unit_and_year() -> None:
    sentence = "revenue rose $ 94 million in 2009 , or 1.6% ( 1.6 % ) , compared to 2008 ."
    ms = text_mentions(sentence, "text_3", "text")
    values = [(m.text, m.value) for m in ms]
    # The bracketed "( 1.6 % )" echo is dropped; the two years are still numbers.
    assert values == [("94", 94.0), ("2009", 2009.0), ("1.6", 1.6), ("2008", 2008.0)]
    money, pct = ms[0], ms[2]
    assert money.unit == 1e6
    assert not money.percent
    assert pct.percent
    assert money.year == 2009
    assert money.fact_id == "text_3"


def test_bracketed_numbers_hint_at_a_negative_value() -> None:
    ms = text_mentions("a loss of ( 512 ) was recorded", "text_0", "text")
    assert ms[0].value == 512.0
    assert ms[0].negative_hint
    assert [v.value for v in variants(ms[0])] == [-512.0]


def test_glued_tokens_are_not_numbers() -> None:
    assert text_mentions("see the 10-k and note 2i", "text_0", "text") == []


def test_thousands_separators_are_read() -> None:
    (m,) = text_mentions("total of 1,234,567 shares", "text_0", "text")
    assert m.value == 1234567.0
    assert m.text == "1234567"


@pytest.mark.parametrize(
    ("cell", "parsed"),
    [
        ("$ 5,829", (5829.0, "5829", False)),
        ("12.5%", (12.5, "12.5", True)),
        ("( 120 )", None),
        ("45 ( 3 )", (45.0, "45", False)),
        ("n/a", None),
        ("", None),
    ],
)
def test_cell_number_reads_cells_like_the_interpreter(
    cell: str, parsed: tuple[float, str, bool] | None
) -> None:
    assert cell_number(cell) == parsed


def test_table_mentions_know_their_row_column_and_year() -> None:
    table = [
        ["( in millions )", "2009", "2008"],
        ["net revenue", "$ 5,829", "$ 5,735"],
        ["margin", "12%", "$ 10.8 million"],
    ]
    ms = table_mentions(table)
    by_text = {m.text: m for m in ms}
    rev = by_text["5829"]
    assert (rev.row, rev.col, rev.row_label, rev.year) == (1, 1, "net revenue", 2009)
    assert rev.unit == 1e6
    assert by_text["12"].percent
    assert by_text["10.8"].fact_id == "table_2"
    # Header years are not mentions.
    assert "2009" not in by_text or by_text["2009"].row != 0


def test_variants_add_percent_and_magnitude_readings() -> None:
    (m,) = text_mentions("rate of 2.5 % last year", "text_0", "text")
    (v,) = variants(m)
    assert v.value == pytest.approx(0.025)
    assert v.text == "2.5%"
    assert v.variant == "pct"


def test_merge_groups_mentions_by_value_in_first_seen_order() -> None:
    a = text_mentions("revenue was 100 and costs were 40", "text_0", "text")
    b = text_mentions("again 100 here", "text_1", "text")
    cands = merge(a + b)
    assert [c.value for c in cands] == [100.0, 40.0]
    assert len(cands[0].mentions) == 2


def test_units_headers_and_years() -> None:
    assert unit_of("( in thousands )") == 1e3
    assert unit_of("dollars in billions") == 1e9
    assert unit_of("shares") == 1.0
    assert normalise_header("Fiscal Year Ended 2009 ( 1 )") == "2009"
    assert years_in("from 2007 to 2009, not 1890 or 2101") == [2007, 2009]
