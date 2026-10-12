from __future__ import annotations

import math

import pytest

from finqa_search.dsl import (
    Const,
    NoArg,
    Num,
    ProgramError,
    Ref,
    Row,
    answers_match,
    canonical,
    equivalent,
    execute,
    execute_steps,
    number_value,
    parse_program,
    round_answer,
    table_row_values,
)

TABLE = [
    ["( in millions )", "2009", "2008", "2007"],
    ["net revenue", "$ 5,829", "$ 5,735", "$ 5,516"],
    ["operating income ( loss )", "-120", "310", "295"],
    ["interest expense", "$ 45 ( 3 )", "40", "35"],
    ["total, net", "1.5", "2.5", "3.5"],
    ["margin", "12%", "13%", "n/a"],
]


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("5829", 5829.0),
        ("5,829", 5829.0),
        ("3.8%", 0.038),
        ("-12.5", -12.5),
        (".5", 0.5),
        ("$ 42", 42.0),
        ("12 %", 0.12),
        ("abc", None),
        ("10-k", None),
        ("", None),
    ],
)
def test_number_value_follows_finqa_rules(text: str, value: float | None) -> None:
    got = number_value(text)
    if value is None:
        assert got is None
    else:
        assert got == pytest.approx(value)


def test_parse_renders_back_to_the_same_text() -> None:
    text = "subtract(5829, 5735), divide(#0, 5735), multiply(#1, const_100)"
    program = parse_program(text)
    assert program.render() == text
    assert len(program) == 3
    assert program.steps[1].a == Ref(0)
    assert program.steps[2].b == Const("const_100")


def test_parse_strips_eof_and_whitespace() -> None:
    program = parse_program("  add(1, 2), EOF ")
    assert program.render() == "add(1, 2)"


def test_table_operations_take_a_row_label_and_none() -> None:
    program = parse_program("table_sum(total, net, none), divide(#0, const_3)")
    step = program.steps[0]
    assert step.a == Row("total, net")
    assert step.b == NoArg()
    assert execute(program, TABLE) == pytest.approx(2.5)


def test_row_label_with_parentheses_parses() -> None:
    program = parse_program("table_max(operating income ( loss ), none)")
    assert program.steps[0].a == Row("operating income ( loss )")
    assert execute(program, TABLE) == pytest.approx(310.0)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("foo(1, 2)", "unknown operation"),
        ("add(1)", "two arguments"),
        ("add(1, #0)", "not computed yet"),
        ("add(1, 2), subtract(#1, 1)", "not computed yet"),
        ("add(1, banana)", "not a number"),
        ("table_sum(net revenue, 3)", "none"),
        ("add(1, 2", "unbalanced"),
    ],
)
def test_parse_rejects_malformed_programs(text: str, message: str) -> None:
    with pytest.raises(ProgramError, match=message):
        parse_program(text)


def test_percent_literals_divide_by_100() -> None:
    program = parse_program("multiply(200, 3.5%)")
    assert program.steps[0].b == Num("3.5%", 0.035)
    assert execute(program, []) == pytest.approx(7.0)


def test_execute_keeps_every_intermediate_result() -> None:
    program = parse_program("subtract(5829, 5735), divide(#0, 5735)")
    results = execute_steps(program, [])
    assert results[0] == pytest.approx(94.0)
    assert results[1] == pytest.approx(94.0 / 5735)


def test_greater_returns_a_boolean() -> None:
    assert execute(parse_program("greater(5, 3)"), []) is True
    assert execute(parse_program("greater(3, 5)"), []) is False


def test_boolean_results_cannot_feed_arithmetic() -> None:
    with pytest.raises(ProgramError, match="yes/no"):
        execute(parse_program("greater(5, 3), add(#0, 1)"), [])


def test_division_by_zero_is_a_program_error() -> None:
    with pytest.raises(ProgramError, match="division by zero"):
        execute(parse_program("divide(5, 0)"), [])


def test_exp_handles_compound_growth_and_rejects_complex_results() -> None:
    assert execute(parse_program("divide(121, 100), exp(#0, 0.5)"), []) == pytest.approx(1.1)
    with pytest.raises(ProgramError):
        execute(parse_program("subtract(0, 4), exp(#0, 0.5)"), [])


def test_table_cells_drop_dollar_signs_and_bracketed_echoes() -> None:
    assert table_row_values(TABLE, "net revenue") == [5829.0, 5735.0, 5516.0]
    assert table_row_values(TABLE, "interest expense") == [45.0, 40.0, 35.0]
    with pytest.raises(ProgramError, match="non-numeric"):
        table_row_values(TABLE, "margin")


def test_table_average_and_min() -> None:
    avg = execute(parse_program("table_average(net revenue, none)"), TABLE)
    assert avg == pytest.approx((5829 + 5735 + 5516) / 3)
    assert execute(parse_program("table_min(net revenue, none)"), TABLE) == 5516.0


def test_missing_row_is_a_program_error() -> None:
    with pytest.raises(ProgramError, match="no table row"):
        execute(parse_program("table_sum(gross margin, none)"), TABLE)


def test_repeated_row_labels_resolve_to_the_last_row() -> None:
    table = [["", "a"], ["total", "1"], ["total", "5"]]
    assert execute(parse_program("table_sum(total, none)"), table) == 5.0


@pytest.mark.parametrize(
    ("pred", "gold", "ok"),
    [
        (0.0164, 0.0164, True),
        (0.016400001, 0.0164, True),
        (0.01641, 0.0164, False),
        (True, "yes", True),
        (False, "yes", False),
        (False, "no", True),
        (1.0, "yes", False),
        (True, 1.0, False),
        (None, 3.0, False),
        (94.0, "94", True),
        (94.0, "n/a", False),
    ],
)
def test_answers_match_uses_five_decimals_and_yes_no_words(
    pred: float | bool | None, gold: float | str, ok: bool
) -> None:
    assert answers_match(pred, gold) is ok


def test_round_answer_keeps_booleans() -> None:
    assert round_answer(True) is True
    assert round_answer(1.234567891) == 1.23457


def test_canonical_form_ignores_argument_order_of_commutative_ops() -> None:
    a = parse_program("add(5, 3), multiply(#0, 2)")
    b = parse_program("add(3, 5), multiply(2, #0)")
    c = parse_program("subtract(5, 3), multiply(#0, 2)")
    assert canonical(a) == canonical(b)
    assert canonical(a) != canonical(c)


def test_canonical_form_identifies_leaves_by_value() -> None:
    assert canonical(parse_program("add(5829.0, 1)")) == canonical(parse_program("add(5829, 1)"))
    assert canonical(parse_program("multiply(2, const_100)")) == canonical(
        parse_program("multiply(2, 100)")
    )


def test_equivalent_sees_through_reassociation() -> None:
    p = parse_program("add(1, 2), add(#0, 4)")
    q = parse_program("add(2, 4), add(1, #0)")
    assert equivalent(p, q)


def test_equivalent_distinguishes_operand_order_of_subtract() -> None:
    assert not equivalent(parse_program("subtract(5, 3)"), parse_program("subtract(3, 5)"))


def test_equivalent_handles_distribution() -> None:
    p = parse_program("add(3, 4), multiply(#0, 5)")
    q = parse_program("multiply(3, 5), multiply(4, 5), add(#0, #1)")
    assert equivalent(p, q)
    assert math.isclose(float(execute(p, [])), float(execute(q, [])))


def test_equivalent_treats_exp_and_tables_structurally() -> None:
    p = parse_program("divide(4, 2), exp(#0, 3)")
    q = parse_program("divide(4, 2), exp(#0, 3)")
    r = parse_program("divide(4, 2), exp(#0, 2)")
    assert equivalent(p, q)
    assert not equivalent(p, r)
    t1 = parse_program("table_sum(net revenue, none)")
    t2 = parse_program("table_average(net revenue, none)")
    assert not equivalent(t1, t2)
