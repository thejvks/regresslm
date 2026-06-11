from regresslm.schema import Case
from regresslm.scorers import (
    Contains, ExactMatch, JsonValid, LabelMatch, NumericTolerance, RegexMatch, MockJudge,
)
from regresslm.scorers.base import ScorerContext


def ctx(**kw):
    return ScorerContext(case=Case(id="c", input="x", **kw))


def test_exact_match():
    s = ExactMatch().score("hello", ctx(reference="hello"))
    assert s.passed and s.value == 1.0
    assert ExactMatch().score("hello", ctx(reference="bye")).passed is False


def test_contains_partial_credit():
    c = ctx(expect={"substrings": ["alpha", "beta"]})
    s = Contains().score("only alpha here", c)
    assert s.value == 0.5 and not s.passed


def test_label_match_case_insensitive():
    s = LabelMatch().score("Billing", ctx(expect={"label": "billing"}))
    assert s.passed


def test_numeric_tolerance():
    assert NumericTolerance().score("10.0", ctx(reference=10.4, expect={"tol": 0.5})).passed
    assert not NumericTolerance().score("10.0", ctx(reference=11.0, expect={"tol": 0.5})).passed


def test_regex():
    assert RegexMatch().score("order #12345", ctx(expect={"regex": r"#\d+"})).passed


def test_json_valid_and_schema():
    schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
    assert JsonValid().score('{"name": "x"}', ctx(expect={"schema": schema})).passed
    assert not JsonValid().score("{not json", ctx()).passed
    assert not JsonValid().score('{"name": 5}', ctx(expect={"schema": schema})).passed


def test_mock_judge_overlap():
    s = MockJudge().score("the cat sat on the mat", ctx(reference="cat on a mat"))
    assert 0.0 < s.value <= 1.0
