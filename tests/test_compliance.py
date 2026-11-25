"""Luhn and PCI detection, PII detection and masking, numeric grounding."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agent_eval import InvariantViolation, Span, SpanKind, Stage, Trajectory
from agent_eval.invariants import (
    BaseInvariant,
    InvariantRegistry,
    assert_no_pii,
    assert_numeric_grounding,
    assert_pci_compliant,
    extract_monetary_figures,
    find_pci,
    find_pii,
    luhn_valid,
    mask_pci,
    mask_pii,
    verhoeff_checksum_valid,
)
from agent_eval.invariants.compliance import aba_checksum_valid, shannon_entropy

TEST_PAN = "4111111111111111"  # well known Luhn valid test number
TEST_AADHAAR = "234567890124"  # Verhoeff valid 12-digit test number


def tool_span(output, *, input=None, name="ledger.get_transaction") -> Span:  # noqa: A002
    return Span(
        kind=SpanKind.TOOL,
        name=f"execute_tool {name}",
        tool_name=name,
        input=input,
        output=output,
        start_ns=1,
        end_ns=2,
    )


# ---- Luhn and PCI ---------------------------------------------------------------

@pytest.mark.parametrize(
    "number",
    [TEST_PAN, "5555555555554444", "378282246310005", "6011111111111117", "4111 1111 1111 1111"],
)
def test_luhn_accepts_valid_numbers(number):
    assert luhn_valid(number)


@pytest.mark.parametrize("number", ["4111111111111112", "1234567890123456", "", "7"])
def test_luhn_rejects_invalid_numbers(number):
    assert not luhn_valid(number)


@pytest.mark.parametrize(
    "text",
    [
        f"card {TEST_PAN} please",
        "card 4111 1111 1111 1111 please",
        "card 4111-1111-1111-1111 please",
        "ref 12 4111 1111 1111 1111",  # PAN inside a longer digit run
    ],
)
def test_pan_detected_in_common_formats(text):
    findings = find_pci(text)
    assert [f.kind for f in findings] == ["pan"]
    assert findings[0].masked == "************1111"


def test_non_luhn_and_masked_numbers_are_not_pans():
    assert find_pci("order 4111111111111112 shipped") == []
    assert find_pci("card ending ************1111") == []
    assert find_pci("0000000000000000") == []


@pytest.mark.parametrize("text", ["CVV 123", "cvv: 1234", "security code is 987", "CVC2=321"])
def test_cvv_patterns_detected(text):
    assert [f.kind for f in find_pci(text)] == ["cvv"]


def test_cvv_needs_a_keyword():
    assert find_pci("the answer is 123") == []


def test_mask_pci_hides_pan_and_cvv():
    masked = mask_pci("card 4111 1111 1111 1111, CVV 123")
    assert "4111 1111" not in masked
    assert "123" not in masked
    assert "************1111" in masked


def test_pci_invariant_scans_prompt_tool_payload_and_completion():
    invariant = assert_pci_compliant()
    assert invariant.evaluate(Trajectory(input=f"pay with {TEST_PAN}"), Stage.INPUT).passed is False
    in_tool = Trajectory(spans=[tool_span({"pan": int(TEST_PAN)})])
    assert invariant.evaluate(in_tool).passed is False
    in_output = Trajectory(final_output={"text": "your cvv 123 is saved"})
    result = invariant.evaluate(in_output)
    assert result.passed is False
    assert all(TEST_PAN not in item and "123" not in item for item in result.evidence)
    assert invariant.evaluate(Trajectory(final_output="card ending 1111")).passed is True


def test_assert_style_call_raises():
    with pytest.raises(InvariantViolation) as info:
        assert_pci_compliant()(f"number {TEST_PAN}")
    assert info.value.report.violated_names == ["pci_compliant"]
    assert TEST_PAN not in str(info.value)


# ---- PII ------------------------------------------------------------------------

def test_formatted_ssn_detected_and_masked():
    findings = find_pii("my ssn is 123-45-6789 ok")
    assert [(f.kind, f.masked) for f in findings] == [("ssn", "***-**-6789")]
    assert mask_pii("my ssn is 123-45-6789 ok") == "my ssn is ***-**-6789 ok"


@pytest.mark.parametrize("text", ["000-12-3456", "666-12-3456", "912-12-3456", "123-00-4567"])
def test_structurally_invalid_ssn_ignored(text):
    assert find_pii(text) == []


def test_bare_nine_digits_need_context_or_checksum():
    assert find_pii("order 123456789 shipped") == []
    assert [f.kind for f in find_pii("SSN 123456789")] == ["ssn"]


def test_routing_number_detected_by_aba_checksum():
    assert aba_checksum_valid("011000015")
    assert not aba_checksum_valid("011000016")
    findings = find_pii("wire to routing 011000015 today")
    assert [(f.kind, f.masked) for f in findings] == [("routing_number", "*****0015")]
    assert [f.kind for f in find_pii("aba_routing_number: 011000015")] == ["routing_number"]


def test_bare_checksum_valid_number_needs_routing_context_by_default():
    # One in ten random nine digit ids passes the ABA checksum.
    assert find_pii("ledger entry 011000015 posted") == []
    assert find_pii("routing 011000016") == []  # keyword but bad checksum
    strict = find_pii("ledger entry 011000015 posted", require_routing_context=False)
    assert [f.kind for f in strict] == ["routing_number"]
    trajectory = Trajectory(final_output="entry 011000015")
    assert assert_no_pii().evaluate(trajectory).passed is True
    assert assert_no_pii(require_routing_context=False).evaluate(trajectory).passed is False


@pytest.mark.parametrize(
    "digits",
    [TEST_AADHAAR, "999988887779", "412345678902", "234555678900"],
)
def test_verhoeff_checksum_accepts_valid_aadhaar_numbers(digits):
    assert verhoeff_checksum_valid(digits)


def test_verhoeff_checksum_rejects_errors_and_invalid_formats():
    # Every 11-digit prefix >= 2 has a unique valid Verhoeff check digit.
    for prefix in ("20000000000", "31415926535", "98765432109"):
        valid_digits = [d for d in "0123456789" if verhoeff_checksum_valid(prefix + d)]
        assert len(valid_digits) == 1

    # All single-digit errors are caught.
    for pos in range(12):
        for digit in "0123456789":
            if digit != TEST_AADHAAR[pos]:
                mutated = TEST_AADHAAR[:pos] + digit + TEST_AADHAAR[pos + 1 :]
                assert not verhoeff_checksum_valid(mutated)

    # All adjacent transpositions of distinct digits are caught.
    for sample in (TEST_AADHAAR, "999988887779"):
        for pos in range(11):
            if sample[pos] != sample[pos + 1]:
                swapped = sample[:pos] + sample[pos + 1] + sample[pos] + sample[pos + 2 :]
                assert not verhoeff_checksum_valid(swapped)

    # Twin errors caught by Verhoeff.
    assert not verhoeff_checksum_valid("911988887779")
    assert not verhoeff_checksum_valid("999933887779")
    assert not verhoeff_checksum_valid("234566678900")

    # Leading 0 or 1 rejected even when mathematically Verhoeff-valid.
    for invalid in (
        "012345678906",
        "112345678907",
        "",
        "23456789012",
        "2345678901240",
        "2345 6789 0124",
        "23456789012a",
        "23456789012²",
        "٢34567890124",
    ):
        assert not verhoeff_checksum_valid(invalid), invalid


@pytest.mark.parametrize(
    "text",
    [
        f"aadhaar {TEST_AADHAAR}",
        "Aadhaar: 2345 6789 0124",
        "UIDAI 2345-6789-0124",
        "aadhar_number: 234567890124",
        "user_uid=2345 6789 0124",
        f"Aadhaar.{TEST_AADHAAR}.",
    ],
)
def test_aadhaar_detected_in_supported_formats_and_contexts(text):
    findings = find_pii(text)
    assert [(f.kind, f.masked) for f in findings] == [("aadhaar", "********0124")]
    assert find_pci(text) == []


@pytest.mark.parametrize(
    "text",
    [
        f"order {TEST_AADHAAR}",
        "id 2345 6789 0124",
        "id 2345-6789-0124",
        "guid: 234567890124",
        "uuid: 2345-6789-0124",
        "aadhaar 2345 6789-0124",  # mixed separators
        "aadhaar 2345-6789 0124",  # mixed separators
        "aadhaar 2345.6789.0124",  # unsupported separator
        "aadhaar 0.234567890124",  # decimal
        f"aadhaar {TEST_AADHAAR}.5",  # decimal
        f"aadhaar {TEST_AADHAAR}5",  # 13 digits
        "aadhaar 012345678906",  # leading 0
        "aadhaar 1123-4567-8907",  # leading 1
        "aadhaar 234567890125",  # bad Verhoeff checksum
    ],
)
def test_aadhaar_rejects_uncontexted_mixed_decimals_and_invalid_numbers(text):
    assert find_pii(text) == []


def test_aadhaar_context_toggle_and_masking():
    assert mask_pii("aadhaar 2345 6789 0124 done") == "aadhaar ********0124 done"
    assert mask_pii("UID 2345-6789-0124 done") == "UID ********0124 done"
    masked = mask_pii(f"aadhaar {TEST_AADHAAR}")
    assert masked == "aadhaar ********0124"
    assert find_pii(masked) == []

    for bare_fmt in (TEST_AADHAAR, "2345 6789 0124", "2345-6789-0124"):
        assert find_pii(f"ref {bare_fmt}") == []
        findings = find_pii(f"ref {bare_fmt}", require_aadhaar_context=False)
        assert [(f.kind, f.masked) for f in findings] == [("aadhaar", "********0124")]
        assert mask_pii(f"ref {bare_fmt}", require_aadhaar_context=False) == "ref ********0124"


def test_no_pii_trajectory_aadhaar_dict_masking_and_scopes():
    invariant = assert_no_pii()
    in_input = Trajectory(input={"aadhaar_number": "2345 6789 0124"})
    assert invariant.evaluate(in_input, Stage.INPUT).passed is False

    in_tool = Trajectory(spans=[tool_span({"aadhaar_number": int(TEST_AADHAAR)})])
    result = invariant.evaluate(in_tool)
    assert result.passed is False
    assert result.metrics["findings"] == 1.0  # not double-flagged as high-entropy secret
    assert result.span_ids == [in_tool.spans[0].span_id]
    assert result.evidence == [
        "aadhaar in execute_tool ledger.get_transaction output: ********0124"
    ]
    assert assert_no_pii(scopes={"input", "output"}).evaluate(in_tool).passed is True
    assert assert_no_pii(ignore_keys={"aadhaar_number"}).evaluate(in_tool).passed is True

    in_output = Trajectory(final_output="UID: 2345-6789-0124")
    assert invariant.evaluate(in_output).passed is False

    clean = Trajectory(final_output=f"order {TEST_AADHAAR}")
    assert invariant.evaluate(clean).passed is True
    assert assert_no_pii(require_aadhaar_context=False).evaluate(clean).passed is False
    assert invariant.evaluate(Trajectory(spans=[tool_span({"guid": TEST_AADHAAR})])).passed is True


def test_email_detected_and_masked():
    text = "reach me at jane.doe@example.com thanks"
    assert [f.kind for f in find_pii(text)] == ["email"]
    assert mask_pii(text) == "reach me at j***@example.com thanks"


def test_entropy_flags_secrets_but_not_ids_or_prose():
    secret = "sk9Xq2LmZ7pRt4VwY8nB3cJh6KdF1gHa"
    assert shannon_entropy(secret) > 4.0
    assert [f.kind for f in find_pii(f"token={secret}")] == ["high_entropy_secret"]
    assert find_pii("trace 4bf92f3577b34da6a3ce929d0e0e4736") == []
    assert find_pii("id 123e4567-e89b-12d3-a456-426614174000") == []
    assert find_pii("payment_dispute_resolution_agent_handoff") == []
    assert find_pii(f"token={secret}", detect_secrets=False) == []


def test_clean_payment_text_has_no_findings():
    text = "Refund of $42.50 for TXN-1001 on 2026-10-03 to the card ending in 1111."
    assert find_pii(text) == []
    assert find_pci(text) == []


def test_pii_invariant_respects_scopes_and_masks_evidence():
    trajectory = Trajectory(
        input="hello",
        spans=[tool_span({"email": "jane.doe@example.com"})],
        final_output="done",
    )
    result = assert_no_pii().evaluate(trajectory)
    assert result.passed is False
    assert result.span_ids == [trajectory.spans[0].span_id]
    assert all("jane.doe" not in item for item in result.evidence)
    assert assert_no_pii(scopes={"input", "output"}).evaluate(trajectory).passed is True


# ---- Numeric grounding ----------------------------------------------------------

def test_extract_monetary_figures_formats():
    text = "Fees: $1,234.56, USD 50, 12.5 dollars, 99 cents, €20, $1.2k and 3 million USD."
    values = [value for _, value in extract_monetary_figures(text)]
    assert values == [
        Decimal("1234.56"),
        Decimal("50"),
        Decimal("12.5"),
        Decimal("0.99"),
        Decimal("20"),
        Decimal("1200"),
        Decimal("3000000"),
    ]


def test_plain_numbers_are_not_money():
    assert extract_monetary_figures("TXN-1001 has 3 items, card ending 1111") == []


def test_grounded_amount_passes():
    trajectory = Trajectory(
        spans=[tool_span({"amount": 42.5, "currency": "USD"})],
        final_output="A refund of $42.50 has been issued.",
    )
    assert assert_numeric_grounding().evaluate(trajectory).passed is True


def test_string_and_formatted_tool_values_ground():
    trajectory = Trajectory(
        spans=[tool_span({"summary": "total 1,280.00 USD", "fee": "3.10"})],
        final_output="Total $1,280 with a $3.10 fee.",
    )
    assert assert_numeric_grounding().evaluate(trajectory).passed is True


def test_hallucinated_amount_fails_with_evidence():
    trajectory = Trajectory(
        spans=[tool_span({"amount": 42.5})],
        final_output="A refund of $425.00 has been issued.",
    )
    result = assert_numeric_grounding().evaluate(trajectory)
    assert result.passed is False
    assert result.evidence == ["ungrounded figure: $425.00"]


def test_money_without_any_tool_call_fails():
    result = assert_numeric_grounding().evaluate(Trajectory(final_output="You get $10."))
    assert result.passed is False


def test_tool_inputs_do_not_ground_by_default():
    trajectory = Trajectory(
        spans=[tool_span({"ok": True}, input={"amount": 500})],
        final_output="Refunded $500.",
    )
    assert assert_numeric_grounding().evaluate(trajectory).passed is False
    assert assert_numeric_grounding(include_tool_inputs=True).evaluate(trajectory).passed is True


def test_user_input_and_minor_units_are_opt_in():
    echoed = Trajectory(input="I was charged $50", final_output="You were charged $50.")
    assert assert_numeric_grounding().evaluate(echoed).passed is False
    assert assert_numeric_grounding(include_user_input=True).evaluate(echoed).passed is True

    cents = Trajectory(spans=[tool_span({"amount_minor": 4250})], final_output="Refund: $42.50")
    assert assert_numeric_grounding().evaluate(cents).passed is False
    assert assert_numeric_grounding(allow_minor_units=True).evaluate(cents).passed is True


def test_output_without_money_passes():
    assert assert_numeric_grounding().evaluate(Trajectory(final_output="All set.")).passed is True


# ---- Fail-closed registry -------------------------------------------------------

class _Crashes(BaseInvariant):
    name = "crashes"

    def check(self, trajectory, stage):
        raise RuntimeError("boom")


class _ReturnsGarbage(BaseInvariant):
    name = "garbage"

    def check(self, trajectory, stage):
        return True


def test_registry_fails_closed():
    empty = InvariantRegistry().evaluate(Trajectory())
    assert empty.passed is False
    assert empty.violated_names == ["registry_not_empty"]

    report = InvariantRegistry([_Crashes(), _ReturnsGarbage()]).evaluate(Trajectory())
    assert report.passed is False
    assert report.violated_names == ["crashes", "garbage"]


def test_registry_rejects_duplicates_and_unknown_names():
    registry = InvariantRegistry([assert_no_pii()])
    with pytest.raises(ValueError):
        registry.register(assert_no_pii())
    with pytest.raises(KeyError):
        registry.get("nope")
    with pytest.raises(TypeError):
        registry.register(object())  # type: ignore[arg-type]


# ---- Review fixes: grounding precision, currencies, incremental scanning --------

def test_ids_and_dates_in_tool_strings_do_not_ground_amounts():
    payload = {"transaction_id": "TXN-1001", "date": "2026-10-03", "time": "12:30", "ref": "#42"}
    for hallucinated in ("$1,001.00", "$2,026.00", "$10", "$3", "$42"):
        trajectory = Trajectory(spans=[tool_span(payload)], final_output=f"Refund: {hallucinated}")
        assert assert_numeric_grounding().evaluate(trajectory).passed is False, hallucinated


def test_standalone_numbers_and_negative_amounts_in_strings_still_ground():
    trajectory = Trajectory(
        spans=[tool_span({"note": "reversal of -42.50 posted, balance 1,280.00 USD"})],
        final_output="We reversed $42.50. Your balance is $1,280.00.",
    )
    assert assert_numeric_grounding().evaluate(trajectory).passed is True


def test_ground_from_strings_can_be_switched_off():
    trajectory = Trajectory(spans=[tool_span({"note": "fee 3.10"})], final_output="Fee: $3.10")
    assert assert_numeric_grounding().evaluate(trajectory).passed is True
    assert assert_numeric_grounding(ground_from_strings=False).evaluate(trajectory).passed is False


def test_more_currencies_and_indian_grouping():
    text = "₹1,00,000, JPY 5000, C$20, 300 yen, CHF 12.50, 75 INR, ¥900"
    values = [value for _, value in extract_monetary_figures(text)]
    assert values == [
        Decimal("100000"), Decimal("5000"), Decimal("20"), Decimal("300"),
        Decimal("12.50"), Decimal("75"), Decimal("900"),
    ]
    assert extract_monetary_figures("please try 5 times, php 7 is old") == []
    assert extract_monetary_figures("€12,50 and 12,50 EUR and €1.234,56") == []


def test_unparseable_money_fails_closed():
    from agent_eval.invariants import find_unparseable_money

    text = "Refund of €12,50 issued, plus 12,50 EUR and €1.234,56."
    assert find_unparseable_money(text) == ["€12,50", "12,50 EUR", "€1.234,56"]
    assert find_unparseable_money("Refund of $1,234.56, ₹1,00,000 and 99 cents.") == []

    trajectory = Trajectory(spans=[tool_span({"amount": 12.5})], final_output="Refund: €12,50")
    result = assert_numeric_grounding().evaluate(trajectory)
    assert result.passed is False
    assert result.evidence == ["unparseable figure: €12,50"]


def test_hallucinated_amount_in_other_currency_fails():
    hallucinated = Trajectory(spans=[tool_span({"amount": 5000})], final_output="Refund: JPY 50000")
    assert assert_numeric_grounding().evaluate(hallucinated).passed is False


def test_span_stage_scans_only_the_span_that_closed():
    dirty = tool_span({"email": "jane.doe@example.com"})
    clean = tool_span({"amount": 1})
    trajectory = Trajectory(spans=[dirty, clean])
    invariant = assert_no_pii()
    assert invariant.evaluate(trajectory, Stage.SPAN, span=clean).passed is True
    assert invariant.evaluate(trajectory, Stage.SPAN, span=dirty).passed is False
    assert invariant.evaluate(trajectory, Stage.SPAN).passed is False   # no focus: full scan
    assert invariant.evaluate(trajectory, Stage.OUTPUT).passed is False  # final verdict: full scan


def test_payload_scans_grow_linearly_with_spans(monkeypatch):
    from agent_eval import EvaluationSession
    from agent_eval.invariants import compliance

    calls = {"pii": 0}
    real = compliance.find_pii

    def counting(text, **kwargs):
        calls["pii"] += 1
        return real(text, **kwargs)

    monkeypatch.setattr(compliance, "find_pii", counting)
    spans = 40
    with EvaluationSession(invariants=[assert_no_pii()]) as session:
        session.set_input("hello")
        for index in range(spans):
            with session.tool("t", input={"i": index}) as call:
                call.set_output({"ok": True})
        session.set_output("done")
    # input gate 1 + two payloads per span + one full pass at the end.
    assert calls["pii"] == 1 + 2 * spans + (1 + 2 * spans + 1)


def test_self_referential_payloads_do_not_recurse_infinitely():
    cyclic: dict = {"amount": 42.5, "note": "mail jane.doe@example.com"}
    cyclic["self"] = cyclic
    trajectory = Trajectory(spans=[tool_span(cyclic)], final_output="Refunded $42.50.")
    assert assert_numeric_grounding().evaluate(trajectory).passed is True
    assert assert_no_pii().evaluate(trajectory).passed is False


@pytest.mark.parametrize(
    "token",
    [
        "AKIA" + "IOSFODNN7EXAMPLE",
        "ghp_" + "1234567890abcdefghij1234567890",
        "github_pat_" + "11ABCDEFG0123456789_abcdef0123456789",
        "sk_live_" + "51NxYzAbCdEfGhIjKlMnOpQr",
        "sk-ant-" + "api03-abcdefghijklmnopqrstuvwx",
        "sk-proj-" + "abcdefghijklmnopqrstuvwx0123",
        "AIza" + "SyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe",
        "xoxb-" + "1234567890-abcdef123456",
    ],
)
def test_known_secret_prefixes_detected_without_duplicates(token):
    findings = find_pii(f"key={token}")
    assert [f.kind for f in findings] == ["known_secret"]
    assert token not in findings[0].masked


def test_allowed_domains_and_ignore_keys():
    payload = {"support": "help@mail.company.com", "trace_key": "AKIA" + "IOSFODNN7EXAMPLE"}
    trajectory = Trajectory(
        spans=[tool_span(payload)],
        final_output="Reach us at help@company.com or ops@mail.company.com",
    )
    assert assert_no_pii().evaluate(trajectory).passed is False
    scoped = assert_no_pii(allowed_domains={"company.com"}, ignore_keys={"trace_key"})
    assert scoped.evaluate(trajectory).passed is True
    # Suffix lookalike must not match subdomain allowlist.
    assert find_pii("hi@notcompany.com", allowed_domains={"company.com"}) != []
    assert mask_pii("ops@mail.company.com", allowed_domains={"company.com"}) == (
        "ops@mail.company.com"
    )

    pci_traj = Trajectory(spans=[tool_span({"external_id": TEST_PAN})])
    assert assert_pci_compliant(ignore_keys={"external_id"}).evaluate(pci_traj).passed is True


def test_dot_separated_pan_and_known_bin_filter():
    assert [f.kind for f in find_pci("card 4111.1111.1111.1111.")] == ["pan"]
    assert [f.kind for f in find_pci(f"Paid.{TEST_PAN}")] == ["pan"]
    # Decimals, timestamps, and large amounts must never trigger PAN detection.
    for clean in (
        "rate: 0.4111111111111111",
        "ts: 1700000123.456789",
        "amount: 41111111111111.11",
        "oid: 1.2.840.113549.1.1.11",
    ):
        assert find_pci(clean) == [], clean

    # Luhn valid 16-digit number starting with 99 (not a major card network BIN).
    non_card = "9900000000000002"
    assert luhn_valid(non_card)
    assert [f.kind for f in find_pci(non_card)] == ["pan"]
    assert find_pci(non_card, require_known_bin=True) == []
    assert mask_pci(non_card, require_known_bin=True) == non_card
    assert [f.kind for f in find_pci(TEST_PAN, require_known_bin=True)] == ["pan"]
    assert assert_pci_compliant(require_known_bin=True).evaluate(
        Trajectory(final_output=non_card)
    ).passed is True
