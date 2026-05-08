"""
Step 4 — Guardrails AI Validators
====================================
TASK:
  1. Build a PIIDetector validator that detects & redacts emails, phone
     numbers, SSNs, and credit card numbers
  2. Build a JSONFormatter validator that auto-repairs malformed JSON
  3. Wrap each with a Guard and test with sample inputs
  4. Run a full demo with 6 PII cases and 5 JSON cases

DELIVERABLE: All test cases pass (PII redacted, JSON repaired)

KEY CONCEPTS:
  - @register_validator — declares a custom validator class
  - Validator.validate() — implement the check + fix logic
  - OnFailAction.FIX — replace output instead of raising an error
  - Guard().use(MyValidator(on_fail=...)) — attach validator to guard
  - guard.validate(text) → ValidationOutcome
    .validation_passed — bool
    .validated_output   — the (possibly repaired) output string

⚠️  IMPORTANT: pass `on_fail` to the VALIDATOR constructor, NOT to Guard.use()
    RIGHT: Guard().use(PIIDetector(on_fail=OnFailAction.FIX))  ← correct
"""

import json
import re
import warnings
from typing import Any, Dict

from guardrails import Guard, OnFailAction
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

warnings.filterwarnings("ignore")


# ── 1. PII Detector Validator ─────────────────────────────────────────────────
@register_validator(name="pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Detects and redacts Personally Identifiable Information (PII).

    Patterns detected:
      - EMAIL: xxx@xxx.xxx
      - PHONE: (123) 456-7890 or 123-456-7890
      - SSN:   123-45-6789
      - CREDIT CARD: 1234 5678 9012 3456 (or dashes)
    """

    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE": r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        """
        Check value for PII; if found, redact and return FailResult with fix_value.
        """
        all_matches = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            for match in re.finditer(pattern, value):
                all_matches.append((match.start(), match.end(), pii_type))

        if not all_matches:
            return PassResult()

        # Sort by start position descending to replace from back to front
        all_matches.sort(key=lambda x: x[0], reverse=True)

        redacted_text = value
        last_start = len(value) + 1
        found_pii_count = 0

        for start, end, pii_type in all_matches:
            # Basic overlap prevention: only replace if this match doesn't overlap with the previous (which starts later)
            if end <= last_start:
                redacted_text = (
                    redacted_text[:start] + "[REDACTED]" + redacted_text[end:]
                )
                last_start = start
                found_pii_count += 1

        if found_pii_count > 0:
            print(f"  ⚠️  Detected {found_pii_count} PII items")
            return FailResult(
                error_message=f"PII detected",
                fix_value=redacted_text,
            )

        return PassResult()


# ── 2. JSON Formatter Validator ───────────────────────────────────────────────
@register_validator(name="json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validates and auto-repairs malformed JSON strings.

    Common repairs:
      - Strip markdown code fences (``` or ```json)
      - Replace single quotes with double quotes
      - Remove trailing commas before } or ]
    """

    @staticmethod
    def _repair(text: str) -> str:
        """
        Attempt to repair a JSON string.
        """
        text = text.strip()

        # Remove markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Single quotes → double quotes
        text = text.replace("'", '"')

        # Remove trailing commas
        text = re.sub(r",\s*([}\]])", r"\1", text)

        return text

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        """
        Try to parse value as JSON. If it fails, try _repair() then parse again.
        """
        try:
            parsed = json.loads(value)
            # Re-serialize for consistent formatting
            repaired = json.dumps(parsed, indent=2)
            return PassResult(value_override=repaired)
        except json.JSONDecodeError:
            pass

        # Try repair
        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            repaired = json.dumps(parsed, indent=2)
            print(f"  🔧 JSON repaired successfully")
            return FailResult(
                error_message="Malformed JSON repaired", fix_value=repaired
            )
        except json.JSONDecodeError as e:
            # If repair fails, return a JSON error object as requested by README
            error_json = json.dumps({"error": "unrecoverable json", "raw": value})
            return FailResult(
                error_message=f"Invalid JSON after repair attempt: {e}",
                fix_value=error_json,
            )


# ── 3. PII Guard demo ────────────────────────────────────────────────────────
def demo_pii_guard():
    """
    Create a Guard with PIIDetector and test 6 sample texts.
    """
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call our support line at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean", "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ Pass" if result.validation_passed else "❌ Fail"

        print(f"\n[{label}] {status}")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 4. JSON Guard demo ────────────────────────────────────────────────────────
def demo_json_guard():
    """
    Create a Guard with JSONFormatter and test 5 sample strings.
    """
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma", '{"key": "value",}'),
        ("Truly invalid", "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ Pass" if result.validation_passed else "❌ Fail"

        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        # Print first 100 chars of output
        output_str = str(result.validated_output).replace("\n", " ")
        print(f"  Output: {output_str[:100]}")


# ── 5. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\n✅ Step 4 complete!")


if __name__ == "__main__":
    main()
