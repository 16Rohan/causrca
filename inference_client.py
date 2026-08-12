from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 600


# =============================================================================
# EXCEPTIONS
# =============================================================================

class InferenceError(RuntimeError):
    pass


class InferenceConfigurationError(InferenceError):
    pass


class InferenceRequestError(InferenceError):
    pass


class InferenceResponseError(InferenceError):
    pass


class SchemaValidationError(InferenceResponseError):
    pass


# =============================================================================
# ENVIRONMENT
# =============================================================================

def _env(
    name: str,
    default: str | None = None,
) -> str:

    value = os.getenv(name, default)

    if value is None or not value.strip():
        raise InferenceConfigurationError(
            f"Missing required environment variable: {name}"
        )

    return value.strip()


def _int_env(
    name: str,
    default: int,
) -> int:

    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        return int(raw)
    except ValueError as exc:
        raise InferenceConfigurationError(
            f"{name} must be an integer: {raw!r}"
        ) from exc


def _float_env(
    name: str,
    default: float,
) -> float:

    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        return float(raw)
    except ValueError as exc:
        raise InferenceConfigurationError(
            f"{name} must be numeric: {raw!r}"
        ) from exc


def _bool_env(
    name: str,
    default: bool,
) -> bool:

    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    value = raw.strip().lower()

    if value in {"1", "true", "yes", "on"}:
        return True

    if value in {"0", "false", "no", "off"}:
        return False

    raise InferenceConfigurationError(
        f"{name} must be true/false: {raw!r}"
    )


def get_inference_config() -> dict[str, Any]:

    max_tokens = _int_env(
        "MAX_TOKENS",
        DEFAULT_MAX_TOKENS,
    )

    if max_tokens < 1:
        raise InferenceConfigurationError(
            "MAX_TOKENS must be >= 1."
        )

    timeout = _int_env(
        "TIMEOUT",
        DEFAULT_TIMEOUT,
    )

    if timeout < 1:
        raise InferenceConfigurationError(
            "TIMEOUT must be >= 1."
        )

    return {
        "api_key": _env("MODEL_API_KEY"),
        "model": _env("MODEL"),
        "base_url": os.getenv(
            "BASE_URL",
            DEFAULT_BASE_URL,
        ).strip().rstrip("/"),
        "temperature": _float_env(
            "TEMPERATURE",
            DEFAULT_TEMPERATURE,
        ),
        "max_tokens": max_tokens,
        "timeout": timeout,
        "json_mode": _bool_env(
            "JSON_MODE",
            True,
        ),
    }


# =============================================================================
# FILE HELPERS
# =============================================================================

def _read_text(
    path: str | Path,
) -> str:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


def _read_json(
    path: str | Path,
) -> Any:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON: {path}: "
            f"{exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc


# =============================================================================
# FIXED OUTPUT CONTRACT VALIDATION
# =============================================================================

def _is_number(
    value: Any,
) -> bool:

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def _matches(
    actual: Any,
    expected: Any,
) -> bool:
    """
    Match the JSON contract semantically.

    Important:
        JSON has one numeric value family for our purposes.

        1820
        1820.0
        1820.5

    are all valid numeric values.

    The example file defines the structure and
    semantic value categories, not arbitrary
    integer/float restrictions.
    """

    # Objects.
    if isinstance(expected, dict):
        return isinstance(actual, dict)

    # Arrays.
    if isinstance(expected, list):
        return isinstance(actual, list)

    # Booleans must be checked before numbers because
    # bool is a subclass of int in Python.
    if isinstance(expected, bool):
        return isinstance(actual, bool)

    # Numeric values.
    if _is_number(expected):
        return _is_number(actual)

    # Strings.
    if isinstance(expected, str):
        return isinstance(actual, str)

    # Null.
    if expected is None:
        return actual is None

    return type(actual) is type(expected)


def _type_name(
    value: Any,
) -> str:

    if value is None:
        return "null"

    if isinstance(value, bool):
        return "boolean"

    if _is_number(value):
        return "number"

    if isinstance(value, str):
        return "string"

    if isinstance(value, list):
        return "array"

    if isinstance(value, dict):
        return "object"

    return type(value).__name__


def _expected_name(
    value: Any,
) -> str:

    if value is None:
        return "null"

    if isinstance(value, bool):
        return "boolean"

    if _is_number(value):
        return "number"

    if isinstance(value, str):
        return "string"

    if isinstance(value, list):
        return "array"

    if isinstance(value, dict):
        return "object"

    return type(value).__name__


def _validate(
    actual: Any,
    expected: Any,
    path: str,
    errors: list[str],
) -> None:

    if not _matches(
        actual,
        expected,
    ):
        errors.append(
            f"{path}: expected "
            f"{_expected_name(expected)}, "
            f"received {_type_name(actual)}"
        )
        return

    # -------------------------------------------------------------------------
    # Object
    # -------------------------------------------------------------------------

    if isinstance(expected, dict):

        actual_keys = set(actual)
        expected_keys = set(expected)

        for key in sorted(
            expected_keys - actual_keys
        ):
            errors.append(
                f"{path}.{key}: missing required key"
            )

        for key in sorted(
            actual_keys - expected_keys
        ):
            errors.append(
                f"{path}.{key}: unexpected key"
            )

        for key in sorted(
            expected_keys & actual_keys
        ):
            _validate(
                actual[key],
                expected[key],
                f"{path}.{key}",
                errors,
            )

        return

    # -------------------------------------------------------------------------
    # Array
    # -------------------------------------------------------------------------

    if isinstance(expected, list):

        # An empty array in the example means:
        # "this value must be an array".
        #
        # There is no item structure to validate.
        if not expected:
            return

        item_schema = expected[0]

        for index, item in enumerate(actual):
            _validate(
                item,
                item_schema,
                f"{path}[{index}]",
                errors,
            )


def validate_output(
    result: Any,
    schema: dict,
) -> None:

    errors: list[str] = []

    _validate(
        result,
        schema,
        "$",
        errors,
    )

    if not errors:
        return

    shown = errors[:100]

    message = "\n".join(
        f"  - {error}"
        for error in shown
    )

    if len(errors) > 100:
        message += (
            f"\n  - ... {len(errors) - 100} "
            "additional errors"
        )

    raise SchemaValidationError(
        "Model output failed the fixed JSON contract:\n"
        + message
    )


# =============================================================================
# MODEL OUTPUT PARSING
# =============================================================================

def _extract_json(
    text: str,
) -> dict:

    text = text.strip()

    # Preferred: pure JSON.
    try:
        result = json.loads(text)

        if not isinstance(
            result,
            dict,
        ):
            raise InferenceResponseError(
                "Model returned JSON, but the root "
                "value is not an object."
            )

        return result

    except json.JSONDecodeError:
        pass

    # Handle Markdown fences.
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")

    if start < 0:
        raise InferenceResponseError(
            "Model response contains no JSON object."
        )

    depth = 0
    in_string = False
    escaped = False

    for index in range(
        start,
        len(text),
    ):

        char = text[index]

        if in_string:

            if escaped:
                escaped = False

            elif char == "\\":
                escaped = True

            elif char == '"':
                in_string = False

            continue

        if char == '"':
            in_string = True

        elif char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:

                candidate = text[
                    start:index + 1
                ]

                try:
                    result = json.loads(
                        candidate
                    )
                except json.JSONDecodeError as exc:
                    raise InferenceResponseError(
                        "Model produced malformed JSON: "
                        f"{exc.msg} "
                        f"(line {exc.lineno}, "
                        f"column {exc.colno})"
                    ) from exc

                if not isinstance(
                    result,
                    dict,
                ):
                    raise InferenceResponseError(
                        "Extracted JSON root "
                        "is not an object."
                    )

                return result

    raise InferenceResponseError(
        "Model returned an unclosed JSON object."
    )


def _message_content(
    response: dict,
) -> str:

    try:
        content = response[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]
    except (
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        raise InferenceResponseError(
            "Could not extract "
            "choices[0].message.content "
            "from model response."
        ) from exc

    if isinstance(
        content,
        str,
    ):
        return content

    if isinstance(
        content,
        list,
    ):

        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
            and isinstance(
                part.get("text"),
                str,
            )
        ]

        if parts:
            return "".join(parts)

    raise InferenceResponseError(
        "Model returned unsupported message content."
    )


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def _system_message(
    system_prompt: str,
    schema: dict,
) -> str:

    example = json.dumps(
        schema,
        indent=2,
        ensure_ascii=False,
    )

    return (
        system_prompt.rstrip()
        + "\n\n"
        "OUTPUT CONTRACT\n"
        "===============\n"
        "Return exactly one JSON object.\n"
        "Follow the supplied output structure exactly.\n"
        "Do not add, remove, rename, or wrap keys.\n"
        "Preserve the object and array structure.\n"
        "Use valid JSON only.\n"
        "Do not return Markdown or explanatory text.\n"
        "Do not invent measurements, timestamps, "
        "entities, or statistics.\n\n"
        "FIXED OUTPUT EXAMPLE:\n"
        + example
    )


def _user_message(
    evidence: dict,
) -> str:

    evidence_json = json.dumps(
        evidence,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    return (
        "Analyze the following deterministic "
        "AnalysisEvidence.\n\n"
        "The original dataset has already been "
        "processed by the Python pipeline. "
        "Use only the information provided below.\n\n"
        "Treat measured values, timestamps, "
        "statistics, anomaly identifiers, "
        "entities, relationships, and evidence "
        "as grounded facts.\n\n"
        "Root-cause explanations are hypotheses "
        "unless directly established by evidence.\n\n"
        "Return the required fixed RCA JSON.\n\n"
        "ANALYSISEVIDENCE:\n"
        + evidence_json
    )


# =============================================================================
# INFERENCE MODEL REQUEST
# =============================================================================

def call_inference_model(
    *,
    evidence: dict,
    system_prompt_path: str | Path,
    output_example_path: str | Path,
) -> dict:

    config = get_inference_config()

    system_prompt = _read_text(
        system_prompt_path
    )

    schema = _read_json(
        output_example_path
    )

    if not isinstance(
        schema,
        dict,
    ):
        raise InferenceConfigurationError(
            "Output example root must be an object."
        )

    # The only data supplied to the model from the analysis
    # pipeline is AnalysisEvidence.
    #
    # No CSV or raw dataset is loaded here.

    try:
        json.dumps(
            evidence,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise InferenceResponseError(
            "AnalysisEvidence is not valid strict JSON."
        ) from exc

    payload: dict[str, Any] = {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": _system_message(
                    system_prompt,
                    schema,
                ),
            },
            {
                "role": "user",
                "content": _user_message(
                    evidence,
                ),
            },
        ],
        "temperature": config["temperature"],

        # Direct configuration value.
        # No token calculation is performed.
        "max_tokens": config["max_tokens"],
    }

    if config["json_mode"]:
        payload["response_format"] = {
            "type": "json_object"
        }

    endpoint = (
        config["base_url"]
        + "/chat/completions"
    )

    headers = {
        "Authorization": (
            "Bearer "
            + config["api_key"]
        ),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=config["timeout"],
        )

    except requests.Timeout as exc:
        raise InferenceRequestError(
            "Inference request timed out after "
            f"{config['timeout']} seconds."
        ) from exc

    except requests.RequestException as exc:
        raise InferenceRequestError(
            f"Inference request failed: {exc}"
        ) from exc

    if not response.ok:

        body = response.text.strip()

        if len(body) > 4000:
            body = body[:4000] + "..."

        raise InferenceRequestError(
            f"Inference endpoint returned HTTP "
            f"{response.status_code}.\n{body}"
        )

    try:
        api_response = response.json()
    except ValueError as exc:
        raise InferenceResponseError(
            "Inference endpoint returned invalid API JSON."
        ) from exc

    content = _message_content(
        api_response
    )

    result = _extract_json(
        content
    )

    # Final structural contract check.
    #
    # Numeric fields intentionally accept both
    # integer and floating-point JSON numbers.
    validate_output(
        result,
        schema,
    )

    # Strict JSON check.
    try:
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise SchemaValidationError(
            "Model output is not valid strict JSON."
        ) from exc

    return result


# =============================================================================
# CONFIGURATION TEST
# =============================================================================

def main() -> None:

    config = get_inference_config()

    print(
        "Inference client configuration OK"
    )
    print(
        f"Model       : {config['model']}"
    )
    print(
        f"Endpoint    : {config['base_url']}"
    )
    print(
        f"Temperature : {config['temperature']}"
    )
    print(
        f"Max tokens  : {config['max_tokens']}"
    )
    print(
        f"Timeout     : {config['timeout']}s"
    )
    print(
        f"JSON mode   : {config['json_mode']}"
    )
    print(
        f".env        : {ROOT / '.env'}"
    )


if __name__ == "__main__":
    main()
