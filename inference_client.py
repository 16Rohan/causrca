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

# Always load the project's .env.
load_dotenv(ROOT / ".env")

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 600


# =============================================================================
# EXCEPTIONS
# =============================================================================


class InferenceError(RuntimeError):
    """Base inference error."""


class InferenceConfigurationError(InferenceError):
    """Invalid inference configuration."""


class InferenceRequestError(InferenceError):
    """HTTP/API request failure."""


class InferenceResponseError(InferenceError):
    """Invalid model response."""


class SchemaValidationError(InferenceResponseError):
    """Model output violates the fixed output contract."""


# =============================================================================
# ENVIRONMENT
# =============================================================================


def _env(
    name: str,
    default: str | None = None,
) -> str:

    value = os.getenv(
        name,
        default,
    )

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
        value = int(raw)

    except ValueError as exc:

        raise InferenceConfigurationError(
            f"{name} must be an integer: {raw!r}"
        ) from exc

    return value


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

        raise InferenceConfigurationError(f"{name} must be numeric: {raw!r}") from exc


def _bool_env(
    name: str,
    default: bool,
) -> bool:

    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    value = raw.strip().lower()

    if value in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if value in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise InferenceConfigurationError(f"{name} must be true/false: {raw!r}")


def get_inference_config() -> dict[str, Any]:

    max_tokens = _int_env(
        "MAX_TOKENS",
        DEFAULT_MAX_TOKENS,
    )

    if max_tokens < 1:

        raise InferenceConfigurationError("MAX_TOKENS must be >= 1.")

    timeout = _int_env(
        "TIMEOUT",
        DEFAULT_TIMEOUT,
    )

    if timeout < 1:

        raise InferenceConfigurationError("TIMEOUT must be >= 1.")

    temperature = _float_env(
        "TEMPERATURE",
        DEFAULT_TEMPERATURE,
    )

    if not 0 <= temperature <= 2:

        raise InferenceConfigurationError("TEMPERATURE must be between 0 and 2.")

    return {
        "api_key": _env("MODEL_API_KEY"),
        "model": _env("MODEL"),
        "base_url": os.getenv(
            "BASE_URL",
            DEFAULT_BASE_URL,
        )
        .strip()
        .rstrip("/"),
        "temperature": temperature,
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

        raise FileNotFoundError(f"File not found: {path}")

    return path.read_text(encoding="utf-8")


def _read_json(
    path: str | Path,
) -> Any:

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(f"JSON file not found: {path}")

    try:

        return json.loads(path.read_text(encoding="utf-8"))

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Invalid JSON: {path}: "
            f"{exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc


# =============================================================================
# FIXED OUTPUT CONTRACT VALIDATION
# =============================================================================


# =============================================================================
# FIXED OUTPUT CONTRACT VALIDATION
# =============================================================================


def _is_number(value: Any) -> bool:
    """
    JSON numeric type.

    int and float are intentionally treated as the same
    JSON type: number.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _type_name(value: Any) -> str:

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


def _expected_type_name(value: Any) -> str:

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


def _matches_type(
    actual: Any,
    expected: Any,
) -> bool:

    # Object.
    if isinstance(expected, dict):
        return isinstance(actual, dict)

    # Array.
    if isinstance(expected, list):
        return isinstance(actual, list)

    # Boolean.
    if isinstance(expected, bool):
        return isinstance(actual, bool)

    # JSON number.
    #
    # IMPORTANT:
    # 123 and 123.0 are both valid JSON numbers.
    if _is_number(expected):
        return _is_number(actual)

    # String.
    if isinstance(expected, str):
        return isinstance(actual, str)

    # Null.
    if expected is None:
        return actual is None

    return type(actual) is type(expected)


def _validate_contract(
    actual: Any,
    template: Any,
    path: str,
    errors: list[str],
) -> None:

    # -------------------------------------------------------------------------
    # TYPE
    # -------------------------------------------------------------------------

    if not _matches_type(
        actual,
        template,
    ):

        errors.append(
            f"{path}: expected "
            f"{_expected_type_name(template)}, "
            f"received {_type_name(actual)}"
        )

        return

    # -------------------------------------------------------------------------
    # OBJECT
    # -------------------------------------------------------------------------

    if isinstance(template, dict):

        template_keys = set(template.keys())

        actual_keys = set(actual.keys())

        # Missing required keys.
        for key in sorted(template_keys - actual_keys):

            errors.append(f"{path}.{key}: " "missing required key")

        # Unexpected keys.
        for key in sorted(actual_keys - template_keys):

            errors.append(f"{path}.{key}: " "unexpected key")

        # Recursively validate shared keys.
        for key in sorted(template_keys & actual_keys):

            _validate_contract(
                actual[key],
                template[key],
                f"{path}.{key}",
                errors,
            )

        return

    # -------------------------------------------------------------------------
    # ARRAY
    # -------------------------------------------------------------------------

    if isinstance(template, list):

        # [] means:
        #
        # "This field must be an array."
        #
        # There is intentionally no item constraint.
        if not template:
            return

        # A populated array in the example provides
        # the schema for each item.
        item_template = template[0]

        for index, item in enumerate(actual):

            _validate_contract(
                item,
                item_template,
                f"{path}[{index}]",
                errors,
            )

        return

    # -------------------------------------------------------------------------
    # SCALAR
    # -------------------------------------------------------------------------

    # Scalar type validation has already happened above.
    #
    # DO NOT compare the actual value against the
    # example value.
    #
    # For example:
    #
    # schema:
    #     "health_score": 72
    #
    # valid output:
    #     "health_score": 91
    #
    # Likewise:
    #
    # schema:
    #     "id": "ANOM-001"
    #
    # valid output:
    #     "id": "ANOM-037"
    #
    return


def validate_output(
    result: Any,
    schema: dict,
) -> None:
    """
    Validate model output against the fixed JSON contract.

    This function performs ONLY structural/type validation.

    It intentionally does NOT enforce:
        - foreign keys
        - cross-array references
        - ID relationships
        - enum values
        - numeric ranges
        - business rules
        - causal consistency
        - semantic relationships

    Those are analytical concerns, not properties of the
    fixed output JSON contract.
    """

    errors: list[str] = []

    _validate_contract(
        result,
        schema,
        "$",
        errors,
    )

    if not errors:
        return

    shown = errors[:100]

    message = "\n".join(f"  - {error}" for error in shown)

    if len(errors) > 100:

        message += f"\n  - ... " f"{len(errors) - 100} " "additional errors"

    raise SchemaValidationError(
        "Model output failed the fixed " "JSON contract:\n" + message
    )


# =============================================================================
# MODEL RESPONSE PARSING
# =============================================================================


def _extract_json(
    text: str,
) -> dict:

    text = text.strip()

    # Direct JSON.
    try:

        result = json.loads(text)

        if not isinstance(result, dict):

            raise InferenceResponseError(
                "Model returned JSON, but " "the root value is not an object."
            )

        return result

    except json.JSONDecodeError:
        pass

    # Markdown code fence.
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced:

        text = fenced.group(1).strip()

    start = text.find("{")

    if start < 0:

        raise InferenceResponseError("Model response contains no JSON object.")

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

                candidate = text[start : index + 1]

                try:

                    result = json.loads(candidate)

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
                        "Extracted JSON root " "is not an object."
                    )

                return result

    raise InferenceResponseError("Model returned an unclosed JSON object.")


def _message_content(
    response: dict,
) -> str:

    try:

        content = response["choices"][0]["message"]["content"]

    except (
        KeyError,
        IndexError,
        TypeError,
    ) as exc:

        raise InferenceResponseError(
            "Could not extract " "choices[0].message.content " "from model response."
        ) from exc

    if isinstance(content, str):

        return content

    if isinstance(content, list):

        parts = [
            part.get("text", "")
            for part in content
            if (
                isinstance(part, dict)
                and isinstance(
                    part.get("text"),
                    str,
                )
            )
        ]

        if parts:
            return "".join(parts)

    raise InferenceResponseError("Model returned unsupported " "message content.")


# =============================================================================
# PROMPTS
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
        system_prompt.rstrip() + "\n\n"
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
        "FIXED OUTPUT EXAMPLE:\n" + example
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
        "The raw dataset is NOT part of this request.\n\n"
        "Use only the information provided below.\n\n"
        "Treat measured values, timestamps, "
        "statistics, event groups, entities, "
        "relationships, and evidence as grounded facts.\n\n"
        "Root-cause explanations are hypotheses "
        "unless directly established by evidence.\n\n"
        "Return the required fixed RCA JSON.\n\n"
        "ANALYSISEVIDENCE:\n" + evidence_json
    )


# =============================================================================
# INFERENCE REQUEST
# =============================================================================


def call_inference_model(
    *,
    evidence: dict,
    system_prompt_path: str | Path,
    output_example_path: str | Path,
) -> dict:

    config = get_inference_config()

    system_prompt = _read_text(system_prompt_path)

    schema = _read_json(output_example_path)

    if not isinstance(
        schema,
        dict,
    ):

        raise InferenceConfigurationError("Output example root must be an object.")

    # -------------------------------------------------------------------------
    # Serialize the compact evidence.
    #
    # This is the ONLY analytical payload sent to the model.
    #
    # There is deliberately NO token arithmetic here.
    # -------------------------------------------------------------------------

    evidence_json = json.dumps(
        evidence,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )

    evidence_bytes = len(evidence_json.encode("utf-8"))

    print(f"Evidence payload: " f"{evidence_bytes:,} bytes")

    # -------------------------------------------------------------------------
    # Request body.
    # -------------------------------------------------------------------------

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
        # Explicit fixed configuration.
        #
        # NEVER calculate this from evidence size.
        "max_tokens": config["max_tokens"],
    }

    if config["json_mode"]:

        payload["response_format"] = {"type": "json_object"}

    endpoint = config["base_url"] + "/chat/completions"

    headers = {
        "Authorization": ("Bearer " + config["api_key"]),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # -------------------------------------------------------------------------
    # Request.
    # -------------------------------------------------------------------------

    try:

        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=config["timeout"],
        )

    except requests.Timeout as exc:

        raise InferenceRequestError(
            "Inference request timed out " f"after {config['timeout']} seconds."
        ) from exc

    except requests.RequestException as exc:

        raise InferenceRequestError(f"Inference request failed: {exc}") from exc

    # -------------------------------------------------------------------------
    # HTTP error.
    # -------------------------------------------------------------------------

    if not response.ok:

        body = response.text.strip()

        if len(body) > 4000:

            body = body[:4000] + "..."

        raise InferenceRequestError(
            "Inference endpoint returned " f"HTTP {response.status_code}.\n" f"{body}"
        )

    # -------------------------------------------------------------------------
    # API response.
    # -------------------------------------------------------------------------

    try:

        api_response = response.json()

    except ValueError as exc:

        raise InferenceResponseError(
            "Inference endpoint returned invalid API JSON."
        ) from exc

    content = _message_content(api_response)

    result = _extract_json(content)

    # -------------------------------------------------------------------------
    # FINAL SCHEMA VALIDATION.
    #
    # This remains completely separate from the intermediate schema.
    # prompts/output_example.json is the final contract.
    # -------------------------------------------------------------------------

    validate_output(
        result,
        schema,
    )

    # Final strict JSON check.
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

        raise SchemaValidationError("Model output is not valid strict JSON.") from exc

    return result


# =============================================================================
# OPTIONAL CONFIGURATION TEST
# =============================================================================


def print_configuration() -> None:

    config = get_inference_config()

    print("Inference client configuration OK")

    print(f"Model       : {config['model']}")

    print(f"Endpoint    : {config['base_url']}")

    print(f"Temperature : {config['temperature']}")

    print(f"Max tokens  : {config['max_tokens']}")

    print(f"Timeout     : {config['timeout']}s")

    print(f"JSON mode   : {config['json_mode']}")

    print(f".env        : {ROOT / '.env'}")


# IMPORTANT:
# There is intentionally NO:
#
#     if __name__ == "__main__":
#         main()
#
# The inference client is a library.
# run.py owns application execution.
