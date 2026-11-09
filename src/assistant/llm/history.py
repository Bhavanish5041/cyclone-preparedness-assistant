"""Safe conversation history management, compaction, and bounds enforcement.

Ensures:
- Historical assistant and user messages never exceed Pydantic schema limits (5,000 characters).
- Preservation of key analytical metrics, community profiles, and headers for downstream context.
- Graceful handling of large structured JSON / tool results without producing malformed syntax.
- Sanitization and graceful filtering of malformed, empty, or orphaned historical records.
- Complete protection against Streamlit interface crashes caused by context overflow.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.assistant.llm.schemas import ChatMessage, MessageRole

logger = logging.getLogger(__name__)

DEFAULT_MAX_HISTORY_CHARS = 4000
HARD_SCHEMA_LIMIT = 5000


def compact_json_structure(
    data: Any,
    max_list_items: int = 5,
    max_dict_keys: int = 10,
    max_str_len: int = 200,
) -> Any:
    """Recursively summarize large dictionary or list to a compact, valid JSON structure.

    Guarantees:
    - String lengths are bounded to max_str_len.
    - List items and dictionary keys are bounded.
    - Prevents blind string slicing that results in invalid JSON syntax.
    """
    if isinstance(data, str):
        if len(data) > max_str_len:
            return data[: max_str_len - 15] + "... [truncated]"
        return data
    elif isinstance(data, list):
        if len(data) > max_list_items:
            compact_list = [
                compact_json_structure(x, max_list_items=2, max_dict_keys=4, max_str_len=max_str_len // 2)
                for x in data[:max_list_items]
            ]
            compact_list.append(f"[+{len(data) - max_list_items} additional records omitted for context economy]")
            return compact_list
        return [compact_json_structure(x, max_list_items, max_dict_keys, max_str_len) for x in data]
    elif isinstance(data, dict):
        compact_dict: Dict[str, Any] = {}
        items = list(data.items())
        for k, v in items[:max_dict_keys]:
            if isinstance(v, (dict, list)):
                compact_dict[k] = compact_json_structure(v, max_list_items=2, max_dict_keys=4, max_str_len=max_str_len // 2)
            elif isinstance(v, str) and len(v) > max_str_len:
                compact_dict[k] = v[: max_str_len - 15] + "... [truncated]"
            else:
                compact_dict[k] = v
        if len(items) > max_dict_keys:
            compact_dict["_omitted_keys"] = f"{len(items) - max_dict_keys} additional fields omitted"
        return compact_dict
    return data


def compact_message_content(content: str, max_chars: int = DEFAULT_MAX_HISTORY_CHARS) -> str:
    """Bound message content string to max_chars while preserving syntactic and semantic integrity.

    Guarantees:
    - Valid JSON inputs are ALWAYS compacted into valid, parseable JSON and never sliced into syntax errors.
    - Markdown code fences are cleanly closed.
    - Natural prose is truncated at paragraph/sentence boundaries with context preservation markers.

    Args:
        content: Raw message text.
        max_chars: Maximum character limit (default 4000, strictly below schema 5000).

    Returns:
        Compacted message text within max_chars.
    """
    if not isinstance(content, str):
        content = str(content)

    clean = content.strip()
    if not clean:
        return "[Empty historical message]"

    if len(clean) <= max_chars:
        return clean

    # 1. Check if the content is pure raw JSON
    is_json = False
    if (clean.startswith("{") and clean.endswith("}")) or (clean.startswith("[") and clean.endswith("]")):
        try:
            parsed = json.loads(clean)
            is_json = True
            for (items, keys, s_len) in [(5, 10, 200), (3, 6, 100), (2, 3, 50)]:
                compacted_obj = compact_json_structure(
                    parsed, max_list_items=items, max_dict_keys=keys, max_str_len=s_len
                )
                serialized = json.dumps(compacted_obj, indent=2)
                if len(serialized) <= max_chars:
                    return serialized
                serialized_flat = json.dumps(compacted_obj)
                if len(serialized_flat) <= max_chars:
                    return serialized_flat

            # Fallback for JSON: return guaranteed valid JSON summary
            summary_dict = {
                "_summary": "Structured data condensed to fit context limit",
                "record_type": type(parsed).__name__,
                "preview_keys": list(parsed.keys())[:5] if isinstance(parsed, dict) else [],
                "item_count": len(parsed) if isinstance(parsed, (dict, list)) else 1,
            }
            return json.dumps(summary_dict)
        except Exception:
            pass

    # If the payload was identified as JSON, never fall through to prose slicing
    if is_json:
        return json.dumps({"_summary": "Structured data condensed to fit context limit"})

    # 2. Check if content contains a fenced JSON/code block that dominates length
    code_block_match = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", clean, re.DOTALL)
    if code_block_match:
        try:
            raw_inner = code_block_match.group(1)
            parsed_inner = json.loads(raw_inner)
            for (items, keys, s_len) in [(4, 8, 150), (2, 4, 80)]:
                compacted_inner = compact_json_structure(
                    parsed_inner, max_list_items=items, max_dict_keys=keys, max_str_len=s_len
                )
                summary_inner = json.dumps(compacted_inner, indent=2)
                clean_sub = clean[:code_block_match.start()] + f"```json\n{summary_inner}\n```" + clean[code_block_match.end():]
                if len(clean_sub) <= max_chars:
                    return clean_sub
        except Exception:
            pass

    # 3. Text/Markdown prose truncation at sensible sentence/paragraph boundary
    marker = "\n\n[... prior response condensed to preserve conversational context ...]"
    target_len = max_chars - len(marker) - 10
    if target_len <= 50:
        target_len = max_chars - len(marker)

    truncated = clean[:target_len]

    # Look for last paragraph or sentence boundary
    boundary_pos = -1
    for sep in ("\n\n", "\n", ". ", "; "):
        pos = truncated.rfind(sep)
        if pos > target_len // 2:
            boundary_pos = pos + len(sep)
            break

    if boundary_pos > 0:
        truncated = truncated[:boundary_pos].rstrip()
    else:
        truncated = truncated.rstrip()

    # If an open markdown code block fence was left unclosed, close it cleanly
    if truncated.count("```") % 2 != 0:
        truncated += "\n```"

    result = truncated + marker

    # Absolute guarantee: never exceed max_chars
    if len(result) > max_chars:
        result = result[: max_chars - len(marker)].rstrip() + marker

    return result


def prepare_conversation_history(
    raw_messages: Optional[List[Any]],
    max_turns: int = 10,
    max_content_chars: int = DEFAULT_MAX_HISTORY_CHARS,
) -> List[ChatMessage]:
    """Extract, sanitize, and bound conversation history into valid ChatMessage objects.

    Guarantees:
    - Never raises Pydantic ValidationError.
    - Each message content length is between 1 and max_content_chars (strictly <= 5000).
    - Preserves user, assistant, system, and tool conversational turns.
    - Preserves tool_call_id and name on tool turns.
    - Gracefully skips malformed or non-dict items.
    - Limits history length to at most max_turns (default 10).

    Args:
        raw_messages: Sequence of message dictionaries from Streamlit session or external caller.
        max_turns: Maximum number of recent messages to retain.
        max_content_chars: Maximum character limit per message.

    Returns:
        List of validated, bounded ChatMessage instances.
    """
    if not raw_messages or not isinstance(raw_messages, list):
        return []

    validated_messages: List[ChatMessage] = []

    for item in raw_messages:
        if not isinstance(item, dict):
            continue

        raw_role = item.get("role")
        if not raw_role or not isinstance(raw_role, str):
            continue

        clean_role = raw_role.strip().lower()
        if clean_role not in ("user", "assistant", "model", "tool", "system"):
            continue

        role_enum = (
            MessageRole.USER
            if clean_role == "user"
            else MessageRole.TOOL
            if clean_role == "tool"
            else MessageRole.SYSTEM
            if clean_role == "system"
            else MessageRole.ASSISTANT
        )

        raw_content = item.get("content")
        if raw_content is None:
            continue

        content_str = str(raw_content).strip()
        if not content_str:
            continue

        compacted = compact_message_content(content_str, max_chars=max_content_chars)

        try:
            msg = ChatMessage(
                role=role_enum,
                content=compacted,
                name=str(item.get("name") or "").strip() if item.get("name") else None,
                tool_call_id=str(item.get("tool_call_id") or "").strip() if item.get("tool_call_id") else None,
            )
            validated_messages.append(msg)
        except Exception as err:
            logger.warning(f"Skipping malformed historical message due to validation failure: {err}")

    # Bounded to the most recent max_turns
    if len(validated_messages) > max_turns:
        validated_messages = validated_messages[-max_turns:]

    return validated_messages
