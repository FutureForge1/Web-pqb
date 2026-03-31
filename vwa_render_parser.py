from __future__ import annotations

import base64
import html
import re
from typing import Any
from urllib.parse import urlparse

RESULT_RE = re.compile(
    r"\[Result\]\s+\((PASS|FAIL)\)\s+config_files/([a-z_]+)_visual/(\d+)\.json"
)
TAG_RE = re.compile(r"<[^>]+>")
NEW_PAGE_MARKER = "<h2>New Page</h2>"


def first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        if text.strip():
            return text
    return None


def domain_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.netloc or None


def parse_results_lookup(results_text: str) -> dict[tuple[str, str], bool]:
    lookup: dict[tuple[str, str], bool] = {}
    for line in results_text.splitlines():
        match = RESULT_RE.search(line)
        if not match:
            continue
        status, site, task_id = match.groups()
        site = site.replace("_visual", "")
        lookup[(site, task_id)] = status == "PASS"
    return lookup


def parse_member_identity(member_name: str) -> tuple[str | None, str | None]:
    parts = member_name.split("/")
    if len(parts) < 3:
        return None, None
    site_dir = parts[-2]
    site = site_dir.removesuffix("_gpt4v_som")
    match = re.search(r"render_(\d+)\.html$", parts[-1])
    task_id = match.group(1) if match else None
    return site, task_id


def parse_header_metadata(html_text: str) -> dict[str, str]:
    start = html_text.find("<pre>")
    if start == -1:
        return {}
    start += len("<pre>")
    end = html_text.find("</pre>", start)
    if end == -1:
        return {}

    metadata: dict[str, str] = {}
    current_key: str | None = None
    pre_text = html.unescape(html_text[start:end])
    for raw_line in pre_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            current_key = key.strip()
            metadata[current_key] = value.strip()
            continue
        if current_key:
            metadata[current_key] = (metadata[current_key] + "\n" + line.strip()).strip()
    return metadata


def _clean_fragment(value: str | None) -> str | None:
    if value is None:
        return None
    text = html.unescape(TAG_RE.sub("", value)).strip()
    return first_non_empty(text)


def _extract_between(text: str, starts: list[str], end: str) -> str | None:
    for marker in starts:
        start = text.find(marker)
        if start == -1:
            continue
        start += len(marker)
        stop = text.find(end, start)
        if stop == -1:
            continue
        return text[start:stop]
    return None


def _extract_image_data_url(text: str) -> str | None:
    for prefix in ("src='data:image", 'src="data:image'):
        start = text.find(prefix)
        if start == -1:
            continue
        quote = text[start + 4]
        data_start = start + 5
        stop = text.find(quote, data_start)
        if stop == -1:
            continue
        return text[data_start:stop]
    return None


def parse_action_text(action_text: str | None) -> dict[str, str | None]:
    action = first_non_empty(action_text)
    if not action:
        return {
            "api_name": None,
            "method": None,
            "selector": None,
            "value": None,
            "text": None,
        }

    method = action.split(" ", 1)[0].strip()
    bracket_values = re.findall(r"\[([^\]]*)\]", action)

    selector = None
    value = None
    if method in {"click", "hover", "press", "scroll"} and bracket_values:
        selector = bracket_values[0]
    elif method in {"type", "fill", "select_option"}:
        if bracket_values:
            selector = bracket_values[0]
        if len(bracket_values) > 1:
            value = bracket_values[1]
    elif method == "stop" and bracket_values:
        value = bracket_values[0]

    return {
        "api_name": method,
        "method": method,
        "selector": selector,
        "value": value,
        "text": action,
    }


def parse_render_pages(html_text: str) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    segments = html_text.split(NEW_PAGE_MARKER)
    for segment in segments[1:]:
        pages.append(
            {
                "url": _clean_fragment(_extract_between(segment, ["URL: "], "</a>")),
                "state_obv": _clean_fragment(
                    _extract_between(
                        segment,
                        [
                            "<div class='state_obv'><pre>",
                            '<div class="state_obv"><pre>',
                        ],
                        "</pre>",
                    )
                ),
                "image_data_url": _extract_image_data_url(segment),
                "prev_action": _clean_fragment(
                    _extract_between(
                        segment,
                        [
                            "<div class='prev_action' style='background-color:pink'>",
                            '<div class="prev_action" style="background-color:pink">',
                            "<div class='prev_action'>",
                            '<div class="prev_action">',
                        ],
                        "</div>",
                    )
                ),
                "raw_prediction": _clean_fragment(
                    _extract_between(
                        segment,
                        [
                            "<div class='raw_parsed_prediction' style='background-color:grey'><pre>",
                            '<div class="raw_parsed_prediction" style="background-color:grey"><pre>',
                            "<div class='raw_parsed_prediction'><pre>",
                            '<div class="raw_parsed_prediction"><pre>',
                        ],
                        "</pre>",
                    )
                ),
                "parsed_action": _clean_fragment(
                    _extract_between(
                        segment,
                        [
                            "<div class='parsed_action' style='background-color:yellow'><pre>",
                            '<div class="parsed_action" style="background-color:yellow"><pre>',
                            "<div class='parsed_action'><pre>",
                            '<div class="parsed_action"><pre>',
                        ],
                        "</pre>",
                    )
                ),
            }
        )

    return pages


def decode_data_url_image(data_url: str | None) -> bytes | None:
    if not data_url or not data_url.startswith("data:image"):
        return None
    try:
        _, payload = data_url.split(",", 1)
        return base64.b64decode(payload)
    except Exception:
        return None
