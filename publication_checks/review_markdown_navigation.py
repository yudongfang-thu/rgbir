"""Check rendered Markdown navigation and invalid nested ordinary links."""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

from markdown_it import MarkdownIt


def nested_links(line):
    stack = []
    findings = []
    position = 0
    while position < len(line):
        char = line[position]
        if char == "\\":
            position += 2
            continue
        if char == "`":
            end = position
            while end < len(line) and line[end] == "`":
                end += 1
            close = line.find(line[position:end], end)
            position = len(line) if close < 0 else close + end - position
            continue
        if char == "[":
            stack.append({"column": position + 1, "image": position > 0 and line[position - 1] == "!", "inner_link": False})
        elif char == "]" and stack:
            item = stack.pop()
            if position + 1 < len(line) and line[position + 1] == "(":
                if item["inner_link"]:
                    findings.append({"column": item["column"], "rule": "ordinary_link_nested_inside_link"})
                if stack and not item["image"]:
                    stack[-1]["inner_link"] = True
                # Do not parse brackets in the destination as label brackets.
                end = position + 2
                depth = 1
                while end < len(line) and depth:
                    if line[end] == "\\":
                        end += 2
                        continue
                    depth += int(line[end] == "(") - int(line[end] == ")")
                    end += 1
                position = end
                continue
            if stack and item["inner_link"]:
                stack[-1]["inner_link"] = True
        position += 1
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    markdown = MarkdownIt("commonmark").enable("table")
    files = sorted(p for p in args.repo.rglob("*.md") if ".git" not in p.relative_to(args.repo).parts)
    nested = []
    missing = []
    count = 0
    external_count = 0
    image_count = 0
    for path in files:
        content = path.read_text(encoding="utf-8-sig")
        rel = path.relative_to(args.repo).as_posix()
        tokens = markdown.parse(content)
        code_lines = set()
        for token in tokens:
            if token.type in {"fence", "code_block"} and token.map:
                code_lines.update(range(token.map[0] + 1, token.map[1] + 1))
        for number, line in enumerate(content.splitlines(), 1):
            if number not in code_lines:
                nested.extend({"path": rel, "line": number, **item} for item in nested_links(line))
        pending = list(tokens)
        while pending:
            token = pending.pop()
            if token.children:
                pending.extend(token.children)
            if token.type not in {"link_open", "image"}:
                continue
            raw = token.attrGet("src" if token.type == "image" else "href")
            if not raw:
                continue
            if raw.startswith(("https:", "http:", "mailto:", "#", "app:", "codex:")):
                external_count += 1
                continue
            count += 1
            image_count += token.type == "image"
            target = unquote(raw.split("#", 1)[0].split("?", 1)[0])
            if re.match(r"^[A-Za-z]:[/\\]", target) or target.startswith(("/mnt/", "/private/", "file:")):
                missing.append({"path": rel, "target": target, "rule": "rendered_absolute_local_target"})
                continue
            resolved = args.repo / target.lstrip("/") if target.startswith("/") else path.parent / target
            if not resolved.exists():
                missing.append({"path": rel, "target": target, "rule": "rendered_missing_target"})
    result = {"created_local": datetime.now().astimezone().isoformat(), "markdown_files": len(files), "rendered_local_link_or_image_count": count, "rendered_local_images": image_count, "external_or_fragment_links": external_count, "nested_link_findings": nested, "rendered_target_findings": missing, "scope": "CommonMark + table tokenization; local target existence and ordinary nested links only; no external HTTP requests or full browser screenshot"}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
