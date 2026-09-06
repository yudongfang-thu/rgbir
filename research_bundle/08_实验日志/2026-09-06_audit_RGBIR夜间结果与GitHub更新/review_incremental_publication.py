"""Read-only review of changed/new publication files; reuses bounded morning scanner."""
import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scanner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("bounded_publication_scanner", args.scanner)
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    modified = [x for x in git(args.repo, "diff", "HEAD", "--name-only", "-z").decode().split("\0") if x]
    untracked = [x for x in git(args.repo, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0") if x]
    names = sorted(set(modified + untracked))
    findings = []
    links = []
    missing = []
    source_checks = []
    changed_raw = []
    for name in names:
        path = args.repo / name
        if not path.is_file():
            missing.append(name)
            continue
        findings.extend(scanner.scan_file(path, name))
        if path.suffix.lower() == ".md":
            links.extend(scanner.markdown_links(path, args.repo))
        if name in modified and path.suffix.lower() in {".json", ".jsonl", ".csv", ".tsv", ".py", ".yaml", ".yml"}:
            # Export manifests are mutable packaging metadata, not raw research evidence.
            if name not in {"BUNDLE_MANIFEST.json", "PACKAGE_INVENTORY.json", "publication_checks/final_file_inventory.json"}:
                changed_raw.append(name)
        if name.startswith("research_bundle/"):
            source = args.source / name.removeprefix("research_bundle/")
            if source.is_file() and path.suffix.lower() != ".md":
                source_checks.append({"path": name, "bytes_equal": path.read_bytes() == source.read_bytes()})
    files = [p for p in args.repo.rglob("*") if p.is_file() and ".git" not in p.relative_to(args.repo).parts]
    result = {"captured_at": datetime.now().astimezone().isoformat(), "scope": "Changed/new files against existing branch HEAD; bounded credential rules reused without displaying values. Existing raw research changes are flagged for manual classification. No new hashes or credential access.",
        "changed_or_new_files": len(names), "changed_paths": modified, "untracked_count": len(untracked),
        "candidate_findings": findings, "markdown_link_findings": links, "missing_paths": missing,
        "changed_existing_non_markdown_candidates": changed_raw,
        "copied_source_byte_checks": source_checks, "copied_source_bytes_all_equal": all(row["bytes_equal"] for row in source_checks),
        "total_repo_files": len(files), "total_repo_bytes": sum(p.stat().st_size for p in files),
        "large_files_20MiB": [p.relative_to(args.repo).as_posix() for p in files if p.stat().st_size > 20 * 1024 ** 2]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"changed_paths", "candidate_findings", "markdown_link_findings", "copied_source_byte_checks"}}, ensure_ascii=True))
    print("candidate_findings=" + str(len(findings)) + " markdown_link_findings=" + str(len(links)))


if __name__ == "__main__":
    main()
