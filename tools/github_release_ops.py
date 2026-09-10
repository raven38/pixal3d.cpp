#!/usr/bin/env python3
"""Small GitHub REST helper for release operations not exposed by every connector.

Uses GH_TOKEN or GITHUB_TOKEN and only Python's standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"


def token() -> str:
    value = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not value:
        raise SystemExit("GH_TOKEN or GITHUB_TOKEN is required")
    return value


def request(method: str, path: str, payload: dict | None = None) -> tuple[int, object | None]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API + path,
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token()}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "pixal3d-release-helper",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            raw = res.read()
            return res.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise SystemExit(f"GitHub API {e.code}: {detail}") from e


def create_release(args: argparse.Namespace) -> None:
    notes = args.notes or ""
    if args.notes_file:
        with open(args.notes_file, "r", encoding="utf-8") as f:
            notes = f.read()
    payload = {
        "tag_name": args.tag,
        "target_commitish": args.target,
        "name": args.title or args.tag,
        "body": notes,
        "draft": args.draft,
        "prerelease": args.prerelease,
        "generate_release_notes": args.generate_notes,
    }
    if args.dry_run:
        print(json.dumps({"operation": "create-release", "repo": args.repo, "payload": payload}, indent=2))
        return
    status, result = request("POST", f"/repos/{args.repo}/releases", payload)
    print(json.dumps({"status": status, "id": result.get("id"), "tag_name": result.get("tag_name"), "html_url": result.get("html_url")}, indent=2))


def dispatch_workflow(args: argparse.Namespace) -> None:
    inputs: dict[str, str] = {}
    for item in args.input:
        if "=" not in item:
            raise SystemExit(f"--input must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit("--input key must not be empty")
        inputs[key] = value
    payload = {"ref": args.ref, "inputs": inputs}
    workflow = urllib.parse.quote(args.workflow, safe="")
    if args.dry_run:
        print(json.dumps({"operation": "dispatch-workflow", "repo": args.repo, "workflow": args.workflow, "payload": payload}, indent=2))
        return
    status, _ = request("POST", f"/repos/{args.repo}/actions/workflows/{workflow}/dispatches", payload)
    print(json.dumps({"status": status, "workflow": args.workflow, "ref": args.ref, "inputs": inputs}, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("create-release", help="Create a GitHub Release")
    r.add_argument("--repo", required=True, help="OWNER/REPO")
    r.add_argument("--tag", required=True)
    r.add_argument("--target", default="main", help="target_commitish (default: main)")
    r.add_argument("--title")
    r.add_argument("--notes")
    r.add_argument("--notes-file")
    r.add_argument("--prerelease", action="store_true")
    r.add_argument("--draft", action="store_true")
    r.add_argument("--generate-notes", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=create_release)

    w = sub.add_parser("dispatch-workflow", help="Trigger workflow_dispatch")
    w.add_argument("--repo", required=True, help="OWNER/REPO")
    w.add_argument("--workflow", required=True, help="workflow file name or workflow id")
    w.add_argument("--ref", default="main")
    w.add_argument("--input", action="append", default=[], metavar="KEY=VALUE")
    w.add_argument("--dry-run", action="store_true")
    w.set_defaults(func=dispatch_workflow)
    return p


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
