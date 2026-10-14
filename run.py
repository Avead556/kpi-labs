#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

from web.labs import discover

ROOT = Path(__file__).resolve().parent


def runnable(manifest):
    return [(demo["id"], Path(manifest["srcDir"]) / demo["program"])
            for demo in manifest.get("demos", ())
            if demo.get("program")]


def show(manifests):
    for manifest in manifests:
        print(f"\n{manifest['slug']}  №{manifest['number']}  {manifest['title']}")
        for demo_id, program in runnable(manifest):
            print(f"    {demo_id:<12} {program.relative_to(ROOT)}")
    print()


def find(manifests, slug):
    for manifest in manifests:
        if manifest["slug"] == slug or manifest["id"] == slug:
            return manifest
    return None


def main(argv):
    manifests = discover()
    if not manifests:
        print("no labs found", file=sys.stderr)
        return 1

    if not argv:
        print("usage: python run.py <lab> <program> [args...]")
        show(manifests)
        return 0

    slug, rest = argv[0], argv[1:]
    manifest = find(manifests, slug)
    if manifest is None:
        print(f"unknown lab '{slug}'", file=sys.stderr)
        show(manifests)
        return 2

    choices = dict(runnable(manifest))
    if not rest:
        print(f"usage: python run.py {manifest['slug']} <{'|'.join(choices)}> [args...]")
        return 0

    name, extra = rest[0], rest[1:]
    if name not in choices:
        print(f"unknown program '{name}' in {manifest['slug']}; "
              f"available: {', '.join(choices)}", file=sys.stderr)
        return 2

    return subprocess.run([sys.executable, str(choices[name]), *extra]).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
