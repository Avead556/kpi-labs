import ast
import importlib.util
import io
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB_MODULE_GLOB = "*/*/src/lab.py"
INTERNAL_KEYS = frozenset({"srcDir"})


def discover(root=ROOT):
    manifests = []
    for module_path in sorted(Path(root).glob(LAB_MODULE_GLOB)):
        try:
            manifests.append(_load_manifest(module_path))
        except Exception as exc:
            print(f"[labs] skipped {module_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return manifests


def _load_manifest(module_path):
    src_dir = module_path.parent
    lab_dir = module_path.parents[1]
    course_dir = module_path.parents[2]
    module_name = f"labhub_{course_dir.name}_{lab_dir.name}".replace("-", "_")

    sys.path.insert(0, str(src_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        manifest = module.manifest()
    finally:
        sys.path.remove(str(src_dir))

    manifest["srcDir"] = str(src_dir)
    manifest["path"] = f"{course_dir.name}/{lab_dir.name}"
    manifest["slug"] = f"{course_dir.name.lower()}/lab{manifest['number']}"
    return manifest


def without_commentary(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines()
    drop = set()

    carriers = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, carriers) or len(node.body) < 2:
            continue
        first = node.body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            drop.update(range(first.lineno, first.end_lineno + 1))

    trimmed = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        row, column = token.start
        code = lines[row - 1][:column].rstrip()
        if code:
            trimmed[row] = code
        else:
            drop.add(row)

    kept = [trimmed.get(number, line)
            for number, line in enumerate(lines, 1) if number not in drop]
    return _tidy(kept)


def _tidy(lines):
    out = []
    blanks = 0
    for line in lines:
        line = line.rstrip()
        if line:
            out.append(line)
            blanks = 0
            continue
        blanks += 1
        if out and blanks <= 2:
            out.append("")

    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n" if out else ""


def default_params(demo):
    return {control["id"]: float(control["value"]) for control in demo.get("controls", [])}


class LabRegistry:

    def __init__(self, manifests=()):
        self._manifests = list(manifests)

    @classmethod
    def discover(cls, root=ROOT):
        return cls(discover(root))

    def __len__(self):
        return len(self._manifests)

    def __iter__(self):
        return iter(self._manifests)

    def lab(self, lab_id):
        for manifest in self._manifests:
            if manifest["id"] == lab_id:
                return manifest
        return None

    def demo(self, manifest, demo_id):
        for demo in manifest.get("demos", ()):
            if demo["id"] == demo_id:
                return demo
        return None

    def has_slug(self, slug):
        return any(manifest.get("slug") == slug for manifest in self._manifests)

    def public(self):
        return [public_view(manifest) for manifest in self._manifests]


def public_view(manifest):
    return {key: value for key, value in manifest.items() if key not in INTERNAL_KEYS}
