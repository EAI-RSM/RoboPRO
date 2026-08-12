#!/usr/bin/env python3
"""Apply and verify the mechanical S5 cleanup of ``benchmark/bench_envs``."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
BENCH_ENVS = REPO_ROOT / "benchmark" / "bench_envs"
SELF = Path(__file__).resolve()

T4_TARGETS = {
    "kitchenl/_kitchen_base_large.py": (
        "_entity_aabb",
        "_init_drawer_states",
        "set_drawer_open",
        "set_drawer_closed",
        "is_drawer_open",
        "_sample_model_id",
    ),
    "kitchenl/pick_boxdrink_from_basket.py": (
        "_world_point_in_entity_local",
        "_ee_pose_above_place_target",
    ),
    "kitchenl/put_milk_box_in_fridge.py": ("_fridge_inside_target_pose",),
    "kitchenl/put_sauce_can_in_cabinet.py": ("_cabinet_inside_target_pose",),
    "utils/create_actor_custom.py": ("create_multiple_obj_actor",),
    "utils/scene_gen_utils.py": (
        "get_random_valid_placement",
        "get_obj_new_pose",
    ),
}

T5_TARGETS = {
    "study/_study_base_task.py",
    "study/move_book_onto_table.py",
    "study/move_seal_onto_book.py",
    "study/put_cup_in_box.py",
}
T5_OLD_KEY = "include_" "collison"

# These files carried legitimate pre-S5 research/base changes relative to dev.
VERIFY_BASE_EXCEPTIONS = {
    "_bench_base_task.py",
    "eval_video.py",
    "study/_study_base_task.py",
}


@dataclass(frozen=True)
class Change:
    transform: str
    path: Path
    line: int
    detail: str

    def render(self) -> str:
        rel = self.path.relative_to(REPO_ROOT)
        return f"{self.transform} {rel}:{self.line}: {self.detail}"


def _python_files() -> list[Path]:
    return sorted(BENCH_ENVS.rglob("*.py"))


def _parse(path: Path, text: str | None = None) -> ast.Module:
    source = path.read_text(encoding="utf-8") if text is None else text
    return ast.parse(source, filename=str(path))


def _line_edits(text: str, edits: Iterable[tuple[int, int, str]]) -> str:
    """Apply 1-based inclusive line edits from bottom to top."""
    lines = text.splitlines(keepends=True)
    for start, end, replacement in sorted(edits, reverse=True):
        replacement_lines = replacement.splitlines(keepends=True)
        if replacement and not replacement.endswith("\n"):
            replacement_lines[-1] += "\n"
        lines[start - 1 : end] = replacement_lines
        if not replacement and end >= len(text.splitlines()):
            while lines and not lines[-1].strip():
                lines.pop()
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
    return "".join(lines)


def _function_span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int]:
    decorators = [item.lineno for item in node.decorator_list]
    return (min([node.lineno, *decorators]), node.end_lineno or node.lineno)


def _canonical_setup_demo(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.name != "setup_demo":
        return False
    if ast.unparse(node.args) != "self, is_test=False, **kwargs":
        return False
    return [ast.unparse(stmt) for stmt in node.body] == [
        "kwargs['collision_cache'] = {'mesh': 100, 'obb': 3}",
        "super()._init_task_env_(**kwargs)",
    ]


def _t1(path: Path, text: str) -> tuple[str, list[Change]]:
    if path.name == "_bench_base_task.py":
        return text, []
    changes: list[Change] = []
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(_parse(path, text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _canonical_setup_demo(node):
            start, end = _function_span(node)
            edits.append((start, end, ""))
            changes.append(Change("T1", path, start, "drop canonical setup_demo"))
    return _line_edits(text, edits), changes


def _repo_symbol_references(symbols: set[str]) -> dict[str, list[str]]:
    references = {symbol: [] for symbol in symbols}
    ignored_parts = {".git", ".venv", "graphify-out", "research_archive", "__pycache__"}
    for path in REPO_ROOT.rglob("*.py"):
        if path.resolve() == SELF or any(part in ignored_parts for part in path.parts):
            continue
        try:
            tree = _parse(path)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            if name in references:
                rel = path.relative_to(REPO_ROOT)
                references[name].append(f"{rel}:{node.lineno}")
    return references


def _t4(path: Path, text: str, *, references: dict[str, list[str]]) -> tuple[str, list[Change]]:
    rel = path.relative_to(BENCH_ENVS).as_posix()
    targets = set(T4_TARGETS.get(rel, ()))
    if not targets:
        return text, []
    edits: list[tuple[int, int, str]] = []
    changes: list[Change] = []
    found: set[str] = set()
    for node in ast.walk(_parse(path, text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in targets:
            found.add(node.name)
            refs = references[node.name]
            if refs:
                joined = ", ".join(refs[:5])
                raise RuntimeError(f"T4 refuses to remove {node.name}; references: {joined}")
            start, end = _function_span(node)
            edits.append((start, end, ""))
            changes.append(Change("T4", path, start, f"drop dead def {node.name}"))
    missing = targets - found
    # Missing targets mean this transform was already applied; a partial target
    # set is still safe because each present definition is independently gated.
    if found and missing:
        names = ", ".join(sorted(missing))
        print(f"T4 note: already absent in {rel}: {names}", file=sys.stderr)
    return _line_edits(text, edits), changes


def _masked_import_text(text: str, tree: ast.Module) -> str:
    lines = text.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for index in range(node.lineno - 1, node.end_lineno or node.lineno):
                lines[index] = "\n" if lines[index].endswith("\n") else ""
    return "".join(lines)


def _render_import(node: ast.Import | ast.ImportFrom, aliases: list[ast.alias]) -> str:
    rendered = ", ".join(
        alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
        for alias in aliases
    )
    if isinstance(node, ast.Import):
        return f"import {rendered}"
    dots = "." * node.level
    module = node.module or ""
    return f"from {dots}{module} import {rendered}"


def _bound_name(alias: ast.alias, node: ast.Import | ast.ImportFrom) -> str:
    if alias.asname:
        return alias.asname
    if isinstance(node, ast.Import):
        return alias.name.split(".", 1)[0]
    return alias.name


def _t2(path: Path, text: str) -> tuple[str, list[Change]]:
    tree = _parse(path, text)
    used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    masked = _masked_import_text(text, tree)
    edits: list[tuple[int, int, str]] = []
    changes: list[Change] = []
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom) and (node.module == "__future__" or any(a.name == "*" for a in node.names)):
            continue
        survivors: list[ast.alias] = []
        removed: list[str] = []
        for alias in node.names:
            bound = _bound_name(alias, node)
            raw_hit = re.search(rf"\b{re.escape(bound)}\b", masked) is not None
            if bound in used_names or raw_hit:
                survivors.append(alias)
            else:
                removed.append(bound)
        if not removed:
            continue
        replacement = _render_import(node, survivors) if survivors else ""
        edits.append((node.lineno, node.end_lineno or node.lineno, replacement))
        changes.append(
            Change("T2", path, node.lineno, f"drop unused import(s): {', '.join(removed)}")
        )
    return _line_edits(text, edits), changes


def _t3(path: Path, text: str) -> tuple[str, list[Change]]:
    try:
        rel = path.relative_to(BENCH_ENVS)
    except ValueError:
        return text, []
    if not rel.parts or rel.parts[0] != "office":
        return text, []
    lines = text.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []
    changes: list[Change] = []
    for index, line in enumerate(lines):
        if line.strip() != "# return self.info":
            continue
        start = index
        while start > 0 and lines[start - 1].strip().startswith("#"):
            start -= 1
        block = "".join(lines[start : index + 1])
        if '# self.info["info"]' not in block:
            continue
        if (
            "Record information about" not in lines[start]
            and '# self.info["info"]' not in lines[start]
        ):
            raise RuntimeError(f"T3 malformed info block at {path}:{index + 1}")
        edits.append((start + 1, index + 1, ""))
        changes.append(Change("T3", path, start + 1, "drop commented-out info block"))
    return _line_edits(text, edits), changes


TRANSFORMS: dict[str, Callable[..., tuple[str, list[Change]]]] = {
    "T1": _t1,
    "T2": _t2,
    "T3": _t3,
}


def _run_transform(name: str, *, apply: bool) -> list[Change]:
    changes: list[Change] = []
    references: dict[str, list[str]] | None = None
    if name == "T4":
        references = _repo_symbol_references(
            {symbol for names in T4_TARGETS.values() for symbol in names}
        )
    for path in _python_files():
        before = path.read_text(encoding="utf-8")
        if name == "T4":
            assert references is not None
            after, found = _t4(path, before, references=references)
        else:
            after, found = TRANSFORMS[name](path, before)
        changes.extend(found)
        if apply and after != before:
            path.write_text(after, encoding="utf-8")
    return changes


def _git_text(ref: str, path: Path) -> str | None:
    rel = path.relative_to(REPO_ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{ref}:{rel}"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout if result.returncode == 0 else None


def _normalize_allowed(path: Path, text: str) -> ast.Module:
    """Remove all AST-visible S5 cleanup candidates from a source snapshot."""
    normalized, _ = _t1(path, text)
    rel = path.relative_to(BENCH_ENVS).as_posix()
    targets = set(T4_TARGETS.get(rel, ()))
    if targets:
        tree = _parse(path, normalized)
        edits = [
            (*_function_span(node), "")
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in targets
        ]
        normalized = _line_edits(normalized, edits)
    normalized, _ = _t2(path, normalized)
    normalized, _ = _t3(path, normalized)
    if rel in T5_TARGETS:
        normalized = normalized.replace(T5_OLD_KEY, "include_collision")
    return _parse(path, normalized)


def _protected_functions(tree: ast.Module) -> dict[str, str]:
    found: dict[str, str] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.name in {"check_success", "play_once"}:
                key = ".".join([*self.scope, node.name])
                found[key] = ast.dump(node, include_attributes=False)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    Visitor().visit(tree)
    return found


def verify(ref: str) -> None:
    manifest_failures: list[str] = []
    protected_failures: list[str] = []
    compared = 0
    protected = 0
    for path in _python_files():
        before = _git_text(ref, path)
        if before is None:
            continue
        after = path.read_text(encoding="utf-8")
        rel = path.relative_to(BENCH_ENVS).as_posix()
        if rel not in VERIFY_BASE_EXCEPTIONS:
            before_norm = ast.dump(_normalize_allowed(path, before), include_attributes=False)
            after_norm = ast.dump(_normalize_allowed(path, after), include_attributes=False)
            compared += 1
            if before_norm != after_norm:
                manifest_failures.append(rel)
        if not path.name.startswith("_") and "/utils/" not in f"/{rel}":
            before_protected = _protected_functions(_parse(path, before))
            after_protected = _protected_functions(_parse(path, after))
            protected += len(before_protected)
            if before_protected != after_protected:
                protected_failures.append(rel)
    if manifest_failures or protected_failures:
        if manifest_failures:
            print("manifest-subtraction mismatch: " + ", ".join(manifest_failures), file=sys.stderr)
        if protected_failures:
            print("check_success/play_once mismatch: " + ", ".join(protected_failures), file=sys.stderr)
        raise SystemExit(1)
    print(f"manifest-subtraction equivalent: {compared} modules")
    print(f"check_success/play_once equivalent: {protected} functions")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report pending transforms")
    mode.add_argument("--apply", action="store_true", help="apply transforms")
    mode.add_argument("--verify", metavar="GIT_REF", help="verify equivalence to a git ref")
    parser.add_argument("--only", choices=("T1", "T2", "T3", "T4"))
    args = parser.parse_args(argv)

    if args.verify:
        if args.only:
            parser.error("--only cannot be combined with --verify")
        verify(args.verify)
        return

    selected = [args.only] if args.only else ["T1", "T4", "T2", "T3"]
    all_changes: list[Change] = []
    for name in selected:
        changes = _run_transform(name, apply=args.apply)
        all_changes.extend(changes)
        for change in changes:
            print(change.render())
    if all_changes:
        action = "applied" if args.apply else "pending"
        print(f"{len(all_changes)} changes {action}")
        if args.check:
            raise SystemExit(1)
    else:
        print("no changes")


if __name__ == "__main__":
    main()
