"""
setup_env.py
------------
Scans the entire project for Python imports, detects third-party packages,
and automatically installs any that are missing.

Usage:
    python setup_env.py
    python setup_env.py --dry-run     # show what would be installed, don't install
    python setup_env.py --dir ./src   # scan a different directory
"""

import ast
import importlib.util
import subprocess
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Standard library module names (Python 3.x built-ins + common stdlib names).
# This list is auto-expanded at runtime using sys.stdlib_module_names (3.10+)
# and augmented with a manual fallback for older versions.
# ---------------------------------------------------------------------------

_STDLIB_FALLBACK = {
    "__future__", "_thread", "abc", "aifc", "argparse", "array", "ast",
    "asynchat", "asyncio", "asyncore", "atexit", "audioop", "base64",
    "bdb", "binascii", "binhex", "bisect", "builtins", "bz2", "calendar",
    "cgi", "cgitb", "chunk", "cmath", "cmd", "code", "codecs", "codeop",
    "colorsys", "compileall", "concurrent", "configparser", "contextlib",
    "contextvars", "copy", "copyreg", "cProfile", "csv", "ctypes",
    "curses", "dataclasses", "datetime", "dbm", "decimal", "difflib",
    "dis", "distutils", "doctest", "email", "encodings", "enum",
    "errno", "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch",
    "fractions", "ftplib", "functools", "gc", "getopt", "getpass",
    "gettext", "glob", "grp", "gzip", "hashlib", "heapq", "hmac",
    "html", "http", "idlelib", "imaplib", "imghdr", "imp",
    "importlib", "inspect", "io", "ipaddress", "itertools", "json",
    "keyword", "lib2to3", "linecache", "locale", "logging", "lzma",
    "mailbox", "mailcap", "marshal", "math", "mimetypes", "mmap",
    "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
    "numbers", "operator", "optparse", "os", "ossaudiodev", "pathlib",
    "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint", "profile",
    "pstats", "pty", "pwd", "py_compile", "pyclbr", "pydoc", "queue",
    "quopri", "random", "re", "readline", "reprlib", "resource",
    "rlcompleter", "runpy", "sched", "secrets", "select", "selectors",
    "shelve", "shlex", "shutil", "signal", "site", "smtpd", "smtplib",
    "sndhdr", "socket", "socketserver", "spwd", "sqlite3", "sre_compile",
    "sre_constants", "sre_parse", "ssl", "stat", "statistics", "string",
    "stringprep", "struct", "subprocess", "sunau", "symtable", "sys",
    "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile",
    "termios", "test", "textwrap", "threading", "time", "timeit",
    "tkinter", "token", "tokenize", "tomllib", "trace", "traceback",
    "tracemalloc", "tty", "turtle", "turtledemo", "types", "typing",
    "unicodedata", "unittest", "urllib", "uu", "uuid", "venv",
    "warnings", "wave", "weakref", "webbrowser", "winreg", "winsound",
    "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile",
    "zipimport", "zlib", "zoneinfo",
}

# Map: import name  →  pip install name  (when they differ)
_IMPORT_TO_PIP = {
    "sklearn":    "scikit-learn",
    "cv2":        "opencv-python",
    "PIL":        "Pillow",
    "yaml":       "PyYAML",
    "bs4":        "beautifulsoup4",
    "dotenv":     "python-dotenv",
    "dateutil":   "python-dateutil",
    "attr":       "attrs",
    "google.protobuf": "protobuf",
    "grpc":       "grpcio",
    "wx":         "wxPython",
    "gi":         "PyGObject",
    "usb":        "pyusb",
    "serial":     "pyserial",
    "Crypto":     "pycryptodome",
    "OpenSSL":    "pyOpenSSL",
    "jose":       "python-jose",
    "jwt":        "PyJWT",
    "aiohttp":    "aiohttp",
    "fastapi":    "fastapi",
    "uvicorn":    "uvicorn",
    "pydantic":   "pydantic",
    "starlette":  "starlette",
    "sqlalchemy": "SQLAlchemy",
    "alembic":    "alembic",
    "celery":     "celery",
    "redis":      "redis",
    "pymongo":    "pymongo",
    "psycopg2":   "psycopg2-binary",
    "boto3":      "boto3",
    "botocore":   "botocore",
    "requests":   "requests",
    "httpx":      "httpx",
    "click":      "click",
    "rich":       "rich",
    "loguru":     "loguru",
    "tqdm":       "tqdm",
    "flwr":       "flwr",
}


def get_stdlib_modules() -> set:
    """Return the full set of standard library module names."""
    if hasattr(sys, "stdlib_module_names"):          # Python 3.10+
        return set(sys.stdlib_module_names)
    return _STDLIB_FALLBACK


def extract_imports_from_file(filepath: Path) -> set:
    """
    Parse a .py file with AST and return all top-level module names imported.
    Falls back to a line-based scan if the file cannot be parsed.
    """
    imports = set()
    source = filepath.read_text(encoding="utf-8", errors="ignore")

    try:
        tree = ast.parse(source, filename=str(filepath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # 'import torch.nn' → top-level is 'torch'
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:  # skip relative imports
                    imports.add(node.module.split(".")[0])
    except SyntaxError:
        # Fallback: naive line scan (handles files with syntax errors)
        for line in source.splitlines():
            line = line.strip()
            if line.startswith("import "):
                mod = line[7:].split()[0].split(".")[0].rstrip(",")
                imports.add(mod)
            elif line.startswith("from ") and " import " in line:
                mod = line[5:line.index(" import ")].strip().split(".")[0]
                if not mod.startswith("."):
                    imports.add(mod)

    return imports


def collect_local_module_names(root: Path, skip_dirs: set) -> set:
    """
    Return names of all local modules/packages in the project so they can be
    excluded from the third-party list.
      - every .py file stem  (e.g. model.py  → 'model')
      - every sub-directory that is a package (contains __init__.py)
    """
    local: set = set()
    for f in root.rglob("*.py"):
        if any(part in skip_dirs for part in f.parts):
            continue
        local.add(f.stem)                      # e.g. 'utils', 'model'
    for d in root.rglob("*"):
        if d.is_dir() and (d / "__init__.py").exists():
            if not any(part in skip_dirs for part in d.parts):
                local.add(d.name)              # e.g. 'client', 'server'
    return local


def scan_project(root: Path) -> tuple:
    """
    Recursively scan all .py files and collect imported module names.
    Returns (all_imports, local_module_names).
    """
    all_imports: set = set()
    py_files = list(root.rglob("*.py"))

    # Skip virtual-environment directories and common noise folders
    skip_dirs = {"venv", ".venv", "env", ".env", "__pycache__", ".git",
                 "node_modules", "dist", "build", ".tox", "site-packages"}

    filtered = [
        f for f in py_files
        if not any(part in skip_dirs for part in f.parts)
    ]

    local_modules = collect_local_module_names(root, skip_dirs)

    print(f"  Found {len(filtered)} Python file(s) to scan.\n")
    for f in filtered:
        all_imports |= extract_imports_from_file(f)

    return all_imports, local_modules


def resolve_pip_name(import_name: str) -> str:
    """Convert an import name to its pip install name."""
    return _IMPORT_TO_PIP.get(import_name, import_name)


def is_installed(import_name: str) -> bool:
    """Return True if the package can be imported (i.e. is installed)."""
    return importlib.util.find_spec(import_name) is not None


def install_package(pip_name: str) -> bool:
    """Run pip install for a single package. Returns True on success."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [ERROR] Failed to install '{pip_name}':")
        print(f"          {result.stderr.strip()}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect and install missing Python dependencies."
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Root directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without actually installing",
    )
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    stdlib = get_stdlib_modules()

    print("=" * 55)
    print("  Dependency Scanner & Auto-Installer")
    print("=" * 55)
    print(f"\nScanning project for dependencies...")
    print(f"  Directory : {root}\n")

    # ── 1. Collect all imports ──────────────────────────────────────────────
    raw_imports, local_modules = scan_project(root)

    # ── 2. Filter out stdlib, built-ins, and local project modules ─────────
    third_party = sorted(
        name for name in raw_imports
        if name
        and name not in stdlib
        and name not in local_modules
        and not name.startswith("_")
    )

    if not third_party:
        print("No third-party imports detected. Nothing to do.")
        return

    print("Detected third-party libraries:")
    for pkg in third_party:
        print(f"  {pkg}")

    # ── 3. Classify: installed vs missing ──────────────────────────────────
    already_installed = []
    to_install = []          # list of (import_name, pip_name) tuples

    for name in third_party:
        if is_installed(name):
            already_installed.append(name)
        else:
            pip_name = resolve_pip_name(name)
            to_install.append((name, pip_name))

    print(f"\nAlready installed ({len(already_installed)}):")
    if already_installed:
        for pkg in already_installed:
            print(f"  [OK] {pkg}")
    else:
        print("  (none)")

    # ── 4. Install missing packages ────────────────────────────────────────
    print(f"\nMissing packages to install ({len(to_install)}):")
    if not to_install:
        print("  (none — everything is already installed!)")
        print("\nInstallation complete.")
        return

    for import_name, pip_name in to_install:
        label = f"{import_name}" if import_name == pip_name else f"{import_name}  (pip: {pip_name})"
        print(f"  {label}")

    if args.dry_run:
        print("\n[Dry-run mode] No packages were installed.")
        return

    print("\nInstalling missing packages...")
    print("-" * 40)

    success, failed = [], []
    for import_name, pip_name in to_install:
        print(f"  Installing '{pip_name}' ...", end=" ", flush=True)
        if install_package(pip_name):
            print("done")
            success.append(pip_name)
        else:
            print("FAILED")
            failed.append(pip_name)

    # ── 5. Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Installation Summary")
    print("=" * 55)
    print(f"  Already installed : {len(already_installed)}")
    print(f"  Newly installed   : {len(success)}")
    print(f"  Failed            : {len(failed)}")

    if failed:
        print(f"\n  The following packages could NOT be installed:")
        for pkg in failed:
            print(f"    - {pkg}")
        print("\n  Try installing them manually:")
        print(f"    pip install {' '.join(failed)}")
        sys.exit(1)
    else:
        print("\nInstallation complete.")


if __name__ == "__main__":
    main()
