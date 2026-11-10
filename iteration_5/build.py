"""Compile the C split engines (blackjack/split_exact.c).

The split EV is too slow in pure Python (see the iteration 5 README for
the numbers), so it lives in C: one shared library that Python loads
through ctypes (see split_exact.py). The file holds two engines: the
sequential decomposition, which is the live one, and the joint recursion,
which is the exact verification tool. Both are the same maths as the
archived Python version, compiled.

Usage: python build.py

Requires a C compiler: gcc, or cc. The output (_split_exact.so, or
_split_exact.dylib on macOS) lands next to split_exact.c and is NOT
committed to the repo -- it is platform-specific, so each checkout builds
its own. Without it, the game still runs, but split odds raise a clear
error telling you to run this script instead of silently giving a wrong
number.
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "blackjack", "split_exact.c")

# The output is _split_exact.so (leading underscore), not split_exact.so:
# a split_exact.so next to split_exact.py would shadow the loader module
# at import time, and Python would try to load it as an extension module.
if sys.platform == "darwin":
    OUT = os.path.join(HERE, "blackjack", "_split_exact.dylib")
    FLAGS = ["-O2", "-dynamiclib"]
elif sys.platform.startswith("win"):
    print("build.py: Windows is not supported yet -- the C helper needs")
    print("a Unix-like compiler (gcc or cc). The game runs without it,")
    print("but split odds will not be available.")
    sys.exit(1)
else:
    OUT = os.path.join(HERE, "blackjack", "_split_exact.so")
    FLAGS = ["-O2", "-shared", "-fPIC"]

CC = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
if CC is None:
    print("build.py: no C compiler found (looked for gcc, then cc).")
    print("Install one and run this again -- the game runs without the")
    print("helper, but split odds will not be available.")
    sys.exit(1)

cmd = [CC, *FLAGS, "-o", OUT, SRC]
print(f"Compiling: {' '.join(cmd)}")
result = subprocess.run(cmd)
if result.returncode != 0:
    print(f"build.py: compilation failed (exit {result.returncode}).")
    sys.exit(result.returncode)
print(f"Built {OUT}")
print("Split odds are now live (near-exact decomposition, exact joint to check it): run the game or the tests.")
