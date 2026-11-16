"""Statically list every global (module.name) a PyTorch checkpoint's pickle references.

Nothing is unpickled or executed: we only walk the opcode stream with pickletools.
A pickle can only call code it imports through GLOBAL / STACK_GLOBAL, so if every
reference is a torch / timm / ultralytics class or a plain container, loading it
is safe from arbitrary-code-execution tricks.

Usage: python -I tools/inspect_pickle.py <checkpoint.pt> [...]
"""
import pickletools
import sys
import zipfile
from collections import Counter


STRING_PUSHES = ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8")
MEMO_MARKERS = ("MEMOIZE", "PUT", "BINPUT", "LONG_BINPUT")  # store a copy; they don't change the stack


def globals_in_pickle(data: bytes) -> Counter:
    """Every GLOBAL, and every STACK_GLOBAL whose module and name are the two string literals pushed right
    before it (protocol 4+). A STACK_GLOBAL fed any other way (e.g. strings fetched back from the memo) is
    reported as unresolved rather than guessed, so it can never hide behind a wrong name."""
    found: Counter = Counter()
    previous: list[tuple[str, object]] = []  # the last two stack-changing opcodes (name, arg)
    for opcode, arg, _pos in pickletools.genops(data):
        name = opcode.name
        if name == "GLOBAL":
            module, attr = arg.split(" ", 1) if " " in arg else arg.split("\n", 1)
            found[f"{module}.{attr}"] += 1
        elif name == "STACK_GLOBAL":
            if len(previous) == 2 and all(op in STRING_PUSHES for op, _ in previous):
                found[f"{previous[0][1]}.{previous[1][1]}"] += 1
            else:
                found["<STACK_GLOBAL with unresolved module/name: inspect manually>"] += 1
        if name not in MEMO_MARKERS:
            previous = (previous + [(name, arg)])[-2:]
    return found


def main() -> None:
    for path in sys.argv[1:]:
        print(f"== {path}")
        with zipfile.ZipFile(path) as zf:
            pkls = [n for n in zf.namelist() if n.endswith(".pkl")]
            for pkl in pkls:
                found = globals_in_pickle(zf.read(pkl))
                print(f"  {pkl}: {len(found)} distinct globals")
                for ref, count in sorted(found.items()):
                    print(f"    {count:5d}  {ref}")


if __name__ == "__main__":
    main()
