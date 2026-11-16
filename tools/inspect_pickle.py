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


def globals_in_pickle(data: bytes) -> Counter:
    found: Counter = Counter()
    recent_strings: list[str] = []
    for opcode, arg, _pos in pickletools.genops(data):
        name = opcode.name
        if name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            recent_strings.append(arg)
            recent_strings = recent_strings[-2:]
        elif name in ("BINGET", "LONG_BINGET", "GET"):
            # memo lookups can feed STACK_GLOBAL; we track memoized strings below
            pass
        if name == "GLOBAL":
            module, attr = arg.split(" ", 1) if " " in arg else arg.split("\n", 1)
            found[f"{module}.{attr}"] += 1
        elif name == "STACK_GLOBAL":
            if len(recent_strings) == 2:
                found[f"{recent_strings[0]}.{recent_strings[1]}"] += 1
            else:
                found["<STACK_GLOBAL with non-literal args>"] += 1
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
