import pickletools
from pathlib import Path
import sys

PKL = Path('enhanced_multimodal_rag_store/stores/model.pkl')

if not PKL.exists():
    print(f"Pickle file not found: {PKL}")
    sys.exit(1)

with open(PKL, 'rb') as f:
    data = f.read()

print(f"Inspecting pickle: {PKL} (size={len(data)} bytes)\n")

globals_found = set()
for op, arg, pos in pickletools.genops(data):
    if op.name == 'GLOBAL':
        globals_found.add(arg)

if not globals_found:
    print("No GLOBAL refs found in pickle (or not a pickle with GLOBAL ops).\n")
    # Fallback: show a short pickletools disassembly to inspect other ops
    import io
    buf = io.StringIO()
    try:
        pickletools.dis(data, out=buf)
        dis = buf.getvalue()
        print("Disassembly (first 4000 chars):\n")
        print(dis[:4000])
    except Exception as e:
        print(f"Could not disassemble pickle: {e}")
else:
    print("GLOBAL references in pickle (module name and attribute):\n")
    for g in sorted(globals_found):
        print(f" - {g}")

print('\nDone.')
