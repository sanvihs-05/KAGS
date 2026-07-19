import sys
import os
import json

# Ensure backend folder is on path (so imports like `core` resolve)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BACKEND = os.path.join(ROOT, 'backend')
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Ensure project root is on sys.path so package imports resolve
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Dynamically load as package modules so relative imports work
encoder_path = os.path.join(ROOT, 'backend', 'agents', 'encoder_agent.py')
spec = importlib.util.spec_from_file_location(
    'backend.agents.encoder_agent', encoder_path
)
encoder_mod = importlib.util.module_from_spec(spec)
sys.modules['backend.agents.encoder_agent'] = encoder_mod
spec.loader.exec_module(encoder_mod)
EncoderAgent = encoder_mod.EncoderAgent

# Load VectorStoreManager
vs_path = os.path.join(ROOT, 'backend', 'database', 'vector_store.py')
spec2 = importlib.util.spec_from_file_location(
    'backend.database.vector_store', vs_path
)
vs_mod = importlib.util.module_from_spec(spec2)
sys.modules['backend.database.vector_store'] = vs_mod
spec2.loader.exec_module(vs_mod)
VectorStoreManager = vs_mod.VectorStoreManager


def main():
    vs = VectorStoreManager()
    encoder = EncoderAgent(vs)

    req = (
        'This house will be built on a 30x40 ft north-facing plot with simple modern styling '
        'and space for one car in front. Setbacks are 1.5 m at the front and 1 m on all other sides. '
        'The home should have a master bedroom (13-15 sqm) with an attached bathroom, cross-ventilation, '
        'and good east/southeast light. A child bedroom (10-12 sqm) should avoid west heat and stay quiet. '
        'The living and dining area (20-22 sqm) must be bright from the north side, seat 5-6 people, and keep bedroom doors out of direct view. '
        'A small study room (8-9 sqm) should work as a home office or guest room with good daylight. '
        'The kitchen (8-9 sqm) should connect to the dining area, stay hidden from the entrance, and have proper ventilation. '
        'There should be a naturally ventilated common bathroom (4-5 sqm) and a master bathroom of similar size. '
        'Adjacency should follow a simple flow: living -> dining -> kitchen, with a grouped master suite and a centrally accessible common toilet. '
        'The design should maximize daylight, enable cross-ventilation, avoid heat gain in the west, maintain privacy from neighbors, and ensure easy circulation. '
        'Structurally, it should use a simple grid, minimize cantilevers, group plumbing lines, and include a small utility/solar inverter space at the rear.'
    )

    print('\n=== Encoder debug run ===')
    print('LLM client:', 'available' if getattr(encoder, 'llm', None) else 'none')
    print('LLM CLI fallback:', getattr(encoder, 'llm_cli', False))

    # Call internal extraction to capture raw response and parsed program
    program = encoder._extract_spatial_program_with_llm(req)

    print('\n--- Parsed spatial program ---')
    print(json.dumps(program, indent=2))

    node = encoder._create_node_from_spatial_program(program, req)

    print('\n--- Created node layout rooms ---')
    if node.layout and node.layout.rooms:
        for r in node.layout.rooms.values():
            print(f"- {r.name}: {r.area} sqm (type={r.room_type})")
    else:
        print('No rooms created in node.layout')

    print('\n=== Debug complete ===')


if __name__ == '__main__':
    main()
