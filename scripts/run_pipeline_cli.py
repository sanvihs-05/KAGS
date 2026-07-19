"""Run the pipeline from the terminal and print summary; intended for testing DB storage.

Usage examples:
  py scripts\run_pipeline_cli.py --requirements "Compact 2BR with workspace" --project_name test1
  py scripts\run_pipeline_cli.py -r "Studio with kitchen" -p demo --top_k 5

The script reads DB connection info from `backend/config/config.yaml` and shows SQL commands
for inspecting stored rows (tables: `projects`, `fbsl_nodes`, `evaluations`).
"""
import argparse
import asyncio
import yaml
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))
import shutil


def load_config():
    # Try to locate backend config
    cfg_paths = [
        ROOT / 'backend' / 'config' / 'config.yaml',
        ROOT / 'backend' / 'config.yaml',
        ROOT / 'backend' / 'config' / 'config.yml',
    ]
    for p in cfg_paths:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    return {}


async def run_orchestrator(req):
    from backend.pipeline.orchestrator import PipelineOrchestrator
    orchestrator = PipelineOrchestrator()
    return await orchestrator.process_design_request(req)


def print_db_instructions(cfg):
    db = cfg.get('database') or cfg.get('db') or {}
    host = db.get('host', 'localhost')
    port = db.get('port', 5432)
    database = db.get('database', 'fbsl_kags')
    user = db.get('user', 'fbsl_user')

    print('\nDatabase connection (from config):')
    print(f"  host: {host}\n  port: {port}\n  database: {database}\n  user: {user}\n")

    print('Check stored rows (psql examples):')
    print(f"  psql -h {host} -p {port} -U {user} -d {database}")
    print('Once in psql, run:')
    print("  SELECT project_id, project_name, created_at FROM projects ORDER BY created_at DESC LIMIT 10;")
    print("  SELECT node_id, project_id, composite_score, created_at FROM fbsl_nodes ORDER BY created_at DESC LIMIT 20;")
    print("  SELECT evaluation_id, node_id, composite_score, evaluated_at FROM evaluations ORDER BY evaluated_at DESC LIMIT 20;\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project_name', '-p', type=str, default=None)
    p.add_argument('--requirements', '-r', type=str, required=True)
    p.add_argument('--top_k', type=int, default=3)
    p.add_argument('--max_alternatives', type=int, default=5)
    p.add_argument('--got_delta', type=float, default=None)
    p.add_argument('--got_patience', type=int, default=None)
    p.add_argument('--got_max_nodes', type=int, default=None)
    p.add_argument('--got_selection_metric', type=str, default=None)
    p.add_argument('--context_file', type=str, default=None, help='JSON file with context (room sizes, constraints)')
    p.add_argument('--force_cpu', action='store_true', help='Force CPU mode for local LLMs (sets CUDA_VISIBLE_DEVICES="")')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    # Build request and avoid sending explicit None for got_* params
    req = {
        'project_name': args.project_name,
        'requirements': args.requirements,
        'context': {},
        'max_alternatives': args.max_alternatives,
        'use_got': True,
        'enable_convergence_loop': True,
    }
    if args.got_delta is not None:
        req['got_delta'] = args.got_delta
    if args.got_patience is not None:
        req['got_patience'] = args.got_patience
    if args.got_max_nodes is not None:
        req['got_max_nodes'] = args.got_max_nodes
    if args.got_selection_metric is not None:
        req['got_selection_metric'] = args.got_selection_metric
    if args.context_file:
        try:
            with open(args.context_file, 'r', encoding='utf-8') as f:
                req['context'] = json.load(f)
        except Exception as e:
            print(f"Warning: failed to read context file {args.context_file}: {e}. Continuing with empty context.")

    print('Running pipeline (this runs the backend orchestrator).')
    print('Requirements:', args.requirements)
    if args.force_cpu:
        print('Force-CPU requested: will set CUDA_VISIBLE_DEVICES="" for this run.')

    # Optionally set CPU-mode env var before first run if requested
    if args.force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

    # Propagate LLM config into environment so worker processes use same model
    llm_cfg = cfg.get('llm', {})
    if llm_cfg:
        model = llm_cfg.get('model')
        base_url = llm_cfg.get('base_url')
        if model and not os.environ.get('OLLAMA_MODEL'):
            os.environ['OLLAMA_MODEL'] = str(model)
            print(f"Set OLLAMA_MODEL={model} from config")
        if base_url and not os.environ.get('OLLAMA_BASE_URL'):
            os.environ['OLLAMA_BASE_URL'] = str(base_url)
            print(f"Set OLLAMA_BASE_URL={base_url} from config")
    # If Ollama CLI is present on PATH and the user hasn't explicitly disabled CLI usage,
    # prefer using the local Ollama CLI for deterministic local models.
    if not os.environ.get('OLLAMA_USE_CLI') and shutil.which('ollama'):
        os.environ['OLLAMA_USE_CLI'] = '1'
        print("Detected 'ollama' on PATH — setting OLLAMA_USE_CLI=1 to prefer local CLI models.")

    try:
        result = asyncio.run(run_orchestrator(req))
    except Exception as e:
        # If LLM/CUDA OOM related, retry once forcing CPU
        err_str = str(e).lower()
        import traceback
        traceback.print_exc()
        if ('cuda' in err_str or 'out of memory' in err_str or 'cudaalloc' in err_str or 'llama' in err_str) and not args.force_cpu:
            print('\nDetected LLM/CUDA error; retrying once with CPU forced (CUDA_VISIBLE_DEVICES="").')
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
            try:
                result = asyncio.run(run_orchestrator(req))
            except Exception:
                print('Retry with CPU also failed. Aborting.')
                traceback.print_exc()
                return 1
        else:
            print('Orchestrator raised an exception:')
            print(e)
            return 1

    # Print a concise summary
    success = result.get('success', False)
    if not success:
        print('Pipeline reported failure:')
        print(json.dumps(result, indent=2))
        return 1

    designs = result.get('designs', [])
    project_id = result.get('project_id') or 'project'

    print('\nPipeline finished — summary:')
    print(f"  Project ID: {project_id}")
    print(f"  Designs returned: {len(designs)}\n")

    for i, d in enumerate(designs[:args.top_k], start=1):
        node_id = d.get('node_id')
        scores = d.get('scores', {})
        print(f"{i}. node_id={node_id}  composite={scores.get('composite', 0):.3f}  functional={scores.get('functional_adequacy', 0):.3f}")
        # If the orchestrator wrote to DB, mention it
        print(f"   Stored: projects/{project_id} and fbsl_nodes/{node_id}")

    # Database inspection guidance
    print_db_instructions(cfg)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
