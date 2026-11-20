"""Apply PPE rules to existing video results without rerunning inference."""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from safety_rules.ppe_rules import evaluate_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New directory under runs/PPE/step5_video')
    args = parser.parse_args()
    root = (PROJECT / 'runs/PPE/step5_video').resolve()
    out = args.output.resolve()
    if out == root or root not in out.parents:
        parser.error('Output must be a new directory under runs/PPE/step5_video')
    if not args.results.is_file():
        parser.error(f'Results file does not exist: {args.results}')
    report = json.loads(args.results.read_text(encoding='utf-8'))
    result = evaluate_report(report)
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f'Output already exists: {out}')
    result['source_results'] = str(args.results.resolve())
    (out / 'incidents.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Status: {result['status']}; {len(result['incidents'])} unverified incidents: {out / 'incidents.json'}")
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
