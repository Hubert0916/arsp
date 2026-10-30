"""Reproduce the published scores from pinned public data and frozen weights."""
import argparse
import gzip
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-only", action="store_true", help="Require all pinned data in the Hugging Face cache")
    parser.add_argument("--prepared", action="store_true", help="Reuse prepared data (its checksum is still verified)")
    parser.add_argument("--legacy", action="store_true", help="Reproduce the historical fixed-subject proxy instead")
    parser.add_argument("--workers", type=int, default=4, help="Independent CPU workers; use 1 on a small machine")
    args = parser.parse_args()
    def run(script, *arguments):
        subprocess.run([sys.executable, str(ROOT / "tools" / script), *arguments], cwd=ROOT, check=True)
    run("verify_artifacts.py")
    if not args.legacy:
        run('test_draw_protocol.py')
        if not args.prepared:
            run('prepare_draws.py', *(['--cache-only'] if args.cache_only else []))
        output=ROOT/'results/check-formative'
        run('evaluate_draws.py','--workers',str(args.workers),'--output',str(output))
        run('score_draws.py',str(output))
        compare_formative(output,ROOT/'results/formative')
        print('Formative reproduction complete: results/check-formative/summary.json')
        return
    run("test_proxy.py")
    if not args.prepared:
        run("prepare_proxy.py", *(["--cache-only"] if args.cache_only else []))
    run("build_submission.py", "asrp")
    run("evaluate_proxy.py", "--workers", str(args.workers))
    run("score_proxy.py", "results/reproduced", "--compare", "results/proxy")
    print("Reproduction complete: results/reproduced/summary.json")


def compare_formative(actual,reference):
    """Stream all rows, including repeated responses, without a large join."""
    from evaluate_draws import METHODS
    max_error=0.;counts={}
    for kind in ('predictions','acquisition'):
        n=0
        with gzip.open(actual/f'{kind}.jsonl.gz','rt') as a,gzip.open(reference/f'{kind}.jsonl.gz','rt') as b:
            for left,right in itertools.zip_longest(a,b):
                if left is None or right is None:raise ValueError('Record counts differ')
                x,y=json.loads(left),json.loads(right)
                if kind=='predictions':
                    for v in METHODS:
                        p,q=x.pop(v),y.pop(v)
                        if len(p)!=6 or len(q)!=6:raise ValueError('Missing budget')
                        max_error=max(max_error,max(abs(u-w) for u,w in zip(p,q)))
                if x!=y:raise ValueError(f'{kind} identity, response, or decision differs at row {n}')
                n+=1
        counts[kind]=n
    if max_error>1e-7:raise ValueError(f'Predictions differ by {max_error}')
    print(f'Reference agreement: {counts}; maximum prediction error {max_error}')


if __name__ == "__main__":
    main()
