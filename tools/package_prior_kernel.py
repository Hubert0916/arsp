"""Package the checked experimental runtime without replacing released ASRP."""
import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from prepare_proxy import ROOT, sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT / 'data/asrp_next_bundle')
    args = p.parse_args()
    folder = ROOT / 'results/asrp_next'
    validation = json.loads((folder / 'validation/summary.json').read_text())
    checked = json.loads((folder / 'runtime_check.json').read_text())
    lock_path = ROOT / 'experiments/asrp_next/locked_candidate.json'
    assert checked['locked_candidate_sha256'] == validation['locked_candidate_sha256'] == sha(lock_path)
    for name, digest in checked['native_files'].items():
        assert sha(args.bundle / name) == digest, name
    assert not (args.bundle / 'difficulty_model.npz').exists()
    assert checked['max_probability_error'] < 1e-10
    assert checked['full_acquisition_trajectories_checked'] == 6
    provenance = {
        'method': 'Experimental no-Rasch ASRP refinement',
        'locked_candidate_sha256': sha(lock_path),
        'all_predefined_validation_gates_passed': validation['passed'],
        'status': 'Experimental candidate; does not replace the released with-Rasch ASRP submission.',
        'scope': validation['scope'],
    }
    output = ROOT / 'dist/asrp_no_rasch_candidate.zip'
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name in sorted(checked['native_files']):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (args.bundle / name).read_bytes())
        info = ZipInfo('candidate_validation.json', date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, json.dumps(provenance, indent=2) + '\n')
    write_json(folder / 'submission.json', dict(provenance, file=output.relative_to(ROOT).as_posix(),
                                               sha256=sha(output), bytes=output.stat().st_size))
    print(f'Built experimental candidate: {output} ({output.stat().st_size:,} bytes)')
    print(f'All stability gates passed: {validation["passed"]}')


if __name__ == '__main__':
    main()
