"""Build an isolated experimental runtime without Rasch code or artifacts."""
import argparse
import ast
import json
from pathlib import Path
import shutil

from prepare_proxy import ROOT, sha, write_json


def selected_definitions(path, names, header):
    source = path.read_text()
    nodes = {node.name: ast.get_source_segment(source, node)
             for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    return header + '\n\n' + '\n\n\n'.join(nodes[name] for name in names) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/asrp_next_bundle')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a new output directory so no existing submission can be overwritten.')
    lock = json.loads((ROOT / 'experiments/asrp_next/locked_candidate.json').read_text())
    assert lock['implementation_sha256'] == sha(ROOT / 'experiments/asrp_next/prior_kernel.py')
    frozen = json.loads((ROOT / 'artifacts/manifest.sha256.json').read_text())
    for name, digest in frozen.items():
        assert sha(ROOT / name) == digest
    args.output.mkdir(parents=True)
    for name in ['kernel_probe.py', 'base_features.py', 'evidence_features.py',
                 'window_features.py', 'project_core.py', 'project_policy.py',
                 'adaptive_gain_policy.py', 'balanced_probe_policy.py', 'linear_gain_policy.py',
                 'labeling.py', 'requirements.txt']:
        shutil.copyfile(ROOT / 'src/asrp' / name, args.output / name)
    for name in ['model.py', 'prior_kernel.py']:
        shutil.copyfile(ROOT / 'experiments/asrp_next' / name, args.output / name)
    (args.output / 'prior_features.py').write_text(selected_definitions(
        ROOT / 'src/asrp/model.py',
        ['_feature_tokens', '_explicit_topics', '_topic_cluster', '_budget_index',
         '_calibration_adjust', '_feature_prior'],
        '"""Frozen metadata priors and topic corrections, without Rasch dependencies."""\n'
        'import re\nimport numpy as np\nBUDGETS = (0, 1, 3, 7, 15, 31)'))
    (args.output / 'text_encoder.py').write_text(selected_definitions(
        ROOT / 'src/asrp/ml_features.py', ['normalize_text', 'MiniLMEncoder'],
        '"""Frozen MiniLM embeddings used for topic assignment."""\nimport numpy as np'))
    for name in ['encoder', 'probe_encoder']:
        shutil.copytree(ROOT / 'artifacts/asrp' / name, args.output / name)
    for name in ['binary_subject_prior.json', 'feature_prior.json', 'topic_calibration.json',
                 'probe_config.json', 'prediction_config.json']:
        shutil.copyfile(ROOT / 'artifacts/asrp' / name, args.output / name)
    write_json(args.output / 'runtime_config.json', lock['config'])
    write_json(args.output / 'config.json', {'method': 'ASRPPriorKernelCandidate', 'acquisition': 'adaptive_4_warm'})
    write_json(args.output / 'candidate_provenance.json', {
        'locked_candidate_sha256': sha(ROOT / 'experiments/asrp_next/locked_candidate.json'),
        'original_inference_manifest_sha256': sha(ROOT / 'artifacts/manifest.sha256.json'),
        'status': 'Experimental; not the released submission.',
    })
    print(f'Built experimental runtime: {args.output}')


if __name__ == '__main__':
    main()
