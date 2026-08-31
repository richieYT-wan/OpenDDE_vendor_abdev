"""
Wrapper to run the full pipeline from reading a config yaml file.
"""

import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from yaml import load as load_yaml, Loader
from typing import Dict
from datetime import datetime as dt
from hashlib import md5

logger = logging.getLogger(__name__)


def parse_args():
    p = ArgumentParser(description="""Full Protenix pipeline wrapper.""")

    p.add_argument('-c', '--config_file', metavar='CONFIG', required=True, type=Path,
                   help='Path to config file')
    p.add_argument('-i', '--input_file', metavar='INPUT_FILE', required=True, type=Path,
                   help='Path to input csv file containing vh sequences')
    p.add_argument('-o', '--output_dir', metavar='OUTPUT_DIR', required=True, type=Path,
                   help='Output directory')
    p.add_argument('-r', '--rows', metavar=('START', 'END'), type=int,
                   default=[None, None], nargs=2)
    p.add_argument('--sabdab-db', type=str,
                   default='~/search_database/sabdab_nano/sabdab_nano_db',
                   help='Path to MMseqs2 OAS nanobody database')
    # === NEW: template search inputs ===
    p.add_argument('--seqres-db', type=str, default=None,
                   help='Path to the sequence database FASTA for template search '
                        '(e.g. /root/search_database/pdb_seqres_2022_09_28.fasta). '
                        'If not provided, Protenix will attempt to download it.')
    p.add_argument('--hmmsearch-binary', type=str, default=None,
                   help='Path to hmmsearch binary (searches PATH if not provided).')
    p.add_argument('--hmmbuild-binary', type=str, default=None,
                   help='Path to hmmbuild binary (searches PATH if not provided).')
    p.add_argument('--msa-server-mode', type=str, default='protenix',
                   choices=['protenix', 'colabfold'],
                   help='MSA server mode (default: protenix)')
    # ===================================
    p.add_argument('--n_jobs', type=int, default=16, help='n_jobs for pre/post processing')
    p.add_argument('--prepare-only', action='store_true',
                   help='Only run the prepare step (target MSA + JSON), then exit.')
    p.add_argument('--skip_vhh_msa', action='store_true',
                   help='Skip the VHH MSA search in make_json_from_csv subprocess.')
    return p.parse_args()


def build_timestamped_hashed_outdir_and_json_stem(args):
    timestamp = dt.now().strftime("%y%m%d_%H%M%S[split]%f")
    hashed = md5(timestamp.encode()).hexdigest()[:8]
    timestamp = timestamp.split('[split]')[0]
    start, end = args.rows[0], args.rows[1]
    if start is not None and end is not None:
        child = f'{timestamp}_{Path(args.input_file).stem}_rows_{start:04}_{end:04}_outputs_{hashed}'
        json_stem = f'{Path(args.input_file).stem}_rows_{start:04}_{end:04}'
    else:
        child = f'{timestamp}_{Path(args.input_file).stem}_rows_full_dataset_outputs_{hashed}'
        json_stem = f'{Path(args.input_file).stem}_rows_full_dataset'
    return str(args.output_dir / child), json_stem


def build_prepare_cmd(config, args, script_dir, json_outfile):
    """
    The output_dir here is shared across different runs to allow easy access and re-use of target MSAs.
    """
    cmd = ['python', str(script_dir / 'make_json_from_csv.py'),
           '-i', str(args.input_file),
           '-o', str(args.output_dir),
           '--output_json', json_outfile,
           '--n_jobs', str(args.n_jobs),
           '--sabdab_db', str(args.sabdab_db)]
    if args.rows and args.rows[0] is not None and args.rows[1] is not None:
        cmd.extend(['-r', str(args.rows[0]), str(args.rows[1])])
    if getattr(args, 'skip_vhh_msa', False):
        cmd.append('--skip_vhh_msa')

    # === NEW: pass template-search paths through ===
    if args.seqres_db:
        cmd.extend(['--seqres_db', str(args.seqres_db)])
    if args.hmmsearch_binary:
        cmd.extend(['--hmmsearch_binary', str(args.hmmsearch_binary)])
    if args.hmmbuild_binary:
        cmd.extend(['--hmmbuild_binary', str(args.hmmbuild_binary)])
    if args.msa_server_mode:
        cmd.extend(['--msa_server_mode', str(args.msa_server_mode)])
    # ================================================

    target_sequences = config['prepare'].get('target_sequences')
    if target_sequences:
        cmd.extend(['-t', target_sequences])

    subset = config['prepare'].get('subset')
    if type(subset == list) and subset[0] is not None and subset[1] is not None:
        cmd.extend(['--subset', *[str(s) for s in subset]])
    seeds = config['prepare'].get("seeds", [0])
    cmd.extend(['-s', *[str(s) for s in seeds]])
    name = config['prepare'].get('name')
    if name:
        cmd.extend(['-n', name])
    logger.info('\n' + '*' * 100 + '\n' +
                f'Running input preparation with command: {" ".join(cmd)}' + '\n' +
                '*' * 100 + '\n')
    return cmd


def build_predict_cmd(config, args, script_dir, output_subdir, json_stem):
    cmd = ['python', str(script_dir.parent / 'runner' / 'batch_inference.py'),
           '-i', str(Path(args.output_dir) / f'{json_stem}.json'),
           '-o', output_subdir,
           '--use_seeds_in_json', 'True']
    model = config['predict'].get("model", 'protenix-v2')
    cmd.extend(['-n', model])
    logger.info('\n' + '*' * 100 + '\n' +
                f'Running batch inference with command: {" ".join(cmd)}' + '\n' +
                '*' * 100 + '\n')
    return cmd


def build_parse_cmd(args, script_dir, output_subdir):
    cmd = ['python', str(script_dir / 'parse_outputs.py'),
           '-i', output_subdir,
           '--n_jobs', str(args.n_jobs)]
    logger.info('\n' + '*' * 100 + '\n' +
                f'Running output parsing with command: {" ".join(cmd)}' + '\n' +
                '*' * 100 + '\n')
    return cmd


def main():
    args = parse_args()
    with open(args.config_file, 'r') as f:
        config: Dict = load_yaml(f, Loader)
    script_dir = Path(__file__).parent
    output_subdir, json_stem = build_timestamped_hashed_outdir_and_json_stem(args)
    subprocess.run(build_prepare_cmd(config, args, script_dir, json_stem), check=True)
    if args.prepare_only:
        logger.info('--prepare-only set; skipping predict + parse steps.')
        return
    subprocess.run(build_predict_cmd(config, args, script_dir, output_subdir, json_stem), check=True)
    subprocess.run(build_parse_cmd(args, script_dir, output_subdir), check=True)


if __name__ == '__main__':
    main()