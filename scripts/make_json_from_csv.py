"""
Generates OpenDDE-compatible JSONs from a CSV of VH sequences against a fixed target.
Target MSA is computed once. VHH MSAs are built from a local SAbDAb nanobody database.
Local database is built from ~/scripts/setup_vhh_db.sh

Example:
    python3 scripts/make_json_from_csv.py \
        -i data/01_raw/<experiment_name>/<input_file>.csv \
        -o data/02_intermediate/<experiment_name> \
        --target_sequences <target_domain_0> <target_domain_1> [...] <target_domain_n> \
        --oas-db <database_path>/sabdab_nano_db \
        -s 0 13 30 42 1213 -r 0 100
"""

import hashlib
import logging
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_file', type=Path, required=True)
    parser.add_argument('-o', '--output_dir', type=Path, required=True)
    parser.add_argument('--output_json', type=Path, default=None,
                        help='Custom JSON basename, without full path or extension.')
    parser.add_argument('-t', '--target_sequences', type=str, nargs='+', default=None)
    parser.add_argument('-s', '--seeds', type=int, nargs='+')
    parser.add_argument('-n', '--name', type=str, required=False, default=None,
                        help='Custom sample name')
    parser.add_argument('-r', '--rows', type=int, default=[None, None], nargs=2,
                        metavar=('START', 'END'))
    parser.add_argument('--subset', type=str, default=(None, None), nargs=2,
                        metavar=('COLUMN', 'VALUE'),
                        help='A tuple pair of (column,value) for querying a subset of the dataframe.')
    parser.add_argument('--sabdab_db', type=str,
                        default='~/search_database/sabdab_nano/sabdab_nano_db',
                        help='Path to MMseqs2 OAS nanobody database')
    # === NEW: pass template DB path explicitly ===
    parser.add_argument('--seqres_db', type=str, default=None,
                        help='Path to the sequence database FASTA for template search '
                             '(e.g. /root/search_database/pdb_seqres_2022_09_28.fasta). '
                             'If not provided, OpenDDE will attempt to download it.')
    parser.add_argument('--hmmsearch_binary', type=str, default=None,
                        help='Path to hmmsearch binary (searches PATH if not provided).')
    parser.add_argument('--hmmbuild_binary', type=str, default=None,
                        help='Path to hmmbuild binary (searches PATH if not provided).')
    parser.add_argument('--msa_server_mode', type=str, default='opendde',
                        choices=['opendde', 'colabfold'],
                        help='MSA server mode (default: opendde)')
    # ============================================
    parser.add_argument('--skip_vhh_msa', action='store_true',
                        help='Skip VHH MSA building (fallback to dummy)')
    parser.add_argument('--force_refresh_target', action='store_true',
                        help='Ignore cached target MSA and re-run search')
    parser.add_argument('--n_jobs', type=int, default=16)
    return parser.parse_args()


def _target_fingerprint(target: Union[List, str]) -> str:
    """Deterministic hash of the target sequence(s) for cache invalidation."""
    if isinstance(target, list):
        payload = "|".join(target)
    else:
        payload = target
    return hashlib.md5(payload.encode()).hexdigest()


def process_target_msa_template(
        target: Union[List, str],
        output_dir: Path,
        seqres_db: Optional[str] = None,
        hmmsearch_binary: Optional[str] = None,
        hmmbuild_binary: Optional[str] = None) -> Dict:
    """Run `opendde mt` on the target to get its MSA + templates. Runs once."""
    protein_chains = []
    if isinstance(target, list):
        for i, t in enumerate(target):
            protein_chains.append(make_protein_chain(t, f'T{i}', 1))
    else:
        protein_chains.append(make_protein_chain(target, 'T', 1))

    entry = [make_entry(protein_chains, 'target')]
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
        json.dump(entry, tmp, indent=2)
        tmp_path = Path(tmp.name)

    target_intermediate_dir = Path(output_dir) / 'target'

    try:
        cmd = [
            "opendde", "mt",
            "-i", str(tmp_path),
            "-o", str(output_dir),
        ]
        # === NEW: pass explicit template search paths if provided ===
        if seqres_db is not None:
            cmd.extend(["--seqres_database_path", str(seqres_db)])
        if hmmsearch_binary is not None:
            cmd.extend(["--hmmsearch_binary_path", str(hmmsearch_binary)])
        if hmmbuild_binary is not None:
            cmd.extend(["--hmmbuild_binary_path", str(hmmbuild_binary)])
        # ===========================================================

        logger.info(f'Running: {" ".join(cmd)}')
        subprocess.run(cmd, text=True, check=True)
    except subprocess.CalledProcessError:
        if target_intermediate_dir.exists():
            logger.warning(
                f"opendde mt failed; removing partial output at {target_intermediate_dir}"
            )
            shutil.rmtree(target_intermediate_dir)
        raise

    target_file = tmp_path.parent / (tmp_path.stem + '-final-updated.json')
    with open(target_file) as f:
        return json.load(f)


def build_vhh_msa(
        sequence: str,
        name: str,
        output_dir: Path,
        sabdab_db: str = "~/search_database/sabdab_nano/sabdab_nano_db",
        sensitivity: float = 7.5,
        max_seqs: int = 5000,
) -> Tuple[Path, Path]:
    """Run MMseqs2 search of a VHH against OAS, produce paired+unpaired A3M files."""
    output_dir = Path(output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        query_fasta = tmpdir / "query.fasta"
        query_fasta.write_text(f">{name}\n{sequence}\n")

        query_db = tmpdir / "queryDB"
        subprocess.run(["mmseqs", "createdb", str(query_fasta), str(query_db)], check=True)

        result_db = tmpdir / "resultDB"
        subprocess.run([
            "mmseqs", "search",
            str(query_db), sabdab_db, str(result_db), str(tmpdir / "search_tmp"),
            "-s", str(sensitivity),
            "--max-seqs", str(max_seqs),
            "-a",
        ], check=True)

        a3m_out = output_dir / "msa.a3m"
        subprocess.run([
            "mmseqs", "result2msa",
            str(query_db), sabdab_db, str(result_db), str(a3m_out),
            "--msa-format-mode", "5",
        ], check=True)

    unpaired_path = output_dir / "non_pairing.a3m"
    paired_path = output_dir / "pairing.a3m"

    a3m_text = a3m_out.read_text()
    unpaired_path.write_text(a3m_text)
    paired_path.write_text(f">{name}\n{sequence}\n")

    return paired_path, unpaired_path


def find_target_col(df: pd.DataFrame, candidates: List[str] = None):
    if candidates is None:
        candidates = ['target', 'target_seq', 'target_sequence', 'sequence',
                      'seq', 'protein_seq', 'protein_sequence', 'amino_acid',
                      'aa_seq', 'fasta']
    cols_lower = df.columns.str.lower()
    for c in candidates:
        matches = [col for col, cl in zip(df.columns, cols_lower) if cl == c]
        if matches:
            return matches[0]
    raise ValueError(f"No target column found in {list(df.columns)}")


def make_dummy_msa(sequence: str, name: str, msa_dir: Path) -> Tuple[Path, Path]:
    """Fallback: dummy single-sequence MSA."""
    (msa_dir / name).mkdir(parents=True, exist_ok=True)
    pairing = msa_dir / name / 'pairing.a3m'
    nonpairing = msa_dir / name / 'non_pairing.a3m'
    pairing.write_text(f'>{name}\n{sequence}\n')
    nonpairing.write_text(f'>{name}\n{sequence}\n')
    return pairing, nonpairing


def make_protein_chain(sequence, id, count=1,
                       pairing_path: Union[str, Path] = None,
                       nonpairing_path: Union[str, Path] = None):
    res = {"proteinChain": {"sequence": sequence, "id": [id], "count": count}}
    if pairing_path and nonpairing_path:
        res['proteinChain']['pairedMsaPath'] = str(pairing_path)
        res['proteinChain']['unpairedMsaPath'] = str(nonpairing_path)
    return res


def make_entry(protein_chains, name, seeds=None, constraints=None):
    entry = {"name": name}
    if seeds:
        entry["modelSeeds"] = seeds
    entry["sequences"] = protein_chains
    if constraints:
        entry["constraints"] = constraints
    return entry


def make_entry_from_row(vh: str, target_json: Dict, name: str, msa_dir: Path,
                        sabdab_db: str, use_real_msa: bool = True,
                        seeds: List[int] = None, constraints: List[str] = None):
    if use_real_msa:
        try:
            pairing_path, nonpairing_path = build_vhh_msa(
                sequence=vh, name=name, output_dir=msa_dir, sabdab_db=sabdab_db
            )
        except Exception as e:
            logger.warning(f"Real MSA failed for {name} ({e}), falling back to dummy")
            pairing_path, nonpairing_path = make_dummy_msa(vh, name, msa_dir)
    else:
        pairing_path, nonpairing_path = make_dummy_msa(vh, name, msa_dir)

    protein_chains = [make_protein_chain(vh, 'H', 1, pairing_path, nonpairing_path)]
    protein_chains.extend(target_json[0]['sequences'])
    return make_entry(protein_chains, name, seeds, constraints)


def wrapper_make_entry(row_tuple, target_json, seeds, msa_dir, sabdab_db,
                       use_real_msa, constraints):
    _, row = row_tuple
    name = f"{row['seq_id']}_{generate_fablab_hash(row['vh'])}"
    return make_entry_from_row(
        vh=row['vh'], target_json=target_json, name=name,
        msa_dir=msa_dir, sabdab_db=sabdab_db, use_real_msa=use_real_msa,
        seeds=seeds, constraints=constraints,
    )


def generate_fablab_hash(sequence, length=12):
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base = len(alphabet)
    md5_hash = hashlib.md5(sequence.encode()).digest()
    hash_int = int.from_bytes(md5_hash, byteorder="big")
    encoded = []
    while hash_int > 0:
        hash_int, remainder = divmod(hash_int, base)
        encoded.append(alphabet[remainder])
    return "".join(reversed(encoded)).rjust(length, "0")[:length]


def main():
    args = parse_args()
    df = pd.read_csv(args.input_file)
    if args.subset[0] is not None and args.subset[1] is not None and \
            args.subset[0] != "None" and args.subset[1] != "None":
        df = df.query(f'{args.subset[0]}=="{args.subset[1]}"')

    start, end = args.rows
    if start is not None and end is not None:
        df = df.iloc[max(start, 0):min(len(df), end)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_json:
        out_file = out_dir / (Path(args.output_json).stem + '.json')
    else:
        out_file = out_dir / (Path(args.input_file).stem + '.json')
    vhh_msa_dir = out_dir / 'vh_msa'

    target_json_path = out_dir / 'target_msa_cached.json'
    target_fp_path = out_dir / 'target_msa_cached.fingerprint'
    target_intermediate_dir = out_dir / 'target'

    # === NEW: Resolve seqres DB path (expand ~, make absolute if provided) ===
    seqres_db: Optional[str] = None
    if args.seqres_db:
        seqres_path = Path(args.seqres_db).expanduser().resolve()
        if not seqres_path.exists():
            logger.warning(
                f"--seqres_db path does not exist: {seqres_path}. "
                f"OpenDDE will attempt to download it."
            )
            seqres_db = None
        else:
            seqres_db = str(seqres_path)
            logger.info(f"Using pre-staged seqres database: {seqres_db}")
    # ========================================================================

    ###################################################
    #    Processing target for MSA/template search    #
    ###################################################

    if args.force_refresh_target:
        for p in (target_json_path, target_fp_path):
            if p.exists():
                p.unlink()
        if target_intermediate_dir.exists():
            shutil.rmtree(target_intermediate_dir)

    if args.target_sequences is None:
        target_col = find_target_col(df)
        target = df[target_col].unique()[0]
    else:
        target = args.target_sequences
    current_fp = _target_fingerprint(target)

    cache_valid = False
    if target_json_path.exists() and target_fp_path.exists():
        cached_fp = target_fp_path.read_text().strip()
        if cached_fp == current_fp:
            cache_valid = True
        else:
            logger.info(
                f"Target changed since last cache ({cached_fp[:8]} → {current_fp[:8]}); invalidating."
            )

    if not cache_valid and target_intermediate_dir.exists():
        logger.warning(
            f"Found stale intermediate dir {target_intermediate_dir} from a previous failed run; removing."
        )
        shutil.rmtree(target_intermediate_dir)
    if not cache_valid and target_json_path.exists():
        target_json_path.unlink()

    if cache_valid:
        logger.info(f"Loading cached target MSA from {target_json_path}")
        with open(target_json_path) as f:
            target_json = json.load(f)
    else:
        logger.info("Running target MSA/template search...")
        target_json = process_target_msa_template(
            target=target,
            output_dir=out_dir,
            seqres_db=seqres_db,
            hmmsearch_binary=args.hmmsearch_binary,
            hmmbuild_binary=args.hmmbuild_binary,
            msa_server_mode=args.msa_server_mode,
        )
        with open(target_json_path, 'w') as f:
            json.dump(target_json, f, indent=2)
        target_fp_path.write_text(current_fp)
        logger.info(f"Cached target MSA at {target_json_path}")

    ###################################################
    #      Processing VHH + target for final JSON     #
    ###################################################
    if args.name:
        df['seq_id'] = [f'{args.name}_id_{i:06}' for i in range(len(df))]
    else:
        df['seq_id'] = [f'{Path(args.input_file).stem}_id_{i:06}' for i in range(len(df))]

    use_real_msa = not args.skip_vhh_msa
    sabdab_db = str(Path(args.sabdab_db).expanduser().absolute())

    if use_real_msa:
        _PRIMARY = Path(sabdab_db)
        _FALLBACK = Path.home() / "search_database" / "sabdab_nano" / "sabdab_nano_db"

        if Path(str(_PRIMARY) + ".dbtype").exists():
            sabdab_db = str(_PRIMARY)
        elif Path(str(_FALLBACK) + ".dbtype").exists():
            logger.info(f"Primary DB {_PRIMARY} not found; using fallback {_FALLBACK}")
            sabdab_db = str(_FALLBACK)
        else:
            logger.warning(
                f"No SAbDab DB found at {_PRIMARY} or {_FALLBACK}; falling back to dummy MSAs"
            )
            use_real_msa = False

    wrapper = partial(
        wrapper_make_entry,
        target_json=target_json,
        seeds=args.seeds,
        msa_dir=vhh_msa_dir,
        sabdab_db=sabdab_db,
        use_real_msa=use_real_msa,
        constraints=None,
    )

    rows = list(df[['vh', 'seq_id']].iterrows())

    n_jobs = args.n_jobs if not use_real_msa else max(1, args.n_jobs // 4)
    logger.info(
        f"Building {len(rows)} entries with {n_jobs} workers (real MSAs: {use_real_msa})"
    )

    with Pool(n_jobs) as p:
        entries = list(tqdm(p.imap(wrapper, rows), total=len(rows)))

    with open(out_file, 'w') as file:
        json.dump(entries, file, indent=2)
    logger.info(f"Wrote {len(entries)} entries to {out_file}")


if __name__ == '__main__':
    main()