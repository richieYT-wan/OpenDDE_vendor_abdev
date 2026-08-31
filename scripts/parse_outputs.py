"""
Parse Protenix outputs → confidence DataFrames + aggregated summaries.

Directory layout expected:
  <indir>/<seq_id>/seed_<S>/<seq_id>_sample_<K>.cif
  <indir>/<seq_id>/seed_<S>/<seq_id>_summary_confidence_sample_<K>.json
"""
import json
import re
import sys
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from protenix.utils.parsing_utils import load_structure, get_sequence, generate_evalengine_hash, generate_fablab_hash


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = ArgumentParser(
        description=(
            'Parse Protenix outputs; build long DataFrame of confidence scores per '
            '(seq_id, seed, sample) and produce aggregate summaries:\n'
            '  - all seeds × all samples (S×N per seq_id)\n'
            '  - all seeds, per sample_idx (S per (seq_id, sample_idx))\n'
            '  - all seeds for the "best" sample (sample_0)\n'
            '  - single best (seed, sample) per seq_id\n'
        )
    )
    p.add_argument('-i', '--input_dir', type=Path, required=True,
                   help='Root output dir (contains <seq_id>/seed_*/*.cif|json).')
    p.add_argument('-o', '--output_dir', type=Path, default=None,
                   help='If set, write CSVs here (default: <input_dir>/aggregates).')
    p.add_argument('-n', '--n_jobs', type=int, default=16,
                   help='Parallel workers for CIF sequence extraction (default: 16).')
    p.add_argument('--rank_by', type=str, default='ranking_score',
                   help='Score to use for "best (seed, sample)" selection.')
    p.add_argument('--best_sample_idx', type=int, default=0,
                   help='Sample index treated as "best" per seed (default: 0).')
    return p.parse_args()


def find_seeds_and_ids(indir: Path) -> Tuple[List[str], List[str]]:
    paths = [Path(p) for p in glob(f'{indir}/**/seed_*', recursive=True) if Path(p).is_dir()]
    unique_seeds   = sorted({p.name for p in paths},
                            key=lambda s: int(s.split('_')[-1]))
    unique_seq_ids = sorted({p.parent.name for p in paths})
    return unique_seeds, unique_seq_ids


_SAMPLE_RE = re.compile(r'sample_(\d+)')
_SEED_RE   = re.compile(r'seed_(\d+)')


def _parse_seed(path: Path) -> int:
    m = _SEED_RE.search(str(path))
    return int(m.group(1)) if m else -1


def _parse_sample(path: Path) -> int:
    m = _SAMPLE_RE.search(path.name)
    return int(m.group(1)) if m else -1


# ─── Sequence extraction (once per seq_id, using seed_0/sample_0) ────────────

def build_seq_df(indir: Path, unique_seeds: List[str], n_jobs: int = 16) -> pd.DataFrame:
    """One CIF per seq_id → extract VH sequence + hashes."""
    ref_seed = unique_seeds[0]
    cif_files = sorted(glob(f'{indir}/**/{ref_seed}/*sample_0*.cif', recursive=True))

    def _seq(cif_file: str) -> Tuple[str, str]:
        s = load_structure(cif_file)
        seq_id = Path(cif_file).parent.parent.name
        # Chain H = antibody (per foldlab canonicalization)
        vh = get_sequence(s[s.chain_id == 'H'])
        return seq_id, vh

    results = Parallel(n_jobs=n_jobs)(delayed(_seq)(f) for f in cif_files)

    df = pd.DataFrame(results, columns=['seq_id', 'vh'])
    df['eval_hash']   = df['vh'].apply(generate_evalengine_hash)
    df['fablab_hash'] = df['vh'].apply(generate_fablab_hash)
    df['mab_id']      = df['eval_hash'] + '_' + df['fablab_hash']
    return df[['seq_id', 'mab_id', 'vh', 'eval_hash', 'fablab_hash']]


# ─── Confidence JSON parsing ─────────────────────────────────────────────────
def _load_confidence(path: Path) -> Dict:
    """Load one summary_confidence JSON, keep only scalar/numeric fields."""
    with path.open() as f:
        data = json.load(f)
    flat = {}
    for k, v in data.items():
        if isinstance(v, (int, float)):
            flat[k] = v
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            arr = np.asarray(v, dtype=float)
            flat[f'{k}_mean'] = float(arr.mean())
            flat[f'{k}_min']  = float(arr.min())
            flat[f'{k}_max']  = float(arr.max())
        # skip nested dicts / string metadata
    return flat


def build_long_df(indir: Path, n_jobs: int = 16) -> pd.DataFrame:
    """Long DataFrame: one row per (seq_id, seed, sample_idx)."""
    json_files = sorted(glob(f'{indir}/**/summary_confidence_sample_*.json', recursive=True))
    if not json_files:
        # fall back to Protenix's actual naming
        json_files = sorted(glob(f'{indir}/**/*summary_confidence_sample_*.json', recursive=True))

    def _row(jf: str) -> Dict:
        jf_path = Path(jf)
        return {
            'seq_id':     jf_path.parent.parent.name,
            'seed':       _parse_seed(jf_path),
            'sample_idx': _parse_sample(jf_path),
            **_load_confidence(jf_path),
        }

    rows = Parallel(n_jobs=n_jobs)(delayed(_row)(f) for f in json_files)
    return pd.DataFrame(rows).sort_values(['seq_id', 'seed', 'sample_idx']).reset_index(drop=True)


# ─── Aggregations ────────────────────────────────────────────────────────────

def _numeric_cols(df: pd.DataFrame, exclude=('seed', 'sample_idx')) -> List[str]:
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def _agg_mean_std(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metrics = _numeric_cols(df)
    g = df.groupby(group_cols, as_index=False)
    out = g[metrics].agg(['mean', 'std', 'count'])
    # flatten MultiIndex columns
    out.columns = [
        c if not isinstance(c, tuple)
        else (c[0] if c[1] == '' else f'{c[0]}_{c[1]}')
        for c in out.columns.to_flat_index()
    ]
    return out


def aggregate(long_df: pd.DataFrame,
              rank_by: str = 'ranking_score',
              best_sample_idx: int = 0) -> Dict[str, pd.DataFrame]:
    """Produce all 4 aggregate DataFrames keyed by seq_id."""
    aggs: Dict[str, pd.DataFrame] = {}

    # 1. all seeds × all samples per seq_id
    aggs['all'] = _agg_mean_std(long_df, ['seq_id'])

    # 2. all seeds, per sample_idx
    aggs['per_sample'] = _agg_mean_std(long_df, ['seq_id', 'sample_idx'])

    # 3. all seeds for the "best" sample_idx (default: sample_0)
    best_sample_df = long_df[long_df['sample_idx'] == best_sample_idx]
    aggs['best_sample'] = _agg_mean_std(best_sample_df, ['seq_id'])

    # 4. single best (seed, sample) per seq_id, ranked by `rank_by`
    if rank_by not in long_df.columns:
        fallback = next((c for c in ('iptm', 'ptm', 'confidence') if c in long_df.columns), None)
        if fallback is None:
            print(f'[warn] rank_by={rank_by!r} not in columns and no fallback found; '
                  f'skipping best_overall aggregate', file=sys.stderr)
            aggs['best_overall'] = pd.DataFrame()
            return aggs
        print(f'[warn] rank_by={rank_by!r} not found, using {fallback!r}', file=sys.stderr)
        rank_by = fallback

    idx = long_df.groupby('seq_id')[rank_by].idxmax()
    aggs['best_overall'] = long_df.loc[idx].reset_index(drop=True)
    return aggs


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    indir = args.input_dir
    out_dir = args.output_dir or (indir / 'aggregates')
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_seeds, unique_seq_ids = find_seeds_and_ids(indir)
    print(f'[info] Found {len(unique_seq_ids)} seq_ids, {len(unique_seeds)} seeds: {unique_seeds}')

    print(f'[info] Extracting sequences from {len(unique_seq_ids)} CIFs…')
    seq_df = build_seq_df(indir, unique_seeds, n_jobs=args.n_jobs)

    print(f'[info] Parsing confidence JSONs…')
    long_df = build_long_df(indir, n_jobs=args.n_jobs)
    print(f'[info] Long DataFrame: {len(long_df)} rows, {len(long_df.columns)} cols')

    # Merge sequence metadata into long_df
    long_df = long_df.merge(seq_df, on='seq_id', how='left')

    print(f'[info] Aggregating…')
    aggs = aggregate(long_df, rank_by=args.rank_by, best_sample_idx=args.best_sample_idx)

    # Merge seq metadata into aggregate frames (except best_overall which already has it)
    for k in ('all', 'per_sample', 'best_sample'):
        aggs[k] = aggs[k].merge(seq_df, on='seq_id', how='left')

    # Write everything
    long_df.to_csv(out_dir / 'confidence_long.csv', index=False)
    for name, df in aggs.items():
        df.to_csv(out_dir / f'confidence_agg_{name}.csv', index=False)

    print(f'[info] Wrote:')
    print(f'  {out_dir / "confidence_long.csv"} ({len(long_df)} rows)')
    for name, df in aggs.items():
        print(f'  {out_dir / f"confidence_agg_{name}.csv"} ({len(df)} rows)')

    # Make ONE merged dataframe with --> best_sample as mean (_mean, of N seeds) + best_overall (as _best)
    # df_best = aggs['best_overall']
    # df_mean = aggs['best_sample']
    df_merged = pd.merge(aggs['best_overall'].rename(columns = {k: f'{k}_best' for k in aggs['best_overall'].columns if k not in ['seq_id', 'mab_id', 'vh', 'eval_hash', 'fablab_hash']}),
                         aggs['best_sample'],
                         on = ['seq_id', 'mab_id', 'vh', 'eval_hash', 'fablab_hash'])
    df_merged.to_csv(f'{out_dir / "confidence_merged.csv"}', index=False)



if __name__ == '__main__':
    main()
