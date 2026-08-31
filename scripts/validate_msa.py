"""
scripts/validate_msa_inputs.py

Validates Protenix input JSONs and their referenced MSA files.
Run this BEFORE launching folding to catch issues early.

Usage:
    python3 scripts/validate_msa_inputs.py -i data/02_intermediate/1XIW_run/rfab_1XIW_eval_input_sequences.json
"""
import json
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")  # 20 canonical + we'll allow X, -, .


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_json', type=Path, required=True,
                        help='Protenix input JSON to validate')
    parser.add_argument('--min-vhh-depth', type=int, default=10,
                        help='Warn if VHH MSA has fewer than N sequences')
    parser.add_argument('--min-target-depth', type=int, default=100,
                        help='Warn if target MSA has fewer than N sequences')
    parser.add_argument('--strict', action='store_true',
                        help='Exit nonzero on warnings, not just errors')
    parser.add_argument('--sample', type=int, default=0,
                        help='Only validate first N entries (0 = all)')
    return parser.parse_args()


def read_a3m(path: Path) -> List[Tuple[str, str]]:
    """Parse A3M file → list of (header, sequence) tuples. Robust to formatting quirks."""
    if not path.exists():
        return []
    records = []
    current_header = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            if line.startswith('>'):
                if current_header is not None:
                    records.append((current_header, ''.join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            records.append((current_header, ''.join(current_seq)))
    return records


def validate_a3m(path: Path, expected_query_seq: str,
                 min_depth: int, label: str) -> Tuple[List[str], List[str]]:
    """Returns (errors, warnings) for a single A3M file."""
    errors, warnings = [], []
    if not path.exists():
        errors.append(f"[{label}] MSA file missing: {path}")
        return errors, warnings

    records = read_a3m(path)
    if not records:
        errors.append(f"[{label}] MSA is empty: {path}")
        return errors, warnings

    # Check 1: query must be first
    first_header, first_seq = records[0]
    # A3M sequences may include lowercase (insertions) and dashes (gaps)
    first_seq_canonical = ''.join(c for c in first_seq if c.isupper() and c != '-')
    expected_canonical = expected_query_seq.upper().replace('-', '')

    if first_seq_canonical != expected_canonical:
        # Try a softer check: maybe just length differs from insertions
        if len(first_seq.replace('-', '').replace('.', '')) != len(expected_query_seq):
            errors.append(
                f"[{label}] Query sequence mismatch in {path}\n"
                f"  Expected: {expected_query_seq[:50]}... ({len(expected_query_seq)} aa)\n"
                f"  Got:      {first_seq[:50]}... ({len(first_seq)} aa)"
            )

    # Check 2: depth
    depth = len(records)
    if depth < min_depth:
        warnings.append(
            f"[{label}] Shallow MSA: {depth} sequences in {path.name} "
            f"(expected >= {min_depth})"
        )

    # Check 3: sequence length consistency (a3m allows lowercase insertions,
    # but uppercase length should match query)
    query_len = len(first_seq_canonical)
    bad_len = 0
    for header, seq in records[1:]:
        upper_len = sum(1 for c in seq if c.isupper() or c == '-')
        if upper_len != query_len:
            bad_len += 1
    if bad_len > 0:
        warnings.append(
            f"[{label}] {bad_len}/{depth - 1} sequences have inconsistent length in {path.name}"
        )

    # Check 4: weird characters
    weird_chars = Counter()
    for _, seq in records:
        for c in seq.upper():
            if c not in VALID_AA and c not in '-.X':
                weird_chars[c] += 1
    if weird_chars:
        warnings.append(
            f"[{label}] Unusual characters in {path.name}: {dict(weird_chars)}"
        )

    return errors, warnings


def validate_protein_chain(chain: Dict, entry_name: str,
                            min_vhh_depth: int, min_target_depth: int
                            ) -> Tuple[List[str], List[str], Dict]:
    """Validate a single proteinChain block. Returns (errors, warnings, stats)."""
    errors, warnings = [], []
    pc = chain['proteinChain']
    seq = pc['sequence']
    cid = pc['id'][0] if isinstance(pc['id'], list) else pc['id']
    label = f"{entry_name}:{cid}"

    stats = {'id': cid, 'length': len(seq), 'paired_depth': None, 'unpaired_depth': None}

    # Sequence sanity
    bad_aa = set(seq.upper()) - VALID_AA
    if bad_aa:
        errors.append(f"[{label}] Invalid amino acids in sequence: {bad_aa}")

    # Heuristic: is this a VHH? (short, starts with QVQL/EVQL, ends ~VSS)
    is_vhh = (
        100 <= len(seq) <= 150
        and seq.upper().startswith(('QVQL', 'EVQL', 'DVQL', 'AVQL'))
        and seq.upper().endswith(('VSS', 'TVSS'))
    )
    # Or chain id 'H' (heavy) by convention
    is_vhh = is_vhh or cid == 'H'
    min_depth = min_vhh_depth if is_vhh else min_target_depth
    chain_type = 'VHH' if is_vhh else 'target'
    stats['type'] = chain_type

    # Cys count for VHH (should have at least 2 for canonical disulfide)
    if is_vhh:
        cys_count = seq.upper().count('C')
        if cys_count < 2:
            warnings.append(
                f"[{label}] VHH has only {cys_count} cysteines (expected >=2 for disulfide)"
            )
        stats['cys_count'] = cys_count

    # MSA files
    paired = pc.get('pairedMsaPath')
    unpaired = pc.get('unpairedMsaPath')

    if not paired and not unpaired:
        # Falls back to pairing_db — that's the protenix default mode
        if 'pairing_db' not in pc:
            warnings.append(f"[{label}] No MSA path and no pairing_db specified")
    else:
        if paired:
            e, w = validate_a3m(Path(paired), seq, min_depth=1, label=f"{label} paired")
            errors.extend(e); warnings.extend(w)
            stats['paired_depth'] = len(read_a3m(Path(paired)))
        if unpaired:
            e, w = validate_a3m(Path(unpaired), seq, min_depth=min_depth,
                                label=f"{label} unpaired")
            errors.extend(e); warnings.extend(w)
            stats['unpaired_depth'] = len(read_a3m(Path(unpaired)))

    return errors, warnings, stats


def main():
    args = parse_args()
    with open(args.input_json) as f:
        entries = json.load(f)

    if args.sample:
        entries = entries[:args.sample]

    print(f"[info] Validating {len(entries)} entries from {args.input_json}\n")

    total_errors, total_warnings = 0, 0
    all_stats = []

    for entry in entries:
        name = entry.get('name', '<unnamed>')
        seqs = entry.get('sequences', [])
        if not seqs:
            print(f"[ERROR] {name}: no sequences")
            total_errors += 1
            continue

        for chain in seqs:
            if 'proteinChain' not in chain:
                continue  # skip ligands, ions, etc.
            errors, warnings, stats = validate_protein_chain(
                chain, name, args.min_vhh_depth, args.min_target_depth
            )
            all_stats.append({'entry': name, **stats})

            for err in errors:
                print(f"  [ERROR]   {err}")
                total_errors += 1
            for warn in warnings:
                print(f"  [WARN]    {warn}")
                total_warnings += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"Total entries validated: {len(entries)}")
    print(f"Total errors:   {total_errors}")
    print(f"Total warnings: {total_warnings}")

    # Aggregate MSA depth stats
    vhh_depths = [s['unpaired_depth'] for s in all_stats
                  if s.get('type') == 'VHH' and s.get('unpaired_depth')]
    target_depths = [s['unpaired_depth'] for s in all_stats
                     if s.get('type') == 'target' and s.get('unpaired_depth')]

    if vhh_depths:
        print(f"\nVHH MSA depth (unpaired): "
              f"min={min(vhh_depths)} median={sorted(vhh_depths)[len(vhh_depths)//2]} "
              f"max={max(vhh_depths)}")
    if target_depths:
        print(f"Target MSA depth (unpaired): "
              f"min={min(target_depths)} median={sorted(target_depths)[len(target_depths)//2]} "
              f"max={max(target_depths)}")

    print("=" * 70)

    if total_errors > 0:
        exit(1)
    if args.strict and total_warnings > 0:
        exit(2)


if __name__ == '__main__':
    main()