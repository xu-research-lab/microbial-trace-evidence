#!/usr/bin/env python3
"""
Prepare and run SourceTracker2 for mixed_signal experiment_category.

Source = skin samples (each individual is a source environment)
Sink = surface_touched samples
"""

import os
import sys
import subprocess
import biom
import pandas as pd
import numpy as np

# === Paths ===
BASE = '../'
BIOM_FP = os.path.join(BASE, 'data/reference-hit.2026.biom')
MAP_FP = os.path.join(BASE, 'data/merged_mapping.txt')
OUT_DIR = os.path.join(BASE, './')

OTU_TABLE_FP = os.path.join(OUT_DIR, 'otu_table.tsv')
MAPPING_FP = os.path.join(OUT_DIR, 'mapping.tsv')

MIN_READS = 50


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load data
    print("Loading BIOM table...")
    table = biom.load_table(BIOM_FP)
    print(f"  Total samples: {len(table.ids(axis='sample'))}")
    print(f"  Total features: {len(table.ids(axis='observation'))}")

    print("Loading mapping file...")
    md = pd.read_table(MAP_FP, index_col=0,
                       dtype={'month': str, 'day': str, 'year': str, 'sample_date': str})

    # 2. Filter for mixed_signal
    sample_ids = table.ids(axis='sample')
    mixed_ids = [s for s in sample_ids if s in md.index and md.loc[s, 'experiment_category'] == 'mixed_signal']
    print(f"\nmixed_signal samples in mapping: {len(mixed_ids)}")

    # 3. Remove BLANK
    non_blank_ids = [s for s in mixed_ids if 'BLANK' not in s.upper() and s not in md.index or
                     (s in md.index and md.loc[s, 'sample_type'] != 'blank')]
    non_blank_ids = [s for s in mixed_ids if 'BLANK' not in s.upper()]
    print(f"After removing BLANK: {len(non_blank_ids)}")

    # 4. Filter by min reads
    data = table.matrix_data.toarray()
    sample_id_to_idx = {s: i for i, s in enumerate(sample_ids)}
    sample_sums = data.sum(axis=0)

    filtered_ids = []
    for s in non_blank_ids:
        idx = sample_id_to_idx[s]
        if sample_sums[idx] >= MIN_READS:
            filtered_ids.append(s)
    print(f"After removing reads < {MIN_READS}: {len(filtered_ids)}")

    # 5. Separate source and sink
    source_ids = []
    sink_ids = []
    for s in filtered_ids:
        stype = md.loc[s, 'sample_type']
        if stype == 'skin':
            source_ids.append(s)
        elif stype == 'surface_touched':
            sink_ids.append(s)

    # Check individuals in source
    source_individuals = md.loc[source_ids, 'individual'].unique()
    print(f"Source individuals (before filtering): {sorted(source_individuals)}")

    # Filter sources: only keep individuals that appear in sink pairs
    sink_pairs_used = set()
    for s in sink_ids:
        for col in ['pair1_resident', 'pair2_perp']:
            val = str(md.loc[s, col])
            if val.strip().lower() not in ('unknown', 'nan', ''):
                sink_pairs_used.add(val)
    valid_source_ids = [s for s in source_ids if md.loc[s, 'individual'] in sink_pairs_used]
    removed = set(source_ids) - set(valid_source_ids)
    print(f"Removed {len(removed)} source samples whose individual not in any sink pair: {sorted(removed)}")
    source_ids = valid_source_ids
    print(f"\nSource (skin) samples after filtering: {len(source_ids)}")
    print(f"Sink (surface_touched) samples: {len(sink_ids)}")

    # 6. Build OTU table (features x samples, only filtered samples)
    all_filtered = source_ids + sink_ids
    feature_ids = table.ids(axis='observation')

    # Get data subset
    filtered_idx = [sample_id_to_idx[s] for s in all_filtered]
    otu_data = data[:, filtered_idx]  # features x filtered_samples

    # Write OTU table TSV
    print(f"\nWriting OTU table to {OTU_TABLE_FP}...")
    with open(OTU_TABLE_FP, 'w') as f:
        # Header
        f.write('#OTU_ID\t' + '\t'.join(all_filtered) + '\n')
        for i, fid in enumerate(feature_ids):
            row = '\t'.join([str(int(otu_data[i, j])) for j in range(len(all_filtered))])
            f.write(fid + '\t' + row + '\n')

    # 7. Build mapping file
    print(f"Writing mapping file to {MAPPING_FP}...")
    with open(MAPPING_FP, 'w') as f:
        f.write('#SampleID\tSourceSink\tEnv\tsurface_type\tpair1_resident\tpair2_perp\tindividual\n')
        for s in all_filtered:
            row = md.loc[s]
            if s in source_ids:
                source_sink = 'source'
                env = row['individual']
            else:
                source_sink = 'sink'
                env = row['sample_type']

            surface_type = str(row.get('surface_type', ''))
            pair1 = str(row.get('pair1_resident', ''))
            pair2 = str(row.get('pair2_perp', ''))
            individual = str(row.get('individual', ''))

            f.write(f'{s}\t{source_sink}\t{env}\t{surface_type}\t{pair1}\t{pair2}\t{individual}\n')

    print(f"\nDone! Files written:")
    print(f"  OTU table: {OTU_TABLE_FP}")
    print(f"  Mapping: {MAPPING_FP}")
    print(f"\nTo run SourceTracker2:")
    print(f"  conda activate st2")
    print(f"  sourcetracker2 -i {OTU_TABLE_FP} -m {MAPPING_FP} -o {os.path.join(OUT_DIR, 'st2_output')}")
if __name__ == '__main__':
    main()