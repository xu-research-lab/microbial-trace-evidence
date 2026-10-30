#!/bin/bash


# Step 1: Convert BIOM observation IDs to uppercase (some IDs contain lowercase like 'gggggggg')
# and save as a new BIOM table
python3 << 'PYEOF'
import biom
table = biom.load_table('normalized.biom')
old_ids = list(table.ids(axis='observation'))
new_ids = [x.upper() for x in old_ids]
table.update_ids({old: new for old, new in zip(old_ids, new_ids)}, axis='observation')
with biom.util.biom_open('normalized_uppercase.biom', 'w') as f:
    table.to_hdf5(f, 'normalized with uppercase IDs')
print(f'Converted {len(old_ids)} observation IDs to uppercase')
PYEOF

# Step 2: Import uppercase BIOM table to QIIME2
qiime tools import \
  --input-path normalized_uppercase.biom \
  --type 'FeatureTable[Frequency]' \
  --input-format BIOMV210Format \
  --output-path table.qza

# Step 3: Extract ASV IDs (which are the sequences) to FASTA format, convert to uppercase
biom table-ids -i normalized_uppercase.biom --observations | \
  awk 'NF>0 {print ">"$1; print $1}' | tr 'a-z' 'A-Z' > rep-seqs.fna

# Step 4: Import sequences as FeatureData[Sequence]
mkdir -p rep-seqs-dir
cp rep-seqs.fna rep-seqs-dir/dna-sequences.fasta
qiime tools import \
  --type 'FeatureData[Sequence]' \
  --input-path rep-seqs-dir \
  --output-path sequences.qza

# Step 5: Phylogeny - align and build tree
qiime phylogeny align-to-tree-mafft-fasttree \
  --i-sequences sequences.qza \
  --p-n-threads 10 \
  --o-alignment aligned-rep-seqs.qza \
  --o-masked-alignment masked-aligned-rep-seqs.qza \
  --o-tree unrooted-tree.qza \
  --o-rooted-tree rooted-tree.qza

# Step 6: Beta diversity metrics
qiime diversity beta \
  --i-table table.qza \
  --p-metric braycurtis \
  --o-distance-matrix bray_curtis_distance_matrix.qza

qiime diversity beta-phylogenetic \
  --i-table table.qza \
  --i-phylogeny rooted-tree.qza \
  --p-metric weighted_unifrac \
  --o-distance-matrix weighted_unifrac_distance_matrix.qza


# Step 7: Extract distance matrices
qiime tools extract \
  --input-path weighted_unifrac_distance_matrix.qza \
  --output-path extracted_qza/weighted_unifrac_distance_matrix


qiime tools extract \
  --input-path bray_curtis_distance_matrix.qza \
  --output-path extracted_qza/bray_curtis_distance_matrix