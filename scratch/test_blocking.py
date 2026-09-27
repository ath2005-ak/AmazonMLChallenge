import os
import sys
import time
import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\code\business_entity_resolution")
from src.preprocessing import clean_business_name, extract_name_fingerprint, clean_address
from src.metrics import evaluate_candidate_recall

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\dataset"

print("--- Loading Validation Subset using Polars ---")
t0 = time.time()

# Load Ground Truth
gt_dict = {}
with open(os.path.join(base_dir, "train", "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        s1_id = parts[0]
        m_set = set(parts[1].split(',')) if (len(parts) > 1 and parts[1]) else set()
        gt_dict[s1_id] = m_set

# Sample 50,000 S1 entities
val_s1_ids = set(list(gt_dict.keys())[:50000])
val_gt_dict = {k: gt_dict[k] for k in val_s1_ids}

# Find all true S2/S3 target IDs in val set
val_target_ids = set()
for m_set in val_gt_dict.values():
    val_target_ids.update(m_set)

print(f"Sampled {len(val_s1_ids):,} S1 entities containing {len(val_target_ids):,} true matched S2/S3 entities. (Loaded GT in {time.time()-t0:.2f}s)")

# Load S1 records using Polars
s1_pl = pl.read_csv(
    os.path.join(base_dir, "train", "train_source1.tsv"),
    separator="\t",
    has_header=True,
    ignore_errors=True
).filter(pl.col("entity_id").is_in(val_s1_ids))

# Load S2 and S3 records using Polars (first 500k each + GT target matches)
s2_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator="\t", has_header=True, ignore_errors=True, n_rows=500000)
s3_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), separator="\t", has_header=True, ignore_errors=True, n_rows=500000)

# Combine candidate pool
cand_pl = pl.concat([s2_pl, s3_pl])

# Check missing GT target IDs and load them if needed
missing_targets = val_target_ids - set(cand_pl["entity_id"].to_list())
if missing_targets:
    print(f"Loading {len(missing_targets):,} missing GT targets into candidate pool...")
    s2_all = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(missing_targets))
    s3_all = pl.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(missing_targets))
    cand_pl = pl.concat([cand_pl, s2_all, s3_all])

print(f"Candidate pool loaded: {len(cand_pl):,} records in {time.time()-t0:.2f}s.")

# Clean names
s1_df = s1_pl.to_pandas()
cand_df = cand_pl.to_pandas()

s1_df['clean_name'] = [clean_business_name(n) for n in s1_df['business_name']]
s1_df['fingerprint'] = [extract_name_fingerprint(n) for n in s1_df['business_name']]

cand_df['clean_name'] = [clean_business_name(n) for n in cand_df['business_name']]
cand_df['fingerprint'] = [extract_name_fingerprint(n) for n in cand_df['business_name']]

# Building Inverted Indexes (Fast Python lists/dicts)
print("\nBuilding inverted indices...")
t1 = time.time()
fingerprint_index = defaultdict(list)
for eid, c, fp in zip(cand_df['entity_id'], cand_df['country'], cand_df['fingerprint']):
    if fp:
        fingerprint_index[(c, fp)].append(eid)

# Token inverted index
token_index = defaultdict(list)
for eid, c, name in zip(cand_df['entity_id'], cand_df['country'], cand_df['clean_name']):
    tokens = [t for t in name.split() if len(t) >= 3]
    for t in set(tokens): # unique tokens in row
        token_index[(c, t)].append(eid)

print(f"Indices built in {time.time()-t1:.2f}s.")

# Querying Blocking Candidates
cand_results = defaultdict(set)
max_bucket_size = 100  # avoid over-frequent generic words like 'store', 'shop'

t2 = time.time()
for s1_id, c, fp, name in zip(s1_df['entity_id'], s1_df['country'], s1_df['fingerprint'], s1_df['clean_name']):
    # 1. Exact Fingerprint Match
    if fp and (c, fp) in fingerprint_index:
        cand_results[s1_id].update(fingerprint_index[(c, fp)])
        
    # 2. Token overlap match (filtering frequent generic tokens)
    tokens = [t for t in name.split() if len(t) >= 3]
    for t in set(tokens):
        bucket = token_index.get((c, t), [])
        if 0 < len(bucket) <= max_bucket_size:
            cand_results[s1_id].update(bucket)

t_query = time.time() - t2
print(f"Querying completed in {t_query:.2f}s.")

res = evaluate_candidate_recall(val_gt_dict, cand_results)
print("\n=== BLOCKING EVALUATION RESULTS ===")
print(f"Recall Ceiling: {res['recall_ceiling']*100:.2f}%")
print(f"Found Matches: {res['found_matches']:,} / {res['total_matches']:,}")
print(f"Avg Candidates per S1 Entity: {res['avg_candidates_per_s1']:.2f}")

