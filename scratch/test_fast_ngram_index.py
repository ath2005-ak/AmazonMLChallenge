import os
import sys
import time
import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict, Counter

sys.path.append(r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\code\business_entity_resolution")
from src.preprocessing import clean_business_name, extract_name_fingerprint, clean_address
from src.metrics import evaluate_candidate_recall

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\dataset"

print("--- Fast Multi-Pass Character & Word Inverted Index Blocking ---")
t0 = time.time()

# Load GT
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

val_target_ids = set()
for m_set in val_gt_dict.values():
    val_target_ids.update(m_set)

# Load S1 records
s1_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source1.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(val_s1_ids))

# Load S2 and S3 records (first 400k each + GT targets)
s2_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator="\t", has_header=True, ignore_errors=True, n_rows=400000)
s3_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), separator="\t", has_header=True, ignore_errors=True, n_rows=400000)

cand_pl = pl.concat([s2_pl, s3_pl])
missing_targets = val_target_ids - set(cand_pl["entity_id"].to_list())
if missing_targets:
    s2_all = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(missing_targets))
    s3_all = pl.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(missing_targets))
    cand_pl = pl.concat([cand_pl, s2_all, s3_all])

s1_df = s1_pl.to_pandas()
cand_df = cand_pl.to_pandas()

print(f"Data loaded: {len(s1_df):,} S1 and {len(cand_df):,} candidates in {time.time()-t0:.2f}s.")

# Clean strings
s1_df['clean_name'] = [clean_business_name(n) for n in s1_df['business_name']]
s1_df['fp'] = [extract_name_fingerprint(n) for n in s1_df['business_name']]
s1_df['clean_addr'] = [clean_address(a) for a in s1_df['business_address']]

cand_df['clean_name'] = [clean_business_name(n) for n in cand_df['business_name']]
cand_df['fp'] = [extract_name_fingerprint(n) for n in cand_df['business_name']]
cand_df['clean_addr'] = [clean_address(a) for a in cand_df['business_address']]

# Helper to extract tokens & 3-grams
def get_blocking_keys(name, fp, addr):
    keys = []
    # 1. Fingerprint key
    if fp and len(fp) >= 3:
        keys.append(('fp', fp))
    
    # 2. Significant word tokens (len >= 3)
    words = [w for w in name.split() if len(w) >= 3]
    for w in words:
        keys.append(('word', w))
        
    # 3. Char 4-grams for typos
    if len(name) >= 4:
        for i in range(len(name) - 3):
            sub = name[i:i+4]
            if not sub.isspace():
                keys.append(('ch4', sub))
                
    return keys

print("\nBuilding multi-key inverted index...")
t_idx = time.time()

# Store indices per country
index_by_country = {}

for country in s1_df['country'].unique():
    cand_sub = cand_df[cand_df['country'] == country]
    
    idx_dict = defaultdict(list)
    for eid, name, fp, addr in zip(cand_sub['entity_id'], cand_sub['clean_name'], cand_sub['fp'], cand_sub['clean_addr']):
        keys = get_blocking_keys(name, fp, addr)
        for k_type, k_val in keys:
            idx_dict[(k_type, k_val)].append(eid)
            
    index_by_country[country] = idx_dict

print(f"Multi-key inverted index built in {time.time()-t_idx:.2f}s.")

# Querying
print("Querying candidate sets...")
t_q = time.time()
cand_results = defaultdict(set)

MAX_BUCKET_LIMITS = {
    'fp': 500,
    'word': 150,
    'ch4': 80
}

TOP_K_PER_S1 = 15

for s1_id, country, name, fp, addr in zip(s1_df['entity_id'], s1_df['country'], s1_df['clean_name'], s1_df['fp'], s1_df['clean_addr']):
    idx_dict = index_by_country.get(country, {})
    keys = get_blocking_keys(name, fp, addr)
    
    candidate_scores = Counter()
    
    for k_type, k_val in keys:
        max_lim = MAX_BUCKET_LIMITS.get(k_type, 100)
        bucket = idx_dict.get((k_type, k_val), [])
        if 0 < len(bucket) <= max_lim:
            weight = 3.0 if k_type == 'fp' else (2.0 if k_type == 'word' else 0.5)
            for cid in bucket:
                candidate_scores[cid] += weight
                
    if candidate_scores:
        # Pick top K candidates by weighted overlap score
        top_cands = [cid for cid, score in candidate_scores.most_common(TOP_K_PER_S1)]
        cand_results[s1_id] = set(top_cands)

print(f"Querying completed in {time.time()-t_q:.2f}s.")

res = evaluate_candidate_recall(val_gt_dict, cand_results)
print("\n=== FAST MULTI-KEY BLOCKING RESULTS ===")
print(f"Recall Ceiling: {res['recall_ceiling']*100:.2f}%")
print(f"Found Matches: {res['found_matches']:,} / {res['total_matches']:,}")
print(f"Avg Candidates per S1 Entity: {res['avg_candidates_per_s1']:.2f}")

