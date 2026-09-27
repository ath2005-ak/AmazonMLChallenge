import os
import sys
import time
import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict, Counter

sys.path.append(r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\code\business_entity_resolution")
from src.preprocessing import clean_business_name, extract_name_fingerprint, clean_address, extract_pincode
from src.metrics import evaluate_candidate_recall

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\dataset"

print("--- Fast Streamlined Multi-Pass Inverted Index Blocking ---")
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

# Load S2 and S3 records (first 500k each + GT targets)
s2_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator="\t", has_header=True, ignore_errors=True, n_rows=500000)
s3_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), separator="\t", has_header=True, ignore_errors=True, n_rows=500000)

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
s1_df['pin'] = [extract_pincode(a) for a in s1_df['business_address']]

cand_df['clean_name'] = [clean_business_name(n) for n in cand_df['business_name']]
cand_df['fp'] = [extract_name_fingerprint(n) for n in cand_df['business_name']]
cand_df['pin'] = [extract_pincode(a) for a in cand_df['business_address']]

# Generic stop words to exclude from individual word indexing
STOP_WORDS = {
    'store', 'shop', 'company', 'ltd', 'pvt', 'inc', 'corp', 'services', 'service',
    'center', 'centre', 'group', 'enterprises', 'traders', 'trading', 'solutions',
    'technologies', 'private', 'limited', 'corporation', 'sales', 'mart', 'market',
    'and', 'the', 'for', 'with', 'from', 'near', 'opp', 'road', 'street', 'main'
}

def get_keys(name, fp, pin):
    keys = []
    # Fingerprint key
    if fp:
        keys.append(('fp', fp))
        
    # Word tokens
    words = [w for w in name.split() if len(w) >= 3 and w not in STOP_WORDS]
    for w in words:
        keys.append(('w', w))
        
    # First 4 chars of core name (prefix blocking)
    if len(name) >= 4:
        keys.append(('p4', name[:4]))
        
    # Pincode
    if pin:
        keys.append(('pin', pin))
        
    return keys

print("\nBuilding inverted index...")
t_idx = time.time()
index_by_country = {}

for country in s1_df['country'].unique():
    cand_sub = cand_df[cand_df['country'] == country]
    idx_dict = defaultdict(list)
    
    for eid, name, fp, pin in zip(cand_sub['entity_id'], cand_sub['clean_name'], cand_sub['fp'], cand_sub['pin']):
        for k_type, k_val in get_keys(name, fp, pin):
            idx_dict[(k_type, k_val)].append(eid)
            
    index_by_country[country] = idx_dict

print(f"Index built in {time.time()-t_idx:.2f}s.")

# Querying
print("Querying...")
t_q = time.time()
cand_results = defaultdict(set)

MAX_LIMITS = {
    'fp': 1000,
    'w': 250,
    'p4': 100,
    'pin': 300
}

TOP_K = 25

for s1_id, country, name, fp, pin in zip(s1_df['entity_id'], s1_df['country'], s1_df['clean_name'], s1_df['fp'], s1_df['pin']):
    idx_dict = index_by_country.get(country, {})
    keys = get_keys(name, fp, pin)
    
    candidate_scores = Counter()
    
    for k_type, k_val in keys:
        bucket = idx_dict.get((k_type, k_val), [])
        max_lim = MAX_LIMITS.get(k_type, 150)
        if 0 < len(bucket) <= max_lim:
            w_score = 4.0 if k_type == 'fp' else (2.0 if k_type == 'w' else 0.5)
            for cid in bucket:
                candidate_scores[cid] += w_score
                
    if candidate_scores:
        top_cands = [cid for cid, s in candidate_scores.most_common(TOP_K)]
        cand_results[s1_id] = set(top_cands)

print(f"Querying completed in {time.time()-t_q:.2f}s.")

res = evaluate_candidate_recall(val_gt_dict, cand_results)
print("\n=== FAST STREAMLINED BLOCKING RESULTS ===")
print(f"Recall Ceiling: {res['recall_ceiling']*100:.2f}%")
print(f"Found Matches: {res['found_matches']:,} / {res['total_matches']:,}")
print(f"Avg Candidates per S1 Entity: {res['avg_candidates_per_s1']:.2f}")

