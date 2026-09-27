import os
import sys
import time
import re
import unicodedata
import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict, Counter

sys.path.append(r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\code\business_entity_resolution")
from src.metrics import evaluate_candidate_recall

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\dataset"

print("--- Testing Enhanced Multi-Pass Blocking (ASCII Unidecode + Address Keys) ---")
t0 = time.time()

# Enhanced Text Normalization Function
DIGIT_TYPOS = {'0': 'o', '8': 'b', '1': 'i', '5': 's'}

LEGAL_SUFFIXES = re.compile(r'\b(pvt|private|ltd|limited|inc|incorporated|corp|corporation|llc|llp|co|company|sarl|sas|gmbh|s\.a\.r\.l|s\.a\.s|s\.a|n\.v|b\.v|aka|f\/k\/a)\b', flags=re.IGNORECASE)

DOMAIN_PATTERN = re.compile(r'\b([\w\-]+)\.(com|org|net|in|co\.in|us|fr)\b', flags=re.IGNORECASE)

def normalize_string(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    # Strip domain extensions e.g. gcozy.com -> gcozy
    s = DOMAIN_PATTERN.sub(r'\1', s)
    # Unicode NFKD decomposition: strip accents (e.g. ó -> o, Í -> I)
    s = unicodedata.normalize('NFKD', s)
    s = "".join([c for c in s if not unicodedata.combining(c)]).lower()
    # Replace & with and
    s = s.replace('&', ' and ')
    # Keep only alphanumeric and whitespace
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def get_name_fp(name: str) -> str:
    norm = normalize_string(name)
    norm_no_legal = LEGAL_SUFFIXES.sub('', norm)
    tokens = sorted([t for t in norm_no_legal.split() if len(t) > 1])
    return " ".join(tokens)

def extract_street_num(addr: str) -> str:
    if not addr or not isinstance(addr, str):
        return ""
    matches = re.findall(r'\b\d{2,6}\b', addr)
    return matches[0] if matches else ""

def extract_addr_tokens(addr: str) -> list:
    norm = normalize_string(addr)
    # filter numbers and words >= 4 chars
    tokens = [t for t in norm.split() if len(t) >= 4 and not t.isdigit()]
    return tokens[:3]

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

# Preprocessing
s1_df['norm_name'] = [normalize_string(n) for n in s1_df['business_name']]
s1_df['fp'] = [get_name_fp(n) for n in s1_df['business_name']]
s1_df['st_num'] = [extract_street_num(a) for a in s1_df['business_address']]
s1_df['addr_toks'] = [extract_addr_tokens(a) for a in s1_df['business_address']]

cand_df['norm_name'] = [normalize_string(n) for n in cand_df['business_name']]
cand_df['fp'] = [get_name_fp(n) for n in cand_df['business_name']]
cand_df['st_num'] = [extract_street_num(a) for a in cand_df['business_address']]
cand_df['addr_toks'] = [extract_addr_tokens(a) for a in cand_df['business_address']]

STOP_WORDS = {'store', 'shop', 'company', 'ltd', 'pvt', 'inc', 'corp', 'services', 'service', 'center', 'centre', 'group', 'enterprises', 'traders', 'trading', 'solutions', 'technologies', 'private', 'limited', 'corporation', 'sales', 'mart', 'market', 'and', 'the', 'for', 'with', 'from', 'near', 'opp', 'road', 'street', 'main'}

def get_record_keys(norm_name, fp, st_num, addr_toks):
    keys = []
    if fp and len(fp) >= 3:
        keys.append(('fp', fp))
        
    words = [w for w in norm_name.split() if len(w) >= 3 and w not in STOP_WORDS]
    for w in words:
        keys.append(('w', w))
        
    if st_num and len(st_num) >= 2:
        keys.append(('st', st_num))
        for atok in addr_toks:
            keys.append(('st_at', f"{st_num}_{atok}"))
            
    return keys

print("\nBuilding inverted index...")
t_idx = time.time()
index_by_country = {}

for country in s1_df['country'].unique():
    cand_sub = cand_df[cand_df['country'] == country]
    idx_dict = defaultdict(list)
    
    for eid, norm_name, fp, st_num, addr_toks in zip(cand_sub['entity_id'], cand_sub['norm_name'], cand_sub['fp'], cand_sub['st_num'], cand_sub['addr_toks']):
        for k_type, k_val in get_record_keys(norm_name, fp, st_num, addr_toks):
            idx_dict[(k_type, k_val)].append(eid)
            
    index_by_country[country] = idx_dict

print(f"Index built in {time.time()-t_idx:.2f}s.")

# Querying
print("Querying candidate sets...")
t_q = time.time()
cand_results = defaultdict(set)

MAX_LIMITS = {
    'fp': 1500,
    'w': 300,
    'st_at': 200,
    'st': 500
}

TOP_K = 30

for s1_id, country, norm_name, fp, st_num, addr_toks in zip(s1_df['entity_id'], s1_df['country'], s1_df['norm_name'], s1_df['fp'], s1_df['st_num'], s1_df['addr_toks']):
    idx_dict = index_by_country.get(country, {})
    keys = get_record_keys(norm_name, fp, st_num, addr_toks)
    
    candidate_scores = Counter()
    
    for k_type, k_val in keys:
        bucket = idx_dict.get((k_type, k_val), [])
        max_lim = MAX_LIMITS.get(k_type, 150)
        if 0 < len(bucket) <= max_lim:
            w_score = 5.0 if k_type == 'fp' else (3.0 if k_type == 'st_at' else (2.0 if k_type == 'w' else 0.8))
            for cid in bucket:
                candidate_scores[cid] += w_score
                
    if candidate_scores:
        top_cands = [cid for cid, s in candidate_scores.most_common(TOP_K)]
        cand_results[s1_id] = set(top_cands)

print(f"Querying completed in {time.time()-t_q:.2f}s.")

res = evaluate_candidate_recall(val_gt_dict, cand_results)
print("\n=== ENHANCED MULTI-PASS BLOCKING RESULTS ===")
print(f"Recall Ceiling: {res['recall_ceiling']*100:.2f}%")
print(f"Found Matches: {res['found_matches']:,} / {res['total_matches']:,}")
print(f"Avg Candidates per S1 Entity: {res['avg_candidates_per_s1']:.2f}")

