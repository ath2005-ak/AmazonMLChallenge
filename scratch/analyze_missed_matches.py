import os
import sys
import pandas as pd
import polars as pl

sys.path.append(r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\code\business_entity_resolution")
from src.preprocessing import clean_business_name, extract_name_fingerprint, clean_address

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\dataset"

# Load GT
gt_dict = {}
with open(os.path.join(base_dir, "train", "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        s1_id = parts[0]
        m_set = set(parts[1].split(',')) if (len(parts) > 1 and parts[1]) else set()
        gt_dict[s1_id] = m_set

# Load first 100 S1 entities that have matches
sample_s1 = {}
for s1_id, m_set in gt_dict.items():
    if len(m_set) > 0:
        sample_s1[s1_id] = m_set
    if len(sample_s1) >= 50:
        break

# Load S1 details
s1_df = pl.read_csv(os.path.join(base_dir, "train", "train_source1.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(set(sample_s1.keys()))).to_pandas()

all_target_ids = set()
for m_set in sample_s1.values():
    all_target_ids.update(m_set)

s2_df = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(all_target_ids)).to_pandas()
s3_df = pl.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(all_target_ids)).to_pandas()

cand_df = pd.concat([s2_df, s3_df], ignore_index=True)

cand_map = {row['entity_id']: row for _, row in cand_df.iterrows()}

print("=== SAMPLE GROUND TRUTH PAIRS ANALYSIS ===")
for _, s1_row in s1_df.iterrows():
    s1_id = s1_row['entity_id']
    s1_name = s1_row['business_name']
    s1_addr = s1_row['business_address']
    
    print(f"\n[S1] {s1_id} | Name: {s1_name!r} | Addr: {s1_addr!r}")
    print(f"     Clean Name: {clean_business_name(s1_name)!r} | FP: {extract_name_fingerprint(s1_name)!r}")
    
    for mid in sample_s1[s1_id]:
        if mid in cand_map:
            m_row = cand_map[mid]
            m_name = m_row['business_name']
            m_addr = m_row['business_address']
            print(f"  -> [{mid}] Name: {m_name!r} | Addr: {m_addr!r}")
            print(f"       Clean Name: {clean_business_name(m_name)!r} | FP: {extract_name_fingerprint(m_name)!r}")
