import os
import sys
import pandas as pd
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\dataset"

print("--- Ground Truth Analysis ---")
gt_path = os.path.join(base_dir, "train", "train_ground_truth.tsv")

match_counts = []
total_s1 = 0
singletons = 0
s2_matches = 0
s3_matches = 0
multi_matches = 0

with open(gt_path, 'r', encoding='utf-8') as f:
    header = f.readline()
    for line in f:
        total_s1 += 1
        parts = line.strip().split('\t')
        s1_id = parts[0]
        matched_str = parts[1] if len(parts) > 1 else ""
        if not matched_str:
            singletons += 1
            match_counts.append(0)
        else:
            m_list = matched_str.split(',')
            match_counts.append(len(m_list))
            for m in m_list:
                if m.startswith('S2-'):
                    s2_matches += 1
                elif m.startswith('S3-'):
                    s3_matches += 1
            if len(m_list) > 1:
                multi_matches += 1

print(f"Total S1 entities in train GT: {total_s1:,}")
print(f"Singletons (0 matches): {singletons:,} ({singletons/total_s1*100:.2f}%)")
print(f"Entities with >=1 match: {total_s1 - singletons:,} ({(total_s1 - singletons)/total_s1*100:.2f}%)")
print(f"Entities with >1 match: {multi_matches:,} ({multi_matches/total_s1*100:.2f}%)")
print(f"Total S2 matched references: {s2_matches:,}")
print(f"Total S3 matched references: {s3_matches:,}")

count_freq = Counter(match_counts)
print("\nMatch count frequency distribution:")
for k in sorted(count_freq.keys())[:15]:
    print(f"  {k} matches: {count_freq[k]:,} entities ({count_freq[k]/total_s1*100:.3f}%)")

print("\n--- Country Distribution Analysis ---")
def analyze_countries(filepath, label):
    countries = Counter()
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                c = parts[3].strip()
                countries[c] += 1
            elif len(parts) == 3: # if address missing
                c = parts[2].strip()
                countries[c] += 1
    print(f"Countries in {label}: {dict(countries)}")
    return countries

analyze_countries(os.path.join(base_dir, "train", "train_source1.tsv"), "Train Source 1")
analyze_countries(os.path.join(base_dir, "train", "train_source2.tsv"), "Train Source 2")
analyze_countries(os.path.join(base_dir, "train", "train_source3.tsv"), "Train Source 3")

analyze_countries(os.path.join(base_dir, "test", "test_source1.tsv"), "Test Source 1")
analyze_countries(os.path.join(base_dir, "test", "test_source2.tsv"), "Test Source 2")
analyze_countries(os.path.join(base_dir, "test", "test_source3.tsv"), "Test Source 3")

