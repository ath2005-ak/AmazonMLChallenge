import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\dataset"

print("Loading S1 countries...")
s1_country = {}
with open(os.path.join(base_dir, "train", "train_source1.tsv"), 'r', encoding='utf-8', errors='ignore') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            s1_country[parts[0]] = parts[3].strip()

print("Loading S2 countries...")
s2_country = {}
with open(os.path.join(base_dir, "train", "train_source2.tsv"), 'r', encoding='utf-8', errors='ignore') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            s2_country[parts[0]] = parts[3].strip()

print("Loading S3 countries...")
s3_country = {}
with open(os.path.join(base_dir, "train", "train_source3.tsv"), 'r', encoding='utf-8', errors='ignore') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            s3_country[parts[0]] = parts[3].strip()

print("Verifying GT country consistency...")
cross_country_matches = 0
total_pairs_checked = 0

with open(os.path.join(base_dir, "train", "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        s1_id = parts[0]
        c1 = s1_country.get(s1_id)
        if len(parts) > 1 and parts[1]:
            m_list = parts[1].split(',')
            for m in m_list:
                total_pairs_checked += 1
                if m.startswith('S2-'):
                    c2 = s2_country.get(m)
                    if c1 != c2:
                        cross_country_matches += 1
                elif m.startswith('S3-'):
                    c3 = s3_country.get(m)
                    if c1 != c3:
                        cross_country_matches += 1

print(f"Total matched pairs checked: {total_pairs_checked:,}")
print(f"Cross-country matches: {cross_country_matches:,} ({cross_country_matches/total_pairs_checked*100:.4f}%)")
