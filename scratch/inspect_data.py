import os
import sys

sys.stdout.reconfigure(encoding='utf-8')


def count_lines(filepath):
    count = 0
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            count += 1
    return count

def inspect_file(filepath, n_samples=3):
    print(f"--- File: {filepath} ---")
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")
    line_cnt = count_lines(filepath)
    print(f"Total lines: {line_cnt:,} (including header: {line_cnt - 1:,} records)")
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip()
        print(f"Header: {header!r}")
        print("Sample rows:")
        for i in range(n_samples):
            line = f.readline()
            if not line:
                break
            print(f"  [{i+1}] {line.strip()!r}")
    print()

def main():
    base_dir = r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\dataset"
    print("=== TRAIN DATASETS ===")
    inspect_file(os.path.join(base_dir, "train", "train_source1.tsv"))
    inspect_file(os.path.join(base_dir, "train", "train_source2.tsv"))
    inspect_file(os.path.join(base_dir, "train", "train_source3.tsv"))
    inspect_file(os.path.join(base_dir, "train", "train_ground_truth.tsv"))

    print("=== TEST DATASETS ===")
    inspect_file(os.path.join(base_dir, "test", "test_source1.tsv"))
    inspect_file(os.path.join(base_dir, "test", "test_source2.tsv"))
    inspect_file(os.path.join(base_dir, "test", "test_source3.tsv"))

if __name__ == "__main__":
    main()
