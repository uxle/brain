#!/usr/bin/env python3
import os
import re
import hashlib
from collections import defaultdict

def normalize_body(body: str) -> str:
    # Strip comments
    body = re.sub(r'//.*', '', body)
    # Strip whitespace
    body = re.sub(r'\s+', ' ', body).strip()
    return body

def audit_directory(root_dir: str):
    duplicate_groups = defaultdict(list)
    total_tests = 0

    test_pattern = re.compile(r'#\[test\]\s*(?:async\s+)?fn\s+([a-zA-Z0-9_]+)\s*\(\)\s*\{', re.MULTILINE)

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.rs'):
                filepath = os.path.join(root, f)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()

                for match in test_pattern.finditer(content):
                    total_tests += 1
                    fn_name = match.group(1)
                    start_idx = match.end()
                    
                    # Extract function body by matching braces
                    depth = 1
                    end_idx = start_idx
                    while end_idx < len(content) and depth > 0:
                        if content[end_idx] == '{':
                            depth += 1
                        elif content[end_idx] == '}':
                            depth -= 1
                        end_idx += 1

                    body = content[start_idx:end_idx - 1]
                    norm = normalize_body(body)
                    
                    # Compute normalized hash (strip trailing numbers in function names if any)
                    h = hashlib.sha256(norm.encode('utf-8')).hexdigest()
                    duplicate_groups[h].append((filepath, fn_name, len(norm)))

    duplicates = {k: v for k, v in duplicate_groups.items() if len(v) > 1}
    return total_tests, duplicates

if __name__ == '__main__':
    crates_dir = 'crates'
    print("=== Brain Framework Test Audit ===")
    total_all_tests = 0
    total_all_dupes = 0

    for crate in sorted(os.listdir(crates_dir)):
        crate_path = os.path.join(crates_dir, crate)
        if os.path.isdir(crate_path):
            total, dupes = audit_directory(crate_path)
            total_all_tests += total
            dupe_count = sum(len(v) - 1 for v in dupes.values())
            total_all_dupes += dupe_count
            print(f"{crate:<25}: Total Tests = {total:<5} | Duplicate Tests = {dupe_count:<4}")

    print("-" * 50)
    print(f"Total Tests across framework : {total_all_tests}")
    print(f"Total Duplicate Tests        : {total_all_dupes}")
