#!/usr/bin/env python3
"""
Audit and Detect Duplicate Test Functions in Rust Codebases.

Usage:
    python3 scripts/audit_test_dupes.py [DIR] [--markdown OUTPUT_FILE] [--check]
"""

import re
import os
import sys
import hashlib
from collections import defaultdict
from typing import List, Dict, Tuple

def extract_test_functions(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    tests = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if line.startswith('#[test]') or line.startswith('#[tokio::test]'):
            attr_line = i + 1
            fn_idx = i + 1
            while fn_idx < n and not re.search(r'\bfn\s+([a-zA-Z0-9_]+)', lines[fn_idx]):
                fn_idx += 1
            if fn_idx < n:
                match = re.search(r'\bfn\s+([a-zA-Z0-9_]+)', lines[fn_idx])
                if match:
                    fn_name = match.group(1)
                    start_line = fn_idx + 1
                    brace_count = 0
                    found_open = False
                    body_lines = []
                    curr = fn_idx
                    while curr < n:
                        for char in lines[curr]:
                            if char == '{':
                                brace_count += 1
                                found_open = True
                            elif char == '}':
                                brace_count -= 1
                        body_lines.append(lines[curr])
                        if found_open and brace_count == 0:
                            break
                        curr += 1

                    end_line = curr + 1
                    body_text = ''.join(body_lines)
                    tests.append({
                        'name': fn_name,
                        'start_line': start_line,
                        'end_line': end_line,
                        'file': file_path,
                        'body': body_text
                    })
                    i = curr + 1
                    continue
        i += 1
    return tests

def normalize_body(body: str) -> str:
    # Strip comments
    body = re.sub(r'//.*', '', body)
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL)
    # Strip function declaration header
    body = re.sub(r'^[^{]*\{', '{', body.strip())
    # Normalize numeric constants and type suffixes to detect template copies
    body = re.sub(r'\b\d+(\.\d+)?(_f64|_f32|_u32|_usize|_i32|_i64)?\b', 'NUM', body)
    # Strip numeric suffixes from identifiers
    body = re.sub(r'\b([a-zA-Z_]+)_\d+\b', r'\1', body)
    # Normalize whitespace
    body = re.sub(r'\s+', ' ', body).strip()
    return body

def audit_directory(directory: str) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    all_tests = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.rs'):
                path = os.path.join(root, file)
                tests = extract_test_functions(path)
                all_tests.extend(tests)

    # Group by normalized hash
    hash_groups = defaultdict(list)
    for t in all_tests:
        norm = normalize_body(t['body'])
        h = hashlib.sha256(norm.encode('utf-8')).hexdigest()
        t['norm_hash'] = h
        hash_groups[h].append(t)

    dupes = {h: grp for h, grp in hash_groups.items() if len(grp) >= 2}
    return all_tests, dupes

def generate_markdown_report(all_tests: List[Dict], dupes: Dict[str, List[Dict]], target_dir: str) -> str:
    total_tests = len(all_tests)
    dupe_groups = len(dupes)
    dupe_functions = sum(len(grp) for grp in dupes.values())
    redundant_count = dupe_functions - dupe_groups

    # Group by file
    file_dupes = defaultdict(int)
    file_total = defaultdict(int)
    for t in all_tests:
        file_total[t['file']] += 1
    for grp in dupes.values():
        for t in grp:
            file_dupes[t['file']] += 1

    md = []
    md.append('# Test Duplication & Inflation Audit Report')
    md.append('')
    md.append(f'- **Target Directory:** `{target_dir}`')
    md.append(f'- **Total Test Functions Scanned:** {total_tests}')
    md.append(f'- **Duplicate / Template Groups:** {dupe_groups}')
    md.append(f'- **Total Padded / Duplicate Test Functions:** {dupe_functions} ({(dupe_functions / max(total_tests, 1) * 100):.1f}% of total tests)')
    md.append(f'- **Redundant Functions Removable:** {redundant_count}')
    md.append('')
    md.append('## Summary by File')
    md.append('')
    md.append('| File | Total Tests | Duplicated Tests | Redundancy Ratio |')
    md.append('|---|---|---|---|')
    for f in sorted(file_total.keys()):
        tot = file_total[f]
        dup = file_dupes[f]
        ratio = f'{(dup / tot * 100):.1f}%' if tot > 0 else '0.0%'
        md.append(f'| `{f}` | {tot} | {dup} | {ratio} |')

    md.append('')
    md.append('## Top Duplicate Groups')
    md.append('')
    sorted_groups = sorted(dupes.values(), key=lambda g: len(g), reverse=True)
    for idx, grp in enumerate(sorted_groups[:20], 1):
        sample = grp[0]
        md.append(f'### Group {idx}: {len(grp)} identical functions (e.g. `{sample["name"]}` in `{sample["file"]}`)')
        md.append(f'- Files involved: {len(set(t["file"] for t in grp))}')
        md.append(f'- Sample definition (`{sample["file"]}:{sample["start_line"]}`):')
        md.append('```rust')
        snippet = '\n'.join(sample['body'].strip().splitlines()[:15])
        if len(sample['body'].strip().splitlines()) > 15:
            snippet += '\n// ... (truncated)'
        md.append(snippet)
        md.append('```')
        md.append('')

    return '\n'.join(md)

def main():
    target_dir = 'crates/brain-core/src'
    md_output = None
    check_mode = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--markdown' and i + 1 < len(args):
            md_output = args[i + 1]
            i += 2
        elif args[i] == '--check':
            check_mode = True
            i += 1
        else:
            target_dir = args[i]
            i += 1

    all_tests, dupes = audit_directory(target_dir)
    total_tests = len(all_tests)
    dupe_groups = len(dupes)
    dupe_functions = sum(len(grp) for grp in dupes.values())

    report = generate_markdown_report(all_tests, dupes, target_dir)

    if md_output:
        os.makedirs(os.path.dirname(os.path.abspath(md_output)), exist_ok=True)
        with open(md_output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'Wrote report to {md_output}')
    else:
        print(report)

    print(f'\n[Summary] Total tests: {total_tests}, Duplicate groups: {dupe_groups}, Duplicate functions: {dupe_functions}')

    if check_mode and dupe_groups > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
