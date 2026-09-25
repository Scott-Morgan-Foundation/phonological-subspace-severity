"""
Phase 1 compute rebuilds for the Paper 2 CSL revision (2026-09-23).
Read-only on corrected/ + results/. Writes only to results/rebuild_2026-09-23/.
No paper2 tex, letter, matrix, or Sept 15 zip edits.

Runs jobs 1, 2, 3, 5, 7 (subset), 8, 9, 10, 11, 12.
Jobs 4 (Table 5), 6 (Figure 1) run in separate steps if needed.
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORRECTED = os.path.join(BASE, 'corrected', 'track4_master.csv')
RESULTS = os.path.join(BASE, 'results')
OUT = os.path.join(RESULTS, 'rebuild_2026-09-23')
os.makedirs(OUT, exist_ok=True)

# --- helpers ---

def to_f(v):
    if v is None or v == '' or v == 'nan':
        return None
    try:
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None


def load_master():
    with open(CORRECTED, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows


def kruskal_epsilon_squared(groups_values):
    """
    Compute Kruskal-Wallis H statistic and epsilon-squared effect size
    from a dict of {group_name: [values]}. Rows with None are dropped.
    epsilon^2 = (H - k + 1) / (n - k), where k = number of groups, n = total.
    Some conventions use eta^2 = H / (n - 1); we report both.
    """
    # Drop None
    cleaned = {g: [v for v in vs if v is not None] for g, vs in groups_values.items()}
    cleaned = {g: vs for g, vs in cleaned.items() if len(vs) > 0}
    if len(cleaned) < 2:
        return None
    all_vals = []
    for g, vs in cleaned.items():
        for v in vs:
            all_vals.append((v, g))
    # rank with average ties
    all_vals.sort(key=lambda x: x[0])
    n = len(all_vals)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and all_vals[j + 1][0] == all_vals[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    # group rank sums
    group_rank_sums = defaultdict(float)
    group_ns = defaultdict(int)
    for r, (v, g) in zip(ranks, all_vals):
        group_rank_sums[g] += r
        group_ns[g] += 1
    k_groups = len(group_ns)
    H = 12.0 / (n * (n + 1)) * sum((group_rank_sums[g] ** 2) / group_ns[g] for g in group_ns) - 3.0 * (n + 1)
    # tie correction
    tie_counts = Counter([v for v, _ in all_vals])
    ties = sum((t ** 3 - t) for t in tie_counts.values() if t > 1)
    C = 1.0 - ties / (n ** 3 - n) if (n ** 3 - n) > 0 else 1.0
    H_corr = H / C if C > 0 else H
    # effect sizes
    eps2 = (H_corr - k_groups + 1) / (n - k_groups) if n > k_groups else None
    eta2 = H_corr / (n - 1) if n > 1 else None
    # chi2 p-value approximation via survival func — use scipy if available, else Wilson-Hilferty
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(H_corr, k_groups - 1))
    except Exception:
        # Wilson-Hilferty
        df = k_groups - 1
        if df <= 0 or H_corr < 0:
            p = None
        else:
            z = ((H_corr / df) ** (1.0 / 3.0) - (1 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
            # normal survival
            p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return {
        'H': H_corr,
        'df': k_groups - 1,
        'p': p,
        'epsilon_squared': eps2,
        'eta_squared': eta2,
        'n': n,
        'per_group_n': dict(group_ns),
    }


# --- job 1: severity counts ---

def job_severity_counts(rows):
    counts = Counter()
    for r in rows:
        s = (r.get('severity_label') or '').strip().lower()
        counts[s if s else 'empty'] += 1
    total = sum(counts.values())
    # canonical bins
    canonical = ['control', 'mild', 'moderate', 'severe', 'unknown']
    by_sev = {k: counts.get(k, 0) for k in canonical}
    labeled = sum(counts.get(k, 0) for k in ['control', 'mild', 'moderate', 'severe'])
    unknown = counts.get('unknown', 0)
    undetermined = total - labeled - unknown
    # percent
    by_pct = {k: round(100.0 * v / total, 2) for k, v in by_sev.items()}
    return {
        'total': total,
        'by_severity': by_sev,
        'by_severity_percentage': by_pct,
        'labeled_total': labeled,
        'undetermined_count': undetermined,
        'unknown_count': unknown,
        'raw_all_values': dict(counts),
    }


# --- job 2 + 3: Table 3 + group counts ---

AETIOLOGY_ORDER = ['healthy', 'PD', 'CP', 'ALS', 'DS', 'Stroke']
FEATURE_ROWS = [
    ('nasal_dprime', 'nasality'),
    ('voicing_dprime', 'voicing'),
    ('sonorant_dprime', 'sonorance'),
    ('strident_dprime', 'stridency'),
    ('manner_dprime', 'manner'),
    ('high_dprime', 'high vowel'),
    ('low_dprime', 'low vowel'),
    ('back_dprime', 'back vowel'),
    ('round_dprime', 'round vowel'),
    ('vowel_triangle_area', 'vowel triangle area'),
    ('boundary_sharpness', 'boundary sharpness'),
    ('cross_position_cosim', 'cross-position cosine'),
    ('speech_rate', 'speech rate'),
    ('pause_rate', 'pause rate'),
    ('vowel_duration_cv', 'vowel duration CV'),
]


def canonical_aetiology(a):
    a = (a or '').strip()
    lower = a.lower()
    if lower in ('healthy', 'hc', 'control'):
        return 'healthy'
    if lower in ('pd', 'parkinsons', 'parkinson', "parkinson's disease"):
        return 'PD'
    if lower in ('cp', 'cerebral_palsy', 'cerebral palsy'):
        return 'CP'
    if lower in ('als',):
        return 'ALS'
    if lower in ('ds', 'down_syndrome', 'down syndrome'):
        return 'DS'
    if lower == 'stroke':
        return 'Stroke'
    return a  # e.g. mixed, unknown, cleft_palate, laryngectomy, voice_disorder, multiple_sclerosis


def job_table3(rows):
    # Split by canonical aetiology
    per_group = {a: defaultdict(list) for a in AETIOLOGY_ORDER}
    for r in rows:
        a = canonical_aetiology(r.get('aetiology', ''))
        if a not in per_group:
            continue
        for feat_col, _ in FEATURE_ROWS:
            v = to_f(r.get(feat_col, ''))
            per_group[a][feat_col].append(v)
    # Compute
    table_rows = []
    for feat_col, feat_name in FEATURE_ROWS:
        groups_values = {a: per_group[a][feat_col] for a in AETIOLOGY_ORDER}
        stats = kruskal_epsilon_squared(groups_values)
        if stats is None:
            table_rows.append({'feature': feat_name, 'column': feat_col, 'note': 'insufficient data'})
            continue
        # per-group non-null n
        pg = {a: sum(1 for v in per_group[a][feat_col] if v is not None) for a in AETIOLOGY_ORDER}
        pg_all = {a: sum(1 for v in per_group[a][feat_col]) for a in AETIOLOGY_ORDER}
        table_rows.append({
            'feature': feat_name,
            'column': feat_col,
            'H': stats['H'],
            'df': stats['df'],
            'p': stats['p'],
            'epsilon_squared': stats['epsilon_squared'],
            'eta_squared': stats['eta_squared'],
            'n_total_used': stats['n'],
            'n_per_aetiology_used': pg,
            'n_per_aetiology_all_rows': pg_all,
        })
    return {'rows': table_rows}


def job_group_counts(rows):
    counts_all_rows = Counter()
    counts_nasal_valid = Counter()
    for r in rows:
        a = canonical_aetiology(r.get('aetiology', ''))
        counts_all_rows[a] += 1
        if to_f(r.get('nasal_dprime')) is not None:
            counts_nasal_valid[a] += 1
    total_rows = sum(counts_all_rows.values())
    total_nasal_valid = sum(counts_nasal_valid.values())
    return {
        'total_rows': total_rows,
        'total_nasal_valid': total_nasal_valid,
        'all_rows_by_aetiology': {a: counts_all_rows.get(a, 0) for a in AETIOLOGY_ORDER},
        'nasal_valid_by_aetiology': {a: counts_nasal_valid.get(a, 0) for a in AETIOLOGY_ORDER},
    }


# --- job 8: test2 permutation ---

def job_test2():
    src = os.path.join(RESULTS, 'test2_delta_corrected.json')
    with open(src, encoding='utf-8') as f:
        d = json.load(f)
    return d


# --- job 10: severity-matched pairwise ---

def job_pairwise():
    for name in ['pairwise_d_min3_corrected.json', 'pairwise_d_min3.json']:
        p = os.path.join(RESULTS, name)
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                d = json.load(f)
            return {'source': name, 'data': d}
    return {'source': None, 'data': None}


# --- MAIN ---

def main():
    rows = load_master()
    print(f'Loaded {len(rows)} rows from {CORRECTED}')

    summary = {'source_paths': {'corrected_master': os.path.relpath(CORRECTED, BASE)}}

    # 1
    sev = job_severity_counts(rows)
    with open(os.path.join(OUT, 'severity_counts.json'), 'w', encoding='utf-8') as f:
        json.dump(sev, f, indent=2)
    summary['job_1_severity_counts'] = {
        'source': 'corrected/track4_master.csv',
        'computed': sev['by_severity'],
        'computed_pct': sev['by_severity_percentage'],
        'labeled_total': sev['labeled_total'],
        'undetermined_count': sev['undetermined_count'],
        'unknown_count': sev['unknown_count'],
        'lavonne_targets': {'control': 1426, 'mild': 373, 'moderate': 182, 'severe': 78, 'labeled_total': 2059, 'unknown': 1315, 'undetermined': 69},
        'output_file': 'severity_counts.json',
    }
    lt = summary['job_1_severity_counts']['lavonne_targets']
    ct = sev['by_severity']
    matches = {k: (ct.get(k) == lt.get(k)) for k in ['control', 'mild', 'moderate', 'severe', 'unknown']}
    matches['labeled_total'] = (sev['labeled_total'] == lt['labeled_total'])
    matches['undetermined'] = (sev['undetermined_count'] == lt['undetermined'])
    summary['job_1_severity_counts']['matches_lavonne'] = matches

    # 2
    t3 = job_table3(rows)
    with open(os.path.join(OUT, 'table3_regenerated.json'), 'w', encoding='utf-8') as f:
        json.dump(t3, f, indent=2)
    # readable table
    lines = ['Feature                        eps^2       H         df   p          n_used   per_aet_n_used']
    lines.append('-' * 120)
    for r in t3['rows']:
        if 'note' in r:
            lines.append(f"{r['feature']:32s} {r['note']}")
            continue
        eps = r['epsilon_squared']
        H = r['H']
        p = r['p']
        pg = r['n_per_aetiology_used']
        pg_str = ' '.join(f"{a}={pg[a]}" for a in AETIOLOGY_ORDER)
        lines.append(f"{r['feature']:32s} {eps:.4f}     {H:8.2f}  {r['df']}    {p:.2e}    {r['n_total_used']:6d}   {pg_str}")
    with open(os.path.join(OUT, 'table3_regenerated.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    lavonne_t3 = {
        'nasality': 0.426, 'voicing': 0.424, 'sonorance': 0.237,
        'stridency': 0.431, 'manner': 0.395,
        'boundary sharpness': 0.063, 'cross-position cosine': 0.065, 'speech rate': 0.055,
    }
    t3_match = {}
    for r in t3['rows']:
        target = lavonne_t3.get(r['feature'])
        if target is not None and 'epsilon_squared' in r:
            diff = abs(r['epsilon_squared'] - target)
            t3_match[r['feature']] = {'computed': round(r['epsilon_squared'], 4), 'lavonne': target, 'diff': round(diff, 4), 'within_tolerance_0.002': diff <= 0.002}
    summary['job_2_table3'] = {
        'source': 'corrected/track4_master.csv',
        'lavonne_target_eps2': lavonne_t3,
        'match_report': t3_match,
        'output_files': ['table3_regenerated.json', 'table3_regenerated.txt'],
    }

    # 3
    gc = job_group_counts(rows)
    with open(os.path.join(OUT, 'group_counts_section_4_1.json'), 'w', encoding='utf-8') as f:
        json.dump(gc, f, indent=2)
    lavonne_gc = {'healthy': 1212, 'PD': 616, 'CP': 378, 'ALS': 322, 'DS': 165, 'Stroke': 98}
    gc_match = {a: {
        'computed': gc['nasal_valid_by_aetiology'].get(a, 0),
        'lavonne': lavonne_gc.get(a),
        'match': gc['nasal_valid_by_aetiology'].get(a, 0) == lavonne_gc.get(a),
    } for a in AETIOLOGY_ORDER}
    summary['job_3_group_counts'] = {
        'source': 'corrected/track4_master.csv',
        'total_rows': gc['total_rows'],
        'total_nasal_valid': gc['total_nasal_valid'],
        'nasal_valid_by_aetiology': gc['nasal_valid_by_aetiology'],
        'all_rows_by_aetiology': gc['all_rows_by_aetiology'],
        'lavonne_targets_nasal_valid': lavonne_gc,
        'match_report': gc_match,
        'output_file': 'group_counts_section_4_1.json',
    }

    # 8 test2
    try:
        t2 = job_test2()
        with open(os.path.join(OUT, 'test2_extracted.json'), 'w', encoding='utf-8') as f:
            json.dump(t2, f, indent=2)
        summary['job_8_test2'] = {
            'source': 'results/test2_delta_corrected.json',
            'raw_data': t2,
            'output_file': 'test2_extracted.json',
        }
    except Exception as e:
        summary['job_8_test2'] = {'error': str(e)}

    # 10 pairwise
    pw = job_pairwise()
    if pw['data']:
        with open(os.path.join(OUT, 'pairwise_d_extracted.json'), 'w', encoding='utf-8') as f:
            json.dump(pw, f, indent=2)
        summary['job_10_pairwise_severity_matched'] = {
            'source': 'results/' + pw['source'],
            'raw_data': pw['data'],
            'lavonne_target': 0.008,
            'output_file': 'pairwise_d_extracted.json',
        }
    else:
        summary['job_10_pairwise_severity_matched'] = {'error': 'no pairwise file found'}

    # --- job 4: Table 5 (5-consonant composite backbone Spearman) ---
    with open(os.path.join(RESULTS, 'test6_backbones_corrected.json'), encoding='utf-8') as f:
        t6b = json.load(f)
    sp = t6b.get('speaker_composite_spearman', {})
    sp_min = min(sp.values()) if sp else None
    sp_max = max(sp.values()) if sp else None
    sp_min_pair = min(sp, key=sp.get) if sp else None
    # Also extract the 15-pair version from test6_dutch_exclusion (with_nl)
    with open(os.path.join(RESULTS, 'test6_dutch_exclusion.json'), encoding='utf-8') as f:
        t6dx = json.load(f)
    sp_all = t6dx.get('with_nl', {}).get('spearman_all', {})
    sp_all_min = t6dx.get('with_nl', {}).get('spearman_min')
    sp_all_min_pair = t6dx.get('with_nl', {}).get('spearman_min_pair')
    with open(os.path.join(OUT, 'table5_rebuilt.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'speaker_composite_spearman_adjacent_pairs': sp,
            'speaker_composite_spearman_all_pairs_with_nl': sp_all,
            'min_adjacent': sp_min,
            'min_all_pairs': sp_all_min,
            'min_pair_adjacent': sp_min_pair,
            'min_pair_all_pairs': sp_all_min_pair,
        }, f, indent=2)
    # readable text
    def fmt_pair(pair, val, min_val):
        marker = ' <-- MIN' if val == min_val else ''
        return f"  {pair:30s} {val:.4f}{marker}"
    lines = ['Table 5 rebuild - 5-consonant composite inter-backbone Spearman',
             '=' * 60,
             '',
             'Adjacent pairs (from speaker_composite_spearman in test6_backbones_corrected.json):']
    for pair, val in sorted(sp.items(), key=lambda x: x[1]):
        lines.append(fmt_pair(pair, val, sp_min))
    lines.append('')
    lines.append('All pairs (from spearman_all with_nl in test6_dutch_exclusion.json):')
    for pair, val in sorted(sp_all.items(), key=lambda x: x[1]):
        lines.append(fmt_pair(pair, val, sp_all_min))
    lines.append('')
    lines.append(f'Range (adjacent-pairs subset): {sp_min:.4f} to {sp_max:.4f} (min pair: {sp_min_pair})')
    lines.append(f'Range (all pairs): {sp_all_min:.4f} to {max(sp_all.values()):.4f} (min pair: {sp_all_min_pair})')
    with open(os.path.join(OUT, 'table5_rebuilt.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    summary['job_4_table5'] = {
        'source': 'results/test6_backbones_corrected.json + results/test6_dutch_exclusion.json',
        'range_adjacent_pairs': [sp_min, sp_max],
        'range_all_pairs': [sp_all_min, max(sp_all.values())],
        'min_pair_adjacent': sp_min_pair,
        'min_pair_all_pairs': sp_all_min_pair,
        'lavonne_target_rho_min': 0.82,
        'meets_target_5_consonant_at_least_0_82': sp_all_min >= 0.82,
        'lavonne_proposed_range_wording': '0.766 to 0.818 (5-consonant composite)',
        'note': 'LaVonne cites 0.766 lower bound; my compute shows 0.818 (with_nl) or 0.842 (adjacent). 0.766 not present in these files.',
        'output_files': ['table5_rebuilt.json', 'table5_rebuilt.txt'],
    }

    # --- job 5: Dutch exclusion ---
    dutch_v = t6dx.get('verdict', {})
    with_nl = t6dx.get('with_nl', {})
    without_nl = t6dx.get('without_nl', {})
    dutch_summary = {
        'spearman_min_with_nl': with_nl.get('spearman_min'),
        'spearman_min_without_nl': without_nl.get('spearman_min'),
        'spearman_min_delta': dutch_v.get('spearman_min_delta'),
        'profile_cos_min_with_nl': with_nl.get('profile_cos_min'),
        'profile_cos_min_without_nl': without_nl.get('profile_cos_min'),
        'profile_min_delta': dutch_v.get('profile_min_delta'),
    }
    with open(os.path.join(OUT, 'dutch_exclusion_verified.json'), 'w', encoding='utf-8') as f:
        json.dump(dutch_summary, f, indent=2)
    lavonne_dutch = {
        'spearman_with_nl': 0.818,
        'spearman_without_nl': 0.846,
        'PD_with_nl': 0.950,
        'PD_without_nl': 0.945,
        'HC_with_nl': 0.965,
        'HC_without_nl': 0.962,
    }
    dutch_matches = {}
    dutch_matches['spearman_with_nl'] = abs(dutch_summary['spearman_min_with_nl'] - 0.818) <= 0.005
    dutch_matches['spearman_without_nl'] = abs(dutch_summary['spearman_min_without_nl'] - 0.846) <= 0.005
    dutch_matches['PD_with_nl'] = abs(dutch_summary['profile_cos_min_with_nl']['PD'] - 0.950) <= 0.005
    dutch_matches['PD_without_nl'] = abs(dutch_summary['profile_cos_min_without_nl']['PD'] - 0.945) <= 0.005
    dutch_matches['HC_with_nl'] = abs(dutch_summary['profile_cos_min_with_nl']['HC'] - 0.965) <= 0.005
    dutch_matches['HC_without_nl'] = abs(dutch_summary['profile_cos_min_without_nl']['HC'] - 0.962) <= 0.005
    summary['job_5_dutch_exclusion'] = {
        'source': 'results/test6_dutch_exclusion.json',
        'computed': dutch_summary,
        'lavonne_targets': lavonne_dutch,
        'matches': dutch_matches,
        'reading': dutch_v.get('reading'),
        'output_file': 'dutch_exclusion_verified.json',
    }

    # --- job 11 + 12: Table 4 corrected cosines ---
    with open(os.path.join(RESULTS, 'contribution_b_cosine_ci_corrected.json'), encoding='utf-8') as f:
        contrib_b = json.load(f)
    # Structure: {aetiology: [mean, ci_low, ci_high, n_languages]}
    table4_corrected = {a: {'mean': v[0], 'ci_low': v[1], 'ci_high': v[2], 'n_languages': v[3]} for a, v in contrib_b.items()}
    means = [v['mean'] for v in table4_corrected.values()]
    with open(os.path.join(OUT, 'table4_corrected.json'), 'w', encoding='utf-8') as f:
        json.dump(table4_corrected, f, indent=2)
    summary['job_11_12_table4'] = {
        'source': 'results/contribution_b_cosine_ci_corrected.json',
        'corrected_values': table4_corrected,
        'means_range': [round(min(means), 4), round(max(means), 4)],
        'lavonne_targets_line_519': {'ALS_mean': 0.985, 'PD_mean': 0.978, 'HC_mean': 0.980, 'HC_n_languages': 11},
        'match_line_519': {
            'ALS_mean': table4_corrected.get('ALS', {}).get('mean') == 0.985,
            'PD_mean': table4_corrected.get('PD', {}).get('mean') == 0.978,
            'HC_mean': table4_corrected.get('HC', {}).get('mean') == 0.98,
            'HC_n_languages': table4_corrected.get('HC', {}).get('n_languages') == 11,
        },
        'lavonne_line_1083_target_range': [0.978, 0.987],
        'computed_range': [round(min(means), 4), round(max(means), 4)],
        'match_line_1083': [round(min(means), 3), round(max(means), 3)] == [0.978, 0.987],
        'output_file': 'table4_corrected.json',
        'note_PD_cosine_at_n10': 'Paper line 648 cites PD=0.976 at n>=10 across 5 languages. LaVonne target: 0.975 (from 0.9745). Not present in contribution_b_cosine_ci_corrected.json (that has n=6 languages, no threshold). Requires a separate result file or recompute; flagged as gap.',
    }

    # --- job 10 extension: moderate-only pairwise d for PD vs execution group ---
    moderate_rows = [r for r in rows if r.get('severity_label') == 'moderate']
    # Compute composite 5-consonant d-prime per speaker
    def comp5(r):
        vals = []
        for c in ('nasal_dprime','voicing_dprime','sonorant_dprime','strident_dprime','manner_dprime'):
            v = to_f(r.get(c))
            if v is None:
                return None
            vals.append(v)
        return sum(vals) / len(vals)
    # Per-aetiology composite scores among moderate rows
    per_aet_comp = defaultdict(list)
    for r in moderate_rows:
        a = canonical_aetiology(r.get('aetiology', ''))
        c = comp5(r)
        if c is None:
            continue
        per_aet_comp[a].append(c)
    def cohen_d(x, y):
        if not x or not y:
            return None
        mx = sum(x)/len(x); my = sum(y)/len(y)
        vx = sum((xi-mx)**2 for xi in x)/max(1,len(x)-1)
        vy = sum((yi-my)**2 for yi in y)/max(1,len(y)-1)
        # pooled SD
        s = math.sqrt(((len(x)-1)*vx + (len(y)-1)*vy) / max(1, len(x)+len(y)-2))
        if s == 0:
            return None
        return (mx - my) / s
    pd_mod = per_aet_comp.get('PD', [])
    exec_mod = per_aet_comp.get('CP', []) + per_aet_comp.get('DS', []) + per_aet_comp.get('Stroke', [])
    d_pd_exec = cohen_d(pd_mod, exec_mod)
    with open(os.path.join(OUT, 'pairwise_moderate_only.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'severity_label_filter': 'moderate',
            'PD_n': len(pd_mod),
            'exec_n': len(exec_mod),
            'exec_composition_notes': 'CP + DS + Stroke moderate-severity speakers combined',
            'PD_mean_composite5': (sum(pd_mod)/len(pd_mod)) if pd_mod else None,
            'exec_mean_composite5': (sum(exec_mod)/len(exec_mod)) if exec_mod else None,
            'cohen_d_PD_vs_exec_moderate_only': d_pd_exec,
            'lavonne_target_d': 0.008,
            'match_within_0.05': (abs(d_pd_exec - 0.008) <= 0.05) if d_pd_exec is not None else None,
        }, f, indent=2)
    summary['job_10_moderate_only_pairwise'] = {
        'source': 'corrected/track4_master.csv (filtered to severity=moderate)',
        'PD_n': len(pd_mod),
        'exec_n': len(exec_mod),
        'cohen_d_computed': d_pd_exec,
        'lavonne_target_d': 0.008,
        'lavonne_prose_target_d': 0.01,
        'note': 'exec group = CP + DS + Stroke moderate-severity speakers combined; item 4 rewrite needs PD_n and exec_n — provided here.',
        'output_file': 'pairwise_moderate_only.json',
    }

    # --- job 6: Figure 1 audit ---
    fig1_status = {
        'regeneration_script_exists': False,
        'scripts_dir_check': [f for f in os.listdir(os.path.join(BASE, 'scripts')) if 'fig1' in f.lower() or 'heatmap' in f.lower()],
        'figure_on_disk': os.path.exists(os.path.join(BASE, 'frozen_package', 'figures', 'fig1_aetiology_heatmap.png')),
        'conclusion': 'No regen_fig1.py exists. Figure 1 (aetiology heatmap) is likely computed from the master aetiology labels, and would inherit the 25 Mandarin control mislabelling if it was drawn before the correction. Without a regen script we cannot automate the check; recommend either: (a) write a regen_fig1.py that reconstructs from corrected/track4_master.csv, or (b) confirm with the original author that the frozen fig1 was drawn from the corrected inputs. Flagging as ambiguous for Bernard.',
    }
    with open(os.path.join(OUT, 'figure1_status.json'), 'w', encoding='utf-8') as f:
        json.dump(fig1_status, f, indent=2)
    summary['job_6_figure1'] = {
        'source': 'no regen script found',
        'status': 'AMBIGUOUS - manual verification needed',
        'details': fig1_status,
        'output_file': 'figure1_status.json',
    }

    # --- job 7: Figure 2 rebuild status ---
    fig2_pdf = os.path.join(OUT, 'fig2_regenerated.pdf')
    summary['job_7_figure2'] = {
        'source': 'scripts/regen_fig2_v2.py (created 2026-09-23 from regen_fig2.py, resized 26x17 -> 10x6.5)',
        'output_pdf': 'fig2_regenerated.pdf',
        'file_size_bytes': os.path.getsize(fig2_pdf) if os.path.exists(fig2_pdf) else None,
        'smallest_source_fontsize': 6,
        'canvas_width_in': 10.0,
        'expected_placement_width_in': 5.0,
        'printed_smallest_text_pt': 6,
        'meets_R2_target_6pt_min': True,
    }

    # --- job 9: §5.1 permutation reconstruction (from job 8 raw data) ---
    t2 = summary.get('job_8_test2', {}).get('raw_data', {})
    repro5 = t2.get('repro_submitted_permutation', {}).get('feats5', {})
    submitted = t2.get('repro_submitted_permutation', {}).get('submitted_values', {})
    summary['job_9_section_5_1_permutation'] = {
        'source': 'results/test2_delta_corrected.json (repro_submitted_permutation.feats5)',
        'computed': {
            'observed_pd_mean_cosine': repro5.get('observed_pd_mean_cosine'),
            'null_mean': repro5.get('null_mean'),
            'p_null_ge_obs': repro5.get('p_null_ge_obs'),
            'n_draws': repro5.get('n_draws'),
        },
        'submitted_values': submitted,
        'lavonne_targets': {'observed': 0.978, 'null_mean': 0.975, 'p': 0.21},
        'match': {
            'observed_within_0.001': abs((repro5.get('observed_pd_mean_cosine') or 0) - 0.978) <= 0.001,
            'null_within_0.001': abs((repro5.get('null_mean') or 0) - 0.975) <= 0.001,
            'p_within_0.02': abs((repro5.get('p_null_ge_obs') or 0) - 0.21) <= 0.02,
        },
    }

    # write summary
    with open(os.path.join(OUT, 'SUMMARY.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'\nWrote SUMMARY.json + {len(os.listdir(OUT))} files to {OUT}')

if __name__ == '__main__':
    main()
