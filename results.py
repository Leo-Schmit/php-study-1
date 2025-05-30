import os
import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

with open('combined_repo_data.json', 'r', encoding='utf-8') as f:
    data_ = json.load(f)

required_fields = [
    'lines_of_code',
    'phpstan_results',
    'exists_analyzer',
    'stars',
    'github_bugs',
    'phan_results',
    'psalm_results'
]

def is_valid_analysis_field(field_data):
    if not isinstance(field_data, dict) or not field_data:
        return False
    return all(isinstance(k, str) and isinstance(v, int) for k, v in field_data.items())

data = {}
for repo, repo_data in data_.items():
    if all(field in repo_data and repo_data[field] not in (None, {}, []) for field in required_fields):
        if all(is_valid_analysis_field(repo_data[field]) for field in ['phpstan_results', 'phan_results', 'psalm_results']):
            data[repo] = repo_data

os.makedirs('results', exist_ok=True)

def filter_by_stars(subset, min_stars=30):
    return [r for r in subset if data[r]['stars'] >= min_stars]

repos = list(data.keys())
has_analyzer = [r for r in repos if data[r]['exists_analyzer']]
no_analyzer  = [r for r in repos if not data[r]['exists_analyzer']]

def build_density_df(results_key):
    all_types = sorted({ wt for r in repos for wt in data[r].get(results_key, {}) })

    rows = []
    for repo in repos:
        loc = data[repo]['lines_of_code']
        scale = loc / 1000
        row = {'repo': repo, 'has_analyzer': data[repo]['exists_analyzer']}
        for wt in all_types:
            count = data[repo].get(results_key, {}).get(wt, 0)
            row[wt] = count / scale
        stars = data[repo]['stars']
        row['bug_per_stars'] = data[repo]['github_bugs'] / stars if stars > 0 else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).set_index('repo')
    for wt in all_types:
        q99 = df[wt].quantile(0.99)
        df[wt] = df[wt].clip(upper=q99)
    return df, all_types

def analyze_subset(df_dens, all_types, subset, label):

    df_sub = df_dens.loc[subset].copy()
    results = []

    for wt in all_types:
        x = df_sub[wt]
        y = df_sub['bug_per_stars']

        if x.std() == 0 or y.std() == 0:
            continue

        rho, pval = spearmanr(x, y)
        results.append({'warning_type': wt, 'rho': rho, 'p_value': pval})

    df_results = pd.DataFrame(results)

    corrected = multipletests(df_results['p_value'], method='fdr_bh')
    df_results['p_value_corrected'] = corrected[1]
    df_results['rho'] = df_results['rho'].round(3)

    sig = df_results[(df_results['p_value_corrected'] < 0.05) & (df_results['rho'].abs() >= 0.35)].copy()
    sig = sig.reindex(sig['rho'].abs().sort_values(ascending=False).index)

    fmt = lambda p: 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    sig['p_value'] = sig['p_value_corrected'].apply(fmt)
    df_results['p_value'] = df_results['p_value_corrected'].apply(fmt)

    sig[['warning_type', 'rho', 'p_value']].to_csv(f"results/{label}_significant_corr.csv", index=False)

def balance_samples(group1, group2):
    if len(group1) > len(group2):
        group1 = group1[:len(group2)]
    elif len(group2) > len(group1):
        group2 = group2[:len(group1)]
    return group1, group2

def compute_stats(subset_with, subset_without):
    stats = {True: {'lines_of_code': 0, 'stars': 0, 'github_bugs': 0, 'repos': 0},
             False:{'lines_of_code': 0, 'stars': 0, 'github_bugs': 0, 'repos': 0}}

    for repo in subset_with:
        rd = data[repo]
        stats[True]['lines_of_code'] += rd['lines_of_code']
        stats[True]['stars'] += rd['stars']
        stats[True]['github_bugs'] += rd['github_bugs']
        stats[True]['repos'] += 1
    for repo in subset_without:
        rd = data[repo]
        stats[False]['lines_of_code'] += rd['lines_of_code']
        stats[False]['stars'] += rd['stars']
        stats[False]['github_bugs'] += rd['github_bugs']
        stats[False]['repos'] += 1

    def safe_div(a, b): return a / b if b else 0
    def fmt_int(n): return f"{int(n):,}"
    def fmt_float(n, digits=2): return f"{n:.{digits}f}"

    table = [
        ("Number of Repos", stats[True]['repos'], stats[False]['repos']),
        ("Total Lines of Code", fmt_int(stats[True]['lines_of_code']), fmt_int(stats[False]['lines_of_code'])),
        ("Total Stars", fmt_int(stats[True]['stars']), fmt_int(stats[False]['stars'])),
        ("Total GitHub Bugs", fmt_int(stats[True]['github_bugs']), fmt_int(stats[False]['github_bugs'])),
        ("GitHub Bugs per 1000 LOC",
         fmt_float(safe_div(stats[True]['github_bugs'] * 1000, stats[True]['lines_of_code'])),
         fmt_float(safe_div(stats[False]['github_bugs'] * 1000, stats[False]['lines_of_code']))),
        ("GitHub Bugs per 1 Star",
         fmt_float(safe_div(stats[True]['github_bugs'], stats[True]['stars']),4),
         fmt_float(safe_div(stats[False]['github_bugs'], stats[False]['stars']),4)),
        ("Avg. Lines of Code",
         fmt_int(safe_div(stats[True]['lines_of_code'], stats[True]['repos'])),
         fmt_int(safe_div(stats[False]['lines_of_code'], stats[False]['repos']))),
        ("Avg. Stars",
         fmt_int(safe_div(stats[True]['stars'], stats[True]['repos'])),
         fmt_int(safe_div(stats[False]['stars'], stats[False]['repos']))),
        ("Avg. GitHub Bugs",
         fmt_int(safe_div(stats[True]['github_bugs'], stats[True]['repos'])),
         fmt_int(safe_div(stats[False]['github_bugs'], stats[False]['repos'])))
    ]

    print(f"\n=== Stats ===")
    print(f"{'Name':<30} | {'With Analyzer':<17} | {'Without Analyzer'}")
    print("-" * 70)
    for name, w, wo in table:
        print(f"{name:<30} | {w:<17} | {wo}")

for analyzer in ['phpstan_results', 'phan_results', 'psalm_results']:
    print(f"\n=== {analyzer} ===")
    df_dens, types = build_density_df(analyzer)

    repos_filtered        = filter_by_stars(repos)
    has_analyzer_filtered = filter_by_stars(has_analyzer)
    no_analyzer_filtered  = filter_by_stars(no_analyzer)

    has_analyzer_balanced, no_analyzer_balanced = balance_samples(has_analyzer_filtered, no_analyzer_filtered)

    analyze_subset(df_dens, types, has_analyzer_balanced, f"{analyzer}_with_analizer")
    analyze_subset(df_dens, types, no_analyzer_balanced,  f"{analyzer}_without_analizer")

compute_stats(has_analyzer_balanced, no_analyzer_balanced)