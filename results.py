import os
import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--min_r_s', type=float, default=0.35, help='Min rho')
args = parser.parse_args()

min_r_s = args.min_r_s

def build_density_df(results_key, data, repos):
    all_types = sorted(
        {wt for r in repos for wt in data[r].get(results_key, {})})
    rows = []
    for repo in repos:
        loc = data[repo]['lines_of_code']
        scale = loc / 1000
        row = {
            'repo': repo,
            'has_analyzer': len(data[repo]['exists_analyzer']) > 0
        }
        for wt in all_types:
            count = data[repo].get(results_key, {}).get(wt, 0)
            row[wt] = count / scale
        stars = data[repo]['stars']
        row['bug_per_stars'] = data[repo]['github_bugs'] / \
            stars if stars > 0 else np.nan
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
    sig = df_results[(df_results['p_value_corrected'] < 0.05)
                     & (df_results['rho'].abs() >= min_r_s)].copy()
    sig = sig.reindex(sig['rho'].abs().sort_values(ascending=False).index)
    def fmt(p): return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    sig['p_value'] = sig['p_value_corrected'].apply(fmt)
    df_results['p_value'] = df_results['p_value_corrected'].apply(fmt)
    sig[['warning_type', 'rho', 'p_value']].to_csv(
        f"results/{label}.csv", index=False)


def write_stats_html(subset_with, subset_without, data, out_file):
    stats = {
        True: {'lines_of_code': 0, 'stars': 0, 'github_bugs': 0, 'repos': 0},
        False: {'lines_of_code': 0, 'stars': 0, 'github_bugs': 0, 'repos': 0}
    }
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
        ("Total Lines of Code", fmt_int(stats[True]['lines_of_code']), fmt_int(
            stats[False]['lines_of_code'])),
        ("Total Stars", fmt_int(stats[True]['stars']), fmt_int(
            stats[False]['stars'])),
        ("Total GitHub Bugs", fmt_int(stats[True]['github_bugs']), fmt_int(
            stats[False]['github_bugs'])),
        ("GitHub Bugs per 1000 LOC",
            fmt_float(safe_div(stats[True]['github_bugs']
                      * 1000, stats[True]['lines_of_code'])),
            fmt_float(safe_div(stats[False]['github_bugs'] * 1000, stats[False]['lines_of_code']))),
        ("Avg. Lines of Code",
            fmt_int(
                safe_div(stats[True]['lines_of_code'], stats[True]['repos'])),
            fmt_int(safe_div(stats[False]['lines_of_code'], stats[False]['repos']))),
        ("Avg. Stars",
            fmt_int(safe_div(stats[True]['stars'], stats[True]['repos'])),
            fmt_int(safe_div(stats[False]['stars'], stats[False]['repos']))),
        ("Avg. GitHub Bugs",
            fmt_int(safe_div(stats[True]['github_bugs'],
                    stats[True]['repos'])),
            fmt_int(safe_div(stats[False]['github_bugs'], stats[False]['repos'])))
    ]
    html = "<h1>Stats</h1>\n"
    html += "<table сellpadding='5' cellspacing='0'>\n"
    html += "  <tr><th>Name</th><th>With Analyzer</th><th>Without Analyzer</th></tr>\n"
    for name, w, wo in table:
        html += f"  <tr><td>{name}</td><td>{w}</td><td>{wo}</td></tr>\n"
    html += "</table>\n"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write("\n<hr/>\n")
        f.write(html)


def process_correlation(analyzer_key, repos, data, has_analyzer, no_analyzer):
    df_dens, types = build_density_df(analyzer_key, data, repos)
    analyze_subset(df_dens, types, has_analyzer,
                   f"{analyzer_key}_with_analizer")
    analyze_subset(df_dens, types, no_analyzer,
                   f"{analyzer_key}_without_analizer")


def plot_pie_chart(labels, sizes, filename, dpi=300, fig_size=(8, 8),
                   autopct='%.1f%%', startangle=140, edgecolor='white',
                   label_fontsize=16, autotext_fontsize=15, bbox_inches=None):
    fig, ax = plt.subplots(figsize=fig_size)
    wedge_props = {'edgecolor': edgecolor} if edgecolor else None
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels,
        autopct=autopct,
        startangle=startangle,
        wedgeprops=wedge_props
    )
    for text in texts:
        text.set_fontsize(label_fontsize)
    for at in autotexts:
        at.set_fontsize(autotext_fontsize)
    ax.axis('equal')
    fig.tight_layout()
    if bbox_inches:
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    else:
        fig.savefig(filename, dpi=dpi)
    plt.close(fig)


def generate_faceted_plots(analyzers, base_dir, width, dpi, fig_width, inch_per_warning, min_fig_height):
    for analyzer in analyzers:
        file_with = os.path.join(
            base_dir, f"{analyzer}_results_with_analizer.csv")
        file_without = os.path.join(
            base_dir, f"{analyzer}_results_without_analizer.csv")
        json_file = os.path.join(base_dir, f"{analyzer}_cat.json")
        df_with = pd.read_csv(file_with)
        df_without = pd.read_csv(file_without)
        warn2cat = json.load(open(json_file, encoding="utf-8"))
        for df in (df_with, df_without):
            df["category"] = df["warning_type"].map(warn2cat).fillna("Other")
        categories = df_with["category"].unique()
        num_cats = len(categories)
        warnings_per_cat = []
        for cat in categories:
            sub_w = df_with[df_with.category == cat]
            sub_wo = df_without[df_without.category == cat]
            warns = set(sub_w.warning_type) | set(sub_wo.warning_type)
            warnings_per_cat.append(len(warns) if warns else 1)
        fig_height = max(min_fig_height, sum(
            warnings_per_cat) * inch_per_warning + 1)
        fig, axes = plt.subplots(
            nrows=num_cats, ncols=1,
            figsize=(fig_width, fig_height), dpi=dpi,
            sharex=True,
            gridspec_kw={'height_ratios': warnings_per_cat}
        )
        for ax, cat in zip(axes, categories):
            sub_w = df_with[df_with.category == cat]
            sub_wo = df_without[df_without.category == cat]
            warns = sorted(set(sub_w.warning_type) | set(sub_wo.warning_type))
            y = np.arange(len(warns))
            rho_w = [sub_w.loc[sub_w.warning_type == w, "rho"].mean(
            ) if w in sub_w.warning_type.values else 0 for w in warns]
            rho_wo = [sub_wo.loc[sub_wo.warning_type == w, "rho"].mean(
            ) if w in sub_wo.warning_type.values else 0 for w in warns]
            ax.barh(y + width/2, rho_w, height=width, label="with analyzer")
            ax.barh(y - width/2, rho_wo, height=width,
                    label="without analyzer")
            ax.set_yticks([])
            ax.set_title(cat)
            ax.invert_yaxis()
            ax.grid(axis="x", linestyle="--", linewidth=0.5)
            ax.axvline(0.4, color="gray", linestyle="--", linewidth=1)
        axes[-1].set_xlabel("r_s")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   ncol=2, bbox_to_anchor=(0.5, 1.02))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(
            base_dir, f"{analyzer}_by_category_faceted.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)


def classify_evans(rho):
    if rho < 0.20:
        return "very weak"
    if rho < 0.35:
        return "weak"
    if rho < 0.40:
        return "near-moderate"
    if rho < 0.60:
        return "moderate"
    if rho < 0.80:
        return "strong"
    return "very strong"


def generate_html_tables(analyzers, base_dir, out_file):
    records = []
    for analyzer in analyzers:
        json_map = os.path.join(base_dir, f"{analyzer}_cat.json")
        warn2cat = json.load(open(json_map, encoding="utf-8"))
        for suffix in ["with_analizer.csv", "without_analizer.csv"]:
            csv_path = os.path.join(base_dir, f"{analyzer}_results_{suffix}")
            if not os.path.isfile(csv_path):
                continue
            df = pd.read_csv(csv_path)
            df["category"] = df["warning_type"].map(warn2cat).fillna("Other")
            df["analyzer"] = analyzer.capitalize()
            records.append(df[["category", "warning_type", "analyzer", "rho"]])
    all_df = pd.concat(records, ignore_index=True)
    agg = all_df.groupby(["category", "warning_type", "analyzer"], as_index=False)[
        "rho"].max()
    agg = agg[agg["rho"] > 0.4].sort_values(
        ["category", "analyzer", "warning_type"])
    html_main = agg.to_html(
        index=False, classes="table table-bordered",
        columns=["category", "warning_type", "analyzer", "rho"],
        header=["Category", "Warning Type", "Analyzer", "ρ"],
        border=0, float_format="%.3f"
    )
    records2 = []
    for analyzer in analyzers:
        json_map = os.path.join(base_dir, f"{analyzer}_cat.json")
        warn2cat = json.load(open(json_map, encoding="utf-8"))
        for suffix, sign in [("with_analizer.csv", "+"), ("without_analizer.csv", "–")]:
            csv_path = os.path.join(base_dir, f"{analyzer}_results_{suffix}")
            if not os.path.isfile(csv_path):
                continue
            df = pd.read_csv(csv_path)
            df["category"] = df["warning_type"].map(warn2cat).fillna("Other")
            df["analyzer"] = analyzer.capitalize()
            df["Analyzer Present"] = sign
            df["ρ"] = df["rho"]
            df["Correlation Group"] = df["ρ"].apply(classify_evans)
            records2.append(df[
                ["analyzer", "warning_type", "ρ", "Analyzer Present",
                    "Correlation Group", "category"]
            ])
    full = pd.concat(records2, ignore_index=True).sort_values(
        ["analyzer", "warning_type"])
    html_per_analyzer = ""
    for name, sub in full.groupby("analyzer"):
        html_per_analyzer += f"<h2>{name}</h2>\n"
        html_per_analyzer += sub.to_html(
            index=False, classes="table table-bordered",
            columns=["warning_type", "ρ", "Analyzer Present",
                     "Correlation Group", "category"],
            header=["Warning Type", "ρ", "Analyzer Present",
                    "Correlation Group", "Category"],
            border=0, float_format="%.3f"
        )
        html_per_analyzer += "\n<br/>\n"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("<h1>By Category (max ρ, ρ > 0.4)</h1>\n")
        f.write(html_main)
        f.write("\n<hr/>\n")
        f.write(html_per_analyzer)


with open('results/data.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
data = {repo: repo_data for repo, repo_data in raw_data.items()}

os.makedirs('results', exist_ok=True)

repos = list(data.keys())
has_analyzer = [r for r in repos if data[r]['exists_analyzer']]
no_analyzer = [r for r in repos if not data[r]['exists_analyzer']]

correlation_keys = ['phpstan_results', 'phan_results', 'psalm_results']
for key in correlation_keys:
    process_correlation(key, repos, data, has_analyzer, no_analyzer)

analyzer_counts = Counter(
    an for repo in repos for an in data[repo]['exists_analyzer'])
display_names = {'phpstan': 'PHPStan', 'psalm': 'Psalm', 'phan': 'Phan'}
keys = list(analyzer_counts.keys())
sizes = [analyzer_counts[k] for k in keys]
labels = [display_names.get(k, k) for k in keys]
plot_pie_chart(
    labels, sizes,
    filename='results/analyzers_distribution_pie.png',
    dpi=300, fig_size=(8, 8),
    autopct='%.1f%%', startangle=140,
    edgecolor='white',
    label_fontsize=16, autotext_fontsize=15
)

analyzers_simple = ["phan", "phpstan", "psalm"]
base_dir = "results"
width = 0.35
dpi = 300
fig_width = 1400 / dpi
inch_per_warning = 0.5
min_fig_height = 3
generate_faceted_plots(analyzers_simple, base_dir, width,
                       dpi, fig_width, inch_per_warning, min_fig_height)

with open('results/data.json', 'r', encoding='utf-8') as f:
    data_for_pie = json.load(f)
all_exists = [an for repo in data_for_pie.values()
              for an in repo.get('exists_analyzer', [])]
counter = Counter(all_exists)
labels2 = list(counter.keys())
sizes2 = list(counter.values())
plot_pie_chart(
    labels2, sizes2,
    filename='results/pie.png',
    dpi=200, fig_size=(8, 8),
    autopct='%1.1f%%', startangle=140,
    edgecolor=None,
    label_fontsize=14, autotext_fontsize=16,
    bbox_inches='tight'
)

out_file = os.path.join('results', 'tables.html')
generate_html_tables(analyzers_simple, base_dir, out_file)
write_stats_html(has_analyzer, no_analyzer, data, out_file)
