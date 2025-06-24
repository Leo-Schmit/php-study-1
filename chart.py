import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

analyzers = ["phan", "phpstan", "psalm"]
base_dir = "results"
width = 0.4

dpi = 300
fig_width = 1900 / dpi

for analyzer in analyzers:
    file_with = os.path.join(
        base_dir, f"{analyzer}_results_with_analizer_significant_corr.csv")
    file_without = os.path.join(
        base_dir, f"{analyzer}_results_without_analizer_significant_corr.csv")

    df_with = pd.read_csv(file_with)
    df_without = pd.read_csv(file_without)

    unique_warnings = list(
        pd.concat([df_with['warning_type'], df_without['warning_type']]).unique()
    )
    plus = df_with.set_index("warning_type")["rho"]
    minus = df_without.set_index("warning_type")["rho"]
    x = np.arange(len(unique_warnings))

    fig_height = max(2, len(unique_warnings)*0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    ax.barh(x - width/2, [plus.get(w, 0) for w in unique_warnings],
            height=width, label='with analyzer', color='#1f77b4')
    ax.barh(x + width/2, [minus.get(w, 0) for w in unique_warnings],
            height=width, label='without analyzer', color='#ff7f0e')

    ax.set_xlabel('ρ')
    ax.set_yticks(x)
    ax.set_yticklabels(unique_warnings)
    ax.invert_yaxis()
    ax.legend()
    ax.set_title('')

    ax.axvline(0.4, color='gray', linestyle='--', linewidth=1, dashes=(2, 2))
    if analyzer == 'phan':
        ax.axvline(-0.4, color='gray', linestyle='--',
                   linewidth=1, dashes=(2, 2))

    plt.tight_layout(pad=0)

    plt.savefig(os.path.join(base_dir, f"{analyzer}_chart.png"))
    plt.close(fig)

