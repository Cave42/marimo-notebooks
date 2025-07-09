import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""### Import Libraries""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    from matplotlib import gridspec
    from scipy.stats import spearmanr
    import json
    import sys
    import argparse
    from augur.utils import annotate_parents_for_tree
    import Bio.Phylo
    from Bio import Phylo
    import matplotlib.colors as mcolors
    from matplotlib import colormaps
    import colorsys
    import matplotlib.cm as cm
    import numpy as np
    import glob
    import re 
    from matplotlib.lines import Line2D
    return (
        Line2D,
        Path,
        cm,
        colorsys,
        glob,
        gridspec,
        json,
        mo,
        np,
        os,
        pd,
        plt,
        re,
        sns,
        spearmanr,
    )


@app.cell
def _(mo):
    mo.md(r"""### Set Working directory outside of notebook folder""")
    return


@app.cell
def _(os):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_directory)
    os.chdir("..")
    return


@app.cell
def _(mo):
    mo.md(r"""### Load base dataframes""")
    return


@app.cell
def _(Path, glob, pd):
    def load_base_results(
        *subfolders,
        base_dir: str = "Flu_Snakemake_Pipeline/results/max_freqs_log_likelyhood",
        verbose: bool = True,
    ) -> pd.DataFrame:

        full_path = Path(base_dir, *subfolders)
        glob_pattern = str(full_path / "*.csv")
        csv_files = glob.glob(glob_pattern)

        if verbose:
            print(f"Looking in: {full_path}")
            print(f"Found {len(csv_files)} CSV files.")

        frames = []
        for fname in csv_files:
            try:
                df = pd.read_csv(fname)

                seg = Path(fname).stem.rsplit("_", 1)[-1].upper()
                df["Segment"] = seg

                metadata = {key: value for f in subfolders if "~" in f for key, value in [f.split("~")]}
                df["Model"] = f"Base_{metadata.get('model')}" if "model" in metadata else None
                df["tree"] = metadata.get("next_tree")
                df["Model_training_time"] = metadata.get("time")

                frames.append(df)

            except Exception as e:
                if verbose:
                    print(f"Error reading {fname}: {e}")

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return (load_base_results,)


@app.cell
def _(load_base_results):
    df_650_Base = load_base_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~All",
    )

    df_3B_Base = load_base_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t36_3B_UR50D",
        "time~All",
    )

    df_15B_Base = load_base_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t48_15B_UR50D",
        "time~All",
    )
    return df_15B_Base, df_3B_Base, df_650_Base


@app.cell
def _():
    ### Combine Base Dataframes
    return


@app.cell
def _(df_15B_Base, df_3B_Base, df_650_Base, pd):
    df_all_base = pd.concat([df_15B_Base,df_3B_Base, df_650_Base], ignore_index=True)

    df_all_base
    return (df_all_base,)


@app.cell
def _():
    ### Create Runtime Figure
    return


@app.cell
def _(df_all_base, os, pd, plt, sns):
    short_names = {
        'Base_esm2_t33_650M_UR50D': '650M',
        'Base_esm2_t36_3B_UR50D':  '3B',
        'Base_esm2_t48_15B_UR50D': '15B'
    }

    df_all_base['Model_short'] = df_all_base['Model'].map(short_names)

    model_order_short = ['650M', '3B', '15B']
    df_all_base['Model_short'] = pd.Categorical(
        df_all_base['Model_short'],
        categories=model_order_short,
        ordered=True
    )


    df_all_sorted = df_all_base.sort_values('Model_short')

    sns.set_theme(style="whitegrid")

    palette_short = {
        '650M': '#0a2463',
        '3B':   '#f4d35e',
        '15B':  '#890304'
    }

    sns.barplot(
        data      = df_all_sorted,
        x         = 'Segment',
        y         = 'runtime',
        hue       = 'Model_short',
        hue_order = model_order_short,
        palette   = palette_short,
        errorbar  = None,
    )

    plt.title("ESM Runtime")
    plt.xlabel("Segment")
    plt.ylabel("Runtime (Seconds)")
    plt.legend(title="Model", frameon=False, loc='upper left')
    plt.tight_layout()

    newpath = "Flu_Figures/" 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    output_path = os.path.join(newpath, "runtime_esm2_base_mdls.png")
    plt.savefig(output_path, dpi=300)

    #plt.close()
    plt.show()
    return model_order_short, newpath, palette_short, short_names


@app.cell
def _(mo):
    mo.md(r"""### Create Three Base Model Comparisons per segment figures""")
    return


@app.cell
def _(df_all_base, gridspec, newpath, os, plt, sns, spearmanr):
    #plot all three models side by side, individual figures per segment

    sns.set_theme(style="whitegrid")
    sns.set_style("ticks")

    def plot_regression(ax, data, x_col, y_col, title, ylabel="", color="#0a2463"):
        sns.regplot(data=data, y=y_col, x=x_col, ax=ax, scatter_kws={'s': 50, 'alpha': 0.35, 'color': color}, line_kws={'color': 'black'})
        ax.set_title(title)
        ax.set_xlabel("")
        spearman_corr, p_value = spearmanr(data[y_col], data[x_col])
        textstr = (
            f'ρ = {spearman_corr:.2f}\n'
            f'P < {p_value:.2f}\n'
        )
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        ax.set_ylabel(ylabel, weight='bold')
        #ax.set_xlim(data[x_col].min(), data[x_col].max())
        ax.set_ylim(0, 1.1)

    def plot_histogram(ax, data, mean_value, title, xlabel="", color="#0a2463"):
        sns.histplot(data=data, x="log_likelyhood", ax=ax, color= color)
        ax.set_title(title)
        ax.axvline(mean_value, color='black', linestyle='-', linewidth=1.5, ymax=0.9)
        ax.set_ylabel("")
        ax.set_xlabel(xlabel, weight='bold' if xlabel else 'normal')

    segment = "MP"

    for segment, group in df_all_base.groupby('Segment'):

        fig = plt.figure(figsize=(5 * 3, 8))
        gs_main = gridspec.GridSpec(3, 3, height_ratios=[1, 0.4, 0.4])

        df = df_all_base[df_all_base['Segment'] == segment]

        if segment == "PA":
            df = df[df['node'] != 'A/Viamao/LACENRS-974/2015']

        #df_650 = df[df['Model'] == "650M"]
        #df_3B = df[df['Model'] == "3B"]
        #df_15B = df[df['Model'] == "15B"]

        df_650 = df[df['Model'] == "Base_esm2_t33_650M_UR50D"]
        df_3B = df[df['Model'] == "Base_esm2_t36_3B_UR50D"]
        df_15B = df[df['Model'] == "Base_esm2_t48_15B_UR50D"]

        df_below_1_650 = df_650[df_650['max_frequency'] < 0.1]
        df_above_1_650 = df_650[df_650['max_frequency'] >= 0.99]
        df_below_1_3B = df_3B[df_3B['max_frequency'] < 0.1]
        df_above_1_3B = df_3B[df_3B['max_frequency'] >= 0.99]
        df_below_1_15B = df_15B[df_15B['max_frequency'] < 0.1]
        df_above_1_15B = df_15B[df_15B['max_frequency'] >= 0.99]

        ax = fig.add_subplot(gs_main[0, 0])
        ax_1 = fig.add_subplot(gs_main[0, 1], sharey=ax)
        ax_2 = fig.add_subplot(gs_main[0, 2], sharey=ax)

        ax1 = fig.add_subplot(gs_main[1, 0], sharex=ax)
        ax2 = fig.add_subplot(gs_main[2, 0], sharex=ax)

        ax1_1 = fig.add_subplot(gs_main[1, 1], sharex=ax_1)
        ax2_1 = fig.add_subplot(gs_main[2, 1], sharex=ax_1)

        ax1_2 = fig.add_subplot(gs_main[1, 2], sharex=ax_2)
        ax2_2 = fig.add_subplot(gs_main[2, 2], sharex=ax_2)

        plot_regression(ax, df_650, "log_likelyhood", "max_frequency", f"{segment.upper()} - 650M Model - Max Freq. vs LL", ylabel="Max Frequency")
        plot_regression(ax_1, df_3B, "log_likelyhood", "max_frequency", f"{segment.upper()} - 3B Model - Max Freq. vs LL", color='#f4d35e')
        plot_regression(ax_2, df_15B, "log_likelyhood", "max_frequency", f"{segment.upper()} - 15B Model - Max Freq. vs LL", color='#890304')

        mean_below_1_650 = df_below_1_650['log_likelyhood'].mean()
        mean_below_1_3B = df_below_1_3B['log_likelyhood'].mean()
        mean_below_1_15B = df_below_1_15B['log_likelyhood'].mean()

        plot_histogram(ax1, df_below_1_650, mean_below_1_650, "max. freq. (0.0, 0.1)")
        plot_histogram(ax1_1, df_below_1_3B, mean_below_1_3B, "max. freq. (0.0, 0.1)", color='#f4d35e')
        plot_histogram(ax1_2, df_below_1_15B, mean_below_1_15B, "max. freq. (0.0, 0.1)", color='#890304')

        mean_above_1_650 = df_above_1_650['log_likelyhood'].mean()
        mean_above_1_3B = df_above_1_3B['log_likelyhood'].mean()
        mean_above_1_15B = df_above_1_15B['log_likelyhood'].mean()

        plot_histogram(ax2, df_above_1_650, mean_above_1_650, "max. freq. (0.99, 1.0)", xlabel="Log Likelyhood")
        plot_histogram(ax2_1, df_above_1_3B, mean_above_1_3B, "max. freq. (0.99, 1.0)", xlabel="Log Likelyhood", color='#f4d35e')
        plot_histogram(ax2_2, df_above_1_15B, mean_above_1_15B, "max. freq. (0.99, 1.0)", xlabel="Log Likelyhood", color='#890304')

        for axis in [ax, ax_1, ax_2, ax1, ax2, ax1_1, ax2_1, ax1_2, ax2_2]:
            axis.spines[['right', 'top']].set_visible(False)

        fig.text(0.01, 0.3, 'Count', va='center', rotation='vertical', fontsize=12, weight='bold')
        plt.tight_layout()

        newpath_ESM_base = "Flu_Figures/ESM_vs_Max_Freq_Plots_Base_Models_Comparison/" 
        if not os.path.exists(newpath):
                os.makedirs(newpath)

        plt.savefig(f"Flu_Figures/ESM_vs_Max_Freq_Plots_Base_Models_Comparison/{segment}_LL_vs_Max_Frequency.png", dpi=300)
        plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### Save Summary Stats for base models""")
    return


@app.cell
def _(df_all_base, newpath, os, pd, spearmanr):
    #Save summary stats for base model comparison

    results = []

    for model, group_all_base in df_all_base.groupby('Model'):
      for segment_all_base, group_all_base in df_all_base.groupby('Segment'):

        df_all_base_filtered = df_all_base[df_all_base['Segment'] == segment_all_base]
        df_all_base_filtered = df_all_base_filtered[df_all_base_filtered['Model'] == model]

        if segment_all_base == "pa":
          df_all_base_filtered = df_all_base_filtered[df_all_base_filtered['node'] != 'A/Viamao/LACENRS-974/2015']

        df_below_01 = df_all_base_filtered[df_all_base_filtered['max_frequency'] < 0.1]
        df_above_1 = df_all_base_filtered[df_all_base_filtered['max_frequency'] >= 0.99]

        spearman_corr, p_value = spearmanr(df_all_base_filtered['max_frequency'], df_all_base_filtered['log_likelyhood'])

        results.append({
            "Model": model,
            "Segment": segment_all_base,
            "Spearman Correlation Coefficient between Max Frequency and LL": spearman_corr,
            "P-value": p_value,
            "Mean ESM LL below 0.1": df_below_01['log_likelyhood'].mean(),
            "Mean ESM LL above 0.99": df_above_1['log_likelyhood'].mean(),
            "Difference in LL ESM Means": df_above_1['log_likelyhood'].mean() - df_below_01['log_likelyhood'].mean()
        })

        results_df = pd.DataFrame(results)

    print(results_df)

    newpath_all_base = "Flu_Summary_Statistics/" 
    if not os.path.exists(newpath):
            os.makedirs(newpath)

    results_df.to_csv("Flu_Summary_Statistics/ESM_vs_Max_Freq_Summary_Statistics_Base_Models_Comparison.csv", index=False)
    return (results_df,)


@app.cell
def _(results_df):
    # Get average of the Spearman correlation coefficients for each model

    results_df.groupby('Model')['Spearman Correlation Coefficient between Max Frequency and LL'].mean()
    return


@app.cell
def _(mo):
    mo.md(r"""### Spearman Correlation Coefficient Figure For Base Models""")
    return


@app.cell
def _(model_order_short, palette_short, pd, plt, results_df, short_names, sns):
    def make_base_speareman_summary_figure(results_df):

        results_df['Model_short'] = results_df['Model'].map(short_names)
        results_df['Model_short'] = pd.Categorical(
            results_df['Model_short'],
            categories=model_order_short,
            ordered=True
        )

        results_df = results_df.dropna(subset=['Model_short'])

        results_df_sorted = results_df.sort_values('Model_short')

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8,5))

        sns.barplot(
            data=results_df_sorted,
            x='Segment',
            y='Spearman Correlation Coefficient between Max Frequency and LL',
            hue='Model_short',           
            hue_order=model_order_short, 
            palette=palette_short,
            errorbar=None
        )

        plt.title("Spearman Correlation between Max Freq. and LL across Models")
        plt.xlabel("Segment")
        plt.ylabel("Spearman Correlation Coefficient")
        plt.legend(title="Model Size", frameon=False, loc='lower left')
        plt.tight_layout()
        plt.savefig("Flu_Figures/Spearman_Correlation_Comparision_Base_Model.png", dpi=300)

        return plt.show()

    make_base_speareman_summary_figure(results_df)
    return


@app.cell
def _(mo):
    mo.md(r"""### Function To Add ESM to Nextstrain Tree""")
    return


@app.cell
def _(json, os, pd):
    #Add ESM LL to Nextstrain Tree

    def add_ESM_LL_to_Nextstrain_Tree(directory, output_dir):
      for filename in os.listdir(directory):
          if filename.endswith(".csv"):
              file_path = os.path.join(directory, filename)

              segment = filename.rsplit('_', 1)[-1].replace('.csv', '')

              LL_freq_file_df = pd.read_csv(file_path)

              if segment == "pa":
                LL_freq_file_df = LL_freq_file_df[LL_freq_file_df['node'] != 'A/Viamao/LACENRS-974/2015']

              formatted_dict = {
                  "nodes": {
                      key: {"ESM_score": value}
                      for key, value in zip(LL_freq_file_df["node"], LL_freq_file_df["log_likelyhood"])
                  }
              }

              with open(f"/Users/Carlos/Desktop/Bedford/esm-selection/Flu_Snakemake_Pipeline/input/trees/h3n2/{segment}.json", 'r') as fh:
              #with open(f"Flu_Snakemake_Pipeline/h3n2_Sequences/h3n2_60y_{segment}.json", 'r') as fh:
                dataset = json.load(fh)


              node_data = formatted_dict
              esm_scores = {name: info["ESM_score"] for name, info in node_data['nodes'].items()}

              def recurse(n):
                if n["name"] in esm_scores:
                  n["node_attrs"]["ESM_score"] = {"value": esm_scores[n["name"]]}
                for c in n.get("children", []):
                  recurse(c)

              recurse(dataset["tree"])

              dataset['meta']["colorings"].insert(0, {"key": "ESM_score", "title": "esm scores", "type": "continuous"})

              os.makedirs(output_dir, exist_ok=True)

              with open(f"{output_dir}/{segment}_ESM_Tree.json", 'w') as fh:
                json.dump(dataset, fh, indent=2)
    return (add_ESM_LL_to_Nextstrain_Tree,)


@app.cell
def _(mo):
    mo.md(r"""### Add ESM Log Likelihood To Nextstrain Tree:""")
    return


@app.cell
def _(mo):
    mo.md(r"""##### 650M ESM Nextstrain Tree""")
    return


@app.cell
def _(add_ESM_LL_to_Nextstrain_Tree, os):
    folder_path_650M = "Flu_Snakemake_Pipeline/results/max_freqs_log_likelyhood/next_tree~h3n2/epochs~1/learning_rate~5e-05/model~esm2_t33_650M_UR50D/time~1990"

    if os.path.isdir(folder_path_650M):

        add_ESM_LL_to_Nextstrain_Tree(
            folder_path_650M,
            f"Flu_Trees/ESM_Trees_650M_base"
        )
    return


@app.cell
def _(mo):
    mo.md(r"""##### 3B ESM Nextstrain Tree""")
    return


@app.cell
def _(add_ESM_LL_to_Nextstrain_Tree, os):
    folder_path_3B = "/Users/Carlos/Desktop/Bedford/esm-selection/Flu_Snakemake_Pipeline/results/max_freqs_log_likelyhood/next_tree~h3n2/epochs~1/learning_rate~5e-05/model~esm2_t36_3B_UR50D/time~1990"

    if os.path.isdir(folder_path_3B):

        add_ESM_LL_to_Nextstrain_Tree(
            folder_path_3B,
            f"Flu_Trees/ESM_Trees_3B_base"
        )
    return


@app.cell
def _(mo):
    mo.md(r"""##### 15B ESM Nextstrain Tree""")
    return


@app.cell
def _(add_ESM_LL_to_Nextstrain_Tree, os):
    folder_path_15B = "/Users/Carlos/Desktop/Bedford/esm-selection/Flu_Snakemake_Pipeline/results/max_freqs_log_likelyhood/next_tree~h3n2/epochs~1/learning_rate~5e-05/model~esm2_t48_15B_UR50D/time~All"

    if os.path.isdir(folder_path_15B):

        add_ESM_LL_to_Nextstrain_Tree(
            folder_path_15B,
            f"Flu_Trees/ESM_Trees_15B_base"
        )
    return


@app.cell
def _(mo):
    mo.md(r"""### Function to load in fine tune dataframes""")
    return


@app.cell
def _(Path, glob, pd):
    def load_fine_tune_results(
        *subfolders,
        base_dir: str = "Flu_Snakemake_Pipeline/results/max_freqs_log_likelyhood_Fine_Tune",
    ) -> pd.DataFrame:

        full_path = Path(base_dir, *subfolders)
        glob_pattern = str(full_path / "*.csv")
        csv_files = glob.glob(glob_pattern)

        frames = []
        for fname in csv_files:
            df = pd.read_csv(fname)

            # Segment from filename
            seg = Path(fname).stem.rsplit("_", 1)[-1].upper()
            df["Segment"] = seg

            # Extract metadata from folder names
            metadata = {key: value for f in subfolders if "~" in f for key, value in [f.split("~")]}

            model = metadata.get("model")
            lr = metadata.get("learning_rate", "")
            epochs = metadata.get("epochs", "")
            df["Model"] = f"Fine_Tune_{model}" if model else None
            df["Learning_rate"] = lr
            df["Epochs"] = epochs
            df["Model_training_time"] = metadata.get("time") if "time" in metadata else None
            df["tree"] = metadata.get("next_tree")

            frames.append(df)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return (load_fine_tune_results,)


@app.cell
def _(mo):
    mo.md(r"""### Load in fine tuned 1990 datasets""")
    return


@app.cell
def _(load_fine_tune_results):
    #Default fine tune

    df_650_FT_DF_Time_1990 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
    )

    df_3B_FT_DF_Time_1990 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )

    # epochs 5

    df_650_FT_DF_Time_1990_EP_5 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~5",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
    )

    df_3B_FT_DF_Time_1990_EP_5 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~5",
        "learning_rate~5e-05",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )

    # learning rate adjustments

    df_650_FT_DF_Time_1990_LR_2_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~2.5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
    )

    df_3B_FT_DF_Time_1990_LR_2_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~2.5e-05",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )

    df_650_FT_DF_Time_1990_LR_1e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~1e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
    )

    df_3B_FT_DF_Time_1990_LR_1e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~1e-05",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )

    df_650_FT_DF_Time_1990_LR_5e_06 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-06",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
    )

    df_3B_FT_DF_Time_1990_LR_5e_06 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-06",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )
    return (
        df_3B_FT_DF_Time_1990,
        df_3B_FT_DF_Time_1990_EP_5,
        df_3B_FT_DF_Time_1990_LR_1e_05,
        df_3B_FT_DF_Time_1990_LR_2_5e_05,
        df_3B_FT_DF_Time_1990_LR_5e_06,
        df_650_FT_DF_Time_1990,
        df_650_FT_DF_Time_1990_EP_5,
        df_650_FT_DF_Time_1990_LR_1e_05,
        df_650_FT_DF_Time_1990_LR_2_5e_05,
        df_650_FT_DF_Time_1990_LR_5e_06,
    )


@app.cell
def _(mo):
    mo.md(r"""### Add Time to dataframes functions""")
    return


@app.cell
def _(json, os, pd):
    #Add time to Model dfs

    def extract_node_times(tree_data, segment):
        node_list = []

        def recurse_nodes(node):
            name = node.get('name')
            num_date = node.get('node_attrs', {}).get('num_date', {}).get('value')
            if name and num_date is not None:
                node_list.append({'Segment': segment, 'node': name, 'time': num_date})
            for child in node.get('children', []):
                recurse_nodes(child)

        root = tree_data.get('tree', tree_data)
        recurse_nodes(root)
        return node_list


    def process_directory(directory):

        all_data = []

        for filename in os.listdir(directory):

            segment = filename[:-5]  # remove the '.json' suffix
            if (filename == f"{filename[:-5]}.json"):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as f:
                    tree_data = json.load(f)

            segment_data = extract_node_times(tree_data, segment)
            all_data.extend(segment_data)
        return pd.DataFrame(all_data)

    def merge_time(models_df, tree):
        directory = f"Flu_Snakemake_Pipeline/input/trees/{tree}/"
        df = process_directory(directory)
        df['Segment'] = df['Segment'].str.upper()
        models_df = models_df.merge(df, on=['Segment', 'node'], how='left')
        #models_df = models_df[models_df['time'] >= 1991]
        return models_df
    return (merge_time,)


@app.cell
def _(mo):
    mo.md(r"""### Rename model names in dataframes function""")
    return


@app.cell
def _(pd):
    def rename_model_dataframes(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Model'] = df['Model'].replace({
            'Fine_Tune_esm2_t36_3B_UR50D': 'Fine_Tune_3B',
            'Fine_Tune_esm2_t33_650M_UR50D': 'Fine_Tune_650M',
            'Base_esm2_t36_3B_UR50D': '3B',
            'Base_esm2_t33_650M_UR50D': '650M',
        })
        return df   
    return (rename_model_dataframes,)


@app.cell
def _(mo):
    mo.md(r"""### Combine Fine Tune With Base function""")
    return


@app.cell
def _(df_3B_Base, df_650_Base, pd, rename_model_dataframes):
    def concat_with_base(df, model):
        if(model == "3B"):
            df = pd.concat([df, df_3B_Base], ignore_index=True)
        if(model == "650M"):
            df = pd.concat([df, df_650_Base], ignore_index=True)
        df = rename_model_dataframes(df)

        return df
    return (concat_with_base,)


@app.cell
def _(mo):
    mo.md(r"""### Combine dataframes fine tune and base""")
    return


@app.cell
def _(
    concat_with_base,
    df_3B_FT_DF_Time_1990,
    df_3B_FT_DF_Time_1990_EP_5,
    df_3B_FT_DF_Time_1990_LR_1e_05,
    df_3B_FT_DF_Time_1990_LR_2_5e_05,
    df_3B_FT_DF_Time_1990_LR_5e_06,
    df_650_FT_DF_Time_1990,
    df_650_FT_DF_Time_1990_EP_5,
    df_650_FT_DF_Time_1990_LR_1e_05,
    df_650_FT_DF_Time_1990_LR_2_5e_05,
    df_650_FT_DF_Time_1990_LR_5e_06,
):
    # ───── 1. Default Fine-Tune (Epoch 1, LR 5e-05) ─────
    df_3B_FT_DF_Time_1990_Combined   = concat_with_base(df_3B_FT_DF_Time_1990, "3B")
    df_650_FT_DF_Time_1990_Combined  = concat_with_base(df_650_FT_DF_Time_1990, "650M")

    # ───── 2. Epochs = 5 ─────
    df_3B_FT_EP_5_DF_Time_Combined   = concat_with_base(df_3B_FT_DF_Time_1990_EP_5, "3B")
    df_650_FT_EP_5_DF_Time_Combined  = concat_with_base(df_650_FT_DF_Time_1990_EP_5, "650M")

    # ───── 3. Learning Rate Adjustments (Epoch 1) ─────

    # LR = 2.5e-05
    df_3B_FT_EP_1_LR_2_5e_05_DF_Time_Combined  = concat_with_base(df_3B_FT_DF_Time_1990_LR_2_5e_05, "3B")
    df_650_FT_EP_1_LR_2_5e_05_DF_Time_Combined = concat_with_base(df_650_FT_DF_Time_1990_LR_2_5e_05, "650M")

    # LR = 1e-05
    df_3B_FT_EP_1_LR_1e_05_DF_Time_Combined  = concat_with_base(df_3B_FT_DF_Time_1990_LR_1e_05, "3B")
    df_650_FT_EP_1_LR_1e_05_DF_Time_Combined = concat_with_base(df_650_FT_DF_Time_1990_LR_1e_05, "650M")

    # LR = 5e-06
    df_3B_FT_EP_1_LR_5e_06_DF_Time_Combined  = concat_with_base(df_3B_FT_DF_Time_1990_LR_5e_06, "3B")
    df_650_FT_EP_1_LR_5e_06_DF_Time_Combined = concat_with_base(df_650_FT_DF_Time_1990_LR_5e_06, "650M")

    return (
        df_3B_FT_DF_Time_1990_Combined,
        df_3B_FT_EP_1_LR_1e_05_DF_Time_Combined,
        df_3B_FT_EP_1_LR_2_5e_05_DF_Time_Combined,
        df_3B_FT_EP_1_LR_5e_06_DF_Time_Combined,
        df_3B_FT_EP_5_DF_Time_Combined,
        df_650_FT_DF_Time_1990_Combined,
        df_650_FT_EP_1_LR_1e_05_DF_Time_Combined,
        df_650_FT_EP_1_LR_2_5e_05_DF_Time_Combined,
        df_650_FT_EP_1_LR_5e_06_DF_Time_Combined,
        df_650_FT_EP_5_DF_Time_Combined,
    )


@app.cell
def _(mo):
    mo.md(r"""### Add time to dataframes""")
    return


@app.cell
def _(
    df_3B_FT_DF_Time_1990_Combined,
    df_3B_FT_EP_1_LR_1e_05_DF_Time_Combined,
    df_3B_FT_EP_1_LR_2_5e_05_DF_Time_Combined,
    df_3B_FT_EP_1_LR_5e_06_DF_Time_Combined,
    df_3B_FT_EP_5_DF_Time_Combined,
    df_650_FT_DF_Time_1990_Combined,
    df_650_FT_EP_1_LR_1e_05_DF_Time_Combined,
    df_650_FT_EP_1_LR_2_5e_05_DF_Time_Combined,
    df_650_FT_EP_1_LR_5e_06_DF_Time_Combined,
    df_650_FT_EP_5_DF_Time_Combined,
    merge_time,
):
    #no inputs fine tune
    df_3B_FT_DF_Time = merge_time(df_3B_FT_DF_Time_1990_Combined, "h3n2")
    df_650_FT_DF_Time = merge_time(df_650_FT_DF_Time_1990_Combined, "h3n2")

    #epochs > 5
    df_3B_FT_EP_5_DF_Time = merge_time(df_3B_FT_EP_5_DF_Time_Combined, "h3n2")
    df_650_FT_EP_5_DF_Time = merge_time(df_650_FT_EP_5_DF_Time_Combined, "h3n2")

    #learning rate adjustments
    df_650_FT_EP_1_LR_2_5e_05_DF_Time = merge_time(df_650_FT_EP_1_LR_2_5e_05_DF_Time_Combined, "h3n2")
    df_650_FT_EP_1_LR_1e_05_DF_Time = merge_time(df_650_FT_EP_1_LR_1e_05_DF_Time_Combined, "h3n2")
    df_650_FT_EP_1_LR_5e_06_DF_Time = merge_time(df_650_FT_EP_1_LR_5e_06_DF_Time_Combined, "h3n2")

    df_3B_FT_EP_1_LR_2_5e_05_DF_Time = merge_time(df_3B_FT_EP_1_LR_2_5e_05_DF_Time_Combined, "h3n2")
    df_3B_FT_EP_1_LR_1e_05_DF_Time = merge_time(df_3B_FT_EP_1_LR_1e_05_DF_Time_Combined, "h3n2")
    df_3B_FT_EP_1_LR_5e_06_DF_Time = merge_time(df_3B_FT_EP_1_LR_5e_06_DF_Time_Combined, "h3n2")

    #filter above 1990
    df_3B_FT_DF_Time_Above_1990 = df_3B_FT_DF_Time[df_3B_FT_DF_Time['time'] >= 1991]
    df_650_FT_DF_Time_Above_1990 = df_650_FT_DF_Time[df_650_FT_DF_Time['time'] >= 1991]

    df_3B_FT_DF_Time_Below_1990 = df_3B_FT_DF_Time[df_3B_FT_DF_Time['time'] <= 1990]
    df_650_FT_DF_Time_Below_1990 = df_650_FT_DF_Time[df_650_FT_DF_Time['time'] <= 1990]

    df_3B_FT_EP_5_DF_Time_Above_1990 = df_3B_FT_EP_5_DF_Time[df_3B_FT_EP_5_DF_Time['time'] >= 1991]
    df_650_FT_EP_5_DF_Time_Above_1990 = df_650_FT_EP_5_DF_Time[df_650_FT_EP_5_DF_Time['time'] >= 1991]

    df_3B_FT_EP_5_DF_Time_Below_1990 = df_3B_FT_EP_5_DF_Time[df_3B_FT_EP_5_DF_Time['time'] <= 1990]
    df_650_FT_EP_5_DF_Time_Below_1990 = df_650_FT_EP_5_DF_Time[df_650_FT_EP_5_DF_Time['time'] <= 1990]
    return (
        df_3B_FT_DF_Time,
        df_3B_FT_DF_Time_Above_1990,
        df_3B_FT_DF_Time_Below_1990,
        df_3B_FT_EP_5_DF_Time,
        df_650_FT_DF_Time,
        df_650_FT_DF_Time_Above_1990,
        df_650_FT_DF_Time_Below_1990,
        df_650_FT_EP_1_LR_1e_05_DF_Time,
        df_650_FT_EP_1_LR_2_5e_05_DF_Time,
        df_650_FT_EP_1_LR_5e_06_DF_Time,
        df_650_FT_EP_5_DF_Time,
    )


@app.cell
def _(mo):
    mo.md(r"""### Pre vs post 1990 figure, fine tuning vs base""")
    return


@app.cell
def _(
    df_3B_FT_DF_Time_Above_1990,
    df_3B_FT_DF_Time_Below_1990,
    df_650_FT_DF_Time_Above_1990,
    df_650_FT_DF_Time_Below_1990,
):
    # Pre post 1990 fine tune vs base figures

    def make_max_freq_LL_figures(dataset, model, pre_1990_dataset):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        import pandas as pd
        from scipy.stats import spearmanr
        from matplotlib import gridspec

        sns.set_theme(style="whitegrid")
        sns.set_style("ticks")

        output_dir = os.path.join(f"Flu_Figures/ESM_vs_Max_Freq_Plots_Fine_Tune_{model}")
        os.makedirs(output_dir, exist_ok=True)

        def plot_regression(ax, data, x_col, y_col, title, ylabel="", color="#0a2463"):
            sns.regplot(data=data, y=y_col, x=x_col, ax=ax, scatter_kws={'s': 50, 'alpha': 0.35, 'color': color}, line_kws={'color': 'black'})
            ax.set_title(title)
            ax.set_xlabel("")
            spearman_corr, p_value = spearmanr(data[y_col], data[x_col])
            textstr = f'ρ = {spearman_corr:.2f}\nP < {p_value:.2f}\n'
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
            ax.set_ylabel(ylabel, weight='bold')
            #ax.set_xlim(data[x_col].min(), data[x_col].max())
            ax.set_ylim(0, 1.1)

        def plot_histogram(ax, data, mean_value, title, xlabel="", color="#0a2463"):
            sns.histplot(data=data, x="log_likelyhood", ax=ax, color=color)
            ax.set_title(title)
            ax.axvline(mean_value, color='black', linestyle='-', linewidth=1.5, ymax=0.9)
            ax.set_ylabel("")
            ax.set_xlabel(xlabel, weight='bold' if xlabel else 'normal')

        for segment, group in dataset.groupby('Segment'):
            fig = plt.figure(figsize=(5 * 4, 8))  # Four plots in a row
            gs_main = gridspec.GridSpec(3, 4, height_ratios=[1, 0.4, 0.4])

            df = dataset[dataset['Segment'] == segment]

            df_pre = pre_1990_dataset[pre_1990_dataset['Segment'] == segment]

            if segment == "PA":
                df = df[df['node'] != 'A/Viamao/LACENRS-974/2015']

            df_650 = df[df['Model'] == model]
            df_650_Fine_Tune = df[df['Model'] == ("Fine_Tune_" + model)]
            df_below_1_650 = df_650[df_650['max_frequency'] < 0.1]
            df_above_1_650 = df_650[df_650['max_frequency'] >= 0.99]
            df_below_1_Fine_Tune = df_650_Fine_Tune[df_650_Fine_Tune['max_frequency'] < 0.1]
            df_above_1_Fine_Tune = df_650_Fine_Tune[df_650_Fine_Tune['max_frequency'] >= 0.99]

            df_pre_650 = df_pre[df_pre['Model'] == model]
            df_pre_Fine_Tune = df_pre[df_pre['Model'] == ("Fine_Tune_" + model)]
            df_below_1_pre_650 = df_pre_650[df_pre_650['max_frequency'] < 0.1]
            df_above_1_pre_650 = df_pre_650[df_pre_650['max_frequency'] >= 0.99]
            df_below_1_pre_Fine_Tune = df_pre_Fine_Tune[df_pre_Fine_Tune['max_frequency'] < 0.1]
            df_above_1_pre_Fine_Tune = df_pre_Fine_Tune[df_pre_Fine_Tune['max_frequency'] >= 0.99]

            ax = fig.add_subplot(gs_main[0, 0])
            ax_1 = fig.add_subplot(gs_main[0, 1])
            ax_pre = fig.add_subplot(gs_main[0, 2])
            ax_pre_1 = fig.add_subplot(gs_main[0, 3])

            plot_regression(ax, df_650, "log_likelyhood", "max_frequency", f"{segment.upper()} - {model} Model - Post 1990", ylabel="Max Frequency")
            plot_regression(ax_1, df_650_Fine_Tune, "log_likelyhood", "max_frequency", f"{segment.upper()} - {model} Fine Tune Model - Post 1990", color='#f4d35e')
            plot_regression(ax_pre, df_pre_650, "log_likelyhood", "max_frequency", f"{segment.upper()} - {model} - Pre-1990")
            plot_regression(ax_pre_1, df_pre_Fine_Tune, "log_likelyhood", "max_frequency", f"{segment.upper()} - Fine Tune - Pre-1990", color="#f4d35e")

            ax1 = fig.add_subplot(gs_main[1, 0])
            ax1_1 = fig.add_subplot(gs_main[1, 1])
            ax1_2 = fig.add_subplot(gs_main[1, 2])
            ax1_3 = fig.add_subplot(gs_main[1, 3])

            plot_histogram(ax1, df_below_1_650, df_below_1_650['log_likelyhood'].mean(), "max. freq. (0.0, 0.1)")
            plot_histogram(ax1_1, df_below_1_Fine_Tune, df_below_1_Fine_Tune['log_likelyhood'].mean(), "max. freq. (0.0, 0.1)", color='#f4d35e')
            plot_histogram(ax1_2, df_below_1_pre_650, df_below_1_pre_650['log_likelyhood'].mean(), "max. freq. (0.0, 0.1)")
            plot_histogram(ax1_3, df_below_1_pre_Fine_Tune, df_below_1_pre_Fine_Tune['log_likelyhood'].mean(), "max. freq. (0.0, 0.1)", color="#f4d35e")

            ax2 = fig.add_subplot(gs_main[2, 0])
            ax2_1 = fig.add_subplot(gs_main[2, 1])
            ax2_2 = fig.add_subplot(gs_main[2, 2])
            ax2_3 = fig.add_subplot(gs_main[2, 3])

            plot_histogram(ax2, df_above_1_650, df_above_1_650['log_likelyhood'].mean(), "max. freq. (0.99, 1.0)", xlabel="Log Likelyhood")
            plot_histogram(ax2_1, df_above_1_Fine_Tune, df_above_1_Fine_Tune['log_likelyhood'].mean(), "max. freq. (0.99, 1.0)", xlabel="Log Likelyhood", color='#f4d35e')
            plot_histogram(ax2_2, df_above_1_pre_650, df_above_1_pre_650['log_likelyhood'].mean(), "max. freq. (0.99, 1.0)", xlabel="Log Likelyhood")
            plot_histogram(ax2_3, df_above_1_pre_Fine_Tune, df_above_1_pre_Fine_Tune['log_likelyhood'].mean(), "max. freq. (0.99, 1.0)", xlabel="Log Likelyhood", color="#f4d35e")

            for axis in [ax, ax_1, ax_pre, ax_pre_1, ax1, ax1_1, ax1_2, ax1_3, ax2, ax2_1, ax2_2, ax2_3]:
                axis.spines[['right', 'top']].set_visible(False)

            fig.text(0.01, 0.3, 'Count', va='center', rotation='vertical', fontsize=12, weight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{segment}_LL_vs_Max_Frequency_Fine_Tune_{model}.png", dpi=300)
            plt.show()

    make_max_freq_LL_figures(df_3B_FT_DF_Time_Above_1990, "3B", df_3B_FT_DF_Time_Below_1990)
    make_max_freq_LL_figures(df_650_FT_DF_Time_Above_1990, "650M", df_650_FT_DF_Time_Below_1990)

    return


@app.cell
def _(mo):
    mo.md(r"""### Calculate Summary Statistics for Models""")
    return


@app.cell
def _(
    df_3B_FT_DF_Time_Above_1990,
    df_3B_FT_DF_Time_Below_1990,
    df_650_FT_DF_Time_Above_1990,
    df_650_FT_DF_Time_Below_1990,
    pd,
    spearmanr,
):
    #calculate summary statistics for fine-tune models

    def summary_stats(model_df, base_name, time_frame):
      results = []

      for model, group in model_df.groupby('Model'):
        for segment, group in model_df.groupby('Segment'):

          df = model_df[model_df['Segment'] == segment]
          df = df[df['Model'] == model]

          if base_name == "PA":
            df = df[df['node'] != 'A/Viamao/LACENRS-974/2015']

          df_below_01 = df[df['max_frequency'] < 0.1]
          df_above_1 = df[df['max_frequency'] >= 0.99]

          spearman_corr, p_value = spearmanr(df['max_frequency'], df['log_likelyhood'])

          results.append({
              "Model": model,
              "Segment": segment,
              "Spearman Correlation Coefficient between Max Frequency and LL": spearman_corr,
              "P-value": p_value,
              "Mean ESM LL below 0.1": df_below_01['log_likelyhood'].mean(),
              "Mean ESM LL above 0.99": df_above_1['log_likelyhood'].mean(),
              "Difference in LL ESM Means": df_above_1['log_likelyhood'].mean() - df_below_01['log_likelyhood'].mean(),
              "Time Frame": time_frame
          })

          results_df = pd.DataFrame(results)

      print("____________________________")
      print(f"Summary Statistics for {base_name} Model - {time_frame}")
      print(results_df.groupby('Model')['Spearman Correlation Coefficient between Max Frequency and LL'].mean())

      #results_df.to_csv(f"Flu_Summary_Statistics/ESM_vs_Max_Freq_Summary_Fine_Tune_{base_name}_Statistics.csv", index=False)
      return results_df

    df_3B_FT_DF_Time_Above_1990_Results_DF = summary_stats(df_3B_FT_DF_Time_Above_1990, "3B", "Post 1990")
    df_650_FT_DF_Time_Above_1990_Results_DF = summary_stats(df_650_FT_DF_Time_Above_1990, "650M", "Post 1990")
    df_3B_FT_DF_Time_Below_1990_Results_DF = summary_stats(df_3B_FT_DF_Time_Below_1990, "3B", "Pre 1990")
    df_650_FT_DF_Time_Below_1990_Results_DF = summary_stats(df_650_FT_DF_Time_Below_1990, "650M", "Pre 1990")

    # Combine all results into a single DataFrame
    combined_results = pd.concat([df_3B_FT_DF_Time_Above_1990_Results_DF, df_650_FT_DF_Time_Above_1990_Results_DF, df_3B_FT_DF_Time_Below_1990_Results_DF, df_650_FT_DF_Time_Below_1990_Results_DF], ignore_index=True)
    # Save the combined results to a CSV file
    combined_results.to_csv("Flu_Summary_Statistics/ESM_vs_Max_Freq_Summary_Fine_Tune_Statistics.csv", index=False)

    return (
        df_3B_FT_DF_Time_Above_1990_Results_DF,
        df_3B_FT_DF_Time_Below_1990_Results_DF,
        df_650_FT_DF_Time_Above_1990_Results_DF,
        df_650_FT_DF_Time_Below_1990_Results_DF,
    )


@app.cell
def _(mo):
    mo.md(r"""### Create fine tune spearman comparisons""")
    return


@app.cell
def _(
    df_3B_FT_DF_Time_Above_1990_Results_DF,
    df_650_FT_DF_Time_Above_1990_Results_DF,
    pd,
    plt,
    sns,
):
    #crete fine-tune model comparison figures spearman averages
    def average_spearman_fine_tune_compare(results_df, base_name):

        model_order = [base_name, f'Fine_Tune_{base_name}']

        palette = {
            base_name: '#0a2463',
            f'Fine_Tune_{base_name}': '#f4d35e',
        }

        results_df['Model'] = pd.Categorical(
            results_df['Model'], 
            categories=model_order, 
            ordered=True
        )

        results_df = results_df.sort_values('Model')

        #sns.color_palette("pastel")

        sns.barplot(
            data=results_df, 
            x='Segment', 
            y='Spearman Correlation Coefficient between Max Frequency and LL', 
            hue='Model', 
            hue_order=model_order,  
            errorbar=None,
            palette=palette
        )

        plt.title(f"Spearman Correlation Coefficient compairing {base_name} fo Fine Tune {base_name}")
        plt.xlabel("Segment")
        plt.ylabel("Spearman CC (between Max Freq. and LL)")
        plt.legend(title="Model",frameon=False, loc='lower left')
        plt.tight_layout()

        #plt.show()
        plt.savefig(f"Flu_Figures/Spearman_Summary_Fine_Tune_{base_name}.png", dpi=300)
        #plt.close()
        plt.show()


    average_spearman_fine_tune_compare(df_3B_FT_DF_Time_Above_1990_Results_DF, "3B")
    average_spearman_fine_tune_compare(df_650_FT_DF_Time_Above_1990_Results_DF, "650M")
    return


@app.cell
def _(mo):
    mo.md(r"""### Spearman Fine Tune Compare 4x4 figure""")
    return


@app.cell
def _(
    df_3B_FT_DF_Time_Above_1990_Results_DF,
    df_3B_FT_DF_Time_Below_1990_Results_DF,
    df_650_FT_DF_Time_Above_1990_Results_DF,
    df_650_FT_DF_Time_Below_1990_Results_DF,
    pd,
    plt,
    sns,
):
    # Combine all result plots into one 4x4 figure, easier to view

    def plot_spearman_barplot(ax, df, model_order, palette, title, xaxis=""):
        df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
        df = df.sort_values('Model')

        sns.barplot(
            data=df,
            x='Segment',
            y='Spearman Correlation Coefficient between Max Frequency and LL',
            hue='Model',
            hue_order=model_order,
            errorbar=None,
            palette=palette,
            ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel(xaxis, weight='bold')
        ax.set_ylabel("Spearman CC (Max Freq. vs LL)", weight='bold')
        ax.legend(title="Model", frameon=False, loc='lower left')

    def combined_average_spearman_fine_tune_compare(df_3B, df_650M, df_3B_FT, df_650M_FT):
        model_order_3B = ['3B', 'Fine_Tune_3B']
        model_order_650M = ['650M', 'Fine_Tune_650M']

        palette_3B = {
            '3B': '#0a2463',
            'Fine_Tune_3B': '#f4d35e',
        }

        palette_650M = {
            '650M': '#0a2463',
            'Fine_Tune_650M': '#f4d35e',
        }

        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

        plot_spearman_barplot(axes[0, 0], df_3B, model_order_3B, palette_3B, "3B vs Fine Tune 3B (Post-1990)", xaxis="")
        plot_spearman_barplot(axes[0, 1], df_650M, model_order_650M, palette_650M, "650M vs Fine Tune 650M (Post-1990)", xaxis="")
        plot_spearman_barplot(axes[1, 0], df_3B_FT, model_order_3B, palette_3B, "3B vs Fine Tune 3B (Pre-1990)", xaxis="Segment")
        plot_spearman_barplot(axes[1, 1], df_650M_FT, model_order_650M, palette_650M, "650M vs Fine Tune 650M (Pre-1990)", xaxis="Segment")

        plt.tight_layout()
        plt.savefig("Flu_Figures/Combined_Spearman_Fine_Tune_Comparison.png", dpi=300)
        plt.show()

    combined_average_spearman_fine_tune_compare(df_3B_FT_DF_Time_Above_1990_Results_DF, df_650_FT_DF_Time_Above_1990_Results_DF, df_3B_FT_DF_Time_Below_1990_Results_DF, df_650_FT_DF_Time_Below_1990_Results_DF)
    return


@app.cell
def _(mo):
    mo.md(r"""### Calculate spearman CC for each time frame""")
    return


@app.cell
def _(
    df_3B_FT_DF_Time,
    df_3B_FT_EP_5_DF_Time,
    df_650_FT_DF_Time,
    df_650_FT_EP_1_LR_1e_05_DF_Time,
    df_650_FT_EP_1_LR_2_5e_05_DF_Time,
    df_650_FT_EP_1_LR_5e_06_DF_Time,
    df_650_FT_EP_5_DF_Time,
    pd,
    spearmanr,
):
    # calculate spearman cc for each time frame

    def spearman_correlation_calculation(df, x_col, y_col, model_name, segment, time_label):
        spearman_corr, p_value = spearmanr(df[x_col], df[y_col])
        return {
            "Model": model_name,
            "Segment": segment,
            "Time_Range": time_label,
            "Spearman_Correlation": spearman_corr,
            "P_Value": p_value
        }

    def spearman_correlation(df, model):
        results = []

        for segment, group in df.groupby('Segment'):
            df_segment = df[df['Segment'] == segment]

            if segment == "PA":
                df_segment = df_segment[df_segment['node'] != 'A/Viamao/LACENRS-974/2015']

            df_segment_FT = df_segment[df_segment["Model"] == f"Fine_Tune_{model}"]
            df_segment_BS = df_segment[df_segment["Model"] == f"{model}"]

            time_ranges = [
                (1970, 1990, "1980"),
                (1980, 2000, "1990"),
                (1990, 2010, "2000"),
                (2000, 2020, "2010"),
                (2010, None, "2020"),
            ]

            for start, end, label in time_ranges:
                if end is None:
                    df_segment_FT_label = df_segment_FT[df_segment_FT['time'] >= start]
                    df_segment_BS_label = df_segment_BS[df_segment_BS['time'] >= start]
                else:
                    df_segment_FT_label = df_segment_FT[(df_segment_FT['time'] >= start) & (df_segment_FT['time'] <= end)]
                    df_segment_BS_label = df_segment_BS[(df_segment_BS['time'] >= start) & (df_segment_BS['time'] <= end)]

                results.append(spearman_correlation_calculation(df_segment_FT_label, "max_frequency", "log_likelyhood", f"Fine_Tune_{model}", segment, label))
                results.append(spearman_correlation_calculation(df_segment_BS_label, "max_frequency", "log_likelyhood", model, segment, label))

        return pd.DataFrame(results)


    df_3B_FT_DF_Time_spearman = spearman_correlation(df_3B_FT_DF_Time, "3B")
    df_650_FT_DF_Time_spearman = spearman_correlation(df_650_FT_DF_Time, "650M")

    df_3B_FT_EP_5_DF_Time_spearman = spearman_correlation(df_3B_FT_EP_5_DF_Time, "3B")
    df_650_FT_EP_5_DF_Time_spearman = spearman_correlation(df_650_FT_EP_5_DF_Time, "650M")

    df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman = spearman_correlation(df_650_FT_EP_1_LR_2_5e_05_DF_Time, "650M")
    df_650_FT_EP_1_LR_1e_05_DF_Time_spearman = spearman_correlation(df_650_FT_EP_1_LR_1e_05_DF_Time, "650M")
    df_650_FT_EP_1_LR_5e_06_DF_Time_spearman = spearman_correlation(df_650_FT_EP_1_LR_5e_06_DF_Time, "650M")

    return (
        df_3B_FT_DF_Time_spearman,
        df_3B_FT_EP_5_DF_Time_spearman,
        df_650_FT_DF_Time_spearman,
        df_650_FT_EP_1_LR_1e_05_DF_Time_spearman,
        df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman,
        df_650_FT_EP_1_LR_5e_06_DF_Time_spearman,
        df_650_FT_EP_5_DF_Time_spearman,
        spearman_correlation,
    )


@app.cell
def _(mo):
    mo.md(r"""### Split base and fine tune dataframes""")
    return


@app.cell
def _(
    df_3B_FT_DF_Time_spearman,
    df_3B_FT_EP_5_DF_Time_spearman,
    df_650_FT_DF_Time_spearman,
    df_650_FT_EP_1_LR_1e_05_DF_Time_spearman,
    df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman,
    df_650_FT_EP_1_LR_5e_06_DF_Time_spearman,
    df_650_FT_EP_5_DF_Time_spearman,
):
    df_650_FT_DF_Time_spearman_Base = df_650_FT_DF_Time_spearman[df_650_FT_DF_Time_spearman['Model'] == '650M']
    df_650_FT_DF_Time_spearman_Fine_Tune = df_650_FT_DF_Time_spearman[df_650_FT_DF_Time_spearman['Model'] == 'Fine_Tune_650M']
    df_3B_FT_DF_Time_spearman_Base = df_3B_FT_DF_Time_spearman[df_3B_FT_DF_Time_spearman['Model'] == '3B']
    df_3B_FT_DF_Time_spearman_Fine_Tune = df_3B_FT_DF_Time_spearman[df_3B_FT_DF_Time_spearman['Model'] == 'Fine_Tune_3B']

    df_650_FT_EP_5_DF_Time_spearman_Base = df_650_FT_EP_5_DF_Time_spearman[df_650_FT_EP_5_DF_Time_spearman['Model'] == '650M']
    df_650_FT_EP_5_DF_Time_spearman_Fine_Tune = df_650_FT_EP_5_DF_Time_spearman[df_650_FT_EP_5_DF_Time_spearman['Model'] == 'Fine_Tune_650M']
    df_3B_FT_EP_5_DF_Time_spearman_Base = df_3B_FT_EP_5_DF_Time_spearman[df_3B_FT_EP_5_DF_Time_spearman['Model'] == '3B']
    df_3B_FT_EP_5_DF_Time_spearman_Fine_Tune = df_3B_FT_EP_5_DF_Time_spearman[df_3B_FT_EP_5_DF_Time_spearman['Model'] == 'Fine_Tune_3B']


    df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman_Base = df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman[df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman['Model'] == '650M']
    df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman_Fine_Tune = df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman[df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman['Model'] == 'Fine_Tune_650M']
    df_650_FT_EP_1_LR_1e_05_DF_Time_spearman_Base = df_650_FT_EP_1_LR_1e_05_DF_Time_spearman[df_650_FT_EP_1_LR_1e_05_DF_Time_spearman['Model'] == '650M']
    df_650_FT_EP_1_LR_1e_05_DF_Time_spearman_Fine_Tune = df_650_FT_EP_1_LR_1e_05_DF_Time_spearman[df_650_FT_EP_1_LR_1e_05_DF_Time_spearman['Model'] == 'Fine_Tune_650M']
    df_650_FT_EP_1_LR_5e_06_DF_Time_spearman_Base = df_650_FT_EP_1_LR_5e_06_DF_Time_spearman[df_650_FT_EP_1_LR_5e_06_DF_Time_spearman['Model'] == '650M']
    df_650_FT_EP_1_LR_5e_06_DF_Time_spearman_Fine_Tune = df_650_FT_EP_1_LR_5e_06_DF_Time_spearman[df_650_FT_EP_1_LR_5e_06_DF_Time_spearman['Model'] == 'Fine_Tune_650M']

    return (
        df_3B_FT_DF_Time_spearman_Base,
        df_3B_FT_DF_Time_spearman_Fine_Tune,
        df_3B_FT_EP_5_DF_Time_spearman_Fine_Tune,
        df_650_FT_DF_Time_spearman_Base,
        df_650_FT_DF_Time_spearman_Fine_Tune,
        df_650_FT_EP_1_LR_1e_05_DF_Time_spearman_Fine_Tune,
        df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman_Fine_Tune,
        df_650_FT_EP_1_LR_5e_06_DF_Time_spearman_Fine_Tune,
        df_650_FT_EP_5_DF_Time_spearman_Fine_Tune,
    )


@app.cell
def _(mo):
    mo.md(r"""### Create spearman CC lineplots base vs ft for 650M and 3B""")
    return


@app.cell
def _(
    df_3B_FT_DF_Time_spearman_Base,
    df_3B_FT_DF_Time_spearman_Fine_Tune,
    df_3B_FT_EP_5_DF_Time_spearman_Fine_Tune,
    df_650_FT_DF_Time_spearman_Base,
    df_650_FT_DF_Time_spearman_Fine_Tune,
    df_650_FT_EP_5_DF_Time_spearman_Fine_Tune,
    np,
    plt,
    sns,
):
    def create_spearman_lineplots():
        def plot_spearman_lineplot(df, model_name, ax, xaxis):

            sns.set_style("whitegrid")
            custom_params = {"axes.spines.right": False, "axes.spines.top": False}
            sns.set_theme(style="ticks", rc=custom_params)
            #sns.set_palette("pastel")

            lineplot = sns.lineplot(
                data=df,
                x='Time_Range',
                y='Spearman_Correlation',
                hue='Segment',
                ax=ax,
                marker="o",
                legend=False,
                zorder=1
            )

            ax.set_title(model_name)
            ax.set_xlabel("")
            ax.set_ylabel("")

            label_positions = []  
            pos = 0

            for line, segment in zip(ax.lines, df['Segment'].unique()):

                if pos == 0:
                    y = line.get_ydata()[-2]
                    x = line.get_xdata()[-2]
                    pos = 1
                elif pos == 1:
                    y = line.get_ydata()[-1]
                    x = line.get_xdata()[-1]
                    pos = 0

                #x = line.get_xdata()[-1]
                if not np.isfinite(y) or not np.isfinite(x):
                    continue

                while any(abs(y - pos) < 0.025 for pos in label_positions):  
                    y += 0.007  

                label_positions.append(y)  

                ax.annotate(
                    segment,
                    xy=(x, y),
                    xytext=(5, 0),  
                    textcoords="offset points",
                    color=line.get_color(),
                    fontsize=12,
                    weight='bold',
                    ha='left',  
                    va='center',
                    zorder=2,
                )



        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

        fig.text(0.5, 0.05, "Time Range", ha='center', va="top", fontsize=12, weight='bold')
        fig.text(0.05, 0.57, "Spearman Correlation", ha='center', va="top", rotation='vertical', fontsize=12, weight='bold')

        fig.suptitle('Spearman Correlation Coefficient for 30 Year Sliding Window Incrementing By 10 Years', fontsize=14, weight='bold',  y=0.95)

        plot_spearman_lineplot(df_650_FT_DF_Time_spearman_Base, " Base - 650M Model", axes[0,0], xaxis="")
        plot_spearman_lineplot(df_650_FT_DF_Time_spearman_Fine_Tune, "Fine Tune Before 1990 - 650M Model", axes[1,0], xaxis="Time Range")
        plot_spearman_lineplot(df_3B_FT_DF_Time_spearman_Base, "Base - 3B Model", axes[0,1], xaxis="")
        plot_spearman_lineplot(df_3B_FT_DF_Time_spearman_Fine_Tune, "Fine Tune Before 1990 - 3B Model", axes[1,1], xaxis="Time Range")

        #plt.tight_layout()
        #plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

        plt.savefig("Flu_Figures/Spearman_Correlation_Comparison_Fine_Tune_time_range.png", dpi=300)
        plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

        fig.text(0.5, 0.05, "Time Range", ha='center', va="top", fontsize=12, weight='bold')
        fig.text(0.05, 0.57, "Spearman Correlation", ha='center', va="top", rotation='vertical', fontsize=12, weight='bold')

        fig.suptitle('Spearman CC 30 Year Sliding Window - Comparing 1 vs 5 Epochs Fine Tune Models', fontsize=14, weight='bold',  y=0.95)

        plot_spearman_lineplot(df_650_FT_DF_Time_spearman_Fine_Tune, "Fine Tune Before 1990 - 650M Model - Epochs 1", axes[0,0], xaxis="Time Range")
        plot_spearman_lineplot(df_650_FT_EP_5_DF_Time_spearman_Fine_Tune, "Fine Tune Before 1990 - 650M Model - Epochs 5", axes[1,0], xaxis="Time Range")
        plot_spearman_lineplot(df_3B_FT_DF_Time_spearman_Fine_Tune, "Fine Tune Before 1990 - 3B Model - Epochs 1", axes[0,1], xaxis="Time Range")
        plot_spearman_lineplot(df_3B_FT_EP_5_DF_Time_spearman_Fine_Tune, "Fine Tune Before 1990 - 3B Model - Epochs 5", axes[1,1], xaxis="Time Range")

        plt.savefig("Flu_Figures/Spearman_Correlation_Comparison_Fine_Tune_time_range_Epoch_5.png", dpi=300)
        plt.show()

    create_spearman_lineplots()
    return


@app.cell
def _(mo):
    mo.md(r"""### Combined Spearman Summary""")
    return


@app.cell
def _(
    df_3B_FT_DF_Time_spearman_Base,
    df_3B_FT_DF_Time_spearman_Fine_Tune,
    df_3B_FT_EP_5_DF_Time_spearman_Fine_Tune,
    df_650_FT_DF_Time_spearman_Base,
    df_650_FT_DF_Time_spearman_Fine_Tune,
    df_650_FT_EP_1_LR_1e_05_DF_Time_spearman_Fine_Tune,
    df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman_Fine_Tune,
    df_650_FT_EP_1_LR_5e_06_DF_Time_spearman_Fine_Tune,
    df_650_FT_EP_5_DF_Time_spearman_Fine_Tune,
    pd,
):
    def mean_spearman_segments(df, model_name):
        df_Fine_Tune_Summary = df.groupby('Time_Range', as_index=False)['Spearman_Correlation'].mean()
        df_Fine_Tune_Summary["Model"] = model_name 

        return df_Fine_Tune_Summary

    df_3B_FT_DF_Time_spearman_Fine_Tune_Summary = mean_spearman_segments(df_3B_FT_DF_Time_spearman_Fine_Tune, "Fine Tune - 3B Model")
    df_650_FT_DF_Time_spearman_Fine_Tune_Summary = mean_spearman_segments(df_650_FT_DF_Time_spearman_Fine_Tune, "Fine Tune - 650M Model")
    df_3B_FT_EP_5_DF_Time_spearman_Fine_Tune_Summary = mean_spearman_segments(df_3B_FT_EP_5_DF_Time_spearman_Fine_Tune, "Fine Tune - 3B Model - Epochs 5")
    df_650_FT_EP_5_DF_Time_spearman_Fine_Tune_Summary = mean_spearman_segments(df_650_FT_EP_5_DF_Time_spearman_Fine_Tune, "Fine Tune - 650M Model - Epochs 5")
    df_3B_DF_Time_Spearman_Summary = mean_spearman_segments(df_3B_FT_DF_Time_spearman_Base, "Base - 3B Model")
    df_650_DF_Time_Spearman_Summary = mean_spearman_segments(df_650_FT_DF_Time_spearman_Base, "Base - 650M Model")

    df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman_Fine_Tune_Summary = mean_spearman_segments(df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman_Fine_Tune, "Fine Tune - 650M Model - LR 2.5e-05")
    df_650_FT_EP_1_LR_1e_05_DF_Time_spearman_Fine_Tune_Summary = mean_spearman_segments(df_650_FT_EP_1_LR_1e_05_DF_Time_spearman_Fine_Tune, "Fine Tune - 650M Model - LR 1e-05")
    df_650_FT_EP_1_LR_5e_06_DF_Time_spearman_Fine_Tune_Summary = mean_spearman_segments(df_650_FT_EP_1_LR_5e_06_DF_Time_spearman_Fine_Tune, "Fine Tune - 650M Model - LR 1e-06")

    combined_spearman_summary = pd.concat([
        df_3B_FT_DF_Time_spearman_Fine_Tune_Summary,
        df_650_FT_DF_Time_spearman_Fine_Tune_Summary,
        df_3B_FT_EP_5_DF_Time_spearman_Fine_Tune_Summary,
        df_650_FT_EP_5_DF_Time_spearman_Fine_Tune_Summary,
        df_3B_DF_Time_Spearman_Summary,
        df_650_DF_Time_Spearman_Summary,
        df_650_FT_EP_1_LR_2_5e_05_DF_Time_spearman_Fine_Tune_Summary,
        df_650_FT_EP_1_LR_1e_05_DF_Time_spearman_Fine_Tune_Summary,
        df_650_FT_EP_1_LR_5e_06_DF_Time_spearman_Fine_Tune_Summary
    ], ignore_index=True)

    combined_spearman_summary
    return combined_spearman_summary, mean_spearman_segments


@app.cell
def _(mo):
    mo.md(r"""### Combined spearman summary plot""")
    return


@app.cell
def _(combined_spearman_summary, np, plt, sns):
    def create_spearman_summary_plot():
        sns.set_style("whitegrid")
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax = sns.lineplot(
            data=combined_spearman_summary,
            x='Time_Range',
            y='Spearman_Correlation',
            hue='Model',
            marker="o",
            legend=False,
            zorder=1,
            ax=ax
        )

        ax.set_title("Spearman CC Summary All Models")
        ax.set_xlabel("Time Range")
        ax.set_ylabel("Spearman CC")

        label_positions = []  
        pos = 0

        for line, model in zip(ax.lines, combined_spearman_summary['Model'].unique()):

            y = line.get_ydata()[-1]
            x = line.get_xdata()[-1]

            if not np.isfinite(y) or not np.isfinite(x):
                continue

            if(model == "Base - 3B Model" or model == "Base - 650M Model"):
                y = line.get_ydata()[-4]
                x = line.get_xdata()[-4]
            if(model == "Fine Tune - 650M Model - LR 2.5e-05" or model == "Fine Tune - 650M Model - LR 1e-05" or model == "Fine Tune - 650M Model - LR 1e-06"):
                y = line.get_ydata()[-3]
                x = line.get_xdata()[-3]
            else:
                while any(abs(y - pos) < 0.025 for pos in label_positions):  
                    y += 0.007  

            label_positions.append(y)  

            ax.annotate(
                model,
                xy=(x, y),
                xytext=(5, 0),  
                textcoords="offset points",
                color=line.get_color(),
                fontsize=12,
                weight='bold',
                ha='left',  
                va='center',
                zorder=2,
            )

        plt.tight_layout()
        plt.show()

        return plt.savefig("Flu_Figures/Spearman_Correlation_Comparison_All_Models.png", dpi=300, bbox_inches='tight')


    create_spearman_summary_plot()
    return


@app.cell
def _(mo):
    mo.md(r"""### Make all time dataframes and dataframes for larger tree""")
    return


@app.cell
def _(load_fine_tune_results):
    df_650_FT_DF_All = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~All",
    )

    df_650_FT_DF_2020_Lg_tree = load_fine_tune_results(
        "next_tree~h3n2-Large",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~2020",
    )

    df_650_FT_DF_2005_Lg_tree = load_fine_tune_results(
        "next_tree~h3n2-Large",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~2005",
    )

    df_650_FT_DF_1990_Lg_tree = load_fine_tune_results(
        "next_tree~h3n2-Large",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
    )
    return (
        df_650_FT_DF_1990_Lg_tree,
        df_650_FT_DF_2005_Lg_tree,
        df_650_FT_DF_2020_Lg_tree,
        df_650_FT_DF_All,
    )


@app.cell
def _(mo):
    mo.md(r"""### Create time series dataframes""")
    return


@app.cell
def _(load_fine_tune_results):
    df_650M_FT_DF_Time_1990_LR_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
    )

    df_650M_FT_DF_Time_1995_LR_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1995",
    )

    df_650M_FT_DF_Time_2000_LR_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~2000",
    )

    df_650M_FT_DF_Time_2005_LR_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~2005",
    )

    df_650M_FT_DF_Time_2010_LR_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~2010",
    )

    df_650M_FT_DF_Time_2010_LR_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~2010",
    )

    df_650M_FT_DF_Time_2020_LR_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~2020",
    )
    return (
        df_650M_FT_DF_Time_1990_LR_5e_05,
        df_650M_FT_DF_Time_1995_LR_5e_05,
        df_650M_FT_DF_Time_2000_LR_5e_05,
        df_650M_FT_DF_Time_2005_LR_5e_05,
        df_650M_FT_DF_Time_2010_LR_5e_05,
        df_650M_FT_DF_Time_2020_LR_5e_05,
    )


@app.cell
def _(mo):
    mo.md(r"""### Add time to trees""")
    return


@app.cell
def _(
    df_650M_FT_DF_Time_1990_LR_5e_05,
    df_650M_FT_DF_Time_1995_LR_5e_05,
    df_650M_FT_DF_Time_2000_LR_5e_05,
    df_650M_FT_DF_Time_2005_LR_5e_05,
    df_650M_FT_DF_Time_2010_LR_5e_05,
    df_650M_FT_DF_Time_2020_LR_5e_05,
    df_650_FT_DF_1990_Lg_tree,
    df_650_FT_DF_2005_Lg_tree,
    df_650_FT_DF_2020_Lg_tree,
    df_650_FT_DF_All,
    merge_time,
):
    df_650_FT_DF_All_Time = merge_time(df_650_FT_DF_All, "h3n2")
    df_650_FT_DF_1990_Time = merge_time(df_650M_FT_DF_Time_1990_LR_5e_05, "h3n2")
    df_650_FT_DF_1995_Time = merge_time(df_650M_FT_DF_Time_1995_LR_5e_05, "h3n2")
    df_650_FT_DF_2000_Time = merge_time(df_650M_FT_DF_Time_2000_LR_5e_05, "h3n2")
    df_650_FT_DF_2005_Time = merge_time(df_650M_FT_DF_Time_2005_LR_5e_05, "h3n2")
    df_650_FT_DF_2010_Time = merge_time(df_650M_FT_DF_Time_2010_LR_5e_05, "h3n2")
    df_650_FT_DF_2020_Time = merge_time(df_650M_FT_DF_Time_2020_LR_5e_05, "h3n2")

    df_650_FT_DF_2020_Lg_tree_Time = merge_time(df_650_FT_DF_2020_Lg_tree, "h3n2-Large")
    df_650_FT_DF_2005_Lg_tree_Time = merge_time(df_650_FT_DF_2005_Lg_tree, "h3n2-Large")
    df_650_FT_DF_1990_Lg_tree_Time = merge_time(df_650_FT_DF_1990_Lg_tree, "h3n2-Large")
    return (
        df_650_FT_DF_1990_Lg_tree_Time,
        df_650_FT_DF_1990_Time,
        df_650_FT_DF_1995_Time,
        df_650_FT_DF_2000_Time,
        df_650_FT_DF_2005_Lg_tree_Time,
        df_650_FT_DF_2005_Time,
        df_650_FT_DF_2010_Time,
        df_650_FT_DF_2020_Lg_tree_Time,
        df_650_FT_DF_2020_Time,
        df_650_FT_DF_All_Time,
    )


@app.cell
def _(df_650_FT_DF_2020_Lg_tree_Time):
    df_650_FT_DF_2020_Lg_tree_Time
    return


@app.cell
def _(mo):
    mo.md(r"""### Calculate spearman CC for lg trees""")
    return


@app.cell
def _(
    df_650_FT_DF_1990_Lg_tree_Time,
    df_650_FT_DF_1990_Time,
    df_650_FT_DF_1995_Time,
    df_650_FT_DF_2000_Time,
    df_650_FT_DF_2005_Lg_tree_Time,
    df_650_FT_DF_2005_Time,
    df_650_FT_DF_2010_Time,
    df_650_FT_DF_2020_Lg_tree_Time,
    df_650_FT_DF_2020_Time,
    df_650_FT_DF_All_Time,
    spearman_correlation,
):
    def cal_spearman_cc(df):
        df['Model'] = df['Model'].str.replace('Fine_Tune_esm2_t33_650M_UR50D', 'Fine_Tune_650M')
        df_spear = spearman_correlation(df, "650M")
        df_spear = df_spear[df_spear['Model'] != '650M']

        return df_spear

    df_650_FT_DF_All_Time_spearman = cal_spearman_cc(df_650_FT_DF_All_Time)

    df_650_FT_DF_1990_Time_spearman = cal_spearman_cc(df_650_FT_DF_1990_Time)
    df_650_FT_DF_1995_Time_spearman = cal_spearman_cc(df_650_FT_DF_1995_Time)
    df_650_FT_DF_2000_Time_spearman = cal_spearman_cc(df_650_FT_DF_2000_Time)
    df_650_FT_DF_2005_Time_spearman = cal_spearman_cc(df_650_FT_DF_2005_Time)
    df_650_FT_DF_2010_Time_spearman = cal_spearman_cc(df_650_FT_DF_2010_Time)
    df_650_FT_DF_2020_Time_spearman = cal_spearman_cc(df_650_FT_DF_2020_Time)

    df_650_FT_DF_2020_Lg_tree_Time_spearman = cal_spearman_cc(df_650_FT_DF_2020_Lg_tree_Time)
    df_650_FT_DF_2005_Lg_tree_Time_spearman = cal_spearman_cc(df_650_FT_DF_2005_Lg_tree_Time)
    df_650_FT_DF_1990_Lg_tree_Time_spearman = cal_spearman_cc(df_650_FT_DF_1990_Lg_tree_Time)
    return (
        df_650_FT_DF_1990_Lg_tree_Time_spearman,
        df_650_FT_DF_1990_Time_spearman,
        df_650_FT_DF_1995_Time_spearman,
        df_650_FT_DF_2000_Time_spearman,
        df_650_FT_DF_2005_Lg_tree_Time_spearman,
        df_650_FT_DF_2005_Time_spearman,
        df_650_FT_DF_2010_Time_spearman,
        df_650_FT_DF_2020_Lg_tree_Time_spearman,
        df_650_FT_DF_2020_Time_spearman,
        df_650_FT_DF_All_Time_spearman,
    )


@app.cell
def _(df_650_FT_DF_1990_Lg_tree_Time):
    #df_650_FT_DF_1990_Lg_tree_Time_spearman
    df_650_FT_DF_1990_Lg_tree_Time.describe()
    return


@app.cell
def _(df_650_FT_DF_1990_Lg_tree_Time):
    df_650_FT_DF_1990_Lg_tree_Time
    return


@app.cell
def _(mo):
    mo.md(r"""### Get spearman fine tune summary""")
    return


@app.cell
def _(
    df_650_FT_DF_1990_Lg_tree_Time_spearman,
    df_650_FT_DF_1990_Time_spearman,
    df_650_FT_DF_1995_Time_spearman,
    df_650_FT_DF_2000_Time_spearman,
    df_650_FT_DF_2005_Lg_tree_Time_spearman,
    df_650_FT_DF_2005_Time_spearman,
    df_650_FT_DF_2010_Time_spearman,
    df_650_FT_DF_2020_Lg_tree_Time_spearman,
    df_650_FT_DF_2020_Time_spearman,
    df_650_FT_DF_All_Time_spearman,
    mean_spearman_segments,
):
    df_650_FT_DF_All_Time_spearman_mean = mean_spearman_segments(df_650_FT_DF_All_Time_spearman, "Fine Tune - 650M Model - Trained on all")

    df_650_FT_DF_1990_Time_spearman_mean = mean_spearman_segments(df_650_FT_DF_1990_Time_spearman, "Fine Tune - 650M Model - Trained on all")
    df_650_FT_DF_1995_Time_spearman_mean = mean_spearman_segments(df_650_FT_DF_1995_Time_spearman, "Fine Tune - 650M Model - Trained on all")
    df_650_FT_DF_2000_Time_spearman_mean = mean_spearman_segments(df_650_FT_DF_2000_Time_spearman, "Fine Tune - 650M Model - Trained on all")
    df_650_FT_DF_2005_Time_spearman_mean = mean_spearman_segments(df_650_FT_DF_2005_Time_spearman, "Fine Tune - 650M Model - Trained on all")
    df_650_FT_DF_2010_Time_spearman_mean = mean_spearman_segments(df_650_FT_DF_2010_Time_spearman, "Fine Tune - 650M Model - Trained on all")
    df_650_FT_DF_2020_Time_spearman_mean = mean_spearman_segments(df_650_FT_DF_2020_Time_spearman, "Fine Tune - 650M Model - Trained on all")

    df_650_FT_DF_2020_Lg_tree_Time_spearman_mean = mean_spearman_segments(df_650_FT_DF_2020_Lg_tree_Time_spearman, "Fine Tune - 650M Model - Trained up to 2020, LG Tree")
    df_650_FT_DF_2005_Lg_tree_Time_spearman_mean = mean_spearman_segments(df_650_FT_DF_2005_Lg_tree_Time_spearman, "Fine Tune - 650M Model - Trained up to 2020, LG Tree")
    df_650_FT_DF_1990_Lg_tree_Time_spearman_mean = mean_spearman_segments(df_650_FT_DF_1990_Lg_tree_Time_spearman, "Fine Tune - 650M Model - Trained up to 1990, LG Tree")
    return (
        df_650_FT_DF_1990_Lg_tree_Time_spearman_mean,
        df_650_FT_DF_1990_Time_spearman_mean,
        df_650_FT_DF_1995_Time_spearman_mean,
        df_650_FT_DF_2000_Time_spearman_mean,
        df_650_FT_DF_2005_Lg_tree_Time_spearman_mean,
        df_650_FT_DF_2005_Time_spearman_mean,
        df_650_FT_DF_2010_Time_spearman_mean,
        df_650_FT_DF_2020_Lg_tree_Time_spearman_mean,
        df_650_FT_DF_2020_Time_spearman_mean,
        df_650_FT_DF_All_Time_spearman_mean,
    )


@app.cell
def _(mo):
    mo.md(r"""### Generate spearman fine tune comparison continuation""")
    return


@app.cell
def _(
    df_650_FT_DF_1990_Lg_tree_Time_spearman_mean,
    df_650_FT_DF_1990_Time_spearman_mean,
    df_650_FT_DF_1995_Time_spearman_mean,
    df_650_FT_DF_2000_Time_spearman_mean,
    df_650_FT_DF_2005_Lg_tree_Time_spearman_mean,
    df_650_FT_DF_2005_Time_spearman_mean,
    df_650_FT_DF_2010_Time_spearman_mean,
    df_650_FT_DF_2020_Lg_tree_Time_spearman_mean,
    df_650_FT_DF_2020_Time_spearman_mean,
    df_650_FT_DF_All_Time_spearman_mean,
    plt,
    sns,
):
    def create_spearman_summary_plot_2(
        df,
        title="Spearman CC Summary All Models",
        xlabel="Time Range",
        ylabel="Spearman CC",
        save_path="Spearman_Correlation_Comparison.png"
    ):

        # Apply Seaborn theme and style:
        sns.set_style("whitegrid")
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        # Create the figure and axis:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Draw the line plot:
        ax = sns.lineplot(
            data=df,
            x="Time_Range",
            y="Spearman_Correlation",
            hue="Model",
            marker="o",
            legend=False,
            zorder=1,
            ax=ax
        )

        # Set titles and labels:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.tight_layout()

        # Save the figure to the specified path:
        plt.savefig(f"Flu_Figures/{save_path}", dpi=300, bbox_inches="tight")
        plt.show()
        #plt.close(fig)

        return save_path

    create_spearman_summary_plot_2(
        df_650_FT_DF_All_Time_spearman_mean,
        title="Spearman CC Summary All Time",
        xlabel="Time Range",
        ylabel="Spearman CC",
        save_path="Spearman_Correlation_Comparison_All_Time.png"
    )

    create_spearman_summary_plot_2(
        df_650_FT_DF_1990_Time_spearman_mean,
        title="Spearman CC Summary 1990 Time",
        xlabel="Time Range",
        ylabel="Spearman CC",
        save_path="Spearman_Correlation_Comparison_1990_Time.png"
    )

    create_spearman_summary_plot_2(
        df_650_FT_DF_1995_Time_spearman_mean,
        title="Spearman CC Summary 1995 Time",
        xlabel="Time Range",
        ylabel="Spearman CC",
        save_path="Spearman_Correlation_Comparison_1995_Time.png"
    )

    create_spearman_summary_plot_2(
        df_650_FT_DF_2000_Time_spearman_mean,
        title="Spearman CC Summary 2000 Time",
        xlabel="Time Range",
        ylabel="Spearman CC",
        save_path="Spearman_Correlation_Comparison_2000_Time.png"
    )

    create_spearman_summary_plot_2(
        df_650_FT_DF_2005_Time_spearman_mean,
        title="Spearman CC Summary 2005 Time",
        xlabel="Time Range",
        ylabel="Spearman CC",
        save_path="Spearman_Correlation_Comparison_2005_Time.png"
    )

    create_spearman_summary_plot_2(
        df_650_FT_DF_2010_Time_spearman_mean,
        title="Spearman CC Summary 2010 Time",
        xlabel="Time Range",
        ylabel="Spearman CC",
        save_path="Spearman_Correlation_Comparison_2010_Time.png"
    )

    create_spearman_summary_plot_2(
        df_650_FT_DF_2020_Time_spearman_mean,
        title="Spearman CC Summary 2020 Time",
        xlabel="Time Range",
        ylabel="Spearman CC",
        save_path="Spearman_Correlation_Comparison_2020_Time.png"
    )

    create_spearman_summary_plot_2(
        df_650_FT_DF_1990_Lg_tree_Time_spearman_mean,
        title="Spearman CC Summary LG Tree 1990",
        xlabel="Time Range",
        ylabel="Spearman CC",
        save_path="Spearman_Correlation_Comparison_LG_Tree_1990.png"
    )

    create_spearman_summary_plot_2(
        df_650_FT_DF_2005_Lg_tree_Time_spearman_mean,
        title="Spearman CC Summary LG Tree 2005",
        xlabel="Time Range",
        ylabel="Spearman CC",
        save_path="Spearman_Correlation_Comparison_LG_Tree_2005.png"
    )

    create_spearman_summary_plot_2(
        df_650_FT_DF_2020_Lg_tree_Time_spearman_mean,
        title="Spearman CC Summary LG Tree 2020",
        xlabel="Time Range",
        ylabel="Spearman CC",
        save_path="Spearman_Correlation_Comparison_LG_Tree_2020.png"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Combine spearman CC comparison plots""")
    return


@app.cell
def _(Line2D, pd, plt, sns):
    def create_combined_spearman_overlay_plot(
        df_list,
        labels,
        title="Combined Spearman CC Summary",
        xlabel="Time Range",
        ylabel="Spearman CC",
        save_path="Combined_Spearman_Correlation_Overlay.png"
    ):
        # Add source labels
        for df, label in zip(df_list, labels):
            df["Source"] = label

        # Combine into one DataFrame
        combined_df = pd.concat(df_list, ignore_index=True)

        # Set Seaborn theme
        sns.set_style("whitegrid")
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        # Set up figure
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # Custom colors and linestyles
        color_map = {
            "1990 Small Tree": "#1f77b4",  # blue
            "2005 Small Tree": "#1f77b4",
            "1990 Large Tree": "#ff7f0e",  # orange
            "2005 Large Tree": "#ff7f0e",
        }

        linestyle_map = {
            "1990 Small Tree": "solid",
            "2005 Small Tree": "dashed",
            "1990 Large Tree": "solid",
            "2005 Large Tree": "dashed",
        }

        # Plot each line manually with markers, but legend will use line-only handles
        for label in labels:
            subset = combined_df[combined_df["Source"] == label]
            sns.lineplot(
                data=subset,
                x="Time_Range",
                y="Spearman_Correlation",
                label=None,  # prevent automatic legend entry
                marker="o",
                ax=ax,
                color=color_map[label],
                linestyle=linestyle_map[label]
            )

        # Custom legend with no markers
        legend_handles = [
            Line2D(
                [0], [0],
                color=color_map[label],
                linestyle=linestyle_map[label],
                linewidth=2,
                label=label
            )
            for label in labels
        ]
        ax.legend(handles=legend_handles, title="Tree & Year")

        # Final formatting
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(f"Flu_Figures/{save_path}", dpi=300, bbox_inches="tight")
        plt.show()

    return (create_combined_spearman_overlay_plot,)


@app.cell
def _(
    create_combined_spearman_overlay_plot,
    df_650_FT_DF_1990_Lg_tree_Time_spearman_mean,
    df_650_FT_DF_1990_Time_spearman_mean,
    df_650_FT_DF_2005_Lg_tree_Time_spearman_mean,
    df_650_FT_DF_2005_Time_spearman_mean,
):
    create_combined_spearman_overlay_plot(
        df_list=[
            df_650_FT_DF_1990_Time_spearman_mean,
            df_650_FT_DF_2005_Time_spearman_mean,
            df_650_FT_DF_1990_Lg_tree_Time_spearman_mean,
            df_650_FT_DF_2005_Lg_tree_Time_spearman_mean
        ],
        labels=[
            "1990 Small Tree",
            "2005 Small Tree",
            "1990 Large Tree",
            "2005 Large Tree"
        ],
        title="Spearman CC 1990, 2005 for large and small H3N2 trees",
        save_path="Spearman_Correlation_Overlay_1990_to_2000.png"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Individual ESM vs Time Scatterplot""")
    return


@app.cell
def _(cm, colorsys, df_3B_FT_DF_Time, df_650_FT_DF_Time, os, plt):
    def esm_vs_time_scatterplot_seperate(model_df, model_name):
        for segment, group in model_df.groupby('Segment'):

                def darken_color(rgb, factor=0.7):
                    h, l, s = colorsys.rgb_to_hls(*rgb)
                    r, g, b = colorsys.hls_to_rgb(h, max(0, l * factor), s)
                    return (r, g, b, 1.0)

                def plot_esm_score(ax, df, title, Fine_Tune=False):
                    norm = plt.Normalize(df["log_likelyhood"].min(), df["log_likelyhood"].max())
                    cmap = plt.get_cmap("viridis")
                    colors = cmap(norm(df["log_likelyhood"]))
                    edgecolors = [darken_color(c[:3], factor=0.7) for c in colors]
                    sc = ax.scatter(
                        df["time"],
                        df["log_likelyhood"],
                        c=colors,
                        edgecolors=edgecolors,
                        linewidths=0.5,
                        alpha=0.7,
                        zorder=1
                    )

                    high_freq_df = df[df["max_frequency"] >= 1].sort_values("time")
                    ax.plot(
                        high_freq_df["time"],
                        high_freq_df["log_likelyhood"],
                        linestyle='-',
                        color='black',
                        linewidth=3,
                        alpha=0.6,
                        label='Max Freq ≥ 0.99',
                        zorder=2
                    )

                    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
                    ax.set_title(title)
                    if Fine_Tune:
                        ax.axvline(1990, color='gray', linestyle='--', linewidth=1.5)
                    ax.set_ylabel("ESM Score")
                    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.75)
                    ax.spines[['right', 'top']].set_visible(False)
                    return ax

                df1 = model_df[model_df['Model'] == f"Fine_Tune_{model_name}"]
                df1 = df1[df1['Segment'] == segment]
                if segment == "PA":
                    df1 = df1[df1['node'] != 'A/Viamao/LACENRS-974/2015']

                df2 = model_df[model_df['Model'] == model_name]
                df2 = df2[df2['Segment'] == segment]

                if segment == "PA":
                    df2 = df2[df2['node'] != 'A/Viamao/LACENRS-974/2015']

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
                plot_esm_score(ax1, df1, f"{segment.upper()} - 3B Fine Tune Model", Fine_Tune=True)
                plot_esm_score(ax2, df2, f"{segment.upper()} - 3B Base Model")
                ax2.set_xlabel("Date")

                os.makedirs(f"Flu_Figures/Combined_{model_name}_v_{model_name}_TN_Scatterplots/", exist_ok=True)
                plt.savefig(f"Flu_Figures/Combined_{model_name}_v_{model_name}_TN_Scatterplots/{segment}_{model_name}_v_{model_name}_TN_Scatterplots.png", dpi=300)

                plt.tight_layout()
                plt.show()

    esm_vs_time_scatterplot_seperate(df_3B_FT_DF_Time, "3B")
    esm_vs_time_scatterplot_seperate(df_650_FT_DF_Time, "650M")
    return


@app.cell
def _(mo):
    mo.md(r"""### Combined ESM vs Time Scatterplot""")
    return


@app.cell
def _(cm, colorsys, df_3B_FT_DF_Time, df_650_FT_DF_Time, plt):
    #combine exm v time into 4x4 grid

    def darken_color(rgb, factor=0.7):
        h, l, s = colorsys.rgb_to_hls(*rgb)
        r, g, b = colorsys.hls_to_rgb(h, max(0, l * factor), s)
        return (r, g, b, 1.0)

    def plot_esm_score(ax, df, title, Fine_Tune=False):
        norm = plt.Normalize(df["log_likelyhood"].min(), df["log_likelyhood"].max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(df["log_likelyhood"]))
        edgecolors = [darken_color(c[:3], factor=0.7) for c in colors]
        sc = ax.scatter(
            df["time"],
            df["log_likelyhood"],
            c=colors,
            edgecolors=edgecolors,
            linewidths=0.5,
            alpha=0.7,
            zorder=1
        )
        high_freq_df = df[df["max_frequency"] >= 1].sort_values("time")
        ax.plot(
            high_freq_df["time"],
            high_freq_df["log_likelyhood"],
            linestyle='-',
            color='black',
            linewidth=3,
            alpha=0.6,
            label='Max Freq ≥ 0.99',
            zorder=2
        )
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation='vertical')
        ax.set_title(title)
        if Fine_Tune:
            ax.axvline(1990, color='gray', linestyle='--', linewidth=1.5)
        ax.set_ylabel("ESM Score")
        ax.grid(True, color='lightgray', linestyle='-', linewidth=0.75)
        ax.spines[['right', 'top']].set_visible(False)
        return ax

    def esm_vs_time_4x4_grid(model_df, model_name):
        segments = sorted(model_df['Segment'].unique())
        fig, axs = plt.subplots(4, 4, figsize=(20, 13), sharex=True)
        axs = axs.flatten()

        for i, segment in enumerate(segments):
            df1 = model_df[(model_df['Model'] == f"Fine_Tune_{model_name}") & (model_df['Segment'] == segment)]
            df2 = model_df[(model_df['Model'] == model_name) & (model_df['Segment'] == segment)]

            if segment == "PA":
                df1 = df1[df1['node'] != 'A/Viamao/LACENRS-974/2015']
                df2 = df2[df2['node'] != 'A/Viamao/LACENRS-974/2015']

            ax1 = axs[2*i]
            ax2 = axs[2*i + 1]
            plot_esm_score(ax1, df1, f"{segment.upper()} - {model_name} Fine Tune", Fine_Tune=True)
            plot_esm_score(ax2, df2, f"{segment.upper()} - {model_name} Base")

            if i >= 6: 
                ax1.set_xlabel("Date")
                ax2.set_xlabel("Date")

        plt.tight_layout()
        #os.makedirs(f"Flu_Figures/Combined_{model_name}_Grid/", exist_ok=True)
        #plt.savefig(f"Flu_Figures/Combined_{model_name}_Grid/{model_name}_4x4_grid.png", dpi=300)
        plt.savefig(f"Flu_Figures/{model_name}_4x4_grid.png", dpi=300)
        plt.show()

    esm_vs_time_4x4_grid(df_3B_FT_DF_Time, "3B")
    esm_vs_time_4x4_grid(df_650_FT_DF_Time, "650M")
    return


@app.cell
def _(mo):
    mo.md(r"""### Combine time series dataframes""")
    return


@app.cell
def _(
    df_650M_FT_DF_Time_1990_LR_5e_05,
    df_650M_FT_DF_Time_1995_LR_5e_05,
    df_650M_FT_DF_Time_2000_LR_5e_05,
    df_650M_FT_DF_Time_2005_LR_5e_05,
    df_650M_FT_DF_Time_2010_LR_5e_05,
    pd,
):
    df_650M_FT_DF_Time_1990_LR_5e_05["Model_training_time"] = 1990
    df_650M_FT_DF_Time_1995_LR_5e_05["Model_training_time"] = 1995
    df_650M_FT_DF_Time_2000_LR_5e_05["Model_training_time"] = 2000
    df_650M_FT_DF_Time_2005_LR_5e_05["Model_training_time"] = 2005
    df_650M_FT_DF_Time_2010_LR_5e_05["Model_training_time"] = 2010

    df_650_FT_DF_Time_Series_Validation = pd.concat([
        df_650M_FT_DF_Time_1990_LR_5e_05,
        df_650M_FT_DF_Time_1995_LR_5e_05,
        df_650M_FT_DF_Time_2000_LR_5e_05,
        df_650M_FT_DF_Time_2005_LR_5e_05,
        df_650M_FT_DF_Time_2010_LR_5e_05,
    ], ignore_index=True)
    return (df_650_FT_DF_Time_Series_Validation,)


@app.cell
def _(mo):
    mo.md(r"""### Add time to time series dataframe""")
    return


@app.cell
def _(df_650_FT_DF_Time_Series_Validation, merge_time):
    df_650_FT_DF_Time_Series_Validation_Time = merge_time(df_650_FT_DF_Time_Series_Validation, "h3n2")
    return (df_650_FT_DF_Time_Series_Validation_Time,)


@app.cell
def _(mo):
    mo.md(r"""### Separate and filter time series dataframe""")
    return


@app.cell
def _(df_650_FT_DF_Time_Series_Validation_Time):
    df_650_FT_DF_Time_Series_Validation_Time_1991_1 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 1991) & (df_650_FT_DF_Time_Series_Validation_Time['time'] <= 2001) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 1991)]
    df_650_FT_DF_Time_Series_Validation_Time_1991_2 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 2001) & (df_650_FT_DF_Time_Series_Validation_Time['time'] <= 2011) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 1991)]
    df_650_FT_DF_Time_Series_Validation_Time_1991_3 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 2011) & (df_650_FT_DF_Time_Series_Validation_Time['time'] <= 2021) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 1991)]

    df_650_FT_DF_Time_Series_Validation_Time_1996_1 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 1996) & (df_650_FT_DF_Time_Series_Validation_Time['time'] <= 2006) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 1996)]
    df_650_FT_DF_Time_Series_Validation_Time_1996_2 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 2006) & (df_650_FT_DF_Time_Series_Validation_Time['time'] <= 2016) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 1996)]
    df_650_FT_DF_Time_Series_Validation_Time_1996_3 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 2016) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 1996)]

    df_650_FT_DF_Time_Series_Validation_Time_2001_1 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 2001) & (df_650_FT_DF_Time_Series_Validation_Time['time'] <= 2011) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 2001)]
    df_650_FT_DF_Time_Series_Validation_Time_2001_2 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 2011) & (df_650_FT_DF_Time_Series_Validation_Time['time'] <= 2021) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 2001)]
    df_650_FT_DF_Time_Series_Validation_Time_2001_3 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 2021) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 2001)]

    df_650_FT_DF_Time_Series_Validation_Time_2006_1 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 2006) & (df_650_FT_DF_Time_Series_Validation_Time['time'] <= 2016) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 2006)]
    df_650_FT_DF_Time_Series_Validation_Time_2006_2 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 2016) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 2006)]

    df_650_FT_DF_Time_Series_Validation_Time_2011_1 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 2011) & (df_650_FT_DF_Time_Series_Validation_Time['time'] <= 2021) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 2011)]
    df_650_FT_DF_Time_Series_Validation_Time_2011_2 = df_650_FT_DF_Time_Series_Validation_Time[(df_650_FT_DF_Time_Series_Validation_Time['time'] >= 2021) & (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == 2011)]
    return


@app.cell
def _(df_650_FT_DF_Time_Series_Validation_Time):
    df_650_FT_DF_Time_Series_Validation_Time
    return


@app.cell
def _(mo):
    mo.md(r"""### Calculate spearman CC for each time frame""")
    return


@app.cell
def _(df_650_FT_DF_Time_Series_Validation_Time, pd, spearmanr):
    def calculate_time_series_cross_df():
        time_bins = {
            "1990": [(1990, 2000), (2000, 2010), (2010, 2020)],
            "1995": [(1995, 2005), (2005, 2015), (2015, None)],
            "2000": [(2000, 2010), (2010, 2020), (2020, None)],
            "2005": [(2005, 2015), (2015, None)],
            "2010": [(2010, 2020), (2020, None)]
        }

        results = []

        for start_year, ranges in time_bins.items():
            for idx, (start, end) in enumerate(ranges, 1):
                if end is None:
                    df_bin = df_650_FT_DF_Time_Series_Validation_Time[
                    (df_650_FT_DF_Time_Series_Validation_Time['time'] >= start) &
                    (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == int(start_year))
                    ]
                else:
                    df_bin = df_650_FT_DF_Time_Series_Validation_Time[
                        (df_650_FT_DF_Time_Series_Validation_Time['time'] >= start) &
                        (df_650_FT_DF_Time_Series_Validation_Time['time'] <= end) &
                        (df_650_FT_DF_Time_Series_Validation_Time['Model_training_time'] == int(start_year))
                    ]


                corr, _ = spearmanr(df_bin['max_frequency'], df_bin['log_likelyhood'])


                results.append({
                    "start_year": start_year,
                    "bin_index": idx,
                    "range": f"{start}-{end if end else '2025'}",
                    "spearman_corr": corr
                })

        spearman_df = pd.DataFrame(results)
        return spearman_df


    spearman_df = calculate_time_series_cross_df()
    return (spearman_df,)


@app.cell
def _(spearman_df):
    spearman_df
    return


@app.cell
def _(mo):
    mo.md(r"""### Create time series cross validation plot""")
    return


@app.cell
def _(plt, sns, spearman_df):
    def create_time_series_plot(spearman_df, name):
        unique_years = spearman_df['start_year'].unique()
        n_years = len(unique_years)

        fig, axes = plt.subplots(1, n_years, figsize=(5 * n_years, 5), sharey=True)

        if n_years == 1:
            axes = [axes]

        for ax, year in zip(axes, unique_years):
            subset = spearman_df[spearman_df['start_year'] == year]
            sns.barplot(data=subset, x='range', y='spearman_corr', hue='range', palette="viridis", errorbar=None, ax=ax)
            ax.set_title(f"Model Trained up to: {int(year)}")
            ax.set_xlabel("Time Range")
            ax.set_ylabel("Spearman Correlation Coefficient")

        plt.tight_layout()

        plt.savefig(f"Flu_Figures/{name}", dpi=300, bbox_inches='tight')

        plt.show()


    create_time_series_plot(spearman_df, "Spearman_Correlation_Comparison_Fine_Tune_time_series_cross_validation.png")
    return (create_time_series_plot,)


@app.cell
def _(mo):
    mo.md(r"""### Time series cross validation dummy plot""")
    return


@app.cell
def _(spearman_df):
    spearman_df
    return


@app.cell
def _(pd):
    data = {
        "start_year": [
            1990, 1990, 1990,
            1995, 1995, 1995,
            2000, 2000, 2000,
            2005, 2005,
            2010, 2010
        ],
        "bin_index": [
            1, 2, 3,
            1, 2, 3,
            1, 2, 3,
            1, 2,
            1, 2
        ],
        "range": [
            "1990-2000", "2000-2010", "2010-2020",
            "1995-2005", "2005-2015", "2015-2025",
            "2000-2010", "2010-2020", "2020-2025",
            "2005-2015", "2015-2025",
            "2010-2020", "2020-2025"
        ],
        "spearman_corr": [
            0.2, 0.1, 0.05,  
            0.35, 0.25, 0.15, 
            0.4, 0.3, 0.2,  
            0.45, 0.35,  
            0.5, 0.45
        ]
    }

    spearman_df_dummy = pd.DataFrame(data)
    print(spearman_df_dummy)

    return (spearman_df_dummy,)


@app.cell
def _(create_time_series_plot, spearman_df_dummy):
    create_time_series_plot(spearman_df_dummy, "Spearman_Correlation_Comparison_Fine_Tune_time_series_cross_validation_dummy.png")
    return


@app.cell
def _(mo):
    mo.md(r"""### Load small and large fine tune trees trained on 2005 and 2020""")
    return


@app.cell
def _(load_fine_tune_results, pd):
    df_SM_Tree_2005 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~2005",
    )

    df_LG_Tree_2005 = load_fine_tune_results(
        "next_tree~h3n2-Large",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~2005",
    )

    df_SM_Tree_2020 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~2020",
    )

    df_LG_Tree_2020 = load_fine_tune_results(
        "next_tree~h3n2-Large",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~2020",
    )

    df_both_trees_2005 = pd.concat([df_SM_Tree_2005, df_LG_Tree_2005], ignore_index=True)
    df_both_trees_2020 = pd.concat([df_SM_Tree_2020, df_LG_Tree_2020], ignore_index=True)
    #df_both_trees_1990 = pd.concat([df_SM_Tree_1990, df_LG_Tree_1990], ignore_index=True)
    return df_both_trees_2005, df_both_trees_2020


@app.cell
def _(mo):
    mo.md(r"""### Add time to small and large trees""")
    return


@app.cell
def _(df_both_trees_2005, df_both_trees_2020, merge_time, pd):
    df_SM_Tree_Time_2005 = merge_time(df_both_trees_2005, "h3n2")
    df_LG_Tree_Time_2005 = merge_time(df_both_trees_2005, "h3n2-Large")
    df_SM_Tree_Time_2020 = merge_time(df_both_trees_2020, "h3n2")
    df_LG_Tree_Time_2020 = merge_time(df_both_trees_2020, "h3n2-Large")
    #df_SM_Tree_Time_1990 = merge_time(df_both_trees_1990, "h3n2")
    #df_LG_Tree_Time_1990 = merge_time(df_both_trees_1990, "h3n2-Large")

    #df_both_trees_Time_1990 = pd.concat([df_SM_Tree_Time_1990, df_LG_Tree_Time_1990], ignore_index=True)
    df_both_trees_Time_2005 = pd.concat([df_SM_Tree_Time_2005, df_LG_Tree_Time_2005], ignore_index=True)
    df_both_trees_Time_2020 = pd.concat([df_SM_Tree_Time_2020, df_LG_Tree_Time_2020], ignore_index=True)
    return (
        df_LG_Tree_Time_2005,
        df_both_trees_Time_2005,
        df_both_trees_Time_2020,
    )


@app.cell
def _(mo):
    mo.md(r"""### Create small and large tree plots for 2005 and 2020""")
    return


@app.cell
def _(
    df_both_trees_Time_2005,
    df_both_trees_Time_2020,
    gridspec,
    os,
    plt,
    re,
    sns,
    spearmanr,
):
    #Create fine tune vs base model figures

    def max_freq_LL_figures_small_vs_large_tree(dataset, model, time):

        sns.set_theme(style="whitegrid")
        sns.set_style("ticks")

        output_dir = os.path.join(f"Flu_Figures/ESM_vs_Max_Freq_Plots_Fine_Tune_Large_V_Small_Tree_{model}_{time}")

        os.makedirs(output_dir, exist_ok=True)

        def plot_regression(ax, data, x_col, y_col, title, ylabel="", color="#0a2463"):

            old_mask = data["time"] < int(time)
            recent_mask = ~old_mask

            ax.scatter(data.loc[old_mask, x_col], data.loc[old_mask, y_col],
                       s=50, alpha=0.35, color="lightgray", label="< int(time)")

            ax.scatter(data.loc[recent_mask, x_col], data.loc[recent_mask, y_col],
                       s=50, alpha=0.35, color=color, label="≥ int(time)")

            light_recent = sns.desaturate(color, 0.5)

            if old_mask.sum() >= 2:
                sns.regplot(data=data.loc[old_mask], x=x_col, y=y_col, ax=ax,
                            scatter=False,
                            line_kws={"color": "gray", "linestyle": "--"}, ci=None)
            if recent_mask.sum() >= 2:
                sns.regplot(data=data.loc[recent_mask], x=x_col, y=y_col, ax=ax,
                            scatter=False,
                            line_kws={"color": light_recent}, ci=None)


            rho_all, p_all = spearmanr(data[y_col], data[x_col])
            rho_old, p_old = spearmanr(data.loc[old_mask, y_col], data.loc[old_mask, x_col])
            rho_recent, p_recent = spearmanr(data.loc[recent_mask, y_col], data.loc[recent_mask, x_col])

            textstr = (
                f"ρ(<{time}) = {rho_old:.2f}")

            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, color="gray",
                    verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.0))

            textstr = (
                f"ρ(≥{time}) = {rho_recent:.2f}")

            ax.text(0.05, 0.85, textstr, transform=ax.transAxes, fontsize=10, color=light_recent,
                    verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.0))

            ax.set_title(title)
            ax.set_xlabel(" ", weight="bold")
            ax.set_ylabel(ylabel, weight="bold")
            ax.set_xlim(data[x_col].min(), data[x_col].max())
            ax.set_ylim(0, 1.1)
            #ax.legend(frameon=False)

        for segment, group in dataset.groupby('Segment'):

            fig = plt.figure(figsize=(4 * 2, 4))

            df = dataset[dataset['Segment'] == segment]

            if segment == "PA":
                df = df[df['node'] != 'A/Viamao/LACENRS-974/2015']

            df = df[df["log_likelyhood"] > -200]

            df_650 = df[df['tree'] == "h3n2"]
            df_650_Fine_Tune = df[df['tree'] == "h3n2-Large"]


            gs_main = gridspec.GridSpec(1, 2)  

            ax = fig.add_subplot(gs_main[0, 0])       
            ax2 = fig.add_subplot(gs_main[0, 1])     

            rmv_uns = re.sub(r'_+', ' ', model) 

            #blue_small_tree = r"$\bf{\color{blue}{Small\ Tree}}$"
            #title_small = f"{segment.upper()} - {rmv_uns} Model - {blue_small_tree}"

            base_title = f"{segment.upper()} – {rmv_uns} Model –           "
            ax.set_title(base_title, fontweight="bold")          

            ax.text(0.93, 1.03, "Small Tree", transform=ax.transAxes,
            ha="center", color="#0a2463", fontweight="bold")

            ax2.text(0.93, 1.03, "Large Tree", transform=ax2.transAxes,
            ha="center", color="#f4d35e", fontweight="bold")

            plot_regression(ax, df_650, "log_likelyhood", "max_frequency", base_title, ylabel="Max Frequency")
            plot_regression(ax2, df_650_Fine_Tune, "log_likelyhood", "max_frequency", base_title, color='#f4d35e')

            fig.text(0.53, 0.048, 'Log Likelihood', va='bottom', ha='center', fontsize=12, weight='bold')

            ax.spines[['right', 'top']].set_visible(False)
            ax2.spines[['right', 'top']].set_visible(False)

            plt.tight_layout()

            plt.savefig(f"{output_dir}/{segment}_LL_vs_Max_Frequency_Fine_Tune_{model}.png", dpi=300)

            plt.show()

    max_freq_LL_figures_small_vs_large_tree(df_both_trees_Time_2005, "Fine_Tune_650M", "2005")
    max_freq_LL_figures_small_vs_large_tree(df_both_trees_Time_2020, "Fine_Tune_650M", "2020")
    return


@app.cell
def _(mo):
    mo.md(r"""### Summary spearman large tree""")
    return


@app.cell
def _(df_LG_Tree_Time_2005, df_both_trees_Time_2005):
    df_LG_Tree_Time_2005

    #filter above 1990
    df_both_trees_Above_2005 = df_both_trees_Time_2005[df_both_trees_Time_2005['time'] >= 2005]
    df_both_trees_below_2005 = df_both_trees_Time_2005[df_both_trees_Time_2005['time'] < 2005]
    return (df_both_trees_below_2005,)


@app.cell
def _(df_both_trees_below_2005):
    df_both_trees_below_2005
    return


@app.cell
def _(mo):
    mo.md(r"""### LOESS Fit""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Recreating Trevor's fix""")
    return


@app.cell
def _(np, plt):
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters for the binormal distribution
    mean = [-100, 0.5]
    var1 = 1
    var2 = 0.5
    corr = 0.9
    cov = corr * np.sqrt(var1 * var2)

    # Covariance matrix
    cov_matrix = [[var1, cov],
                  [cov, var2]]

    # Generate 1000 samples from the bivariate normal distribution
    data_fake = np.random.multivariate_normal(mean, cov_matrix, size=1000)

    # Clip the second dimension values to the range [0, 1]
    data_fake[:, 1] = np.clip(data_fake[:, 1], 0, 1)

    # Separate the dimensions for plotting
    esm_scores = data_fake[:, 0]
    max_freq = data_fake[:, 1]

    # Plotting
    plt.figure()
    plt.scatter(esm_scores, max_freq, s=10)
    plt.xlabel("ESM score")
    plt.ylabel("max freq")
    plt.title("Bivariate Normal Samples with Clipped max freq")
    plt.grid(True)
    plt.show()

    return data_fake, esm_scores, max_freq


@app.cell
def _(esm_scores, max_freq, spearmanr):
    print(spearmanr(esm_scores, max_freq))
    return


@app.cell
def _(data_fake):
    data_fake
    return


@app.cell
def _(esm_scores, max_freq, np, plt):
    # Add a secular trend to the ESM score based on a random year between 1970 and 2020
    years = np.random.uniform(1970, 2020, size=1000)
    adjusted_esm_scores = esm_scores + 0.25 * (years - 1970)

    # Combine into adjusted data format: (ESM, max_freq, year)
    adj_data = list(zip(adjusted_esm_scores, max_freq, years))

    # Separate trunk (max_freq == 1) and side branches (max_freq < 1)
    # Since values were clipped to [0, 1], equality check works
    sidebranch_points = [(year, esm) for esm, freq, year in adj_data if freq < 1]
    trunk_points = [(year, esm) for esm, freq, year in adj_data if freq == 1]

    # Plot the points
    plt.figure(figsize=(10, 6))
    if sidebranch_points:
        plt.scatter(*zip(*sidebranch_points), color='gray', s=10, label='Side branches')
    if trunk_points:
        plt.scatter(*zip(*trunk_points), color='red', s=10, label='Trunk')
    plt.xlabel("Year")
    plt.ylabel("Adjusted ESM score")
    plt.title("ESM Score with Secular Trend (Trunk vs. Side Branches)")
    plt.legend()
    plt.grid(True)
    plt.show()
    return (adj_data,)


@app.cell
def _(adj_data, pd):
    # Convert to Pandas DataFrame
    df_fake = pd.DataFrame(adj_data, columns=["esm_score", "max_freq", "year"])

    # Optionally, mark whether each point is trunk or side branch
    df_fake["branch_type"] = df_fake["max_freq"].apply(lambda x: "trunk" if x == 1 else "side")

    df_fake
    return (df_fake,)


@app.cell
def _(df_fake, spearmanr):
    print(spearmanr(df_fake["esm_score"], df_fake["max_freq"]))
    return


@app.cell
def _(df_fake, plt):
    plt.figure()
    plt.scatter(df_fake["esm_score"], df_fake["max_freq"], s=10)
    plt.xlabel("ESM score")
    plt.ylabel("max freq")
    plt.title("Bivariate Normal Samples with Clipped max freq")
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(np):
    def tricube_weight(x):
        """Tricube kernel function"""
        x = np.abs(x)
        return np.where(x < 1, (1 - x ** 3) ** 3, 0)

    def loess_distance(x0, x, alpha):
        """Distance scale factor for LOESS"""
        n = len(x)
        span = max(1, int(np.ceil(alpha * n)))
        distances = np.abs(x - x0)
        sorted_distances = np.sort(distances)
        return sorted_distances[span - 1]  # the span-th smallest distance

    def loess_weights(x0, x, alpha):
        """Tricube weights based on scaled distances"""
        d = loess_distance(x0, x, alpha)
        if d == 0:
            # Avoid divide-by-zero when all x == x0
            return np.where(x == x0, 1.0, 0.0)
        scaled = (x - x0) / d
        return tricube_weight(scaled)

    def weighted_least_squares(x, y, w, degree=1):
        """Weighted polynomial fit at a single x0"""
        X = np.vstack([x ** i for i in range(degree + 1)]).T  # Design matrix
        W = np.diag(w)
        beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
        return beta

    def loess_single(x0, x, y, alpha=0.75, degree=1):
        """Compute LOESS estimate at a single point x0"""
        w = loess_weights(x0, x, alpha)
        beta = weighted_least_squares(x, y, w, degree)
        X0 = np.array([x0 ** i for i in range(degree + 1)])
        return X0 @ beta

    def loess_fit(x_values, data_x, data_y, alpha=0.75, degree=1):
        """Full LOESS curve"""
        return np.array([loess_single(x0, data_x, data_y, alpha, degree) for x0 in x_values])

    return (loess_fit,)


@app.cell
def _(df_fake, loess_fit, np, plt):
    # x and y data
    x_data = df_fake["year"].values
    y_data = df_fake["esm_score"].values

    # X values where we want the smoothed curve
    x_fit = np.linspace(1970, 2020, 300)
    y_fit = loess_fit(x_fit, x_data, y_data, alpha=0.15, degree=1)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Scatter trunk/side points
    side = df_fake[df_fake["branch_type"] == "side"]
    trunk = df_fake[df_fake["branch_type"] == "trunk"]
    plt.scatter(side["year"], side["esm_score"], color='gray', s=10, label="Side branches")
    plt.scatter(trunk["year"], trunk["esm_score"], color='red', s=10, label="Trunk")

    # LOESS line
    plt.plot(x_fit, y_fit, color='black', linewidth=2, label="Custom LOESS")

    plt.xlabel("Year")
    plt.ylabel("Adjusted ESM score")
    plt.title("ESM Score with Custom LOESS Fit")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(df_fake, loess_fit, plt):
    # Compute LOESS-smoothed values for each year in the data
    loess_smoothed = loess_fit(df_fake["year"].values, df_fake["year"].values, df_fake["esm_score"].values, alpha=0.15, degree=1)

    # Compute residuals: corrected ESM score = original - LOESS-smoothed
    df_fake["corrected_esm_score"] = df_fake["esm_score"] - loess_smoothed

    # Extract corrected sidebranch and trunk points
    side_corr = df_fake[df_fake["branch_type"] == "side"]
    trunk_corr = df_fake[df_fake["branch_type"] == "trunk"]

    # Plot the corrected residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(side_corr["year"], side_corr["corrected_esm_score"], color='gray', s=10, label='Side branches')
    plt.scatter(trunk_corr["year"], trunk_corr["corrected_esm_score"], color='red', s=10, label='Trunk')

    plt.xlabel("Year")
    plt.ylabel("Corrected ESM score")
    plt.title("ESM Residuals After Regressing Out LOESS Trend")
    plt.legend()
    plt.grid(True)
    plt.show()

    return


@app.cell
def _(df_fake, plt):
    plt.figure(figsize=(8, 6))
    plt.scatter(df_fake["corrected_esm_score"], df_fake["max_freq"], s=10)
    plt.xlabel("Corrected ESM score")
    plt.ylabel("Max freq")
    plt.title("Corrected ESM vs Max Frequency")
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(df_fake, spearmanr):
    print(spearmanr(df_fake["corrected_esm_score"], df_fake["max_freq"]))
    return


@app.cell
def _(mo):
    mo.md(r"""### LOESS Function fix""")
    return


@app.cell
def _(df_650_FT_DF_Time):
    df_650_FT_DF_Time_ONLY_FT_HA = df_650_FT_DF_Time[(df_650_FT_DF_Time['Segment'] == 'HA') & (df_650_FT_DF_Time['Model'] == 'Fine_Tune_650M')]
    return (df_650_FT_DF_Time_ONLY_FT_HA,)


@app.cell
def _(
    df_650_FT_DF_1990_Lg_tree_Time,
    df_650_FT_DF_1990_Time,
    df_650_FT_DF_2005_Lg_tree_Time,
    df_650_FT_DF_2005_Time,
    pd,
):
    combined_LG_SM_1990_2005 = pd.concat([df_650_FT_DF_1990_Time, df_650_FT_DF_2005_Time, df_650_FT_DF_1990_Lg_tree_Time, df_650_FT_DF_2005_Lg_tree_Time], ignore_index=True)
    return (combined_LG_SM_1990_2005,)


@app.cell
def _(
    combined_LG_SM_1990_2005,
    df_3B_FT_DF_Time,
    df_650_FT_DF_Time,
    df_650_FT_DF_Time_Series_Validation_Time,
):
    df_650_FT_DF_Time_export = df_650_FT_DF_Time.rename(columns={'log_likelyhood': 'log_likelihood'})
    df_650_FT_DF_Time_export.to_csv("Notebooks/Dataframes/650M_Fine_Tune_Up_To_1990.csv", index = 0)

    df_3B_FT_DF_Time_export = df_3B_FT_DF_Time.rename(columns={'log_likelyhood': 'log_likelihood'})
    df_3B_FT_DF_Time_export.to_csv("Notebooks/Dataframes/3B_Fine_Tune_Up_To_1990.csv", index = 0)

    combined_LG_SM_1990_2005_export = combined_LG_SM_1990_2005.rename(columns={'log_likelyhood': 'log_likelihood'})
    combined_LG_SM_1990_2005_export.to_csv("Notebooks/Dataframes/LG_SM_1990_2005.csv", index = 0)

    df_650_FT_DF_Time_Series_Validation_Time_export = df_650_FT_DF_Time_Series_Validation_Time.rename(columns={'log_likelyhood': 'log_likelihood'})
    df_650_FT_DF_Time_Series_Validation_Time_export.to_csv("Notebooks/Dataframes/df_650_FT_DF_Time_Series_Validation.csv", index = 0)
    return


@app.cell
def _(plt, sns, spearmanr):
    def plot_regression_corr(data, x_col, y_col, title, time, ylabel="", color="#0a2463", ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        old_mask = data["time"] < int(time)
        recent_mask = ~old_mask

        # Scatter plots
        ax.scatter(data.loc[old_mask, x_col], data.loc[old_mask, y_col],
                   s=50, alpha=0.35, color="lightgray", label=f"< {int(time)}")

        ax.scatter(data.loc[recent_mask, x_col], data.loc[recent_mask, y_col],
                   s=50, alpha=0.35, color=color, label=f"≥ {int(time)}")

        light_recent = sns.desaturate(color, 0.5)

        # Regression lines
        if old_mask.sum() >= 2:
            sns.regplot(data=data.loc[old_mask], x=x_col, y=y_col, ax=ax,
                        scatter=False,
                        line_kws={"color": "gray", "linestyle": "--"}, ci=None)
        if recent_mask.sum() >= 2:
            sns.regplot(data=data.loc[recent_mask], x=x_col, y=y_col, ax=ax,
                        scatter=False,
                        line_kws={"color": light_recent}, ci=None)

        # Correlations
        rho_old, _ = spearmanr(data.loc[old_mask, y_col], data.loc[old_mask, x_col])
        rho_recent, _ = spearmanr(data.loc[recent_mask, y_col], data.loc[recent_mask, x_col])

        ax.text(0.05, 0.95, f"ρ(<{int(time)}) = {rho_old:.2f}", transform=ax.transAxes, fontsize=10, color="gray",
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.0))

        ax.text(0.05, 0.85, f"ρ(≥{int(time)}) = {rho_recent:.2f}", transform=ax.transAxes, fontsize=10, color=light_recent,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.0))

        # Axes labels and limits
        ax.set_title(title)
        ax.set_xlabel(" ", weight="bold")
        ax.set_ylabel(ylabel, weight="bold")
        ax.set_xlim(data[x_col].min(), data[x_col].max())
        ax.set_ylim(0, 1.1)
        ax.legend()

        return ax
    return (plot_regression_corr,)


@app.cell
def _(df_650_FT_DF_Time_ONLY_FT_HA, plot_regression_corr, plt):
    plot_regression_corr(df_650_FT_DF_Time_ONLY_FT_HA, x_col="log_likelyhood", y_col="max_frequency", title="Spearman CC 650M Fine Tune for HA (Trained up to 1990)", ylabel="", color="#0a2463", time=1990)

    plt.show()
    return


@app.cell
def _(df_650_FT_DF_Time_ONLY_FT_HA, loess_fit):
    def loess_smoothing(df):
        loess_smoothed = loess_fit(df["time"].values, df["time"].values, df["log_likelyhood"].values, alpha=0.15, degree=1)

        return loess_smoothed

    # Compute residuals: corrected ESM score = original - LOESS-smoothed

    loess_smoothing_HA = loess_smoothing(df_650_FT_DF_Time_ONLY_FT_HA)
    return (loess_smoothing_HA,)


@app.cell
def _(loess_smoothing_HA):
    loess_smoothing_HA
    return


@app.cell
def _(df_650_FT_DF_Time_ONLY_FT_HA, loess_smoothing_HA):
    df_650_FT_DF_Time_ONLY_FT_HA["corrected_esm_score"] = df_650_FT_DF_Time_ONLY_FT_HA["log_likelyhood"] - loess_smoothing_HA
    return


@app.cell
def _(df_650_FT_DF_Time_ONLY_FT_HA, sns):
    sns.scatterplot(data = df_650_FT_DF_Time_ONLY_FT_HA, x = "time", y = "log_likelyhood", s=50, alpha=0.35, label="Data")
    return


@app.cell
def _(df_650_FT_DF_Time_ONLY_FT_HA, sns):
    sns.scatterplot(data = df_650_FT_DF_Time_ONLY_FT_HA, x = "time", y = "corrected_esm_score", s=50, alpha=0.35, label="Data")
    return


@app.cell
def _(plt, sns, spearmanr):
    def plot_regression_corr_no_time(data, x_col, y_col, title, color="#0a2463", ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot
        ax.scatter(data[x_col], data[y_col], s=50, alpha=0.35, color=color, label="Data")

        # Regression line
        sns.regplot(data=data, x=x_col, y=y_col, ax=ax,
                    scatter=False, line_kws={"color": color}, ci=None)

        # Correlation
        rho, _ = spearmanr(data[y_col], data[x_col])
        ax.text(0.05, 0.95, f"ρ = {rho:.2f}", transform=ax.transAxes, fontsize=10, color=color,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.0))

        # Axes labels and limits
        ax.set_title(title)
        ax.set_xlabel(x_col, weight="bold")
        ax.set_ylabel(y_col, weight="bold")
        ax.set_xlim(data[x_col].min(), data[x_col].max())
        ax.set_ylim(0, 1.1)
        ax.legend()

        return ax

    return (plot_regression_corr_no_time,)


@app.cell
def _(df_650_FT_DF_Time_ONLY_FT_HA, plot_regression_corr_no_time):
    plot_regression_corr_no_time(df_650_FT_DF_Time_ONLY_FT_HA, x_col="log_likelyhood", y_col="max_frequency", title="Spearman CC 650M Fine Tune for HA (Trained up to 1990)", color="#0a2463")
    return


@app.cell
def _(df_650_FT_DF_Time_ONLY_FT_HA, plot_regression_corr_no_time):
    plot_regression_corr_no_time(df_650_FT_DF_Time_ONLY_FT_HA, x_col="corrected_esm_score", y_col="max_frequency", title="Spearman CC 650M Fine Tune for HA (Trained up to 1990)", color="#0a2463")
    return


if __name__ == "__main__":
    app.run()
