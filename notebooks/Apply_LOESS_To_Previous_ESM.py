import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Applying LOESS to previous ESM work""")
    return


@app.cell
def _(mo):
    mo.md(r"""In this notebook I will be reproducing some of my previous work I presented at lab meeting with LOESS correction applied, to remove the negative trend of ESM score vs Time.""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Install Libraries""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import seaborn as sns
    from scipy.stats import spearmanr
    import colorsys
    import matplotlib.cm as cm
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.ticker import FormatStrFormatter
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sklearn.linear_model import LinearRegression
    return (
        LinearRegression,
        ScalarFormatter,
        cm,
        colorsys,
        mo,
        np,
        pd,
        plt,
        sns,
        spearmanr,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Patch to fix LOESS package 

    Areas where data similarity is high there is a division by zero error
    """
    )
    return


@app.cell
def _(np):
    class polyfit1d:

        def __init__(self, x, y, degree, weights):

            sqw = np.sqrt(weights)
            a = x[:, None]**np.arange(degree + 1)
            self.degree = degree
            self.coeff = np.linalg.lstsq(a*sqw[:, None], y*sqw, rcond=None)[0]
            self.yfit = a @ self.coeff

        def eval(self, x):

            a = x**np.arange(self.degree + 1)
            yout = a @ self.coeff

            return yout

    def biweight_sigma(y, zero=False):

        y = np.ravel(y)
        if zero:
            d = y
        else:
            d = y - np.median(y)

        mad = np.median(np.abs(d))
        u2 = (d / (9.*mad))**2  # c = 9
        good = u2 < 1.
        u1 = 1. - u2[good]
        num = y.size * ((d[good]*u1**2)**2).sum()
        den = (u1*(1. - 5.*u2[good])).sum()
        sigma = np.sqrt(num/(den*(den - 1.)))  # see note in above reference

        return sigma


    def rotate_points(x, y, ang):
        theta = np.radians(ang)
        xNew = x*np.cos(theta) - y*np.sin(theta)
        yNew = x*np.sin(theta) + y*np.cos(theta)

        return xNew, yNew

    def loess_1d(x, y, xnew=None, degree=1, frac=0.5, npoints=None, rotate=False, sigy=None):

        if frac == 0:
            return y, np.ones_like(y)

        assert x.size == y.size, 'Input vectors (X, Y) must have the same size'

        if npoints is None:
            npoints = int(np.ceil(frac*x.size))

        if rotate:

            assert xnew is None, "`rotate` not supported with `xnew`"

            # Robust calculation of the axis of maximum variance
            #
            nsteps = 180
            angles = np.arange(nsteps)
            sig = np.zeros(nsteps)
            for j, ang in enumerate(angles):
                x2, y2 = rotate_points(x, y, ang)
                sig[j] = biweight_sigma(x2)
            k = np.argmax(sig)  # Find index of max value
            x, y = rotate_points(x, y, angles[k])

        if xnew is None:

            xnew = x

        ynew = np.empty_like(xnew, dtype=float)
        wout = np.empty_like(ynew)

        for j, xj in enumerate(xnew):

            dist = np.abs(x - xj)
            w = np.argsort(dist)[:npoints]
            dist_weights = (1 - (dist[w]/dist[w[-1]])**3)**3  # tricube function distance weights
            yfit = polyfit1d(x[w], y[w], degree, dist_weights).yfit

            # Robust fit from Sec.2 of Cleveland (1979)
            # Use errors if those are known.
            #
            bad = None
            for p in range(10):  # do at most 10 iterations

                if sigy is None:                # Errors are unknown
                    aerr = np.abs(yfit - y[w])  # Note ABS()
                    mad = np.median(aerr)       # Characteristic scale

                    if mad == 0:
                        #mad = np.finfo(float).tiny
                        mad = np.maximum(mad, 1e-10)
                    uu = (aerr/(6*mad))**2      # For a Gaussian: sigma=1.4826*MAD
                else:                           # Errors are assumed known
                    uu = ((yfit - y[w])/(4*sigy[w]))**2  # 4*sig ~ 6*mad

                uu = uu.clip(0, 1)
                biweights = (1 - uu)**2
                tot_weights = dist_weights*biweights
                poly = polyfit1d(x[w], y[w], degree, tot_weights)
                yfit = poly.yfit
                badOld = bad
                bad = biweights < 0.34    # 99% confidence outliers
                if np.array_equal(badOld, bad):
                    break

            if np.array_equal(x, xnew):
                ynew[j] = yfit[0]
                wout[j] = biweights[0]
            else:
                ynew[j] = poly.eval(xj)
                wout[j] = 1

        if rotate:
            xnew, ynew = rotate_points(xnew, ynew, -angles[k])
            j = np.argsort(xnew)
            xnew, ynew = xnew[j], ynew[j]

        return xnew, ynew, wout
    return (loess_1d,)


@app.cell
def _(mo):
    mo.md(r"""## Regenerating ESM vs Time Plots""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Load dataframes trained up to 1990""")
    return


@app.cell
def _(pd):
    df_650_FT_DF = pd.read_csv('Dataframes/650M_Fine_Tune_Up_To_1990.csv', keep_default_na=False)
    df_3B_FT_DF = pd.read_csv('Dataframes/3B_Fine_Tune_Up_To_1990.csv', keep_default_na=False)
    return df_3B_FT_DF, df_650_FT_DF


@app.cell
def _(mo):
    mo.md(r"""### Rewrite LOESS application to work with any dataframe and store as a function""")
    return


@app.cell
def _(loess_1d, pd):
    def apply_loess_to_segment(
        df: pd.DataFrame,
        x_col: str = "time",
        y_col: str = "log_likelihood",
        degree: int = 2,
        frac: float = 0.15,
    ) -> pd.DataFrame:


        x = df[x_col].values
        y = df[y_col].values

        _, y_smoothed, w_smoothed = loess_1d(
            x=x,
            y=y,
            xnew=x,
            degree=degree,
            frac=frac,
        )

        df.loc[:, f"{y_col}_LOESS"] = y_smoothed
        df.loc[:, "loess_weight"] = w_smoothed

        return df


    def apply_loess_to_finetune_models(
        df: pd.DataFrame,
        x_col: str = "time",
        y_col: str = "log_likelihood",
        degree: int = 2,
        frac: float = 0.15,
    ) -> pd.DataFrame:


        df = df.copy()

        fine_tune_mask = df["Model"].str.startswith("Fine_Tune")
        fine_tune_df = df.loc[fine_tune_mask].copy()

        smoothed_parts: list[pd.DataFrame] = []

        for (_, _), group in fine_tune_df.groupby(["Segment", "Model"], sort=False):
            group = apply_loess_to_segment(
                group, x_col=x_col, y_col=y_col, degree=degree, frac=frac
            )
            smoothed_parts.append(
                group[["Segment", "Model", x_col, f"{y_col}_LOESS", "loess_weight"]]
            )

        if smoothed_parts:
            smoothed_df = pd.concat(smoothed_parts, ignore_index=True)
            smoothed_df = (
                smoothed_df.groupby(["Segment", "Model", x_col], as_index=False).first()
            )
        else:
            smoothed_df = pd.DataFrame(
                columns=["Segment", "Model", x_col, f"{y_col}_LOESS", "loess_weight"]
            )


        df = df.merge(
            smoothed_df,
            on=["Segment", "Model", x_col],
            how="left",
            validate="many_to_one",
        )

        df[f"{y_col}_LOESS"] = pd.to_numeric(df[f"{y_col}_LOESS"], errors="coerce")
        df["corrected_log_likelihood"] = df[y_col] - df[f"{y_col}_LOESS"]

        return df

    return (apply_loess_to_finetune_models,)


@app.cell
def _(mo):
    mo.md(r"""### Apply LOESS to Fine Tune ESM Models (650M and 3B parameters) trained up to 1990""")
    return


@app.cell
def _(apply_loess_to_finetune_models, df_3B_FT_DF, df_650_FT_DF):
    df_650_FT_DF_with_loess = apply_loess_to_finetune_models(df_650_FT_DF)
    df_3B_FT_DF_with_loess = apply_loess_to_finetune_models(df_3B_FT_DF)
    return df_3B_FT_DF_with_loess, df_650_FT_DF_with_loess


@app.cell
def _(mo):
    mo.md(r"""### Generate time vs ESM score (log likelihood) plots""")
    return


@app.cell
def _(ScalarFormatter, cm, colorsys, np, plt, sns):
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(style='ticks', palette='Set2')

    def darken_color(rgb, factor=0.7):
        h, l, s = colorsys.rgb_to_hls(*rgb)
        r, g, b = colorsys.hls_to_rgb(h, max(0, l * factor), s)
        return (r, g, b, 1.0)

    def plot_esm_score(ax, df, title, Fine_Tune=False, LOESS=False):

        if(LOESS == False):
            ll_col = "log_likelihood" 
        else: 
            ll_col = "corrected_log_likelihood"

        norm = plt.Normalize(df[ll_col].min(), df[ll_col].max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(df[ll_col]))
        edgecolors = [darken_color(c[:3], factor=0.7) for c in colors]

        ax.scatter(
            df["time"],
            df[ll_col],
            c=colors,
            edgecolors=edgecolors,
            linewidths=0.5,
            alpha=0.7,
            zorder=1
        )

        high_freq_df = df[df["max_frequency"] >= 1].sort_values("time")
        ax.plot(
            high_freq_df["time"],
            high_freq_df[ll_col],
            linestyle='-',
            color='black',
            linewidth=3,
            alpha=0.6,
            label='Max Freq ≥ 0.99',
            zorder=2
        )

        ax.yaxis.offsetText.set_visible(False)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical',
                            pad=0.02,        
                            extend='both'
                           )   

        cbar.ax.yaxis.offsetText.set_visible(False)

        ax.set_title(title, fontsize=10)

        if Fine_Tune:
            ax.axvline(1990, color='gray', linestyle='--', linewidth=1.5)

        ax.set_ylabel("ESM Score", fontsize=8)
        ax.grid(True, color='lightgray', linestyle='-', linewidth=0.75)
        ax.spines[['right', 'top']].set_visible(False)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style='plain', axis='x')
        ax.set_xlim(1965, 2025)

        y_min, y_max = df[ll_col].min(), df[ll_col].max()
        pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)

        return ax

    def esm_vs_time_3x8_grid(model_df, model_name):
        segments = sorted(model_df['Segment'].unique())
        fig, axs = plt.subplots(len(segments), 3, figsize=(15, 30), sharex=True, sharey=False)

        for i, segment in enumerate(segments):
            df_ft   = model_df[(model_df['Model'] == f"Fine_Tune_{model_name}") & (model_df['Segment'] == segment)]
            df_base = model_df[(model_df['Model'] ==       model_name    ) & (model_df['Segment'] == segment)]

            if segment == "PA":
                df_ft   = df_ft[df_ft['node'] != 'A/Viamao/LACENRS-974/2015']
                df_base = df_base[df_base['node'] != 'A/Viamao/LACENRS-974/2015']

            ax1, ax2, ax3 = axs[i, 0], axs[i, 1], axs[i, 2]

            plot_esm_score(ax1, df_base, f"{segment.upper()} • {model_name} Base")
            plot_esm_score(ax2, df_ft,   f"{segment.upper()} • {model_name} FT", Fine_Tune=True)
            plot_esm_score(ax3, df_ft, f"{segment.upper()} • {model_name} LOESS", Fine_Tune=True, LOESS=True)

            if i == len(segments) - 1:
                for ax in (ax1, ax2, ax3):
                    ax.set_xlabel("Year", fontsize=8)


            years = np.arange(1960, 2021, 20)    
            for ax in axs.flat:                    
                ax.set_xticks(years)               
                ax.set_xticklabels(years,         
                                   rotation=0,    
                                   ha='right',
                                   fontsize=10
                                  )
                ax.tick_params(axis='x',
                               which='major',
                               labelbottom=True)   

        plt.tight_layout(h_pad=2, w_pad=1)
        plt.show()
    return (esm_vs_time_3x8_grid,)


@app.cell
def _(mo):
    mo.md(r"""### Generate time vs ESM score plots for 650M parameter model trained up to 1990""")
    return


@app.cell
def _(df_650_FT_DF_with_loess, esm_vs_time_3x8_grid):
    esm_vs_time_3x8_grid(df_650_FT_DF_with_loess, "650M")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    _In this figure column 1 are the ESM scores generated from the base 650M parameter model. Column 2 is the 650M model fine tuned to each individual segement. Column 3 is the same fine tuned results with LOESS correction applied._

    Shows us that the LOESS correction is working properly as each segment, the ESM score is now relatively the same across time. For some segments, such as MP, which were relatively flat before LOESS correction, there is little difference between the  fine tune and LOESS corrected fine tune plots, but average ESM score is now much closer to zero. Points that fall before 1990, the training period are less impacted by LOESS corrected than test period points - post 1990.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Generate time vs ESM score plots for 3B parameter model trained up to 1990""")
    return


@app.cell
def _(df_3B_FT_DF_with_loess, esm_vs_time_3x8_grid):
    esm_vs_time_3x8_grid(df_3B_FT_DF_with_loess, "3B")
    return


@app.cell
def _(mo):
    mo.md(r"""## Compare pre and post LOESS spearman correlation coefficient """)
    return


@app.cell
def _(mo):
    mo.md(r"""### Calculate spearman correlation coefficient for each segment for 650M and 3B models""")
    return


@app.cell
def _(df_3B_FT_DF_with_loess, df_650_FT_DF_with_loess, pd, spearmanr):
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

          spearman_corr, p_value = spearmanr(df['max_frequency'], df['log_likelihood'])

          results.append({
              "Model": model,
              "Segment": segment,
              "Spearman Correlation Coefficient between Max Frequency and LL": spearman_corr,
              "P-value": p_value,
              "Mean ESM LL below 0.1": df_below_01['log_likelihood'].mean(),
              "Mean ESM LL above 0.99": df_above_1['log_likelihood'].mean(),
              "Difference in LL ESM Means": df_above_1['log_likelihood'].mean() - df_below_01['log_likelihood'].mean(),
              "Time Frame": time_frame
          })

          results_df = pd.DataFrame(results)

          if(model == "Fine_Tune_3B" or model == "Fine_Tune_650M"):

              spearman_corr, p_value = spearmanr(df['max_frequency'], df['corrected_log_likelihood'])

              results.append({
                  "Model": f"LOESS_{model}",
                  "Segment": segment,
                  "Spearman Correlation Coefficient between Max Frequency and LL": spearman_corr,
                  "P-value": p_value,
                  "Mean ESM LL below 0.1": df_below_01['corrected_log_likelihood'].mean(),
                  "Mean ESM LL above 0.99": df_above_1['corrected_log_likelihood'].mean(),
                  "Difference in LL ESM Means": df_above_1['corrected_log_likelihood'].mean() - df_below_01['corrected_log_likelihood'].mean(),
                  "Time Frame": time_frame
              })

              results_df = pd.DataFrame(results)

      print("____________________________")
      print(f"Summary Statistics for {base_name} Model - {time_frame}")
      print(results_df.groupby('Model')['Spearman Correlation Coefficient between Max Frequency and LL'].mean())

      #results_df.to_csv(f"Flu_Summary_Statistics/ESM_vs_Max_Freq_Summary_Fine_Tune_{base_name}_Statistics.csv", index=False)
      return results_df

    df_3B_FT_DF_Time_Above_1990 = df_3B_FT_DF_with_loess[df_3B_FT_DF_with_loess['time'] >= 1991]
    df_650_FT_DF_Time_Above_1990 = df_650_FT_DF_with_loess[df_650_FT_DF_with_loess['time'] >= 1991]
    df_3B_FT_DF_Time_Below_1990 = df_3B_FT_DF_with_loess[df_3B_FT_DF_with_loess['time'] <= 1990]
    df_650_FT_DF_Time_Below_1990 = df_650_FT_DF_with_loess[df_650_FT_DF_with_loess['time'] <= 1990]

    df_3B_FT_DF_Time_Above_1990_Results_DF = summary_stats(df_3B_FT_DF_Time_Above_1990, "3B", "Post 1990")
    df_650_FT_DF_Time_Above_1990_Results_DF = summary_stats(df_650_FT_DF_Time_Above_1990, "650M", "Post 1990")
    df_3B_FT_DF_Time_Below_1990_Results_DF = summary_stats(df_3B_FT_DF_Time_Below_1990, "3B", "Pre 1990")
    df_650_FT_DF_Time_Below_1990_Results_DF = summary_stats(df_650_FT_DF_Time_Below_1990, "650M", "Pre 1990")

    # Combine all results into a single DataFrame
    combined_results = pd.concat([df_3B_FT_DF_Time_Above_1990_Results_DF, df_650_FT_DF_Time_Above_1990_Results_DF, df_3B_FT_DF_Time_Below_1990_Results_DF, df_650_FT_DF_Time_Below_1990_Results_DF], ignore_index=True)
    return (
        df_3B_FT_DF_Time_Above_1990_Results_DF,
        df_3B_FT_DF_Time_Below_1990_Results_DF,
        df_650_FT_DF_Time_Above_1990_Results_DF,
        df_650_FT_DF_Time_Below_1990_Results_DF,
    )


@app.cell
def _(mo):
    mo.md(r"""### Plot spearman correlation coefficient for each model with and without LOESS""")
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
        model_order_3B = ['3B', 'Fine_Tune_3B', 'LOESS_Fine_Tune_3B']
        model_order_650M = ['650M', 'Fine_Tune_650M', 'LOESS_Fine_Tune_650M']

        palette_3B = {
            '3B': '#0a2463',
            'Fine_Tune_3B': '#f4d35e',
            'LOESS_Fine_Tune_3B': '#890304'
        }

        palette_650M = {
            '650M': '#0a2463',
            'Fine_Tune_650M': '#f4d35e',
            'LOESS_Fine_Tune_650M': '#890304',
        }

        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

        plot_spearman_barplot(axes[0, 0], df_3B, model_order_3B, palette_3B, "3B - Fine Tune vs LOESS (Post-1990)", xaxis="")
        plot_spearman_barplot(axes[0, 1], df_650M, model_order_650M, palette_650M, "650M - Fine Tune vs LOESS (Post-1990)", xaxis="")
        plot_spearman_barplot(axes[1, 0], df_3B_FT, model_order_3B, palette_3B, "3B - Fine Tune vs LOESS (Pre-1990)", xaxis="Segment")
        plot_spearman_barplot(axes[1, 1], df_650M_FT, model_order_650M, palette_650M, "650M - Fine Tune vs LOESS (Pre-1990)", xaxis="Segment")

        plt.tight_layout()
        plt.show()

    combined_average_spearman_fine_tune_compare(df_3B_FT_DF_Time_Above_1990_Results_DF, df_650_FT_DF_Time_Above_1990_Results_DF, df_3B_FT_DF_Time_Below_1990_Results_DF, df_650_FT_DF_Time_Below_1990_Results_DF)
    return


@app.cell
def _(mo):
    mo.md(r"""## Comparing maximum frequency to time""")
    return


@app.cell
def _(LinearRegression, df_650_FT_DF_with_loess, np, pd, plt, sns, spearmanr):
    def time_maxfreq_spear(model_df):
    
        for segment, group in model_df.groupby('Segment'):
        
            df = model_df[model_df['Segment'] == segment]

            #df = df[df["time"] > 1990]
        
            rho, pval = spearmanr(df["time"], 
                                 df["max_frequency"])
            print(f"Spearman ρ = {rho:.3f}, p = {pval:.3g}")
        
            df = df.copy()
            df["time_rank"] = df["time"].rank()
            df["freq_rank"] = df["max_frequency"].rank()

            X = df[["time_rank"]].values
            y = df["freq_rank"].values
            lm = LinearRegression().fit(X, y)
        
            sns.scatterplot(x="time", y="max_frequency", data=df)
        
            x_line = np.linspace(df["time"].min(), df["time"].max(), 100)
            x_line_rank = pd.Series(x_line).rank(method="first", pct=False).values
            y_line_rank = lm.predict(x_line_rank.reshape(-1, 1))
            y_line = np.percentile(df["max_frequency"], 100 * (y_line_rank - 1) / (len(df) - 1))
        
            plt.plot(x_line, y_line, color="red", linestyle="--",
                     label=f"Spearman fit (ρ={rho:.2f})")
            plt.legend()
        
            plt.title(f"{segment} time vs maximum frequency")
        
            plt.show()

    time_maxfreq_spear(df_650_FT_DF_with_loess)
    return


@app.cell
def _(mo):
    mo.md(r"""explains why spearman higher with post training dataset""")
    return


@app.cell
def _(pd, spearmanr):
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
            #df_segment_FT_LS = df_segment[df_segment["Model"] == f"LOESS_Fine_Tune_{model}"]
            #df_segment_BS = df_segment[df_segment["Model"] == f"{model}"]

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
                    #df_segment_BS_label = df_segment_BS[df_segment_BS['time'] >= start]
                    #df_segment_FT_LS_label = df_segment_FT_LS[df_segment_FT_LS['time'] >= start]
                else:
                    df_segment_FT_label = df_segment_FT[(df_segment_FT['time'] >= start) & (df_segment_FT['time'] <= end)]
                    #df_segment_BS_label = df_segment_BS[(df_segment_BS['time'] >= start) & (df_segment_BS['time'] <= end)]
                    #df_segment_FT_LS_label = df_segment_FT_LS[(df_segment_FT_LS['time'] >= start) & (df_segment_FT_LS['time'] <= end)]

                results.append(spearman_correlation_calculation(df_segment_FT_label, "max_frequency", "log_likelihood", f"Fine_Tune_{model}", segment, label))
                #results.append(spearman_correlation_calculation(df_segment_BS_label, "max_frequency", "log_likelihood", model, segment, label))
                results.append(spearman_correlation_calculation(df_segment_FT_label, "max_frequency", "corrected_log_likelihood", f"LOESS_Fine_Tune_{model}", segment, label))

        return pd.DataFrame(results)
    return (spearman_correlation,)


@app.cell
def _(df_3B_FT_DF_with_loess, df_650_FT_DF_with_loess, spearman_correlation):
    df_3B_FT_DF_Time_spearman = spearman_correlation(df_3B_FT_DF_with_loess, "3B")
    df_650_FT_DF_Time_spearman = spearman_correlation(df_650_FT_DF_with_loess, "650M")
    return df_3B_FT_DF_Time_spearman, df_650_FT_DF_Time_spearman


@app.cell
def _(df_3B_FT_DF_Time_spearman, df_650_FT_DF_Time_spearman):
    df_650_FT_DF_Time_spearman_Base = df_650_FT_DF_Time_spearman[df_650_FT_DF_Time_spearman['Model'] == '650M']
    df_650_FT_DF_Time_spearman_Fine_Tune = df_650_FT_DF_Time_spearman[df_650_FT_DF_Time_spearman['Model'] == 'Fine_Tune_650M']
    df_650_FT_LS_DF_Time_spearman_Fine_Tune = df_650_FT_DF_Time_spearman[df_650_FT_DF_Time_spearman['Model'] == 'LOESS_Fine_Tune_650M']

    df_3B_FT_DF_Time_spearman_Base = df_3B_FT_DF_Time_spearman[df_3B_FT_DF_Time_spearman['Model'] == '3B']
    df_3B_FT_DF_Time_spearman_Fine_Tune = df_3B_FT_DF_Time_spearman[df_3B_FT_DF_Time_spearman['Model'] == 'Fine_Tune_3B']
    df_3B_FT_LS_DF_Time_spearman_Fine_Tune = df_3B_FT_DF_Time_spearman[df_3B_FT_DF_Time_spearman['Model'] == 'LOESS_Fine_Tune_3B']
    return (
        df_3B_FT_DF_Time_spearman_Base,
        df_3B_FT_DF_Time_spearman_Fine_Tune,
        df_3B_FT_LS_DF_Time_spearman_Fine_Tune,
        df_650_FT_DF_Time_spearman_Base,
        df_650_FT_DF_Time_spearman_Fine_Tune,
        df_650_FT_LS_DF_Time_spearman_Fine_Tune,
    )


@app.function
def mean_spearman_segments(df, model_name):
    df_Fine_Tune_Summary = df.groupby('Time_Range', as_index=False)['Spearman_Correlation'].mean()
    df_Fine_Tune_Summary["Model"] = model_name 

    return df_Fine_Tune_Summary


@app.cell
def _(
    df_3B_FT_DF_Time_spearman_Base,
    df_3B_FT_DF_Time_spearman_Fine_Tune,
    df_3B_FT_LS_DF_Time_spearman_Fine_Tune,
    df_650_FT_DF_Time_spearman_Base,
    df_650_FT_DF_Time_spearman_Fine_Tune,
    df_650_FT_LS_DF_Time_spearman_Fine_Tune,
    pd,
):
    df_3B_FT_DF_Time_spearman_Fine_Tune_Summary = mean_spearman_segments(df_3B_FT_DF_Time_spearman_Fine_Tune, "Fine Tune - 3B Model")
    df_650_FT_DF_Time_spearman_Fine_Tune_Summary = mean_spearman_segments(df_650_FT_DF_Time_spearman_Fine_Tune, "Fine Tune - 650M Model")

    df_3B_DF_Time_Spearman_Summary = mean_spearman_segments(df_3B_FT_DF_Time_spearman_Base, "Base - 3B Model")
    df_650_DF_Time_Spearman_Summary = mean_spearman_segments(df_650_FT_DF_Time_spearman_Base, "Base - 650M Model")

    combined_spearman_summary = pd.concat([
        df_3B_FT_DF_Time_spearman_Fine_Tune_Summary,
        df_650_FT_DF_Time_spearman_Fine_Tune_Summary,
        df_3B_DF_Time_Spearman_Summary,
        df_650_DF_Time_Spearman_Summary,
        df_650_FT_LS_DF_Time_spearman_Fine_Tune,
        df_3B_FT_LS_DF_Time_spearman_Fine_Tune
    ], ignore_index=True)
    return (combined_spearman_summary,)


@app.cell
def _(np, plt, sns):
    def create_spearman_summary_plot(df):
        sns.set_style("whitegrid")
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax = sns.lineplot(
            data=df,
            x='Time_Range',
            y='Spearman_Correlation',
            hue='Model',
            marker="o",
            legend=False,
            zorder=1,
            ax=ax,
            errorbar=None,   
        )

        ax.set_title("Spearman CC Summary All Models")
        ax.set_xlabel("Time Range")
        ax.set_ylabel("Spearman CC")

        label_positions = []
        for line, model in zip(ax.lines, df['Model'].unique()):
            y = line.get_ydata()[-1]
            x = line.get_xdata()[-1]

            if not np.isfinite(y) or not np.isfinite(x):
                continue

            if model in ("Base - 3B Model", "Base - 650M Model"):
                y = line.get_ydata()[-4]
                x = line.get_xdata()[-4]
            elif model in (
                "Fine Tune - 650M Model - LR 2.5e-05",
                "Fine Tune - 650M Model - LR 1e-05",
                "Fine Tune - 650M Model - LR 1e-06"
            ):
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
    return (create_spearman_summary_plot,)


@app.cell
def _(combined_spearman_summary, create_spearman_summary_plot):
    create_spearman_summary_plot(combined_spearman_summary)
    return


@app.cell
def _(pd):
    combined_LG_SM_1990_2005 = pd.read_csv('Dataframes/LG_SM_1990_2005.csv', keep_default_na=False)
    return (combined_LG_SM_1990_2005,)


@app.cell
def _(combined_LG_SM_1990_2005):
    LG_1990 = combined_LG_SM_1990_2005[(combined_LG_SM_1990_2005["Model_training_time"] == 1990) & (combined_LG_SM_1990_2005["tree"] == "h3n2-Large")]
    LG_2005 = combined_LG_SM_1990_2005[(combined_LG_SM_1990_2005["Model_training_time"] == 2005) & (combined_LG_SM_1990_2005["tree"] == "h3n2-Large")]
    SM_1990 = combined_LG_SM_1990_2005[(combined_LG_SM_1990_2005["Model_training_time"] == 1990) & (combined_LG_SM_1990_2005["tree"] == "h3n2")]
    SM_2005 = combined_LG_SM_1990_2005[(combined_LG_SM_1990_2005["Model_training_time"] == 2005) & (combined_LG_SM_1990_2005["tree"] == "h3n2")]
    return LG_1990, LG_2005, SM_1990, SM_2005


@app.cell
def _(LG_1990, LG_2005, SM_1990, SM_2005, apply_loess_to_finetune_models):
    LG_1990_with_loess = apply_loess_to_finetune_models(LG_1990)
    LG_2005_with_loess = apply_loess_to_finetune_models(LG_2005)
    SM_1990_with_loess = apply_loess_to_finetune_models(SM_1990)
    SM_2005_with_loess = apply_loess_to_finetune_models(SM_2005)
    return (
        LG_1990_with_loess,
        LG_2005_with_loess,
        SM_1990_with_loess,
        SM_2005_with_loess,
    )


@app.cell
def _(
    LG_1990_with_loess,
    LG_2005_with_loess,
    SM_1990_with_loess,
    SM_2005_with_loess,
    spearman_correlation,
):
    LG_1990_with_loess_spearman = spearman_correlation(LG_1990_with_loess, "650M")
    LG_2005_with_loess_spearman = spearman_correlation(LG_2005_with_loess, "650M")
    SM_1990_with_loess_spearman = spearman_correlation(SM_1990_with_loess, "650M")
    SM_2005_with_loess_spearman = spearman_correlation(SM_2005_with_loess, "650M")
    return (
        LG_1990_with_loess_spearman,
        LG_2005_with_loess_spearman,
        SM_1990_with_loess_spearman,
        SM_2005_with_loess_spearman,
    )


@app.cell
def _(
    LG_1990_with_loess_spearman,
    LG_2005_with_loess_spearman,
    SM_1990_with_loess_spearman,
    SM_2005_with_loess_spearman,
):
    LG_1990_spearman_FT = LG_1990_with_loess_spearman[LG_1990_with_loess_spearman['Model'] == 'Fine_Tune_650M']
    LG_1990_spearman_FT_LS = LG_1990_with_loess_spearman[LG_1990_with_loess_spearman['Model'] == 'LOESS_Fine_Tune_650M']

    LG_2005_spearman_FT = LG_2005_with_loess_spearman[LG_2005_with_loess_spearman['Model'] == 'Fine_Tune_650M']
    LG_2005_spearman_FT_LS = LG_2005_with_loess_spearman[LG_2005_with_loess_spearman['Model'] == 'LOESS_Fine_Tune_650M']

    SM_1990_spearman_FT = SM_1990_with_loess_spearman[SM_1990_with_loess_spearman['Model'] == 'Fine_Tune_650M']
    SM_1990_spearman_FT_LS = SM_1990_with_loess_spearman[SM_1990_with_loess_spearman['Model'] == 'LOESS_Fine_Tune_650M']

    SM_2005_spearman_FT = SM_2005_with_loess_spearman[SM_2005_with_loess_spearman['Model'] == 'Fine_Tune_650M']
    SM_2005_spearman_FT_LS = SM_2005_with_loess_spearman[SM_2005_with_loess_spearman['Model'] == 'LOESS_Fine_Tune_650M']
    return (
        LG_1990_spearman_FT,
        LG_1990_spearman_FT_LS,
        LG_2005_spearman_FT,
        LG_2005_spearman_FT_LS,
        SM_1990_spearman_FT,
        SM_1990_spearman_FT_LS,
        SM_2005_spearman_FT,
        SM_2005_spearman_FT_LS,
    )


@app.cell
def _(
    LG_1990_spearman_FT,
    LG_1990_spearman_FT_LS,
    LG_2005_spearman_FT,
    LG_2005_spearman_FT_LS,
    SM_1990_spearman_FT,
    SM_1990_spearman_FT_LS,
    SM_2005_spearman_FT,
    SM_2005_spearman_FT_LS,
    pd,
):
    LG_1990_spearman_FT_Summary = mean_spearman_segments(LG_1990_spearman_FT, "1990 Large Tree")
    LG_1990_spearman_FT_LS_Summary = mean_spearman_segments(LG_1990_spearman_FT_LS, "1990 Large Tree with LOESS")

    LG_2005_spearman_FT_Summary = mean_spearman_segments(LG_2005_spearman_FT, "2005 Large Tree")
    LG_2005_spearman_FT_LS_Summary = mean_spearman_segments(LG_2005_spearman_FT_LS, "2005 Large Tree with LOESS")

    SM_1990_spearman_FT_Summary = mean_spearman_segments(SM_1990_spearman_FT, "1990 Small Tree")
    SM_1990_spearman_FT_LS_Summary = mean_spearman_segments(SM_1990_spearman_FT_LS, "1990 Small Tree with LOESS")

    SM_2005_spearman_FT_Summary = mean_spearman_segments(SM_2005_spearman_FT, "2005 Small Tree")
    SM_2005_spearman_FT_LS_Summary = mean_spearman_segments(SM_2005_spearman_FT_LS, "2005 Small Tree with LOESS")

    LG_combined_spearman_summary = pd.concat([
        LG_1990_spearman_FT_Summary,
        LG_1990_spearman_FT_LS_Summary,
        LG_2005_spearman_FT_Summary,
        LG_2005_spearman_FT_LS_Summary,
    ], ignore_index=True)

    SM_combined_spearman_summary = pd.concat([
        SM_1990_spearman_FT_Summary,
        SM_1990_spearman_FT_LS_Summary,
        SM_2005_spearman_FT_Summary,
        SM_2005_spearman_FT_LS_Summary
    ], ignore_index=True)
    return (
        LG_1990_spearman_FT_LS_Summary,
        LG_combined_spearman_summary,
        SM_combined_spearman_summary,
    )


@app.cell
def _(LG_1990_spearman_FT_LS_Summary):
    LG_1990_spearman_FT_LS_Summary
    return


@app.cell
def _(LG_combined_spearman_summary, create_spearman_summary_plot):
    create_spearman_summary_plot(LG_combined_spearman_summary)
    return


@app.cell
def _(SM_combined_spearman_summary, create_spearman_summary_plot):
    create_spearman_summary_plot(SM_combined_spearman_summary)
    return


@app.cell
def _(pd):
    df_650_FT_DF_Time_series_Validation = pd.read_csv('Dataframes/df_650_FT_DF_Time_Series_Validation.csv', keep_default_na=False)
    return (df_650_FT_DF_Time_series_Validation,)


@app.cell
def _():
    #apply_loess_to_finetune_models(df_650_FT_DF_Time_series_Validation)
    return


@app.cell
def _(apply_loess_to_finetune_models, df_650_FT_DF_Time_series_Validation, pd):
    time_vals = df_650_FT_DF_Time_series_Validation["Model_training_time"].unique()
    smoothed_parts = []
    for t in time_vals:
        subset = df_650_FT_DF_Time_series_Validation[
            df_650_FT_DF_Time_series_Validation["Model_training_time"] == t
        ].copy()
        smoothed = apply_loess_to_finetune_models(subset)
        smoothed_parts.append(smoothed)

    df_650_FT_DF_Time_series_Validation_LOESS = pd.concat(smoothed_parts, ignore_index=True)
    return (df_650_FT_DF_Time_series_Validation_LOESS,)


@app.cell
def _(pd, spearmanr):
    def calculate_time_series_cross_df(df, ll_column: str = 'log_likelihood'):
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
                    df_bin = df[
                    (df['time'] >= start) &
                    (df['Model_training_time'] == int(start_year))
                    ]
                else:
                    df_bin = df[
                        (df['time'] >= start) &
                        (df['time'] <= end) &
                        (df['Model_training_time'] == int(start_year))
                    ]


                corr, _ = spearmanr(df_bin['max_frequency'], df_bin[ll_column])


                results.append({
                    "start_year": start_year,
                    "bin_index": idx,
                    "range": f"{start}-{end if end else '2025'}",
                    "spearman_corr": corr
                })

        spearman_df = pd.DataFrame(results)
        return spearman_df
    return (calculate_time_series_cross_df,)


@app.cell
def _(
    calculate_time_series_cross_df,
    df_650_FT_DF_Time_series_Validation_LOESS,
):
    spearman_df = calculate_time_series_cross_df(df_650_FT_DF_Time_series_Validation_LOESS)
    spearman_df_LOESS = calculate_time_series_cross_df(df_650_FT_DF_Time_series_Validation_LOESS, "corrected_log_likelihood")
    return spearman_df, spearman_df_LOESS


@app.cell
def _(plt, sns):
    def create_time_series_plot(spearman_df):
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

        plt.show()
    return (create_time_series_plot,)


@app.cell
def _(create_time_series_plot, spearman_df):
    create_time_series_plot(spearman_df)
    return


@app.cell
def _(create_time_series_plot, spearman_df_LOESS):
    create_time_series_plot(spearman_df_LOESS)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
