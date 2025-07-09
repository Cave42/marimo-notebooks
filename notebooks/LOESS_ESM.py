import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Testing LOESS fit on secular trends - with real data""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Initial Setup""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Import Libraries""")
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
    #from loess.loess_1d import loess_1d
    return mo, np, pd, plt, sns, spearmanr


@app.cell
def _(mo):
    mo.md(r"""## Rerun trvrb sample analysis (but convert to python)""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Some notes

    - I'm not as familiar with mathematica and haven't used LOESS before.
    - I spent some time reading to familiarize myself with LOESS and some of the mathematica syntax, I also used chat gpt to help with the mathematica to python conversion
    - For this reason I've rerun the same analysis from trvrb's notebook to help validate my process.
    - All code comments are my own to help practice explaining my work process and to help with reproducibility
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Patch to fix LOESS package for data similarity (division by zero error)""")
    return


@app.cell
def _(np):
    class polyfit1d:

        def __init__(self, x, y, degree, weights):
            """
            Fit a univariate polynomial of given DEGREE to a set of points
            (X, Y), assuming errors in the Y variable only and weights=1/sigy^2.

            For example with DEGREE=1 this function fits a straight line

               y = a + b*x

            while with DEGREE=2 the function fits a parabola

               y = a + b*x + c*x^2

            """
            sqw = np.sqrt(weights)
            a = x[:, None]**np.arange(degree + 1)
            self.degree = degree
            self.coeff = np.linalg.lstsq(a*sqw[:, None], y*sqw, rcond=None)[0]
            self.yfit = a @ self.coeff


        def eval(self, x):
            """Evaluate at the coordinate x the polynomial previously fitted"""

            a = x**np.arange(self.degree + 1)
            yout = a @ self.coeff

            return yout


    ################################################################################


    def biweight_sigma(y, zero=False):
        """
        Biweight estimate of the scale (standard deviation).
        Implements the approach described in
        "Understanding Robust and Exploratory Data Analysis"
        Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417

        """
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


    ################################################################################


    def rotate_points(x, y, ang):
        """
        Rotates points counter-clockwise by an angle ANG in degrees.
        Michele cappellari, Paranal, 10 November 2013

        """
        theta = np.radians(ang)
        xNew = x*np.cos(theta) - y*np.sin(theta)
        yNew = x*np.sin(theta) + y*np.cos(theta)

        return xNew, yNew


    ################################################################################


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
                        mad = np.finfo(float).tiny
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
    mo.md(
        r"""
    ### Create a fake dataset of strong spearman correlation

    Original code:

    ``` mathematica
    data= RandomVariate[BinormalDistribution[{- 100, 0.5}, {1, 0.5}, 0.9], {1000}]; 
    ```

    Creating a dataset of 1000 points, y axis (max freq) is between 0 and 1 and, with a mean of 0.5 and the x axis (ESM LL) mean at -100. This fake dataset has a strong correlation.
    """
    )
    return


@app.cell
def _(np, plt):
    np.random.seed(42)

    mean = [-100, 0.5]
    var1 = 1
    var2 = 0.5
    corr = 0.9
    cov = corr * np.sqrt(var1 * var2)


    cov_matrix = [[var1, cov],
                  [cov, var2]]

    data_fake = np.random.multivariate_normal(mean, cov_matrix, size=1000)

    data_fake[:, 1] = np.clip(data_fake[:, 1], 0, 1)

    esm_scores = data_fake[:, 0]
    max_freq = data_fake[:, 1]

    plt.style.use("seaborn-v0_8-whitegrid")
    return esm_scores, max_freq


@app.cell
def _(mo):
    mo.md(r"""### Produce figure of strong correlation ESM LL vs max freq of fake data""")
    return


@app.cell
def _(esm_scores, max_freq, plt):
    def create_scatterplot(esm_scores, max_freq, title):
        plt.figure()
        plt.scatter(esm_scores, max_freq, s=2, color='darkblue')
        plt.xlabel("ESM score")
        plt.ylabel("max freq")
        plt.title(title)
        plt.show()

    create_scatterplot(esm_scores, max_freq, "ESM LL vs max freq fake data with strong trend")
    return (create_scatterplot,)


@app.cell
def _(mo):
    mo.md(r"""### Check spearman cc""")
    return


@app.cell
def _(esm_scores, max_freq, spearmanr):
    print(spearmanr(esm_scores, max_freq))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Add secular trend to fake data

    Here ESM score increases each year, but in the real dataset, we generally see the opposite trend.

    We also label trunk points (max freq = 1)

    original code: 

    ``` mathematica
    adjData= Map[With[{year= RandomReal[{1970, 2020}]}, {#〚1〛 + 0.25 * (year- 1970), #〚2〛, year}] &, data];
    sidebranchPoints= Cases[adjData, x /; x〚2〛 < 1]〚All, {3, 1}〛;
    trunkPoints= Cases[adjData, x /; x〚2〛 ⩵ 1]〚All, {3, 1}〛;
    ```
    """
    )
    return


@app.cell
def _(esm_scores, max_freq, np):
    years = np.random.uniform(1970, 2020, size=1000)
    adjusted_esm_scores = esm_scores + 0.25 * (years - 1970)

    adj_data = list(zip(adjusted_esm_scores, max_freq, years))

    sidebranch_points = [(year, esm) for esm, freq, year in adj_data if freq < 1]
    trunk_points = [(year, esm) for esm, freq, year in adj_data if freq == 1]
    return adj_data, sidebranch_points, trunk_points


@app.cell
def _(mo):
    mo.md(r"""### Plot fake data with secular trend""")
    return


@app.cell
def _(plt, sidebranch_points, trunk_points):
    plt.figure(figsize=(10, 6))
    if sidebranch_points:
        plt.scatter(*zip(*sidebranch_points), color='gray', s=10, label='Side branches')
    if trunk_points:
        plt.scatter(*zip(*trunk_points), color='red', s=10, label='Trunk')
    plt.xlabel("Year")
    plt.ylabel("ESM score")
    plt.title("ESM Score vs Year with Secular Trend")
    plt.legend()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Check spearman CC of fake data with secular trend

    Switch from numpy arrays to pandas dataframes, my real dataframes are in pandas df.
    """
    )
    return


@app.cell
def _(adj_data, pd, spearmanr):
    df_fake = pd.DataFrame(adj_data, columns=["esm_score", "max_freq", "year"])
    df_fake["branch_type"] = df_fake["max_freq"].apply(lambda x: "trunk" if x == 1 else "side")

    print(spearmanr(df_fake["esm_score"], df_fake["max_freq"]))
    return (df_fake,)


@app.cell
def _(mo):
    mo.md(r"""### Plot fake dataset of ESM LL and max freq with secular trend applied""")
    return


@app.cell
def _(create_scatterplot, df_fake):
    create_scatterplot(df_fake["esm_score"], df_fake["max_freq"], "ESM LL vs max freq fake data with secular trend applied")
    return


@app.cell
def _(mo):
    mo.md(r"""### Correcting secular trend on fake data with LOESS""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### LOESS function

    I used the LOESS python package, I converted trvrb's manual LOESS function to python - which yielded similar results - but was significantly slower that the package. 

    original code:

    ``` mathematica
    LoessFit[(x_) ? VectorQ, data_, α_: 0.75, λ_: 1] :=Table[LoessFit[x〚i〛, data, α, λ], {i, Length[x]}]
    LoessFit[(x_) ? NumberQ, data_, α_: 0.75, λ_WLSFit[data, LoessWts[x, data, α], λ, x];: 1] :=
    WLSFit[data_, wts_, ldegree_: 1, x_] :=Fit[Transpose[(wts * #1 &) /@ Join[{Table[1, {Length[data]}]}, Transpose[data]]],Join[{u}, Table[v ^ i, {i, ldegree}]], {u, v}] /. {u → 1, v → x}
    LoessWts[x_, data_, α_] :=Tricube[(x- First[Transpose[data]]) / LoessDistance[x, data, α]]
    Tricube= Compile[{{x, _Real, 1}}, If[Abs[#] < 1, (1- Abs[#] ^ 3) ^ 3, 0] & /@ x];
    LoessDistance[x_, data_, α_] := Module[{A = Max[1, α], X = First[Transpose[data]], q},q= Min[Length[X], Ceiling[α * Length[X]]];A * Sort[Abs[X- x]]〚q〛]
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### LOESS function 

    Here trvrb's comment say LOESS was fit "with y = max freq and x = year", I am assuming that y should be esm score given the code references year and esm score position:

    ``` mathematica
    loess[year_] := LoessFit[year, adjData〚All, {3, 1}〛, 0.15]
    ```

    Where 1 = esm score and 3 = year
    """
    )
    return


@app.cell
def _(df_fake, loess_1d):
    x = df_fake["year"].values
    y = df_fake["esm_score"].values

    xout, yout, wout = loess_1d(
        x=x,
        y=y,
        xnew=x,       
        degree=2,     
        frac=0.15    
    )

    df_fake["esm_loess"] = yout
    df_fake["loess_weight"] = wout
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Plotting ESM score vs Year with LOESS fit

    Here we can see ESM score steadily increasing with each year.
    """
    )
    return


@app.cell
def _(df_fake, plt):
    plt.scatter(df_fake["year"], df_fake["esm_score"], color='gray', s=10, label="Side branches")

    df_trunk = df_fake[df_fake["branch_type"] == "trunk"]
    plt.scatter(df_trunk["year"], df_trunk["esm_score"], color='red', s=10, label="Trunk")

    plt.scatter(df_fake["year"], df_fake["esm_loess"], color='black', s=1)

    plt.xlabel("Year")
    plt.ylabel("ESM score")
    plt.title("ESM Score with LOESS Fit")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Correcting ESM score secular trend

    We can apply our LOESS function to the ESM score, and the new plotted results show the trend removed with a much flatter ESM vs Time trend.
    """
    )
    return


@app.cell
def _(df_fake, plt):
    df_fake["corrected_esm_score"] = df_fake["esm_score"] - df_fake["esm_loess"]

    side_corr = df_fake[df_fake["branch_type"] == "side"]
    trunk_corr = df_fake[df_fake["branch_type"] == "trunk"]

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
def _(mo):
    mo.md(
        r"""
    ### ESM score vs max frequency with LOESS correction

    Max frequency and ESM score have a much stronger correlation after accounting for the secular trend with ESM score and time and correcting with LOESS.
    """
    )
    return


@app.cell
def _(create_scatterplot, df_fake):
    create_scatterplot(df_fake["corrected_esm_score"], df_fake["max_freq"], "ESM LL vs max freq fake with LOESS fix applied")
    return


@app.cell
def _(df_fake, spearmanr):
    print(spearmanr(df_fake["corrected_esm_score"], df_fake["max_freq"]))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Applying LOESS correction to real data

    The dataset I am using is the 650M fine tuned model trained up 1990

    (I am working with the base H3N2 tree, not the larger tree)
    """
    )
    return


@app.cell
def _(pd):
    df_650_FT_DF = pd.read_csv('Dataframes/650M_Fine_Tune_Up_To_1990.csv', keep_default_na=False)
    return (df_650_FT_DF,)


@app.cell
def _(df_650_FT_DF):
    segments = ["HA", "NA", "NP", "PA", "PB1", "PB2", "MP", "NS"]

    # Filter and store each segment's DataFrame in a dictionary
    segment_dfs = {
        seg: df_650_FT_DF[
            (df_650_FT_DF['Segment'] == seg) &
            (df_650_FT_DF['Model'] == 'Fine_Tune_650M')
        ].copy()
        for seg in segments
    }
    return (segment_dfs,)


@app.cell
def _(segment_dfs):
    segment_dfs["PA"] = segment_dfs["PA"][
        segment_dfs["PA"]["node"] != "A/Viamao/LACENRS-974/2015"
    ]
    return


@app.cell
def _(mo):
    mo.md(r"""### Looking at ESM LL and max frequency we see a weaker correlation""")
    return


@app.cell
def _(create_scatterplot, segment_dfs):
    for seg, df in segment_dfs.items():
        create_scatterplot(
            df["log_likelihood"],
            df["max_frequency"],
            f"ESM LL vs max freq for {seg}"
        )
    return


@app.cell
def _(pd, segment_dfs, spearmanr):
    def compute_spearman():
        results = []
        for seg, df in segment_dfs.items():
            spearman, p = spearmanr(df["log_likelihood"], df["max_frequency"])
            results.append({"segment": seg, "spearman_cc": spearman, "p_value": p})

        return pd.DataFrame(results)

    spearman_df = compute_spearman()
    spearman_df
    return (spearman_df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Plotting log likelihood vs time 

    We are plotting esm score vs time to vizualize the trend before applying any fixes. We see an overall decrease in ESM score as time increases
    """
    )
    return


@app.cell
def _(plt, segment_dfs):
    def plot_branch_vs_trunk(
        df,
        x_col,
        y_col,
        filter_col,
        xlabel="X",
        ylabel="Y",
        title="Scatter Plot",
        trunk_label="Trunk",
        side_label="Side branches"
    ):

        plt.scatter(df[x_col], df[y_col], color='gray', s=10, label=side_label)

        trunk_df = df[df[filter_col] > 0.99]
        plt.scatter(trunk_df[x_col], trunk_df[y_col], color='red', s=10, label=trunk_label)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_ll_v():
        for seg, df in segment_dfs.items():
            plot_branch_vs_trunk(
                df,
                x_col="time",
                y_col="log_likelihood",
                filter_col="max_frequency",
                xlabel="Year",
                ylabel="ESM score",
                title=f"ESM Score vs Time {seg}"
            )

    plot_ll_v()
    return


@app.cell
def _(mo):
    mo.md(r"""### Apply LOESS correction to real data""")
    return


@app.cell
def _(loess_1d, segment_dfs):
    def apply_loess_to_segment(df, x_col="time", y_col="log_likelihood", degree=2, frac=0.15):
        x = df[x_col].values
        y = df[y_col].values

        xout, yout, wout = loess_1d(
            x=x,
            y=y,
            xnew=x,
            degree=degree,
            frac=frac
        )

        df.loc[:, f"{y_col}_LOESS"] = yout
        df.loc[:, "loess_weight"] = wout

        return df

    for seg_1, df_1 in segment_dfs.items():
        print(seg_1)
        #if(seg_1 != "PB1" and seg_1 != "MP"):
        segment_dfs[seg_1] = apply_loess_to_segment(df_1)
    return


@app.cell
def _(mo):
    mo.md(r"""### Plotting ESM Score vs Time with LOESS line""")
    return


@app.cell
def _(plt, segment_dfs):
    def plot_year_v_ESM_LOESS():
        for seg, df in segment_dfs.items():
            if "log_likelihood_LOESS" in df.columns:

                is_freq_one = df["max_frequency"] >= 0.99

                plt.scatter(df["time"][~is_freq_one], df["log_likelihood"][~is_freq_one],
                            color="grey", s=5)

                plt.scatter(df["time"][is_freq_one], df["log_likelihood"][is_freq_one],
                            color="red", s=5)

                plt.scatter(df["time"], df["log_likelihood_LOESS"], color="black", label="LOESS", s=1)

                plt.xlabel("Year")
                plt.ylabel("ESM score")
                plt.title(f"ESM Score vs Time with LOESS: {seg}")

                plt.show()

            else:
                print(f"Skipping {seg} — no LOESS column found.")


    plot_year_v_ESM_LOESS()
    return


@app.cell
def _(mo):
    mo.md(r"""### ESM score vs Time with LOESS correction""")
    return


@app.cell
def _(plt, segment_dfs):
    def plot_year_v_ESM_LOESS_corrected():
        for seg, df in segment_dfs.items():
            if "log_likelihood_LOESS" in df.columns:

                is_freq_one = df["max_frequency"] >= 0.99

                df["corrected_log_likelihood"] = df["log_likelihood"] - df["log_likelihood_LOESS"]

                plt.scatter(df["time"][~is_freq_one], df["corrected_log_likelihood"][~is_freq_one],
                            color="grey", s=5)

                plt.scatter(df["time"][is_freq_one], df["corrected_log_likelihood"][is_freq_one],
                            color="red", s=5)

                plt.xlabel("Year")
                plt.ylabel("ESM score")
                plt.title(f"ESM Score vs Time with LOESS: {seg}")

                plt.show()

            else:
                print(f"Skipping {seg} — no LOESS column found.")


    plot_year_v_ESM_LOESS_corrected()
    return


@app.cell
def _(mo):
    mo.md(r"""### ESM score with LOESS Correction vs maximum frequency""")
    return


@app.cell
def _(create_scatterplot, segment_dfs):
    def esm_v_max_freq_LOESS():
        for seg, df in segment_dfs.items():
            if "log_likelihood_LOESS" in df.columns:
                create_scatterplot(
                    df["corrected_log_likelihood"],
                    df["max_frequency"],
                    f"ESM LL vs max freq for {seg}"
                )

    esm_v_max_freq_LOESS()
    return


@app.cell
def _(pd, segment_dfs, spearmanr):
    def compute_spearman_LOESS():
        results = []
        for seg, df in segment_dfs.items():
            if "log_likelihood_LOESS" in df.columns:
                spearman, p = spearmanr(df["corrected_log_likelihood"], df["max_frequency"])
                results.append({"segment": seg, "spearman_LOESS": spearman, "p_LOESS": p})
        return pd.DataFrame(results)

    df_loess = compute_spearman_LOESS()

    df_loess
    return (df_loess,)


@app.cell
def _(df_loess, pd, spearman_df):
    combined_spearman_df = pd.merge(spearman_df, df_loess, on="segment", how="outer")
    combined_spearman_df
    return


@app.cell
def _(plt, sns, spearmanr):
    def plot_regression_corr(data, x_col, y_col, title, time, ylabel="", xlabel="", color="#0a2463", ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        old_mask = data["time"] < int(time)
        recent_mask = ~old_mask

        ax.scatter(data.loc[old_mask, x_col], data.loc[old_mask, y_col],
                   s=50, alpha=0.35, color="lightgray", label=f"< {int(time)}")

        ax.scatter(data.loc[recent_mask, x_col], data.loc[recent_mask, y_col],
                   s=50, alpha=0.35, color=color, label=f"≥ {int(time)}")

        light_recent = sns.desaturate(color, 0.5)

        if old_mask.sum() >= 2:
            sns.regplot(data=data.loc[old_mask], x=x_col, y=y_col, ax=ax,
                        scatter=False,
                        line_kws={"color": "gray", "linestyle": "--"}, ci=None)
        if recent_mask.sum() >= 2:
            sns.regplot(data=data.loc[recent_mask], x=x_col, y=y_col, ax=ax,
                        scatter=False,
                        line_kws={"color": light_recent}, ci=None)

        rho_old, _ = spearmanr(data.loc[old_mask, y_col], data.loc[old_mask, x_col])
        rho_recent, _ = spearmanr(data.loc[recent_mask, y_col], data.loc[recent_mask, x_col])

        ax.text(0.05, 0.95, f"ρ(<{int(time)}) = {rho_old:.2f}", transform=ax.transAxes, fontsize=10, color="gray",
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.0))

        ax.text(0.05, 0.85, f"ρ(≥{int(time)}) = {rho_recent:.2f}", transform=ax.transAxes, fontsize=10, color=light_recent,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.0))

        ax.set_title(title)
        ax.set_xlabel(xlabel, weight="bold")
        ax.set_ylabel(ylabel, weight="bold")
        ax.set_xlim(data[x_col].min(), data[x_col].max())
        ax.set_ylim(0, 1.1)
        ax.legend()

        return ax
    return (plot_regression_corr,)


@app.cell
def _(mo):
    mo.md(r"""### Spearman CC Max frequency vs ESM Score""")
    return


@app.cell
def _(plot_regression_corr, plt, segment_dfs):
    for seg_2, df_2 in segment_dfs.items():
        #if(seg_2 != "PB1" and seg_2 != "MP"):
            #print(seg_2)
        plot_regression_corr(df_2, x_col="log_likelihood", y_col="max_frequency", title=f"Spearman CC 650M Fine Tune for {seg_2} (Trained up to 1990)", ylabel="Maximum Frequency", xlabel="ESM Score", color="#0a2463", time=1990)

        plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### Spearman CC Max frequency vs ESM Score with LOESS Correction""")
    return


@app.cell
def _(plot_regression_corr, plt, segment_dfs):
    for seg_3, df_3 in segment_dfs.items():
        if(seg_3 != "PB1" and seg_3 != "MP"):
            #print(seg_2)
            plot_regression_corr(df_3, x_col="corrected_log_likelihood", y_col="max_frequency", title=f"Spearman CC 650M Fine Tune for {seg_3} (Trained up to 1990) with LOESS Correction", ylabel="Maximum Frequency", xlabel="ESM Score", color="#0a2463", time=1990)

        plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### Spearman Summary table""")
    return


@app.cell
def _(pd, segment_dfs, spearmanr):
    def compute_spearman_full():
        for seg_4, df_4 in segment_dfs.items():

                results = []

                for seg, df in segment_dfs.items():
                    if "log_likelihood_LOESS" in df.columns:

                        df_before_1990 = df[df["time"] < 1990]
                        df_after_1990 = df[df["time"] >= 1990]

                        spearman, p = spearmanr(df["log_likelihood"], df["max_frequency"])
                        spearman_LOESS, p = spearmanr(df["corrected_log_likelihood"], df["max_frequency"])

                        spearman_before_1990, p = spearmanr(df_before_1990["log_likelihood"], df_before_1990["max_frequency"])
                        spearman_LOESS_before_1990, p = spearmanr(df_before_1990["corrected_log_likelihood"], df_before_1990["max_frequency"])

                        spearman_after_1990, p = spearmanr(df_after_1990["log_likelihood"], df_after_1990["max_frequency"])
                        spearman_LOESS_after_1990, p = spearmanr(df_after_1990["corrected_log_likelihood"], df_after_1990["max_frequency"])

                        results.append({"segment": seg, "spearman_cc_ALL_TIME": spearman, "spearman_cc_LOESS_ALL_TIME": spearman_LOESS, "spearman_cc_Before_1990": spearman_before_1990, "spearman_cc_After_1990": spearman_after_1990, "spearman_cc_LOESS_Before_1990": spearman_LOESS_before_1990, "spearman_cc_LOESS_After_1990": spearman_after_1990, "spearman_cc_LOESS_After_1990": spearman_LOESS_after_1990})


                return pd.DataFrame(results)

    spearman_df_full = compute_spearman_full()
    spearman_df_full
    return (spearman_df_full,)


@app.cell
def _(mo):
    mo.md(r"""### Spearman Summary Plot with training and testing spearman CC separated""")
    return


@app.cell
def _(np, plt, spearman_df_full):
    def spearman_total_figure():

        triplets = [
            ("spearman_cc_ALL_TIME", "spearman_cc_LOESS_ALL_TIME"),
            ("spearman_cc_Before_1990", "spearman_cc_LOESS_Before_1990"),
            ("spearman_cc_After_1990", "spearman_cc_LOESS_After_1990")
        ]

        segments = spearman_df_full["segment"].astype(str)
        x = np.arange(len(segments))

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)

        for i, (ax, (cond1, cond2)) in enumerate(zip(axes, triplets), start=1):
            values1 = spearman_df_full[cond1]
            values2 = spearman_df_full[cond2]

            total_width = 0.8
            bar_width = total_width / 2

            ax.bar(x - bar_width/2, values1, width=bar_width, color='#005AB5', label='No LOESS')
            ax.bar(x + bar_width/2, values2, width=bar_width, color='#DC3220', label='With LOESS correction')

            ax.set_ylabel('Spearman CC')

            if i == 1:
                ax.set_title('Spearman CC - All Data')
            elif i == 2:
                ax.set_title('Spearman CC - Training Data (Before 1990)')
            elif i == 3:
                ax.set_title('Spearman CC - Test Data (After 1990)')

            if i == 1:
                ax.legend()

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(segments)
        axes[-1].set_xlabel('Segment')

        plt.tight_layout()
        plt.show()

        print()
    

    spearman_total_figure()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    - After LOESS Correction, Spearman CC is ~0, for data after 1990.
    - Spearman CC is >0 and positive on training dataset (before 1990).
    """
    )
    return


if __name__ == "__main__":
    app.run()
