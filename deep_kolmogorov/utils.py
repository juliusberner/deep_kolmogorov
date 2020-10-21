import os
import torch
import pandas as pd
from ray import tune
from copy import deepcopy
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from ipywidgets import widgets
from IPython.core import display
from .trainer import HYPERCONFIGS, Trainer


Col = namedtuple("column", "name, precision, type, show_std, map_name")


DF_KEYS = {"params": "initial_stats/params", "seed": "config/seed"}
BEST_COLS = [
    Col("train/current/overall time", 0, "Int32", True, "avg. time (s) +/- std."),
    Col("val/best/L1", 4, float, True, "avg. best L^1 +/- std."),
    Col("initial_stats/params", 0, "Int32", False, "#parameters"),
    Col("train/current/overall steps", 0, "Int32", False, "#steps"),
    Col("val/last improve/L1", 0, "Int32", True, "last L^1 improve (it)"),
]
SHOW_COLS = [
    Col("train/current/overall steps", 0, "Int32", False, "step"),
    Col("train/current/overall time", 0, "Int32", True, "avg. time (s) +/- std."),
    Col("val/current/L1", 4, float, True, "avg. L^1 +/- std."),
    Col("val/current/L2", 4, float, True, "avg. L^2 +/- std."),
]


def initial_map(df):
    return {
        "train/current/overall time": 0.0,
        "train/current/overall steps": 0,
        "val/current/L1": df["initial_stats/val_initial/L1"][0],
        "val/current/L2": df["initial_stats/val_initial/L2"][0],
    }


def dims_to_x_y(best, remove_first_n=None, remove_last_n=None):
    for k in ["mean", "std"]:
        best[k]["dims"] = best[k].index.map(lambda x: int(x.split("_")[-1][:-1]))
        best[k].sort_values(by="dims", inplace=True)
    plot_values = {
        "x": {
            "value": np.array(best["mean"]["dims"] ** 2 + best["mean"]["dims"] + 1)[
                remove_first_n:remove_last_n
            ],
            "name": r"$\operatorname{dim}_\operatorname{in}\Phi$",
        }
    }
    params = best["mean"].loc[:, "initial_stats/params"]
    for k, name in zip(
        ["mean", "std"],
        [r"#parameters $\times$ avg. #steps", r"$\pm\ \operatorname{std.}$"],
    ):
        plot_values[k] = {
            "value": np.array(best[k].loc[:, "train/current/overall steps"] * params)[
                remove_first_n:remove_last_n
            ],
            "name": name,
        }
    return plot_values


def get_run_widget():
    exp = widgets.Dropdown(options=HYPERCONFIGS.keys(), description="experiment")
    gpus = widgets.Dropdown(options=list(range(8)), description="gpus per trial")
    return widgets.HBox([exp, gpus])


def get_exp_widget():
    exp_dir = os.path.join(os.getcwd(), "exp")
    analysis_list = []
    for d in os.listdir(exp_dir):
        try:
            analysis = tune.Analysis(os.path.join(exp_dir, d))
            analysis_list.append((d, analysis))
        except tune.TuneError:
            pass
    return widgets.Dropdown(options=analysis_list, description="experiment")


def combine_tests(tests, add_test):
    for t in tests:
        if add_test["config"] == t["config"]:
            t["paths"].append(add_test["paths"][0])
            return
    tests.append(add_test)


def extract_tests(analysis):
    tests = []
    for path, config in analysis.get_all_configs().items():
        config.pop("seed")
        add_test = {"config": config, "paths": [path]}
        combine_tests(tests, add_test)
    return tests


def add_meta_information(t, analysis, show_every_nth, cut_idx):
    df = analysis.dataframe()[analysis.dataframe()["logdir"].isin(t["paths"])]
    t["paths"] = df["logdir"].to_list()
    t["seeds"] = df[DF_KEYS["seed"]].to_list()
    t["models"] = [analysis.get_trial_checkpoints_paths(p) for p in t["paths"]]
    t["trial_dfs"] = [
        merge_initial(analysis.trial_dataframes[p])[:cut_idx:show_every_nth]
        for p in t["paths"]
    ]
    t["df"] = df
    df_row = df.iloc[0]
    test_tag = df_row["experiment_tag"].split(",")
    t["tune_hp"] = [t.split("=")[-1] for t in test_tag if "seed" not in t]
    t["parameters"] = df_row[DF_KEYS["params"]]


def merge_initial(trial_df):
    initial_stats = initial_map(trial_df)
    df = pd.DataFrame.from_records([initial_stats], columns=trial_df.columns)
    return pd.concat([df, trial_df], ignore_index=True)


def format_col(mean, std, col, latex=False):
    mean = mean.loc[:, col.name].round(col.precision).astype(col.type).astype(str)
    if col.show_std and not std.loc[:, col.name].isnull().all():
        std = std.loc[:, col.name].round(col.precision).astype(col.type).astype(str)
        sep = r" $\pm$ " if latex else " +/- "
        return (mean + sep + std).to_frame(col.map_name)
    else:
        return mean.to_frame(col.map_name)


def prepare_df_group(df, cols, groupby=None, latex_index=False):
    col_names = [col.name for col in cols]
    if groupby is None:
        groupby = df.index
    else:
        col_names.append(groupby)
    df = df.loc[:, col_names]
    tables = {
        "mean": df.groupby(groupby).mean(),
        "std": df.groupby(groupby).std(),
    }
    for key in ["table", "latex"]:
        table = pd.concat(
            [
                format_col(tables["mean"], tables["std"], col, latex=(key == "latex"))
                for col in cols
            ],
            axis=1,
            sort=False,
        )
        if key == "latex":
            table = table.to_latex(index=latex_index, bold_rows=True, escape=False)
        tables[key] = table
    return tables


def prepare(analysis, show_every_nth=1, cut_idx=None):
    tests = extract_tests(analysis)
    for t in tests:
        add_meta_information(t, analysis, show_every_nth, cut_idx)
        conc = pd.concat(t["trial_dfs"])
        t.update(prepare_df_group(conc, SHOW_COLS))
    conc_all = pd.concat(
        [t["df"] for t in tests],
        keys=["/".join(t["tune_hp"]) for t in tests],
        names=["setting", "idx"],
    ).reset_index()
    best = prepare_df_group(conc_all, BEST_COLS, groupby="setting", latex_index=True)
    return tests, best


def plot_fits(
    plot_values, ax, show_exp=False, show_poly=True, xscale="linear", yscale="linear"
):
    x = plot_values["x"]["value"]
    y = plot_values["mean"]["value"]
    evaluate = np.linspace(min(x), max(x))
    if show_exp:
        fit = np.polyfit(x, np.log(y), deg=1)
        y_fit = np.exp(fit[1]) * np.exp(evaluate * fit[0])
        ax.plot(
            evaluate,
            y_fit,
            "--",
            color="orange",
            label="$y={0:.1f}e^{{{1:.2f}x}}$".format(np.exp(fit[1]), fit[0]),
            lw=2.5,
        )
    if show_poly:
        fit = np.polyfit(np.log(x), np.log(y), deg=1)
        y_fit = np.exp(fit[1]) * np.power(evaluate, fit[0])
        ax.plot(
            evaluate,
            y_fit,
            "--b",
            label="$y={0:.1f}x^{{{1:.2f}}}$".format(np.exp(fit[1]), fit[0]),
            lw=2.5,
        )
    ax.plot(x, y, "-or", lw=2.5, label=plot_values["mean"]["name"])
    ax.fill_between(
        x,
        y - plot_values["std"]["value"],
        y + plot_values["std"]["value"],
        color="r",
        alpha=0.1,
        label=plot_values["std"]["name"],
    )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


def get_dims_plot(best, remove_first_n=None, remove_last_n=None, save=True):
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_values = dims_to_x_y(
        best, remove_first_n=remove_first_n, remove_last_n=remove_last_n
    )
    plot_fits(plot_values, ax)
    ax.set_xlabel(plot_values["x"]["name"])
    ax.legend(loc="lower right")
    in_ax = inset_axes(ax, width="50%", height=2.0, loc=2)
    plot_fits(plot_values, in_ax, xscale="log", yscale="log")
    in_ax.yaxis.set_label_position("right")
    in_ax.yaxis.tick_right()
    plt.show()
    if save:
        fig.savefig(
            os.path.join("figures", f"cost_vs_dim.pdf"),
            bbox_inches="tight",
            pad_inches=0,
        )


def visualize_exp(
    analysis,
    show_only_final=True,
    show_seeds=True,
    show_every_nth=1,
    cut_idx=None,
    latex=False,
    save=False,
):
    tests, best = prepare(analysis, show_every_nth, cut_idx)
    if not show_only_final:
        for t in tests:
            print(f'configuration: {t["config"]}\n')
            if show_seeds:
                print(f'seeds: {t["seeds"]}\n')
            print("results:")
            display.display(t["table"])
            if latex:
                print(t["latex"])
        print("\nfinal:")
    display.display(best["table"])
    if latex:
        print(best["latex"])
    if all(t["config"]["mode"] == "dims_heat_paraboloid" for t in tests):
        get_dims_plot(best, save=save)


def get_bs_widget(analysis):
    plt.rcParams.update({"font.size": 22})
    tests, _ = prepare(analysis)
    assert (
        len(tests) == 1 and tests[0]["config"]["mode"] == "avg_bs"
    ), "choose avg_bs experiment"
    test = tests[0]
    model_paths = test["models"]
    model_paths = [p[0][0] for p in model_paths]
    assert len(model_paths) == len(test["seeds"]), "please check your model checkpoints"
    nets = []
    test["config"]["gpus"] = 0
    tr = Trainer(test["config"])
    for p in model_paths:
        tr._restore(p)
        nets.append(deepcopy(tr.net))

    def get_prediction_plot(ax, linspace, y, y_preds):
        mean = y_preds.mean(dim=1)
        ax.plot(linspace, mean, "-r", linewidth=2.5, label="avg. prediction")
        ax.plot(linspace, y, "-b", linewidth=2.5, label="solution")
        if y_preds.shape[1] > 1:
            std = y_preds.std(dim=1)
            ax.fill_between(
                linspace.squeeze(),
                mean - std,
                mean + std,
                color="r",
                alpha=0.1,
                label=r"$\pm\ \operatorname{std.}$",
            )
        return ax

    def get_error_plot(ax, linspace, y, y_preds):
        errors = (y_preds - y).abs() / (1 + y.abs())
        mean_error = errors.mean(dim=1)
        ax.plot(
            linspace, mean_error, "-", color="orange", linewidth=2.5, label="avg. error"
        )
        if y_preds.shape[1] > 1:
            std_error = errors.std(dim=1)
            ax.fill_between(
                linspace.squeeze(),
                mean_error - std_error,
                mean_error + std_error,
                color="orange",
                alpha=0.1,
                label=r"$\pm\ \operatorname{std.}$",
            )
        return ax

    def plot_bs(t, x, sigma, K, free, vega=False, error=False, save=False):
        inputs = {"t": t, "x": x, "sigma": sigma, "K": K}
        linspace = torch.linspace(*tr.pde.hypercubes[free].interval).unsqueeze(1)
        batch = {key: val * torch.ones_like(linspace) for key, val in inputs.items()}
        batch[free] = linspace
        preds_list = []
        if vega:
            y = tr.pde.vega_solution(batch)
            for net in nets:
                batch["sigma"] = batch["sigma"].clone().detach().requires_grad_(True)
                batch_flat = tr.pde.normalize_and_flatten(batch)
                net.eval()
                pred = net(batch_flat)
                pred.backward(torch.ones_like(linspace))
                preds_list.append(batch["sigma"].grad)
        else:
            with torch.no_grad():
                y = tr.pde.solution(batch)
                batch_flat = tr.pde.normalize_and_flatten(batch)
                for net in nets:
                    net.eval()
                    preds_list.append(net(batch_flat))
        preds = torch.cat(preds_list, dim=1)
        specs = "vega" if vega else "option price"
        if error:
            specs = f"error ({specs})"
        specs = [specs] + [
            f"{key}={value}".replace(".", "_")
            for key, value in inputs.items()
            if not key == free
        ]
        fig, ax = plt.subplots(figsize=(10, 7))
        if error:
            ax = get_error_plot(ax, linspace, y, preds)
        else:
            ax = get_prediction_plot(ax, linspace, y, preds)
        ax.set_xlabel(free)
        ax.set_ylabel(specs[0])
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()
        if save:
            name = "__".join(specs).replace(" ", "_")
            path = os.path.join(os.getcwd(), "figures")
            os.makedirs(path, exist_ok=True)
            fig.savefig(os.path.join(path, f"{name}.pdf"))
            print("figure saved!")

    widgets.interact(
        plot_bs,
        free=list(tr.pde.params),
        t=tuple(tr.pde.hypercubes["t"].interval),
        x=tuple(tr.pde.hypercubes["x"].interval),
        sigma=tuple(tr.pde.hypercubes["sigma"].interval),
        K=tuple(tr.pde.hypercubes["K"].interval),
    )
