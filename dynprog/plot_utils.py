import matplotlib.pyplot as plt
import pandas as pd

from dynprog import utils
from dynprog.value_iterator import ValueIterator
from typing import Optional


def get_iterator_frame(iterator: ValueIterator, iterator_name: Optional[str] = None) -> pd.DataFrame:
    value_dd = iterator.get_param(param="value_dd")
    value_df = utils.convert_output_dict(output_dd=value_dd, output_name=iterator_name)
    return value_df


def add_future_cash_on_hand(df: pd.DataFrame) -> pd.DataFrame:
    stack_df = df.stack(level="scenario")
    stack_df["H"] = 1.3 + stack_df["savings"]
    stack_df["L"] = 0.7 + stack_df["savings"]
    concat_df = stack_df.unstack(level="scenario")
    filter_ls = ["H", "L"]
    filter_ss = concat_df.columns.get_level_values("variable").isin(filter_ls)
    plot_df = concat_df.loc[:, filter_ss]
    return plot_df


def plot_variable(output_dd: dict,
                  plot_col: str,
                  x_lim: Optional[float] = None,
                  output_name: Optional[str] = None) -> None:
    concat_ls = [get_iterator_frame(iterator=i, iterator_name=n) for n, i in output_dd.items()]
    concat_df = pd.concat(concat_ls, axis="columns")
    concat_df.columns.names = ["scenario", "variable"]

    if plot_col == "future_cash_on_hand":
        plot_df = add_future_cash_on_hand(df=concat_df)
    else:
        plot_df = concat_df.xs(key=plot_col, axis="columns", level="variable")

    drop_df = plot_df.dropna()

    if x_lim is not None:
        drop_df = drop_df.loc[-x_lim:x_lim].copy()

    drop_df.plot(title=plot_col)
    if output_name is not None:
        plt.savefig(output_name)
