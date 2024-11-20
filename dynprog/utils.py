import pandas as pd

from typing import Optional


def _filter_dict(dd: dict, filter_key: str) -> dict:
    filter_dd = dict()
    for state, value_dd in dd.items():
        filter_dd.setdefault(state, dict())
        update_dd = {k: v[filter_key] for k, v in value_dd.items()}
        filter_dd[state].update(update_dd)
    return filter_dd


def convert_dict_to_frame(dd: dict, key: str, filter_key: Optional[str] = None) -> pd.DataFrame:
    if filter_key is not None:
        dd = _filter_dict(dd=dd, filter_key=filter_key)

    df = pd.DataFrame(dd)
    df = pd.concat([df], axis="columns", keys=[key])
    return df


def convert_output_dict(output_dd: dict, output_name: Optional[str] = None) -> pd.DataFrame:
    concat_dd = {state: pd.DataFrame(dd).T for state, dd in output_dd.items()}
    concat_ls = [pd.concat([df], axis="index", keys=[state]) for state, df in concat_dd.items()]
    concat_df = pd.concat(concat_ls)
    concat_df.index.names = ["state", "cash_on_hand"]
    average_df = concat_df.groupby(level="cash_on_hand").mean()

    if output_name is not None:
        average_df = pd.concat([average_df], axis="columns", keys=[output_name])
    return average_df
