
from itertools import accumulate
import torch
import fsrs_optimizer

def cum_concat(x):
    return list(accumulate(x))

def create_time_series(df):
    df["review_th"] = range(1, df.shape[0] + 1)
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df.drop(df[~df["rating"].isin([1, 2, 3, 4])].index, inplace=True)
    df["i"] = df.groupby("card_id").cumcount() + 1
    df.drop(df[df["i"] > 64 * 2].index, inplace=True)
    card_id_to_first_rating = df.groupby("card_id")["rating"].first().to_dict()
    if (
        "delta_t" not in df.columns
        and "elapsed_days" in df.columns
        and "elapsed_seconds" in df.columns
    ):
        df["delta_t"] = df["elapsed_days"]
    t_history_list = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[max(0, i)] for i in x])
    )
    r_history_list = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history_list for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history_list for item in sublist
    ]
    df["tensor"] = [
        torch.tensor((t_item[:-1], r_item[:-1])).transpose(0, 1)
        for t_sublist, r_sublist in zip(t_history_list, r_history_list)
        for t_item, r_item in zip(t_sublist, r_sublist)
    ]
    last_rating = []
    for t_sublist, r_sublist in zip(t_history_list, r_history_list):
        for t_history, r_history in zip(t_sublist, r_sublist):
            flag = True
            for t, r in zip(reversed(t_history[:-1]), reversed(r_history[:-1])):
                if t > 0:
                    last_rating.append(r)
                    flag = False
                    break
            if flag:
                last_rating.append(r_history[0])
    df["last_rating"] = last_rating
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    df.drop(df[df["elapsed_days"] == 0].index, inplace=True)
    df["i"] = df.groupby("card_id").cumcount() + 1
    df["first_rating"] = df["card_id"].map(card_id_to_first_rating).astype(str)

    filtered_dataset = (
        df[df["i"] == 2]
        .groupby(by=["first_rating"], as_index=False, group_keys=False)[df.columns]
        .apply(fsrs_optimizer.remove_outliers)
    )
    if filtered_dataset.empty:
        return pd.DataFrame()
    df[df["i"] == 2] = filtered_dataset
    df.dropna(inplace=True)
    df = df.groupby("card_id", as_index=False, group_keys=False)[df.columns].apply(
        fsrs_optimizer.remove_non_continuous_rows
    )
    return df[df["elapsed_days"] > 0].sort_values(by=["review_th"])
