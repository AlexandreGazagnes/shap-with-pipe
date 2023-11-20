import pandas as pd


def resultize(grid):
    """takes a gridsearchcv object and returns a dataframe with the results"""

    res = pd.DataFrame(grid.cv_results_)
    cols = [i for i in res.columns if "split" not in i]
    res = res.loc[:, cols]

    res = res.sort_values(by="mean_test_score", ascending=False)
    return res.round(2)
