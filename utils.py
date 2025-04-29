from statsmodels.nonparametric.smoothers_lowess import lowess

def moving_average(x, y):
    points = lowess(y, x, it=3, frac=0.1)
    return points[: ,0], points[:, 1]