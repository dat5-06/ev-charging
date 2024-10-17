import math
from core.util.io import read_csv
from scipy import stats


def avg_loss(x: list, y: list, regression: callable, loss_function: callable) -> float:
    """Calculate average loss for regression function."""
    loss = 0
    for i in range(len(x)):
        loss += loss_function(y[i], regression(x[i]))
    return loss / len(x)


def get_naive_loss(loss_func: callable)-> float:
    """Make linear regression and returns the average loss."""
    normalised_data = read_csv("processed/park_timeseries_lin.csv")

    #Names of all park titles
    columns = normalised_data.columns.to_list()

    #Adds all values to coordinate lists x and y
    x,y = [], []
    for index, row in normalised_data.iterrows():
        for title in columns:
            if not math.isnan(row[title]):
                x.append(index)
                y.append(row[title])

    #Makes linear regression
    slope, intercept, _, _, _ = stats.linregress(x,y)
    def regression(x: float) -> float:
        """Make a function for the regression."""
        return slope * x + intercept

    return avg_loss(x,y,regression,loss_func)
