import torch
import numpy as np
import pandas as pd

from core.util.io import read_csv


def normalize_trefor_park(park_data: pd.DataFrame) -> pd.DataFrame:
    capacities = [800, 2500, 2000, 800, 900, 1300, 700]
    for i, capacity in enumerate(capacities, 1):
        # normalize based on capacity to get relative (%) values
        park_data[f"Ladepark {i}"] = park_data[f"Ladepark {i}"] / capacity

    return park_data


def get_park_dataset(
    lookback: int, lookahead: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_val, x_test, y_train, y_val, y_test = (
        np.empty((1, lookback, 1)),
        np.empty((1, lookback, 1)),
        np.empty((1, lookback, 1)),
        np.empty((1, lookahead)),
        np.empty((1, lookahead)),
        np.empty((1, lookahead)),
    )

    for i in range(1, 7):
        park = read_csv(f"processed/park_{i}.csv")
        park = park.drop(["Date", "Time"], axis=1)
        x, y = split_sequences(park.to_numpy(), park.to_numpy(), lookback, lookahead)

        if i == 1:
            x_train = x
            y_train = y
        elif i <= 4:
            x_train = np.concatenate((x_train, x), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)
        elif i == 5:
            x_val = x
            y_val = y
        elif i == 6:
            x_test = x
            y_test = y

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )


# split a multivariate sequence past, future samples (X and y)
def split_sequences(features, targets, lookback, lookahead):
    x, y = list(), list()  # instantiate X and y
    for i in range(len(features)):
        # find the end of the input, output sequence
        end_ix = i + lookback
        out_end_ix = end_ix + lookahead
        # check if we are beyond the dataset
        if out_end_ix > len(features):
            break
        # gather input and output of the pattern
        seq_x, seq_y = features[i:end_ix], targets[end_ix:out_end_ix, -1]
        x.append(seq_x), y.append(seq_y)
        # if i == 1:
        #     print(x)
        #     print(type(x))
        #     print(y)
        #     print(type(y))
    return np.array(x), np.array(y)


# X_ss, y_mm = split_sequences(X_trans, y_trans, 100, 50)
# print(X_ss.shape, y_mm.shape)


def apply_sliding_window(
    timeseries: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply sliding window of size n.

    Arguments:
    ---------
        timeseries: The time series data
        n: Size of the sliding window

    """
    # Initialize lists to hold features and target
    x = []
    y = []

    # Add n datapoints to x then add target to y.
    for i in range(n, len(timeseries)):
        x.append(timeseries[i - n : i])
        y.append([timeseries[i]])
    return np.array(x), np.array(y)


def get_trefor_timeseries() -> np.ndarray:
    """Get processed trefor timeseries data as numpy array."""
    # Read trefor data csv
    data = read_csv("processed/trefor_final.csv")
    # Get just the total consumption, should be further processed with sliding window
    timeseries = data["Total_Consumption"].astype(float).to_numpy().reshape(-1, 1)

    return timeseries


def get_timeseries_dataset(
    timeseries: pd.DataFrame, n: int, m: int
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Get timeseries dataset with 80:10:10 train:val:test split.

    Arguments:
    ---------
        timeseries: The time series data
        n: Size of the sliding window

    """
    # Create sliding window of n size on the timeseries
    x, y = apply_sliding_window(timeseries, n)
    # Create indexes for 80% training, 10% validation, and 10% testing
    split_test = int(len(x) * 0.9)
    split_val = int(len(x) * 0.8)

    # split into train and test
    np_x_train, np_y_train, np_x_test, np_y_test = (
        x[:split_test],
        y[:split_test],
        x[split_test:],
        y[split_test:],
    )

    # split train into train and val
    np_x_train, np_y_train, np_x_val, np_y_val = (
        np_x_train[:split_val],
        np_y_train[:split_val],
        np_x_train[split_val:],
        np_y_train[split_val:],
    )

    # Create tensors from splits
    x_train = torch.tensor(data=np_x_train).float()
    y_train = torch.tensor(data=np_y_train).float()

    x_val = torch.tensor(data=np_x_val).float()
    y_val = torch.tensor(data=np_y_val).float()

    x_test = torch.tensor(data=np_x_test).float()
    y_test = torch.tensor(data=np_y_test).float()

    # Return tensors and squeeze y tensors to fit dimensions
    return (
        x_train,
        x_val,
        x_test,
        y_train.squeeze(1),
        y_val.squeeze(1),
        y_test.squeeze(1),
    )


def get_trefor_park_as_tensor(
    timeseries: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get timeseries dataset as tensor.

    Arguments:
    ---------
        timeseries: The time series data

    """
    x = []
    y = []

    time = timeseries.drop(["Dato", "Time"], axis=1).to_numpy()

    # loop through each row in timeseries
    for i in range(len(time)):
        x_temp = []
        y_temp = []
        # add t-24 to t-1 to x, add t+0 to t+23 to y
        for j in range(24):
            x_temp.append([time[i][j]])
            y_temp.append([time[i][24 + j]])

        # append values to array
        x.append(x_temp)
        y.append(y_temp)

    # convert list to numpy and float
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)

    # Create tensors of shape (len(x), 24, 1)
    x_tensor = torch.tensor(data=x).float()
    y_tensor = torch.tensor(data=y).float()

    return (
        x_tensor,
        y_tensor.squeeze(1),
    )
