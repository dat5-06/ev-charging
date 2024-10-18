import torch
import numpy as np
import pandas as pd

from core.util.io import read_csv


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
    timeseries: np.ndarray, n: int
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


def denormalize_trefor(multiply: int, tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize trefor data."""
    return tensor * multiply
