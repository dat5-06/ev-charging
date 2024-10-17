from core.preprocessing.sliding_window import apply_sliding
from core.util.io import read_csv, write_csv, read_xlsx
import pandas as pd
import numpy as np

capacities = [800, 2500, 2000, 800, 900, 1300, 700]

def park_preprocess_lin() -> None:
    """Process public charginc station data."""
    original = read_csv("interim/trefor_park.csv")

    for i, capacity in enumerate(capacities, 1):
        # normalize based on capacity to get relative (%) values
        original[f"Ladepark {i}"] = original[f"Ladepark {i}"] / capacity

    original = original.drop(columns=["Dato", "Time"])

    write_csv(original, "processed/park_timeseries_lin.csv")


def park_cleaning() -> None:
    """Process public charging station data."""
    original = read_xlsx("raw/trefor_park.xlsx")

    # crate deep copy that we can manupulate
    data = original.copy(deep=True)
    data = data.drop(columns=["Dato", "Time"])

    # remove values until first non-zero
    cleaned_data = data.apply(lambda col: col.iloc[col.ne(0).idxmax() :])

    # merge back together and save
    combined = pd.concat([original[["Dato", "Time"]], cleaned_data], axis=1)
    write_csv(combined, "interim/trefor_park.csv")


def park_preprocess(backward: int, forward: int) -> None:
    """Process public charginc station data."""
    original = read_csv("interim/trefor_park.csv")

    columns = np.array(
        ["Dato", "Time"]
        + [f"t-{backward-i}" for i in range(backward)]
        + [f"t+{i}" for i in range(forward)]
    )
    merged = np.array([], dtype=object).reshape(0, 2 + backward + forward)

    for i, capacity in enumerate(capacities, 1):
        # normalize based on capacity to get relative (%) values
        original[f"Ladepark {i}"] = original[f"Ladepark {i}"] / capacity

        (x, y), index = apply_sliding(original[f"Ladepark {i}"], backward, forward)

        # Save corresponding time
        time = original[
            original.index.isin(
                range(backward + index, len(original.index) - forward + 1)
            )
        ][["Dato", "Time"]]
        m = np.concatenate([time, x, y], axis=1)
        if i <= len(capacities) - 2:
            merged = np.concatenate([merged, m])
        elif i == len(capacities) - 1:
            write_csv(
                pd.DataFrame(m, columns=columns),
                "processed/park_testing.csv",
            )
        elif i == len(capacities):
            write_csv(
                pd.DataFrame(m, columns=columns),
                "processed/park_validation.csv",
            )

    write_csv(pd.DataFrame(merged, columns=columns), "processed/park_training.csv")


def household_cleaning() -> None:
    """Clean the Trefor household data."""
    # Loading the original data into a pandas dataframe
    file_path = "raw/trefor_raw.csv"
    consumption = read_csv(file_path)

    # Dropping date and time, as this is not needed when cleaning
    df_original = consumption.drop(columns=["Dato", "Time"])

    # Copy is disconnected from the orignal df
    df_cleaning = df_original.copy(deep=True)

    # Only take valuse that are greater than 0
    non_zeroes = df_cleaning.apply(lambda series: series[series > 0])
    # And only take values that are less than 100
    # (some of it is weird, according to Trefor)
    cleaned = non_zeroes.apply(lambda series: series[series < 100])

    # remove households with maximum consumption less than 2kWh
    cleaned = cleaned.loc[:, (cleaned.max(axis=0) > 2)]
    # remove households with mean consumption less than 0.1kWh
    cleaned = cleaned.loc[:, (cleaned.mean(axis=0) > 0.1)]

    # Reads the date and time before transfering to csv file
    cleaned = pd.concat([consumption[["Dato", "Time"]], cleaned], axis=1)

    # Return the cleaned data into another csv
    output_path = "interim/trefor_cleaned.csv"
    write_csv(cleaned, output_path)


def household_preprocessing() -> None:
    """Preprocess the Trefor household data."""
    # Loading the cleaned data into a pandas dataframe
    file_path = "interim/trefor_cleaned.csv"
    trefor_data = read_csv(file_path)

    # Add a total consumption column to the dataframe and remove
    # all individual households
    trefor_data["Total_Consumption"] = trefor_data.sum(axis=1, numeric_only=True)
    trefor_data = trefor_data.filter(["Dato", "Time", "Total_Consumption"], axis=1)

    # Outputs the processed data as csv.
    output_path = "processed/trefor_final.csv"
    write_csv(trefor_data, output_path)


def trefor(backward: int, forward: int) -> None:
    """Preprocess all Trefor data."""
    park_cleaning()
    park_preprocess_lin()
    park_preprocess(backward, forward)
    household_cleaning()
    household_preprocessing()
    print("Processed Trefor data")



if __name__ == "__main__":
    trefor(24, 24)
