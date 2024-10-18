from core.util.io import read_csv, write_csv, read_xlsx
import pandas as pd


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


def park_preprocess() -> None:
    """Process public charginc station data."""
    original = read_csv("interim/trefor_park.csv")

    for park_num in range(1, 7):
        park = pd.DataFrame(original[["Dato", "Time", f"Ladepark {park_num}"]].dropna())
        park_renamed = park.rename(
            columns={f"Ladepark {park_num}": "Consumption", "Dato": "Date"}
        )
        write_csv(park_renamed, f"processed/park_{park_num}.csv")

    print("Processed Trefor park data")


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

    print("Processed Trefor household data")


def trefor() -> None:
    """Preprocess all Trefor data."""
    park_cleaning()
    park_preprocess()
    household_cleaning()
    household_preprocessing()


if __name__ == "__main__":
    trefor()
