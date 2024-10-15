from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd


def read_xlsx(file_path: str) -> pd.DataFrame:
    """Read xlsx file to pandas dataframe.

    Arguments:
    ---------
        file_path: Path to xlsx file relative to project root file to be read.

    """
    data_root = get_data_root()
    path = data_root / file_path
    return pd.read_excel(path)


def read_csv(file_path: str) -> pd.DataFrame:
    """Read csv file to pandas dataframe.

    Reading csv files with ';' as separator and ',' decimals.

    Arguments:
    ---------
        file_path: Path to csv relative to project root file to be read.

    """
    data_root = get_data_root()
    path = data_root / file_path
    return pd.read_csv(path, sep=";", decimal=",")


def write_csv(df: pd.DataFrame, file_path: str, force: bool = False) -> None:
    """Write csv file to specified path.

    Formatting csv files with ';' as separator and ',' decimals.

    Arguments:
    ---------
        file_path: relative output path from project root for the csv file.
        df: DataFrame to convert to csv.
        force: If True, overwrite existing file.

    """
    data_root = get_data_root()
    path = data_root / file_path
    if path.exists() and not force:
        return

    # Make directories if they don't exist.
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, sep=";", decimal=",", index=False)


def save_fig(file_path: str, force: bool = False) -> None:
    """Save matplotlib figure.

    Arguments:
    ---------
        file_path: relative output path from project root for the figure.
        force: If True, overwrite existing file.

    """
    figures_root = get_figures_root()
    path = figures_root / file_path
    if path.exists() and not force:
        return

    # Make directories if they don't exist.
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(path)


def get_project_root() -> Path:
    """Return path of the project root directory."""
    cur_path = Path(__file__).resolve()
    root_dir = "zap"

    # Traverse up the directories until root directory is reached
    for parent in cur_path.parents:
        if parent.name == root_dir:
            return parent

    # Raise exception if root directory is not found
    raise FileNotFoundError(
        root_dir + " directory was not found in the path hierarchy."
    )


def get_data_root() -> Path:
    """Return path of the data root directory."""
    return get_project_root() / "core/data"


def get_figures_root() -> Path:
    """Return path of the figures root directory."""
    return get_project_root() / "core/figures"
