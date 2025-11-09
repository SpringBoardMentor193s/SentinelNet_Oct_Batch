"""datasetinfo.py

Runnable dataset summary script.

Place `kdd_train.csv` and `kdd_test.csv` in the repository root and run:

    python datasetinfo.py

It prints size, shape, null counts and a small preview for each dataset.
"""

import os
"""datasetinfo.py

Runnable dataset summary script.

Place `kdd_train.csv` and `kdd_test.csv` in the repository root and run:

    python datasetinfo.py

It prints size, shape, null counts and a small preview for each dataset.
"""

import os
from typing import Optional

import pandas as pd


def summarize_df(df: pd.DataFrame, name: str) -> None:
    print("=" * 60)
    print(name)
    print("=" * 60)
    print("size:", df.size)
    print("shape:", df.shape)
    print("null counts:")
    print(df.isna().sum())
    total_missing = df.isna().sum().sum()
    pct_missing = (total_missing / df.size) * 100 if df.size else 0.0
    print(f"total missing value percentage: {pct_missing:.4f}%")
    print("preview:")
    print(df.head())
    print()


def find_file(candidate: str, base: Optional[str] = None) -> Optional[str]:
    if base is None:
        base = os.path.dirname(__file__)
    path = os.path.join(base, candidate)
    return path if os.path.exists(path) else None


def main() -> int:
    base = os.path.dirname(__file__)

    train_file = find_file("kdd_train.csv", base)
    test_file = find_file("kdd_test.csv", base)

    if not train_file or not test_file:
        print("Error: expected 'kdd_train.csv' and 'kdd_test.csv' in the repository root.")
        if not train_file:
            print(" - missing kdd_train.csv")
        if not test_file:
            print(" - missing kdd_test.csv")
        return 1

    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
    except Exception as e:
        print("Failed to read CSVs:", e)
        return 2

    summarize_df(df_train, "Dataset Train")
    summarize_df(df_test, "Dataset Test")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import pandas as pd


def summarize_df(df: pd.DataFrame, name: str) -> None:
    print("=" * 60)
    print(name)
    print("=" * 60)
    print("size:", df.size)
    print("shape:", df.shape)
    print("null counts:")
    print(df.isna().sum())
    total_missing = df.isna().sum().sum()
    pct_missing = (total_missing / df.size) * 100 if df.size else 0.0
    print(f"total missing value percentage: {pct_missing:.4f}%")
    print("preview:")
    print(df.head())
    print()


def find_file(candidate: str, base: Optional[str] = None) -> Optional[str]:
    if base is None:
        base = os.path.dirname(__file__)
    path = os.path.join(base, candidate)
    return path if os.path.exists(path) else None


def main() -> int:
    base = os.path.dirname(__file__)

    train_file = find_file("kdd_train.csv", base)
    test_file = find_file("kdd_test.csv", base)

    if not train_file or not test_file:
        print("Error: expected 'kdd_train.csv' and 'kdd_test.csv' in the repository root.")
        if not train_file:
            print(" - missing kdd_train.csv")
        if not test_file:
            print(" - missing kdd_test.csv")
        return 1

    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
    except Exception as e:
        print("Failed to read CSVs:", e)
        return 2

    summarize_df(df_train, "Dataset Train")
    summarize_df(df_test, "Dataset Test")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""datasetinfo.py

Runnable dataset summary script.

Put `kdd_train.csv` and `kdd_test.csv` in the repository root and run:
    python datasetinfo.py

It prints size, shape, null counts and a small preview.
"""

import os
from typing import Optional

import pandas as pd


def summarize_df(df: pd.DataFrame, name: str) -> None:
    print("=" * 60)
    print(name)
    print("=" * 60)
    print("size:", df.size)
    print("shape:", df.shape)
    print("null counts:")
    print(df.isna().sum())
    total_missing = df.isna().sum().sum()
    pct_missing = (total_missing / df.size) * 100 if df.size else 0.0
    print(f"total missing value percentage: {pct_missing:.4f}%")
    print("preview:")
    print(df.head())
    print()


def find_file(candidate: str, base: Optional[str] = None) -> Optional[str]:
    if base is None:
        base = os.path.dirname(__file__)
    path = os.path.join(base, candidate)
    return path if os.path.exists(path) else None


def main() -> int:
    base = os.path.dirname(__file__)

    train_file = find_file("kdd_train.csv", base)
    test_file = find_file("kdd_test.csv", base)

    if not train_file or not test_file:
        print("Error: expected 'kdd_train.csv' and 'kdd_test.csv' in the repository root.")
        if not train_file:
            print(" - missing kdd_train.csv")
        if not test_file:
            print(" - missing kdd_test.csv")
        return 1

    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
    except Exception as e:
        print("Failed to read CSVs:", e)
        return 2

    summarize_df(df_train, "Dataset Train")
    """datasetinfo.py

    Runnable dataset summary script.

    Place `kdd_train.csv` and `kdd_test.csv` in the repository root and run:

        python datasetinfo.py

    It prints size, shape, null counts and a small preview for each dataset.
    """

    import os
    from typing import Optional

    import pandas as pd


    def summarize_df(df: pd.DataFrame, name: str) -> None:
        print("=" * 60)
        print(name)
        print("=" * 60)
        print("size:", df.size)
        print("shape:", df.shape)
        print("null counts:")
        print(df.isna().sum())
        total_missing = df.isna().sum().sum()
        pct_missing = (total_missing / df.size) * 100 if df.size else 0.0
        print(f"total missing value percentage: {pct_missing:.4f}%")
        print("preview:")
        print(df.head())
        print()


    def find_file(candidate: str, base: Optional[str] = None) -> Optional[str]:
        if base is None:
            base = os.path.dirname(__file__)
        path = os.path.join(base, candidate)
        return path if os.path.exists(path) else None


    def main() -> int:
        base = os.path.dirname(__file__)

        train_file = find_file("kdd_train.csv", base)
        test_file = find_file("kdd_test.csv", base)

        if not train_file or not test_file:
            print("Error: expected 'kdd_train.csv' and 'kdd_test.csv' in the repository root.")
            if not train_file:
                print(" - missing kdd_train.csv")
            if not test_file:
                print(" - missing kdd_test.csv")
            return 1

        try:
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)
        except Exception as e:
            print("Failed to read CSVs:", e)
            return 2

        summarize_df(df_train, "Dataset Train")
        summarize_df(df_test, "Dataset Test")

        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    except Exception as e:
