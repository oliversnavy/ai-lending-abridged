import pandas as pd


# Loan statuses that indicate a default event
_DEFAULT_STATUSES = {
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
}

# Loan statuses we include in the dataset (exclude anything ambiguous or in-progress)
_INCLUDED_STATUSES = {
    "Fully Paid",
    "Charged Off",
    "Current",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "In Grace Period",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
}


def add_survival_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `event` and `observed_time` columns to a LendingClub dataframe.

    event (int): 1 = default (Charged Off), 0 = censored
    observed_time (float): months from issue_d to last_pymnt_d

    Rows with unrecognized loan_status are dropped.
    """
    df = df.copy()

    # Drop rows with unrecognized statuses (e.g. "Default" — only 40 rows)
    df = df[df["loan_status"].isin(_INCLUDED_STATUSES)].copy()

    # Event indicator
    df["event"] = df["loan_status"].isin(_DEFAULT_STATUSES).astype(int)

    # Parse dates
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")

    # Observed time in months — fall back to 0 if last_pymnt_d is missing
    df["observed_time"] = (
        (df["last_pymnt_d"] - df["issue_d"]) / pd.Timedelta(days=30.44)
    ).clip(lower=0).fillna(0)

    return df
