from pathlib import Path
import sqlite3

import pandas as pd


DATABASE_PATH = Path("data/quantalpha.db")


def get_connection() -> sqlite3.Connection:
    """Create and return a connection to the QuantAlpha database."""

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    return sqlite3.connect(DATABASE_PATH)


def initialize_database() -> None:
    """Create all required database tables."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            UNIQUE(date, symbol)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            total_return REAL,
            annualized_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            trades INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def insert_market_data(data: pd.DataFrame) -> None:
    """Insert market data into the database."""

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Symbol",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    connection = get_connection()

    records = [
        (
            row["Date"].strftime("%Y-%m-%d"),
            row["Symbol"],
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            float(row["Volume"]),
        )
        for _, row in data.iterrows()
    ]

    connection.executemany(
        """
        INSERT OR IGNORE INTO market_data
        (date, symbol, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )

    connection.commit()
    connection.close()


def get_market_data(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Retrieve market data from SQLite."""

    query = """
        SELECT
            date AS Date,
            open AS Open,
            high AS High,
            low AS Low,
            close AS Close,
            volume AS Volume,
            symbol AS Symbol
        FROM market_data
        WHERE symbol = ?
    """

    parameters = [symbol]

    if start_date:
        query += " AND date >= ?"
        parameters.append(start_date)

    if end_date:
        query += " AND date <= ?"
        parameters.append(end_date)

    query += " ORDER BY date"

    connection = get_connection()

    data = pd.read_sql_query(
        query,
        connection,
        params=parameters,
    )

    connection.close()

    if not data.empty:
        data["Date"] = pd.to_datetime(data["Date"])

    return data