from pathlib import Path

DATABASE_PATH = Path("agrimlops.db")


def get_database_path() -> Path:
    return DATABASE_PATH
