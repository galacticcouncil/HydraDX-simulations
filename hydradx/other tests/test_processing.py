from hydradx.model import processing
from datetime import datetime, timezone, timedelta
import os
import shutil

def test_get_current_balance():
    now = datetime.now(timezone.utc)
    current_dot = processing.get_historical_omnipool_balance(tkn='DOT', date=now)
    print(f"{current_dot} DOT in the pool today.")

def test_get_history_today():
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_dot = processing.get_historical_omnipool_balance(tkn='DOT', date=today)
    print(f"{today_dot} DOT in the pool today.")

def test_get_balance_yesterday():
    yesterday = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    yesterday_dot = processing.get_historical_omnipool_balance(tkn='DOT', date=yesterday)
    print(f"{yesterday_dot} DOT in the pool yesterday.")

def test_get_balance_last_week():
    last_week = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=7)
    last_week_dot = processing.get_historical_omnipool_balance(tkn='DOT', date=last_week)
    print(f"{last_week_dot} DOT in the pool last week.")


def test_download_history_files():
    # rename history folder
    while not os.path.exists("./model"):
        os.chdir("..")
    os.chdir("model/data")
    os.rename('Omnipool Balance History', 'history_backup')

    # download history files
    processing.download_history_files()
    processing.get_historical_omnipool_balance(
        tkn='DOT',
        date="2024-09-21T16:04:12.000Z"
    )

    # restore history folder
    os.chdir("..")
    shutil.rmtree('Omnipool Balance History')
    os.rename('history_backup', 'Omnipool Balance History')
