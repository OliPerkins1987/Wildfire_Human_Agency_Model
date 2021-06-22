# -*- coding: utf-8 -*-
import sharepy
import io
import pandas as pd
import os
import json
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

env_path = Path("..") / ".env"  # move up one directory
load_dotenv(dotenv_path=env_path)
print(os.getenv("SERVER"))

def read_shpt_data(file_path: str, date_cols=None, date_format=None, types=None, low_memory=True):
    """Takes in file name and returns dataframe, uses sharepy to authenticate
    session
    Args:
        file_path: path of file using Documents as root"""

    s = sharepy.connect(
        os.getenv("SERVER"), os.getenv("SP_USERNAME"), os.getenv("PASSWORD")
    )  # reconnect new instance in case of concurrency
    file_base = "https://emckclac-my.sharepoint.com/personal/k1758409_kcl_ac_uk/"
    r = s.get(file_base + file_path)
    if r.status_code == 200:
        if file_path[-3:] == "csv":
            # for csv files
            f = io.BytesIO(r.content)
            df = pd.read_csv(f, encoding="cp1252", parse_dates=date_cols, date_parser=date_format, dtype=types, low_memory=low_memory)   # windows encoding for csv
            return df
        else:
            # non-csv files
            f = io.BytesIO(r.content)
            return f
    else:
        print(f"Error, Status Code: {r.status_code}")


def shpt_file_list() -> list:
    """Prints all files and folders in Documents
    Clips the full filepath for readability
    """

    s = sharepy.connect(
        os.getenv("SERVER"), os.getenv("SP_USERNAME"), os.getenv("PASSWORD")
    )  # reconnect new instance in case of concurrency
    site = "https://emckclac-my.sharepoint.com/personal/k1758409_kcl_ac_uk/"
    library = "Documents"
    # Get all files and folders recursively
    files = s.get(
        f"{site}/_api/web/lists/getbyTitle('{library}')/Items?$select=FileLeafRef,FileRef,Id&$top=5000"
    ).json()["d"]["results"]

    file_store = []
    for file in files:  # array of dict, just want file names
        filepath = file["FileRef"]
        filename = filepath[29:] # remove the full path
        file_store.append(filename)
    return sorted(file_store)


def write_shpt_data(data, file_path, encoding='utf-8'):
    """Uploads files, tested with csv and txt
    Args:
        data: pd.DataFrame or data object in binary form to be written
        file_path: destination file path include file extension
    """

    if type(data) == pd.DataFrame:
        data = data.to_csv(mode="wb")  # convert df automatically
    else:
        data = data

    s = sharepy.connect(
        os.getenv("SERVER"), os.getenv("SP_USERNAME"), os.getenv("PASSWORD")
    )  # reconnect new instance in case of concurrency
    site = "https://emckclac-my.sharepoint.com/personal/k1758409_kcl_ac_uk/"
    library = "Documents"

    r = s.post(
        f"{site}/_api/web/lists/getbyTitle('{library}')/Files/add(overwrite=true, name='{file_path}')",
        data=data.encode(encoding),
    )

    if r.status_code in (200, 204):
        print(f"Successfully uploaded file: {file_path}")
    else:
        print("error")
        print(json.dumps(r.json(), indent=4, sort_keys=True))


