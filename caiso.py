import httpx
import pandas as pd
import zipfile
import io

def format_time(year: int, month: int, day: int, hour: int) -> str:
    # Always use -0000 for UTC per CAISO API docs
    return f"{year:04d}{month:02d}{day:02d}T{hour:02d}:00-0000"

CAISO_BASE_URL='https://oasis.caiso.com/oasisapi/SingleZip'

response = httpx.get(
    CAISO_BASE_URL,
    params = {
        'queryname': 'PRC_LMP',
        'version': '12',
        'startdatetime': format_time(2025, 6, 20, 8),
        'enddatetime': format_time(2025, 6, 20, 12),
        'market_run_id': 'DAM',
        'resultformat': 6,
        # 'node': 'LAPLMG1_7_B2'
        'grp_type': 'ALL_APNODES'
    },
    timeout=120
)

response.raise_for_status()


with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
    # Dynamically get the first file in the archive (in case name changes)
    csv_name = zip_file.namelist()[0]
    with zip_file.open(csv_name) as csv_file:
        df = pd.read_csv(csv_file)
        print(df.shape)
        print(df.columns)
        print(df[['INTERVALSTARTTIME_GMT', 'INTERVALENDTIME_GMT', 'LMP_TYPE', 'NODE_ID_XML', 'MARKET_RUN_ID', 'MW']].head())