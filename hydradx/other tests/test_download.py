import requests, os
from zipfile import ZipFile

# find the data folder
# while not os.path.exists("./data"):
#     cwd = os.getcwd()
#     os.chdir("..")
#     if cwd == os.getcwd():
#         raise FileNotFoundError("Could not find the data folder")


cwd = os.getcwd()
os.chdir("../notebooks/Omnipool/")
cwd = os.getcwd()

tkn = 'DOT'
stablecoin = 'USDT'
date = '2023-05'
file = f"{tkn}{stablecoin}-1s-{date}"
url = f"https://data.binance.vision/?prefix=data/spot/monthly/klines/{tkn}{stablecoin}/1s/{file}.zip"
response = requests.get(url)
full_path = f"./data/{file}.zip"
full_path = cwd + f"/data/{file}.zip"
test = response.headers.get('Content-Type')
is_file = os.path.isfile(full_path)
with open(full_path, 'wb') as f:
    f.write(response.content)
is_file = os.path.isfile(full_path)
with ZipFile(f"./data/{file}.zip", 'r') as zipObj:
    zipObj.extractall(path='./data')