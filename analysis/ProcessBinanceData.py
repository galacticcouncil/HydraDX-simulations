from csv import DictReader, writer
from pprint import pprint
import os
import glob
from dataclasses import dataclass

prefix = 'DOTBUSD'
tick = '1s'

path = 'input/'
extension = 'csv'
os.chdir(path)
result = glob.glob(prefix + '-' + tick + '-' + '*.{}'.format(extension))
os.chdir('..')

# input_filename = 'input/DOTBUSD-1s-2022-11-22.csv'
input_filename = result[0]
# input_filename = 'input/test.csv'
output_filename = 'output/' + prefix + "-" + tick + "-output.csv"

@dataclass
class PriceTick:
    timestamp: int
    price: float

def import_price_data(input_path: str) -> list[PriceTick]:
    price_data = []
    for input_filename in result:
        with open('input/' + input_filename, newline='') as input_file:
            fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                          'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            reader = DictReader(input_file, fieldnames=fieldnames)
            # reader = DictReader(input_file)
            for row in reader:
                price_data.append(PriceTick(int(row["timestamp"]), float(row["open"])))

    price_data.sort(key=lambda x: x.timestamp)
    return price_data



def write_price_data(price_data: list[PriceTick]) -> None:
    with open(output_filename, 'w', newline='') as output_file:
        fieldnames = ['timestamp', 'price']
        csvwriter = writer(output_file)
        csvwriter.writerow(fieldnames)
        for row in price_data:
            csvwriter.writerow([row.timestamp, row.price])

price_data = import_price_data(path)
write_price_data(price_data)
