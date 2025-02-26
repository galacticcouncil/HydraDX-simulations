import copy
import json
import sys
from pprint import pprint
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from mpmath import mpf
from hydradx.model.amm.omnix_solver_simple import  find_solution_outer_approx

def amount_to_float(amount, decimals):
    return amount / 10**decimals

def setup_agent(holding):
    tkn = holding[0]
    amount = holding[1]
    return Agent(holdings={tkn: amount})

class AMMStore:
    def __init__(self, data):
        self._data = data
        self._omnipool_assets = None
        self._assets = None
        self.__parse()

    def tkn_symbol(self, asset_id: int):
        if asset_id == 0:
            return "HDX"
        else:
            return "TKN" + str(asset_id)

    def tkn_decimals(self, tkn: str):
        return self._assets[tkn]["decimals"]


    def __parse(self):
        assets = {}
        omnipool_assets= []
        for entry in self._data:
            if "Omnipool" in entry:
                asset = entry["Omnipool"]
                symbol = self.tkn_symbol(asset["asset_id"])
                asset["asset_id"] = symbol
                omnipool_assets.append(asset)
                assets[asset["asset_id"]] = {"decimals": asset["decimals"]}
            elif "Stableswap" in entry:
                print(f"Stableswap")
                raise Exception("not yet available")
            else:
                raise Exception("unsupported pool")
        self._omnipool_assets = omnipool_assets
        self._assets = assets

    def omnipool_state(self):
        liquidity = {}
        lrna = {}
        fee = {}
        lrna_fee = {}

        def convert_fee(fee):
            return fee[0] / fee[1]
        for asset in self._omnipool_assets:
            liquid = amount_to_float(asset["reserve"], asset["decimals"])
            hub_liquid = amount_to_float(asset["hub_reserve"], 12)
            liquidity[asset["asset_id"]] = liquid
            lrna[asset["asset_id"]] = hub_liquid
            fee[asset["asset_id"]] = convert_fee(asset["fee"])
            lrna_fee[asset["asset_id"]] = convert_fee(asset["hub_fee"])

        initial_state = OmnipoolState(
            tokens={
                tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
            },
            asset_fee=mpf(0.0025),
            lrna_fee=mpf(0.0005)
        )
        initial_state.last_fee = fee
        initial_state.last_lrna_fee = lrna_fee

        return initial_state


def serialize_intents(data, amm_data: AMMStore):
    intents = []
    for intent in data:
        tkn_sell = amm_data.tkn_symbol(intent["asset_in"])
        tkn_buy = amm_data.tkn_symbol(intent["asset_out"])
        tkn_sell_decimals = amm_data.tkn_decimals(tkn_sell)
        tkn_buy_decimals = amm_data.tkn_decimals(tkn_buy)
        intents.append(
            {
                'agent': setup_agent((tkn_sell, amount_to_float(intent['amount_in'], tkn_sell_decimals))),
                'sell_quantity': amount_to_float(intent['amount_in'], tkn_sell_decimals),
                'buy_quantity': amount_to_float(intent['amount_out'], tkn_buy_decimals),
                'tkn_sell': tkn_sell,
                'tkn_buy': tkn_buy,
                'partial': intent['partial']
            }
        )
    return intents

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m hydradx <intents> <data>")
        sys.exit(1)
    intent_input= sys.argv[1]
    data_input = sys.argv[2]
    amm_data = json.load(open(data_input))
    amm_store = AMMStore(amm_data)
    amm_store.omnipool_state()

    intent_data = json.load(open(intent_input))
    intents = serialize_intents(intent_data, amm_store)
    initial_state = amm_store.omnipool_state()
    intents = copy.deepcopy(intents)
    x = find_solution_outer_approx(initial_state, intents)
    pprint(f"Solution: {x}")
