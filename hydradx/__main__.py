import copy
import json
import sys
from pprint import pprint
import random

import pytest
from hypothesis import given, strategies as st, assume, settings, Verbosity, Phase, reproduce_failure

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from mpmath import mp, mpf
import highspy
import numpy as np

from hydradx.model.amm.omnix import validate_and_execute_solution
from hydradx.model.amm.omnix_solver_simple import find_solution, \
    _find_solution_unrounded, add_buy_deltas, round_solution, find_solution_outer_approx, _solve_inclusion_problem, \
    ICEProblem

def asset_symbol_decimals(asset_id):
    return {
        0: ('HDX', 12),
        5: ('DOT', 10),
        8: ('PHA', 12),
        9: ('ASTR', 18),
        12: ('ZTG', 10),
        13: ('CFG', 18),
        14: ('BNC', 12),
        15: ('vDOT', 10),
        16: ('GLMR', 18),
        17: ('INTR', 10),
        20: ('WETH', 18),
        27: ('CRU', 12),
        28: ('KILT', 15),
        31: ('RING', 18),
        33: ('ASTR', 18),
        100: ('4-Pool', 18),
        101: ('2-Pool', 18),
        102: ('2-Pool-1', 18),
    }[asset_id]

def amount_to_float(amount, decimals):
    return amount / 10**decimals

def setup_agent(holding):
    tkn = holding[0]
    amount = holding[1]
    return Agent(holdings={tkn: amount})

def convert_to_intents(data):
    intents = []
    for intent in data:
        (tkn_sell, tkn_sell_decimals)= asset_symbol_decimals(intent['asset_in'])
        (tkn_buy, tkn_buy_decimals)= asset_symbol_decimals(intent['asset_out'])
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

    if len(sys.argv) != 2:
        print("Usage: python -m hydradx <filename>")
        sys.exit(1)
    filename = sys.argv[1]
    # load from json data file into init_intents
    data = json.load(open(filename))

    intents = convert_to_intents(data)

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool-1': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool-1': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    intents = copy.deepcopy(intents)
    x = find_solution_outer_approx(initial_state, intents)
    print(f"Solution: {x[0]}")
