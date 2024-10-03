from pprint import pprint

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from mpmath import mp, mpf

from hydradx.model.amm.omnix import validate_and_execute_solution
from hydradx.model.amm.omnix_solver_simple import find_solution


def test_convex():

    agents = [
        Agent(holdings={'DOT': 100}),
        Agent(holdings={'USDT': 1500}),
        Agent(holdings={'USDT': 400}),
        Agent(holdings={'HDX': 100}),
    ]

    intents = [
        {'sell_quantity': 100, 'buy_quantity': 700, 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0]},  # selling DOT for $7
        {'sell_quantity': 1500, 'buy_quantity': 100000, 'tkn_sell': 'USDT', 'tkn_buy': 'HDX', 'agent': agents[1]},  # buying HDX for $0.015
        {'sell_quantity': 400, 'buy_quantity': 50, 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[2]},  # buying DOT for $8
        {'sell_quantity': 100, 'buy_quantity': 100, 'tkn_sell': 'HDX', 'tkn_buy': 'USDT', 'agent': agents[3]},  # selling HDX for $1
    ]

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.003) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0) for tkn in lrna}

    intent_deltas = find_solution(initial_state, intents)

    assert validate_and_execute_solution(initial_state, intents, intent_deltas)

    pprint(intent_deltas)