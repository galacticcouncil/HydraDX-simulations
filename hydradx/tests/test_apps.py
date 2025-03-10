from hydradx.apps.gigadot_modeling.utils import simulate_route
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.agents import Agent

def test_liquidity():
    import hydradx.apps.gigadot_modeling.liquidity  # throws error if liquidity.py has error

def test_simulate_route():
    assets = {
        'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
        'USDT': {'liquidity': 1000000, 'LRNA': 1000000},
        'DOT': {'liquidity': 1000000, 'LRNA': 1000000}
    }
    omnipool = OmnipoolState(assets)
    ss_assets = {'DOT': 1000000, 'vDOT': 800000, 'aDOT': 1000000}
    peg = ss_assets['DOT'] / ss_assets['vDOT']
    stableswap = StableSwapPoolState(ss_assets, 100, peg=[peg, 1])
    agent = Agent(enforce_holdings=False)

    # within Omnipool

    buy_amt = 1
    routes = [
        [{'tkn_sell': 'HDX', 'tkn_buy': 'USDT', 'pool': "omnipool"}],  # within Omnipool
        [{'tkn_sell': 'DOT', 'tkn_buy': 'vDOT', 'pool': "gigaDOT"}],  # within StableSwap
        [{'tkn_sell': 'DOT', 'tkn_buy': 'aDOT', 'pool': "money market"}],  # within Money Market
        [
            {'tkn_sell': 'DOT', 'tkn_buy': 'aDOT', 'pool': "money market"},
            {'tkn_sell': 'aDOT', 'tkn_buy': 'vDOT', 'pool': "gigaDOT"}
        ],
        [
            {'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'pool': "omnipool"},
            {'tkn_sell': 'DOT', 'tkn_buy': 'aDOT', 'pool': "money market"},
            {'tkn_sell': 'aDOT', 'tkn_buy': 'vDOT', 'pool': "gigaDOT"}
        ]
    ]

    expected_sells = [1, 10/8, 1, 10/8, 10/8]

    for i, route in enumerate(routes):
        new_omnipool, new_stableswap, new_agent = simulate_route(omnipool, stableswap, agent, buy_amt, route)
        does_route_use_moneymarket = False
        for step in route:
            if step['pool'] == "money market":
                does_route_use_moneymarket = True
                break
        # assert abs(new_agent.get_holdings('HDX') + buy_amt) < 1e-4
        tkn_sell = route[0]['tkn_sell']
        tkn_buy = route[-1]['tkn_buy']
        assert new_agent.get_holdings(tkn_buy) == buy_amt
        for tkn in list(assets.keys()) + list(ss_assets.keys()):
            if tkn in ['DOT', 'aDOT'] and does_route_use_moneymarket:  # combine DOT and aDOT
                tkn_init = agent.get_holdings('DOT') + agent.get_holdings('aDOT')
                tkn_after = new_agent.get_holdings('DOT') + new_agent.get_holdings('aDOT')
                for tkn in ['DOT', 'aDOT']:
                    if tkn in omnipool.liquidity:
                        tkn_init += omnipool.liquidity[tkn]
                    if tkn in stableswap.liquidity:
                        tkn_init += stableswap.liquidity[tkn]
                    if tkn in new_omnipool.liquidity:
                        tkn_after += new_omnipool.liquidity[tkn]
                    if tkn in new_stableswap.liquidity:
                        tkn_after += new_stableswap.liquidity[tkn]
            else:
                tkn_init = agent.get_holdings(tkn)
                if tkn in omnipool.liquidity:
                    tkn_init += omnipool.liquidity[tkn]
                if tkn in stableswap.liquidity:
                    tkn_init += stableswap.liquidity[tkn]
                tkn_after = new_agent.get_holdings(tkn)
                if tkn in new_omnipool.liquidity:
                    tkn_after += new_omnipool.liquidity[tkn]
                if tkn in new_stableswap.liquidity:
                    tkn_after += new_stableswap.liquidity[tkn]
            assert tkn_init == tkn_after

        sell_amt = -1 * new_agent.get_holdings(tkn_sell)
        assert abs(sell_amt - expected_sells[i]) < 1e-5
