import json
from datetime import timedelta

from hypothesis import given, strategies as st, settings, reproduce_failure, Verbosity, Phase

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.arbitrage_agent_new import (
    calculate_profit, calculate_arb_amount_bid, calculate_arb_amount_ask, combine_swaps
)
from hydradx.model.amm.arbitrage_agent_new import get_arb_swaps, execute_arb
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.centralized_market import OrderBook, CentralizedMarket
from hydradx.model.processing import save_market_config, load_market_config


def test_calculate_profit():
    init_agent = Agent(holdings={'USDT': 1000000, 'DOT': 2000000, 'HDX': 3000000}, unique_id='bot')
    agent = init_agent.copy()
    profits = {'USDT': 100, 'DOT': 200, 'HDX': 0}
    agent.holdings['USDT'] += profits['USDT']
    agent.holdings['DOT'] += profits['DOT']
    agent.holdings['HDX'] += profits['HDX']
    calculated_profit = calculate_profit(init_agent, agent)
    for tkn in init_agent.holdings:
        assert calculated_profit[tkn] == profits[tkn]


def test_calculate_profit_with_mapping():
    init_agent = Agent(holdings={'USDT': 1000000, 'DOT': 2000000, 'HDX': 3000000, 'DAI': 4000000}, unique_id='bot')
    agent = init_agent.copy()
    profits = {'USDT': 100, 'DOT': 200, 'HDX': 0, 'DAI': 100}
    agent.holdings['USDT'] += profits['USDT']
    agent.holdings['DAI'] += profits['DAI']
    agent.holdings['DOT'] += profits['DOT']
    agent.holdings['HDX'] += profits['HDX']
    mapping = {'DAI': 'USD', 'USDT': 'USD'}
    calculated_profit = calculate_profit(init_agent, agent, mapping)

    assert calculated_profit['USD'] == 200
    assert calculated_profit['DOT'] == 200
    assert calculated_profit['HDX'] == 0


@given(
    dotusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    dot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=8, max_size=8),
    hdxdot_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxdot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=4, max_size=4),
    hdxusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxusd_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=8, max_size=8),
)
def test_get_arb_swaps(
        dotusd_price_mult: float,
        dot_amts: list,
        hdxdot_price_mult: float,
        hdxdot_amts: list,
        hdxusd_price_mult: float,
        hdxusd_amts: list
):

    tokens = {
        'USDT': {
            'liquidity': 2062772,
            'LRNA': 2062772
        },
        'DOT': {
            'liquidity': 350000,
            'LRNA': 1456248
        },
        'HDX': {
            'liquidity': 108000000,
            'LRNA': 494896
        }
    }

    lrna_fee = 0.0005
    asset_fee = 0.0025
    cex_fee = 0.0016

    op_state = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )

    dotusd_spot = op_state.price(op_state, 'DOT', 'USDT')
    dotusd_spot_adj = dotusd_spot * dotusd_price_mult

    dot_usdt_order_book = {
        'bids': [{'price': dotusd_spot_adj * 0.999, 'amount': dot_amts[0]},
                 {'price': dotusd_spot_adj * 0.99, 'amount': dot_amts[1]},
                 {'price': dotusd_spot_adj * 0.9, 'amount': dot_amts[2]},
                 {'price': dotusd_spot_adj * 0.8, 'amount': dot_amts[3]}],
        'asks': [{'price': dotusd_spot_adj * 1.001, 'amount': dot_amts[4]},
                 {'price': dotusd_spot_adj * 1.01, 'amount': dot_amts[5]},
                 {'price': dotusd_spot_adj * 1.1, 'amount': dot_amts[6]},
                 {'price': dotusd_spot_adj * 1.2, 'amount': dot_amts[7]}]
    }

    dot_usdt_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in dot_usdt_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in dot_usdt_order_book['asks']])

    hdxusd_spot = op_state.price(op_state, 'HDX', 'USDT')
    hdxusd_spot_adj = hdxusd_spot * hdxusd_price_mult

    hdx_usdt_order_book = {
        'bids': [{'price': hdxusd_spot_adj * 0.999, 'amount': hdxusd_amts[0]},
                 {'price': hdxusd_spot_adj * 0.99, 'amount': hdxusd_amts[1]},
                 {'price': hdxusd_spot_adj * 0.9, 'amount': hdxusd_amts[2]},
                 {'price': hdxusd_spot_adj * 0.8, 'amount': hdxusd_amts[3]}],
        'asks': [{'price': hdxusd_spot_adj * 1.001, 'amount': hdxusd_amts[4]},
                 {'price': hdxusd_spot_adj * 1.01, 'amount': hdxusd_amts[5]},
                 {'price': hdxusd_spot_adj * 1.1, 'amount': hdxusd_amts[6]},
                 {'price': hdxusd_spot_adj * 1.2, 'amount': hdxusd_amts[7]}]
    }

    hdx_usdt_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in hdx_usdt_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in hdx_usdt_order_book['asks']])

    hdxdot_spot = op_state.price(op_state, 'HDX', 'DOT')
    hdxdot_spot_adj = hdxdot_spot * hdxdot_price_mult

    hdx_dot_order_book = {
        'bids': [{'price': hdxdot_spot_adj * 0.99, 'amount': hdxdot_amts[0]},
                 {'price': hdxdot_spot_adj * 0.9, 'amount': hdxdot_amts[1]}],
        'asks': [{'price': hdxdot_spot_adj * 1.01, 'amount': hdxdot_amts[2]},
                 {'price': hdxdot_spot_adj * 1.1, 'amount': hdxdot_amts[3]}]
    }

    hdx_dot_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in hdx_dot_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in hdx_dot_order_book['asks']])

    order_book = {
        ('DOT', 'USDT'): dot_usdt_order_book_obj,
        ('HDX', 'USDT'): hdx_usdt_order_book_obj,
        ('HDX','DOT'): hdx_dot_order_book_obj
    }

    cex = CentralizedMarket(
        order_book=order_book,
        asset_list=['USDT', 'DOT', 'HDX'],
        trade_fee=cex_fee
    )

    # get_arb_swaps(op_state, cex, order_book_map, buffer=0.0, max_trades={}, iters=20)

    order_book_map = {k: k for k in order_book}

    arb_swaps = get_arb_swaps(op_state, cex, order_book_map)
    initial_agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
    agent = initial_agent.copy()

    execute_arb(op_state, cex, agent, arb_swaps)

    profit = calculate_profit(initial_agent, agent)
    for tkn in profit:
        if profit[tkn] / initial_agent.holdings[tkn] < -1e-10:
            raise


def test_save():
    save_market_config()


def test_load():
    omnipool, cex, order_book_map = load_market_config()
    arb_swaps = get_arb_swaps(omnipool, cex, order_book_map)
    initial_agent = Agent(holdings={tkn: 10000000000 for tkn in omnipool.asset_list + cex.asset_list}, unique_id='bot')
    agent = initial_agent.copy()

    execute_arb(omnipool, cex, agent, arb_swaps)

    asset_map = {}
    for tkn_pair1, tkn_pair2 in order_book_map.items():
        if tkn_pair1[0] != tkn_pair2[0]:
            asset_map[tkn_pair1[0]] = tkn_pair2[0]
        if tkn_pair1[1] != tkn_pair2[1]:
            asset_map[tkn_pair1[1]] = tkn_pair2[1]

    profit = calculate_profit(initial_agent, agent, asset_map)
    for tkn in profit:
        if profit[tkn] / initial_agent.holdings[tkn] < -1e-10:
            raise AssertionError('Loss detected.')


def test_combine_step():
    test_save()
    dex, cex, order_book_map = load_market_config()
    arb_swaps = get_arb_swaps(dex, cex, order_book_map)
    asset_map = {}
    for tkn_pair1, tkn_pair2 in order_book_map.items():
        if tkn_pair1[0] != tkn_pair2[0]:
            asset_map[tkn_pair1[0]] = tkn_pair2[0]
        if tkn_pair1[1] != tkn_pair2[1]:
            asset_map[tkn_pair1[1]] = tkn_pair2[1]
    asset_map.update({
        'WETH': 'ETH',
        'XETH': 'ETH',
        'XXBT': 'BTC',
        'WBTC': 'BTC',
        'ZUSD': 'USD',
        'USDT': 'USD',
        'USDC': 'USD',
        'DAI': 'USD',
        'USDT001': 'USD',
        'DAI001': 'USD',
        'WETH001': 'ETH',
        'WBTC001': 'BTC',
        'iBTC': 'BTC',
        'XBT': 'BTC'
    })
    initial_agent = Agent(
        holdings={
            tkn: 10000000000
            for tkn in dex.asset_list + cex.asset_list + list(asset_map.values())
        }
    )
    # initial_holdings_total = cex.value_assets(initial_agent.holdings, asset_map)

    test_dex, test_cex, test_agent = dex.copy(), cex.copy(), initial_agent.copy()
    execute_arb(test_dex, test_cex, test_agent, arb_swaps)
    profit = calculate_profit(initial_agent, test_agent, asset_map)
    profit_total = test_cex.value_assets(profit, asset_map)
    print('profit: ', profit_total)

    combine_dex, combine_cex, combine_agent = dex.copy(), cex.copy(), initial_agent.copy()
    combined_swaps = combine_swaps(combine_dex, combine_cex, combine_agent, arb_swaps, asset_map)
    execute_arb(combine_dex, combine_cex, combine_agent, combined_swaps)
    combined_profit = calculate_profit(initial_agent, combine_agent, asset_map)
    combined_profit_total = combine_cex.value_assets(combined_profit, asset_map)

    iter_dex, iter_cex, iter_agent = dex.copy(), cex.copy(), initial_agent.copy()
    arb_swaps = get_arb_swaps(iter_dex, iter_cex, order_book_map)
    itered_swaps = combine_swaps(iter_dex, iter_cex, iter_agent, arb_swaps, asset_map)
    execute_arb(iter_dex, iter_cex, iter_agent, itered_swaps)
    iter_profit = calculate_profit(initial_agent, iter_agent, asset_map)
    iter_profit_total = iter_cex.value_assets(iter_profit, asset_map)

    for tkn in profit:
        if profit[tkn] / initial_agent.holdings[tkn] < -1e-10:
            raise AssertionError('Loss detected.')

    for tkn in combined_profit:
        if combined_profit[tkn] < -1e-10:
            raise AssertionError('Loss detected.')

    if profit_total > combined_profit_total:
        raise AssertionError('Loss detected.')
    else:
        print(f"extra profit obtained: {combined_profit_total - profit_total}")
        if iter_profit_total > combined_profit_total:
            print(f'Second iteration also gained {iter_profit_total - combined_profit_total}.')
        else:
            print('Iteration did not improve profit.')
