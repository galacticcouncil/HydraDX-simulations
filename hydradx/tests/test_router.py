import copy
import pytest
from hypothesis import given, strategies as st, settings
from mpmath import mpf, mp
from datetime import timedelta

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.omnipool_router import OmnipoolRouter
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.tests.strategies_omnipool import fee_strategy

mp.dps = 50

asset_quantity_strategy = st.floats(min_value=100000, max_value=10000000)


@given(st.lists(asset_quantity_strategy, min_size=11, max_size=11))
def test_price_route(assets: list[float]):
    tokens = {
        "HDX": {'liquidity': assets[0], 'LRNA': assets[1]},
        "USDT": {'liquidity': assets[2], 'LRNA': assets[3]},
        "DOT": {'liquidity': assets[4], 'LRNA': assets[5]},
        "stablepool": {'liquidity': assets[6], 'LRNA': assets[7]},
    }
    omnipool = OmnipoolState(
        tokens=tokens,
        preferred_stablecoin="USDT",
        asset_fee=0.0025,
        lrna_fee=0.0005,
    )
    sp_tokens = {
        "stable1": assets[8],
        "stable2": assets[9],
        "stable3": assets[10],
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=0.0004,
        unique_id="stablepool"
    )
    exchanges = {
        "omnipool": omnipool,
        "stablepool": stablepool
    }
    router = OmnipoolRouter(exchanges)

    op_price_hdx = omnipool.price("HDX", "USDT")
    op_price_lp = omnipool.price("stablepool", "USDT")
    sp_price = stablepool.price("stable1", "stable3")
    share_price = stablepool.share_price("stable1")

    router_price_hdx = router.price_route("HDX", "USDT", "omnipool", "omnipool")
    if router_price_hdx != op_price_hdx:
        raise ValueError(f"router price {router_price_hdx} != omnipool price {op_price_hdx}")

    router_price_lp = router.price_route("stablepool", "USDT", "omnipool", "omnipool")
    if router_price_lp != op_price_lp:
        raise ValueError(f"router price {router_price_lp} != omnipool price {op_price_lp}")

    router_price_inside_stablepool = router.price_route("stable1", "stable3", "stablepool", "stablepool")
    if router_price_inside_stablepool != pytest.approx(sp_price, rel=1e-15):
        raise ValueError(f"router price {router_price_inside_stablepool} != stablepool price {sp_price}")

    router_price_outside_stablepool = router.price_route("stable1", "USDT", "stablepool", "omnipool")
    if router_price_outside_stablepool != op_price_lp / share_price:
        raise ValueError(
            f"router price {router_price_outside_stablepool} != stablepool price {op_price_lp / share_price}")

    router_price_lp_share = router.price_route("stablepool", "stable1", "omnipool", "stablepool")
    if router_price_lp_share != share_price:
        raise ValueError(f"router price {router_price_lp_share} != stablepool price {share_price}")

    router_price_lp_share_reverse = router.price_route("stable1", "stablepool", "stablepool", "omnipool")
    if router_price_lp_share_reverse != pytest.approx(1 / share_price, rel=1e-12):
        raise ValueError(f"router price {router_price_lp_share_reverse} != stablepool price {1 / share_price}")


def test_price_route_example():
    # This test exists to make sure we are not making same mistake in test_price and in the price function
    tokens = {
        "HDX": {'liquidity': 10000000, 'LRNA': 1000000},
        "USDT": {'liquidity': 1000000, 'LRNA': 1000000},
        "DOT": {'liquidity': 100000, 'LRNA': 1000000},
        "stablepool": {'liquidity': 1000000, 'LRNA': 1000000},
    }
    omnipool = OmnipoolState(
        tokens=tokens,
        preferred_stablecoin="USDT",
        asset_fee=0.0025,
        lrna_fee=0.0005,
    )
    sp_tokens = {
        "stable1": 320000,
        "stable2": 330000,
        "stable3": 350000,
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=0.0004,
        unique_id="stablepool"
    )
    exchanges = {
        "omnipool": omnipool,
        "stablepool": stablepool
    }
    router = OmnipoolRouter(exchanges)

    router_price_outside_stablepool = router.price_route("DOT", "stable1", "omnipool", "stablepool")
    # if we have the math right, the DOT price denominated in stable1 should be in the ballpark of 10.
    if router_price_outside_stablepool != pytest.approx(10, rel=1e-3):
        raise ValueError(f"router price {router_price_outside_stablepool} is not correct")


@given(
    st.lists(asset_quantity_strategy, min_size=14, max_size=14),
    st.floats(min_value=0.0001, max_value=0.1),
)
def test_swap_omnipool(assets: list[float], trade_size_mult: float):
    tokens = {
        "HDX": {'liquidity': assets[0], 'LRNA': assets[1]},
        "USDT": {'liquidity': assets[2], 'LRNA': assets[3]},
        "DOT": {'liquidity': assets[4], 'LRNA': assets[5]},
        "stablepool": {'liquidity': assets[6], 'LRNA': assets[7]},
    }
    omnipool = OmnipoolState(tokens, preferred_stablecoin="USDT", asset_fee=0.0025, lrna_fee=0.0005)
    sp_tokens = {"stable1": assets[8], "stable2": assets[9], "stable3": assets[10]}
    stablepool = StableSwapPoolState(sp_tokens, 1000, trade_fee=0.0004, unique_id="stablepool")
    sp_tokens2 = {"stable2": assets[11], "stable3": assets[12], "USDT": assets[13]}
    stablepool2 = StableSwapPoolState(sp_tokens2, 1000, trade_fee=0.0004, unique_id="stablepool2")
    exchanges = {"omnipool": omnipool, "stablepool": stablepool, "stablepool2": stablepool2}
    router = OmnipoolRouter(exchanges)
    omnipool2 = omnipool.copy()
    agent1 = Agent(holdings={"DOT": 1000000, "USDT": 1000000})
    agent2 = Agent(holdings={"DOT": 1000000, "USDT": 1000000})
    trade_size = trade_size_mult * min(assets[4], assets[2])

    # test buy
    router.swap_route(agent1, "DOT", "USDT", buy_quantity=10000, buy_pool_id="omnipool", sell_pool_id="omnipool")
    omnipool2.swap(agent2, "DOT", "USDT", buy_quantity=10000)

    for token in omnipool.asset_list:
        if omnipool.liquidity[token] != omnipool2.liquidity[token]:
            raise ValueError(f"omnipool liquidity {omnipool.liquidity[token]} != {omnipool2.liquidity[token]}")
        if omnipool.lrna[token] != omnipool2.lrna[token]:
            raise ValueError(f"omnipool lrna {omnipool.lrna[token]} != {omnipool2.lrna[token]}")

    # test sell
    router.swap_route(agent1, "DOT", "USDT", sell_quantity=trade_size, buy_pool_id="omnipool", sell_pool_id="omnipool")
    omnipool2.swap(agent2, "DOT", "USDT", sell_quantity=trade_size)

    for token in omnipool.asset_list:
        if omnipool.liquidity[token] != omnipool2.liquidity[token]:
            raise ValueError(f"omnipool liquidity {omnipool.liquidity[token]} != {omnipool2.liquidity[token]}")
        if omnipool.lrna[token] != omnipool2.lrna[token]:
            raise ValueError(f"omnipool lrna {omnipool.lrna[token]} != {omnipool2.lrna[token]}")


@given(
    st.lists(asset_quantity_strategy, min_size=16, max_size=16),
    st.floats(min_value=0.0001, max_value=0.1),
)
def test_swap_stableswap(assets: list[float], trade_size_mult: float):
    tokens = {
        "HDX": {'liquidity': assets[0], 'LRNA': assets[1]},
        "USDT": {'liquidity': assets[2], 'LRNA': assets[3]},
        "DOT": {'liquidity': assets[4], 'LRNA': assets[5]},
        "stablepool": {'liquidity': assets[6], 'LRNA': assets[7]},
        "stablepool2": {'liquidity': assets[14], 'LRNA': assets[15]},
    }
    omnipool = OmnipoolState(tokens, preferred_stablecoin="USDT", asset_fee=0.0025, lrna_fee=0.0005)
    sp_tokens = {"stable1": assets[8], "stable2": assets[9], "stable3": assets[10]}
    stablepool = StableSwapPoolState(sp_tokens, 1000, trade_fee=0.0004, unique_id="stablepool")
    sp_tokens2 = {"stable2": assets[11], "stable3": assets[12], "USDT": assets[13]}
    stablepool2 = StableSwapPoolState(sp_tokens2, 1000, trade_fee=0.0004, unique_id="stablepool2")
    exchanges = {"omnipool": omnipool, "stablepool": stablepool, "stablepool2": stablepool2}
    router = OmnipoolRouter(exchanges)
    omnipool2 = omnipool.copy()
    stablepool2_copy = stablepool2.copy()
    agent1 = Agent(holdings={"DOT": 10000000, "USDT": 10000000})
    agent2 = Agent(holdings={"DOT": 10000000, "USDT": 10000000})
    trade_size = trade_size_mult * min(assets[4], assets[2])

    buy_quantity = 1000

    # test buy
    router.swap_route(agent1, tkn_sell="DOT", tkn_buy="USDT", buy_quantity=buy_quantity, buy_pool_id="stablepool2",
                      sell_pool_id="omnipool")
    # calculate how many shares to buy
    delta_shares = stablepool2_copy.calculate_withdrawal_shares("USDT", buy_quantity)
    # buy shares
    omnipool2.swap(agent2, tkn_buy="stablepool2", tkn_sell="DOT", buy_quantity=delta_shares)
    # withdraw USDT
    stablepool2_copy.withdraw_asset(agent2, buy_quantity, "USDT")

    for token in omnipool.asset_list:
        if omnipool.liquidity[token] != omnipool2.liquidity[token]:
            raise ValueError(f"omnipool liquidity {omnipool.liquidity[token]} != {omnipool2.liquidity[token]}")
        if omnipool.lrna[token] != omnipool2.lrna[token]:
            raise ValueError(f"omnipool lrna {omnipool.lrna[token]} != {omnipool2.lrna[token]}")
    for token in stablepool2.asset_list:
        if stablepool2.liquidity[token] != stablepool2_copy.liquidity[token]:
            raise ValueError(
                f"stablepool2 liquidity {stablepool2.liquidity[token]} != {stablepool2_copy.liquidity[token]}")
    for token in agent1.holdings:
        if agent1.holdings[token] != agent2.holdings[token]:
            raise ValueError(f"agent1 holdings {agent1.holdings[token]} != {agent2.holdings[token]}")

    # test sell
    router.swap_route(agent1, tkn_sell="DOT", tkn_buy="USDT", sell_quantity=trade_size, buy_pool_id="stablepool2",
                      sell_pool_id="omnipool")
    # sell DOT for shares
    omnipool2.swap(agent2, "stablepool2", "DOT", sell_quantity=trade_size)
    # withdraw USDT
    stablepool2_copy.remove_liquidity(agent2, agent2.holdings["stablepool2"], "USDT")

    for token in omnipool.asset_list:
        if omnipool.liquidity[token] != omnipool2.liquidity[token]:
            raise ValueError(f"omnipool liquidity {omnipool.liquidity[token]} != {omnipool2.liquidity[token]}")
        if omnipool.lrna[token] != omnipool2.lrna[token]:
            raise ValueError(f"omnipool lrna {omnipool.lrna[token]} != {omnipool2.lrna[token]}")
    for token in stablepool2.asset_list:
        if stablepool2.liquidity[token] != stablepool2_copy.liquidity[token]:
            raise ValueError(
                f"stablepool2 liquidity {stablepool2.liquidity[token]} != {stablepool2_copy.liquidity[token]}")
    for token in agent1.holdings:
        if agent1.holdings[token] != agent2.holdings[token]:
            raise ValueError(f"agent1 holdings {agent1.holdings[token]} != {agent2.holdings[token]}")


@given(st.lists(asset_quantity_strategy, min_size=16, max_size=16))
def test_swap_stableswap2(assets: list[float]):
    tokens = {
        "HDX": {'liquidity': assets[0], 'LRNA': assets[1]},
        "USDT": {'liquidity': assets[2], 'LRNA': assets[3]},
        "DOT": {'liquidity': assets[4], 'LRNA': assets[5]},
        "stablepool": {'liquidity': assets[6], 'LRNA': assets[7]},
        "stablepool2": {'liquidity': assets[14], 'LRNA': assets[15]},
    }

    buy_tkn = "USDT"
    sell_tkn = "stable1"

    omnipool = OmnipoolState(tokens, preferred_stablecoin="USDT", asset_fee=0.0025, lrna_fee=0.0005)
    sp_tokens = {"stable1": assets[8], "stable2": assets[9], "stable3": assets[10]}
    stablepool = StableSwapPoolState(sp_tokens, 1000, trade_fee=0.0100, unique_id="stablepool")
    sp_tokens2 = {"stable2": assets[11], "stable3": assets[12], "USDT": assets[13]}
    stablepool2 = StableSwapPoolState(sp_tokens2, 1000, trade_fee=0.0000, unique_id="stablepool2")
    exchanges = {"omnipool": omnipool, "stablepool": stablepool, "stablepool2": stablepool2}
    router = OmnipoolRouter(exchanges)
    init_holdings = {"DOT": 1000000, "USDT": 1000000, "stable1": 1000000}
    agent1 = Agent(holdings={tkn: init_holdings[tkn] for tkn in init_holdings})
    trade_size = 1000

    # test buy
    router.swap_route(agent1, buy_tkn, sell_tkn, buy_quantity=trade_size, buy_pool_id="stablepool2",
                      sell_pool_id="stablepool")
    for tkn in list(agent1.holdings.keys()) + list(init_holdings.keys()):
        if tkn == buy_tkn:
            if agent1.holdings[tkn] != pytest.approx(init_holdings[tkn] + trade_size, rel=1e-12):
                raise ValueError(f"agent1 holdings {agent1.holdings[tkn]} != {init_holdings[tkn] + trade_size}")
        elif tkn == sell_tkn:
            if agent1.holdings[tkn] >= init_holdings[tkn]:
                raise ValueError(f"agent1 holdings {agent1.holdings[tkn]} >= {init_holdings[tkn]}")
        else:
            if tkn not in init_holdings and (tkn in agent1.holdings and agent1.holdings[tkn] != 0):
                raise ValueError(f"agent1 holdings {agent1.holdings[tkn]} != 0")
            elif tkn in agent1.holdings and tkn in init_holdings and agent1.holdings[tkn] != init_holdings[tkn]:
                raise ValueError(f"agent1 holdings {agent1.holdings[tkn]} != {init_holdings[tkn]}")

    sell_init_holdings = copy.deepcopy(agent1.holdings)

    # test sell
    router.swap_route(agent1, tkn_sell="stable1", tkn_buy="USDT", sell_quantity=trade_size, buy_pool_id="stablepool2",
                      sell_pool_id="stablepool")
    for tkn in list(agent1.holdings.keys()) + list(sell_init_holdings.keys()):
        if tkn == buy_tkn:
            if agent1.holdings[tkn] <= sell_init_holdings[tkn]:
                raise ValueError(f"agent1 holdings {agent1.holdings[tkn]} <= {sell_init_holdings[tkn]}")
        elif tkn == sell_tkn:
            if agent1.holdings[tkn] + trade_size != sell_init_holdings[tkn]:
                raise ValueError(f"agent1 holdings {agent1.holdings[tkn] + trade_size} != {init_holdings[tkn]}")
        else:
            if tkn not in sell_init_holdings and (tkn in agent1.holdings and agent1.holdings[tkn] != 0):
                raise ValueError(f"agent1 holdings {agent1.holdings[tkn]} != 0")
            elif tkn in agent1.holdings and tkn in sell_init_holdings and agent1.holdings[tkn] != sell_init_holdings[
                tkn]:
                raise ValueError(f"agent1 holdings {agent1.holdings[tkn]} != {sell_init_holdings[tkn]}")


def test_swap():
    sp_tokens = {"stable1": 400000, "stable3": 400000}
    stablepool = StableSwapPoolState(sp_tokens, 1000, trade_fee=0.0000, unique_id="stablepool")
    sp_tokens2 = {"stable2": 300000, "stable3": 300000, "USDT": 400000}  # USDT cheaper in stablepool2
    stablepool2 = StableSwapPoolState(sp_tokens2, 1000, trade_fee=0.0000, unique_id="stablepool2")

    tokens = {
        "HDX": {'liquidity': 10000000, 'LRNA': 1000000},
        "USDT": {'liquidity': 1000000, 'LRNA': 1000000},
        "DOT": {'liquidity': 100000, 'LRNA': 1000000},
        "stablepool": {'liquidity': stablepool.shares, 'LRNA': 1000000},
        "stablepool2": {'liquidity': stablepool2.shares, 'LRNA': 1000000},
    }

    buy_tkn = "USDT"
    sell_tkn = "stable1"

    omnipool = OmnipoolState(tokens, preferred_stablecoin="USDT", asset_fee=0.0000, lrna_fee=0.0000)
    exchanges = {"omnipool": omnipool, "stablepool": stablepool, "stablepool2": stablepool2}
    router = OmnipoolRouter(exchanges)
    trade_size = 1
    agent1 = Agent(holdings={sell_tkn: trade_size})

    best_route = router.find_best_route(buy_tkn, sell_tkn)
    if best_route != ("stablepool", "stablepool2"):
        raise ValueError(f"best route {best_route} != ('stablepool', 'stablepool2')")

    new_router, new_agent = router.simulate_swap(
        agent1,
        tkn_buy=buy_tkn,
        tkn_sell=sell_tkn,
        sell_quantity=trade_size
    )
    new_router2, new_agent2 = router.simulate_swap_route(
        agent1,
        tkn_sell=sell_tkn,
        tkn_buy=buy_tkn,
        sell_quantity=trade_size,
        buy_pool_id="stablepool2",
        sell_pool_id="stablepool"
    )
    for tkn in new_agent.holdings:
        if new_agent.holdings[tkn] != new_agent2.holdings[tkn]:
            raise ValueError(f"new_agent holdings {new_agent.holdings[tkn]} != {new_agent2.holdings[tkn]}")
    for ex_id in new_router.exchanges:
        ex = new_router.exchanges[ex_id]
        for tkn in ex.liquidity:
            if ex.liquidity[tkn] != new_router2.exchanges[ex_id].liquidity[tkn]:
                raise ValueError(f"ex liquidity {ex.liquidity[tkn]} != {new_router2.exchanges[ex_id].liquidity[tkn]}")


def test_swap2():
    sp_tokens = {"stable1": 400000, "stable3": 400000}
    stablepool = StableSwapPoolState(sp_tokens, 1000, trade_fee=0.0000, unique_id="stablepool")
    sp_tokens2 = {"stable2": 300000, "stable3": 300000, "USDT": 200000}  # USDT more expensive in stablepool2
    stablepool2 = StableSwapPoolState(sp_tokens2, 1000, trade_fee=0.0000, unique_id="stablepool2")

    tokens = {
        "HDX": {'liquidity': 10000000, 'LRNA': 1000000},
        "USDT": {'liquidity': 1000000, 'LRNA': 1000000},
        "DOT": {'liquidity': 100000, 'LRNA': 1000000},
        "stablepool": {'liquidity': stablepool.shares, 'LRNA': 1000000},
        "stablepool2": {'liquidity': stablepool2.shares, 'LRNA': 1000000},
    }

    buy_tkn = "USDT"
    sell_tkn = "stable1"

    omnipool = OmnipoolState(tokens, preferred_stablecoin="USDT", asset_fee=0.0000, lrna_fee=0.0000)
    exchanges = {"omnipool": omnipool, "stablepool": stablepool, "stablepool2": stablepool2}
    router = OmnipoolRouter(exchanges)
    trade_size = 1
    agent1 = Agent(holdings={sell_tkn: trade_size})

    best_route = router.find_best_route(buy_tkn, sell_tkn)
    if best_route != ("stablepool", "omnipool"):
        raise ValueError(f"best route {best_route} != ('stablepool', 'omnipool')")

    new_router, new_agent = router.simulate_swap(
        agent1,
        tkn_buy=buy_tkn,
        tkn_sell=sell_tkn,
        sell_quantity=trade_size
    )
    new_router2, new_agent2 = router.simulate_swap_route(
        agent1,
        tkn_sell=sell_tkn,
        tkn_buy=buy_tkn,
        sell_quantity=trade_size,
        buy_pool_id="omnipool",
        sell_pool_id="stablepool"
    )
    for tkn in new_agent.holdings:
        if new_agent.holdings[tkn] != new_agent2.holdings[tkn]:
            raise ValueError(f"new_agent holdings {new_agent.holdings[tkn]} != {new_agent2.holdings[tkn]}")
    for ex_id in new_router.exchanges:
        ex = new_router.exchanges[ex_id]
        for tkn in ex.liquidity:
            if ex.liquidity[tkn] != new_router2.exchanges[ex_id].liquidity[tkn]:
                raise ValueError(f"ex liquidity {ex.liquidity[tkn]} != {new_router2.exchanges[ex_id].liquidity[tkn]}")


def check_agent_holdings_equal(agent1, agent2):
    for tkn in agent1.holdings:
        if agent1.holdings[tkn] != agent2.holdings[tkn]:
            raise ValueError(f"agent holdings {agent1.holdings[tkn]} != {agent2.holdings[tkn]}")


def check_liquidity_equal(pool1, pool2):
    for tkn in pool1.liquidity:
        if pool1.liquidity[tkn] != pool2.liquidity[tkn]:
            raise ValueError(f"pool liquidity {pool1.liquidity[tkn]} != {pool2.liquidity[tkn]}")


@given(st.lists(asset_quantity_strategy, min_size=10, max_size=10))
def test_swap_shares(assets: list[float]):
    tokens = {
        "HDX": {'liquidity': 1000000, 'LRNA': 1000000},
        "USDT": {'liquidity': assets[0], 'LRNA': assets[1]},
        "DOT": {'liquidity': 1000000, 'LRNA': 1000000},
        "stablepool": {'liquidity': 1000000, 'LRNA': assets[2]},
        "stablepool2": {'liquidity': 1000000, 'LRNA': assets[3]},
    }

    buy_tkn = "USDT"
    sell_tkn = "stablepool2"
    buy_tkn_pool = "stablepool2"
    sell_tkn_pool = "omnipool"

    omnipool = OmnipoolState(tokens, preferred_stablecoin="USDT", asset_fee=0.0025, lrna_fee=0.0005)
    sp_tokens = {"stable1": assets[4], "stable2": assets[5], "stable3": assets[6]}
    stablepool = StableSwapPoolState(sp_tokens, 1000, trade_fee=0.0100, unique_id="stablepool")
    sp_tokens2 = {"stable2": assets[7], "stable3": assets[8], "USDT": assets[9]}
    stablepool2 = StableSwapPoolState(sp_tokens2, 1000, trade_fee=0.0000, unique_id="stablepool2")
    exchanges = {"omnipool": omnipool, "stablepool": stablepool, "stablepool2": stablepool2}
    router = OmnipoolRouter(exchanges)
    init_holdings = {"stablepool2": 1000000, "USDT": 1000000}
    agent1 = Agent(holdings={tkn: init_holdings[tkn] for tkn in init_holdings})
    agent2 = Agent(holdings={tkn: init_holdings[tkn] for tkn in init_holdings})
    stablepool2_copy = stablepool2.copy()
    trade_size = 1000

    # test buy tkn
    router.swap_route(
        agent1,
        tkn_sell=sell_tkn,
        tkn_buy=buy_tkn,
        buy_quantity=trade_size,
        buy_pool_id=buy_tkn_pool,
        sell_pool_id=sell_tkn_pool
    )
    stablepool2_copy.withdraw_asset(agent2, trade_size, "USDT")
    check_agent_holdings_equal(agent1, agent2)
    check_liquidity_equal(stablepool2, stablepool2_copy)

    # test sell shares
    router.swap_route(
        agent1,
        tkn_sell=sell_tkn,
        tkn_buy=buy_tkn,
        sell_quantity=trade_size,
        buy_pool_id=buy_tkn_pool,
        sell_pool_id=sell_tkn_pool
    )
    stablepool2_copy.remove_liquidity(agent2, trade_size, "USDT")
    check_agent_holdings_equal(agent1, agent2)
    check_liquidity_equal(stablepool2, stablepool2_copy)

    # try the other way around
    sell_tkn, buy_tkn = buy_tkn, sell_tkn

    # test sell tkn
    router.swap_route(
        agent1,
        tkn_buy=buy_tkn,
        tkn_sell=sell_tkn,
        sell_quantity=trade_size,
        buy_pool_id=sell_tkn_pool,
        sell_pool_id=buy_tkn_pool
    )
    stablepool2_copy.add_liquidity(agent2, trade_size, "USDT")
    check_agent_holdings_equal(agent1, agent2)
    check_liquidity_equal(stablepool2, stablepool2_copy)

    # test buy shares
    router.swap_route(
        agent1,
        tkn_buy=buy_tkn,
        tkn_sell=sell_tkn,
        buy_quantity=trade_size,
        buy_pool_id=sell_tkn_pool,
        sell_pool_id=buy_tkn_pool
    )
    stablepool2_copy.buy_shares(agent2, trade_size, "USDT")
    check_agent_holdings_equal(agent1, agent2)
    check_liquidity_equal(stablepool2, stablepool2_copy)


@given(st.lists(asset_quantity_strategy, min_size=6, max_size=6))
def test_spot_prices_omnipool_to_self(assets: list[float]):
    tokens = {
        "HDX": {'liquidity': mpf(assets[0]), 'LRNA': mpf(assets[1])},
        "USDT": {'liquidity': mpf(assets[2]), 'LRNA': mpf(assets[3])},
        "DOT": {'liquidity': mpf(assets[4]), 'LRNA': mpf(assets[5])},
    }

    tkn_buy = "USDT"
    tkn_sell = "DOT"

    omnipool = OmnipoolState(tokens, preferred_stablecoin="USDT", asset_fee=0.0025, lrna_fee=0.0005)
    exchanges = {"omnipool": omnipool}
    router = OmnipoolRouter(exchanges)
    initial_agent = Agent(
        holdings={"DOT": mpf(1000000), "USDT": mpf(1000000)}
    )
    trade_size = 1e-12
    buy_spot = router.buy_spot(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
    sell_spot = router.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    test_router, test_agent = router.simulate_swap(initial_agent, tkn_buy, tkn_sell, buy_quantity=trade_size)
    execution_price = ((initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell])
                       / (test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]))
    if buy_spot != pytest.approx(execution_price, rel=1e-12):
        raise ValueError(f"spot price {buy_spot} != execution price {execution_price}")
    if sell_spot != pytest.approx(1 / execution_price, rel=1e-12):
        raise ValueError(f"spot price {sell_spot} != execution price {1 / execution_price}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=2, max_size=2),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_spot_prices_stableswap_to_self(assets, lrna_fee, asset_fee, trade_fee):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': 1000000, 'LRNA': 1000000},
            "USDT": {'liquidity': 1000000, 'LRNA': 1000000}
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool1 = StableSwapPoolState(
        tokens={"stable1": mpf(assets[0]), "stable2": mpf(assets[1])},
        amplification=100,
        trade_fee=trade_fee, unique_id="stablepool1",
        precision=1e-08
    )
    router = OmnipoolRouter({"omnipool": omnipool, "stablepool1": stablepool1})

    tkn_sell = "stable1"
    tkn_buy = "stable2"
    trade_size = 0.001
    initial_agent = Agent(
        holdings={"stable1": mpf(1), "stable2": mpf(0)}
    )
    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy, tkn_sell, sell_quantity=trade_size
    )

    sell_spot = router.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    buy_spot = router.buy_spot(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
    sell_quantity = initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
    buy_quantity = test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]
    sell_ex = buy_quantity / sell_quantity
    buy_ex = sell_quantity / buy_quantity

    if sell_quantity != trade_size:
        raise ValueError(f"actually bought {sell_quantity} != trade size {trade_size}")
    if sell_spot != pytest.approx(sell_ex, rel=1e-08):
        raise ValueError(f"spot price {sell_spot} != execution price {sell_ex}")
    if buy_spot != pytest.approx(buy_ex, rel=1e-08):
        raise ValueError(f"spot price {buy_spot} != execution price {buy_ex}")


@settings(deadline=timedelta(milliseconds=500))
@given(
    assets=st.lists(asset_quantity_strategy, min_size=6, max_size=6),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_buy_spot_buy_stableswap_sell_stableswap(assets, lrna_fee, asset_fee, trade_fee):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': 1000000, 'LRNA': 1000000},
            "USDT": {'liquidity': 1000000, 'LRNA': 1000000},
            "stablepool1": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[0])},
            "stablepool2": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[1])}
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool1 = StableSwapPoolState(
        tokens={"stable1": mpf(1000000), "stable2": mpf(assets[2])},
        amplification=100,
        trade_fee=trade_fee, unique_id="stablepool1",
        precision=1e-08
    )
    stablepool2 = StableSwapPoolState(
        tokens={"stable3": mpf(1000000), "stable4": mpf(assets[3])},
        amplification=1000,
        trade_fee=trade_fee, unique_id="stablepool2",
        precision=1e-08
    )
    initial_agent = Agent(
        holdings={"stable1": mpf(1), "stable3": mpf(0)}
    )
    tkn_sell = "stable1"
    tkn_buy = "stable3"
    trade_size = mpf(1e-07)

    router = OmnipoolRouter({"omnipool": omnipool, "stablepool1": stablepool1, "stablepool2": stablepool2})

    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy, tkn_sell, buy_quantity=trade_size
    )

    # debugging stuff
    # shares_bought = stablepool2.calculate_withdrawal_shares(tkn_buy, trade_size)
    # shares_sold = omnipool.calculate_sell_from_buy(
    #     tkn_buy=stablepool2.unique_id,
    #     tkn_sell=stablepool1.unique_id,
    #     buy_quantity=shares_bought
    # )
    # step_1_buy_spot = stablepool1.buy_shares_spot(tkn_add=tkn_sell)
    # step_1_agent = initial_agent.copy()
    # stablepool1.copy().buy_shares(step_1_agent, quantity=shares_sold, tkn_add=tkn_sell)
    # step_1_buy_ex = (initial_agent.holdings[tkn_sell] - step_1_agent.holdings[tkn_sell]) / step_1_agent.holdings['stablepool1']
    #
    # step_2_buy_spot = omnipool.buy_spot(tkn_sell='stablepool1', tkn_buy='stablepool2')
    # step_2_agent = step_1_agent.copy()
    # omnipool.copy().swap(step_2_agent, tkn_sell='stablepool1', tkn_buy='stablepool2', buy_quantity=shares_bought)
    # step_2_buy_ex = step_1_agent.holdings['stablepool1'] / step_2_agent.holdings['stablepool2']
    #
    # step_3_buy_spot = 1 / stablepool2.withdraw_asset_spot(tkn_remove=tkn_buy)
    # step_3_agent = step_2_agent.copy()
    # stablepool2.copy().withdraw_asset(step_3_agent, quantity=trade_size, tkn_remove=tkn_buy)
    # step_3_buy_ex = step_2_agent.holdings['stablepool2'] / step_3_agent.holdings[tkn_buy]

    buy_spot = router.buy_spot(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
    sell_quantity = initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
    buy_quantity = test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]
    buy_ex = sell_quantity / buy_quantity

    if buy_quantity != trade_size:
        raise ValueError(f"actually bought {buy_quantity} != trade size {trade_size}")
    if buy_spot != pytest.approx(buy_ex, rel=1e-06):
        raise ValueError(f"spot price {buy_spot} != execution price {buy_ex}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=4, max_size=4),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_sell_spot_buy_stableswap_sell_stableswap(assets, lrna_fee, asset_fee, trade_fee):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': 1000000, 'LRNA': 1000000},
            "USDT": {'liquidity': 1000000, 'LRNA': 1000000},
            "stablepool1": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[0])},
            "stablepool2": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[1])}
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool1 = StableSwapPoolState(
        tokens={"stable1": mpf(1000000), "stable2": mpf(1000000)},
        amplification=100,
        trade_fee=trade_fee, unique_id="stablepool1",
        precision=1e-08
    )
    stablepool2 = StableSwapPoolState(
        tokens={"stable3": mpf(1000000), "stable4": mpf(1000000)},
        amplification=1000,
        trade_fee=trade_fee, unique_id="stablepool2",
        precision=1e-08
    )
    initial_agent = Agent(
        holdings={"stable1": mpf(1), "stable3": mpf(0)}
    )
    tkn_sell = "stable1"
    tkn_buy = "stable3"
    trade_size = mpf(1e-07)

    # debugging stuff
    # step_1_sell_spot = 1 / stablepool1.add_liquidity_spot(tkn_add=tkn_sell)
    # step_1_agent = initial_agent.copy()
    # stablepool1.copy().add_liquidity(step_1_agent, quantity=trade_size, tkn_add=tkn_sell)
    # step_1_sell_ex = step_1_agent.holdings['stablepool1'] / trade_size
    #
    # step_2_sell_spot = omnipool.sell_spot(tkn_sell='stablepool1', tkn_buy='stablepool2')
    # step_2_agent = step_1_agent.copy()
    # omnipool.copy().swap(step_2_agent, tkn_sell='stablepool1', tkn_buy='stablepool2', sell_quantity=step_1_agent.holdings['stablepool1'])
    # step_2_sell_ex = step_2_agent.holdings['stablepool2'] / step_1_agent.holdings['stablepool1']
    #
    # step_3_sell_spot = stablepool2.remove_liquidity_spot(tkn_remove=tkn_buy)
    # step_3_agent = step_2_agent.copy()
    # stablepool2.copy().remove_liquidity(step_3_agent, shares_removed=step_2_agent.holdings['stablepool2'], tkn_remove=tkn_buy)
    # step_3_sell_ex = step_3_agent.holdings[tkn_buy] / step_2_agent.holdings['stablepool2']

    router = OmnipoolRouter({"omnipool": omnipool, "stablepool1": stablepool1, "stablepool2": stablepool2})

    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy, tkn_sell, sell_quantity=trade_size
    )

    sell_spot = router.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    sell_quantity = initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
    buy_quantity = test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]
    sell_ex = buy_quantity / sell_quantity

    if sell_quantity != trade_size:
        raise ValueError(f"actually sold {sell_quantity} != trade size {trade_size}")
    if sell_spot != pytest.approx(sell_ex, rel=1e-06):
        raise ValueError(f"spot price {sell_spot} != execution price {sell_ex}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=4, max_size=4),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_buy_spot_buy_stableswap_sell_omnipool(assets, lrna_fee, asset_fee, trade_fee):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "USDT": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "DOT": {'liquidity': 1000000, 'LRNA': mpf(assets[0])},
            "stablepool": {'liquidity': 1000000, 'LRNA': mpf(assets[1])},
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool = StableSwapPoolState(
        tokens={"stable1": mpf(assets[2]), "stable2": mpf(assets[3])},
        amplification=1000,
        trade_fee=trade_fee, unique_id="stablepool",
        precision=1e-08,
        spot_price_precision=1e-12
    )
    router = OmnipoolRouter({"omnipool": omnipool, "stablepool": stablepool})

    tkn_sell = "DOT"
    tkn_buy = "stable1"
    trade_size = 1e-08
    initial_agent = Agent(
        holdings={"DOT": mpf(1), "stable1": mpf(0)}
    )
    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy, tkn_sell, buy_quantity=trade_size
    )

    buy_spot = router.buy_spot(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
    buy_quantity = test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]
    sell_quantity = initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
    buy_ex = sell_quantity / buy_quantity

    if buy_quantity != trade_size:
        raise ValueError(f"actually bought {buy_quantity} != trade size {trade_size}")
    if buy_spot != pytest.approx(buy_ex, rel=1e-08):
        raise ValueError(f"spot price {buy_spot} != execution price {buy_ex}")
    if test_agent.holdings['stablepool'] > 0:
        # 1e-50 would be still acceptable, but we're leaving this at 0 because it seems to work
        raise ValueError(f"leftover stablepool shares: {test_agent.holdings['stablepool']}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=4, max_size=4),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_sell_spot_buy_stableswap_sell_omnipool(assets, lrna_fee, asset_fee, trade_fee):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "USDT": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "DOT": {'liquidity': 1000000, 'LRNA': mpf(assets[0])},
            "stablepool": {'liquidity': 1000000, 'LRNA': mpf(assets[1])},
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool = StableSwapPoolState(
        tokens={"stable1": mpf(assets[2]), "stable2": mpf(assets[3])},
        amplification=1000,
        trade_fee=trade_fee, unique_id="stablepool",
        precision=1e-08,
        spot_price_precision=1e-12
    )
    router = OmnipoolRouter({"omnipool": omnipool, "stablepool": stablepool})

    tkn_sell = "DOT"
    tkn_buy = "stable1"
    trade_size = 1e-08
    initial_agent = Agent(
        holdings={"DOT": mpf(1), "stable1": mpf(0)}
    )
    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy, tkn_sell, sell_quantity=trade_size
    )

    sell_spot = router.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    buy_quantity = test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]
    sell_quantity = initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
    sell_ex = buy_quantity / sell_quantity

    if sell_quantity != trade_size:
        raise ValueError(f"actually sold {sell_quantity} != trade size {trade_size}")
    if sell_spot != pytest.approx(sell_ex, rel=1e-08):
        raise ValueError(f"spot price {sell_spot} != execution price {sell_ex}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=4, max_size=4),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_sell_spot_sell_stableswap_buy_omnipool(
        assets: list[float], lrna_fee: float, asset_fee: float, trade_fee: float
):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "USDT": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "DOT": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[0])},
            "stablepool": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[1])},
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool = StableSwapPoolState(
        tokens={"stable1": mpf(assets[2]), "stable2": mpf(assets[3])},
        amplification=1000,
        trade_fee=trade_fee, unique_id="stablepool",
        precision=1e-08,
        spot_price_precision=1e-12
    )
    router = OmnipoolRouter({"omnipool": omnipool, "stablepool": stablepool})

    tkn_sell = "stable1"
    tkn_buy = "DOT"
    trade_size = 1e-08
    initial_agent = Agent(
        holdings={"DOT": mpf(0), "stable1": mpf(1)}
    )
    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=trade_size
    )

    sell_spot = router.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    sell_quantity = initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
    buy_quantity = test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]
    sell_ex = buy_quantity / sell_quantity

    if sell_quantity != trade_size:
        raise ValueError(f"actually sold {sell_quantity} != trade size {trade_size}")
    if sell_spot != pytest.approx(sell_ex, rel=1e-08):
        raise ValueError(f"spot price {sell_spot} != execution price {sell_ex}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=4, max_size=4),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_buy_spot_sell_stableswap_buy_omnipool(
        assets: list[float], lrna_fee: float, asset_fee: float, trade_fee: float
):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "USDT": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "DOT": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[0])},
            "stablepool": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[1])},
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool = StableSwapPoolState(
        tokens={"stable1": mpf(assets[2]), "stable2": mpf(assets[3])},
        amplification=1000,
        trade_fee=trade_fee, unique_id="stablepool",
        precision=1e-08,
        spot_price_precision=1e-12
    )
    router = OmnipoolRouter({"omnipool": omnipool, "stablepool": stablepool})

    tkn_sell = "stable1"
    tkn_buy = "DOT"
    trade_size = 1e-08
    initial_agent = Agent(
        holdings={"DOT": mpf(0), "stable1": mpf(1)}
    )
    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=trade_size
    )

    buy_spot = router.buy_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    sell_quantity = initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
    buy_quantity = test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]
    buy_ex = sell_quantity / buy_quantity

    if buy_quantity != trade_size:
        raise ValueError(f"actually bought {buy_quantity} != trade size {trade_size}")
    if buy_spot != pytest.approx(buy_ex, rel=1e-08):
        raise ValueError(f"spot price {buy_spot} != execution price {buy_ex}")
