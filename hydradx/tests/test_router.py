import pytest, copy
from hypothesis import given, strategies as st, assume, settings, Verbosity

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.omnipool_router import OmnipoolRouter
from hydradx.model.amm.stableswap_amm import StableSwapPoolState

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

    router_price_hdx = router.price_route("HDX", "USDT", "omnipool", "omnipool")
    op_price_hdx = omnipool.price(omnipool, "HDX", "USDT")
    if router_price_hdx != op_price_hdx:
        raise ValueError(f"router price {router_price_hdx} != omnipool price {op_price_hdx}")

    router_price_lp = router.price_route("stablepool", "USDT", "omnipool", "omnipool")
    op_price_lp = omnipool.price(omnipool, "stablepool", "USDT")
    if router_price_lp != op_price_lp:
        raise ValueError(f"router price {router_price_lp} != omnipool price {op_price_lp}")

    router_price_inside_stablepool = router.price_route("stable1", "stable3", "stablepool", "stablepool")
    sp_price = stablepool.price("stable1", "stable3")
    if router_price_inside_stablepool != pytest.approx(sp_price, rel=1e-15):
        raise ValueError(f"router price {router_price_inside_stablepool} != stablepool price {sp_price}")

    router_price_outside_stablepool = router.price_route("stable1", "USDT", "stablepool", "omnipool")
    share_price = stablepool.share_price("stable1")
    op_price_share = omnipool.price(omnipool, "stablepool", "USDT")
    if router_price_outside_stablepool != op_price_share / share_price:
        raise ValueError(f"router price {router_price_outside_stablepool} != stablepool price {op_price_share / share_price}")

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
    agent1 = Agent(holdings={"DOT": 1000000, "USDT": 1000000})
    agent2 = Agent(holdings={"DOT": 1000000, "USDT": 1000000})
    trade_size = trade_size_mult * min(assets[4], assets[2])

    buy_quantity = 1000

    # test buy
    router.swap_route(agent1, "DOT", "USDT", buy_quantity=buy_quantity, buy_pool_id="stablepool2", sell_pool_id="omnipool")
    # calculate how many shares to buy
    delta_shares = stablepool2_copy.calculate_withdrawal_shares("USDT", buy_quantity)
    # buy shares
    omnipool2.swap(agent2, "stablepool2", "DOT", buy_quantity=delta_shares)
    # withdraw USDT
    stablepool2_copy.remove_liquidity(agent2, agent2.holdings["stablepool2"], "USDT")

    for token in omnipool.asset_list:
        if omnipool.liquidity[token] != omnipool2.liquidity[token]:
            raise ValueError(f"omnipool liquidity {omnipool.liquidity[token]} != {omnipool2.liquidity[token]}")
        if omnipool.lrna[token] != omnipool2.lrna[token]:
            raise ValueError(f"omnipool lrna {omnipool.lrna[token]} != {omnipool2.lrna[token]}")
    for token in stablepool2.asset_list:
        if stablepool2.liquidity[token] != stablepool2_copy.liquidity[token]:
            raise ValueError(f"stablepool2 liquidity {stablepool2.liquidity[token]} != {stablepool2_copy.liquidity[token]}")
    for token in agent1.holdings:
        if agent1.holdings[token] != agent2.holdings[token]:
            raise ValueError(f"agent1 holdings {agent1.holdings[token]} != {agent2.holdings[token]}")

    # test sell
    router.swap_route(agent1, "DOT", "USDT", sell_quantity=trade_size, buy_pool_id="stablepool2", sell_pool_id="omnipool")
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
            raise ValueError(f"stablepool2 liquidity {stablepool2.liquidity[token]} != {stablepool2_copy.liquidity[token]}")
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
    router.swap_route(agent1, sell_tkn, buy_tkn, buy_quantity=trade_size, buy_pool_id="stablepool2", sell_pool_id="stablepool")
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
    router.swap_route(agent1, "stable1", "USDT", sell_quantity=trade_size, buy_pool_id="stablepool2", sell_pool_id="stablepool")
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
            elif tkn in agent1.holdings and tkn in sell_init_holdings and agent1.holdings[tkn] != sell_init_holdings[tkn]:
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

    best_route = router.find_best_route( buy_tkn, sell_tkn)
    if best_route != ("stablepool", "stablepool2"):
        raise ValueError(f"best route {best_route} != ('stablepool', 'stablepool2')")

    new_router, new_agent = router.simulate_swap(agent1, buy_tkn, sell_tkn, sell_quantity=trade_size)
    new_router2, new_agent2 = router.simulate_swap_route(agent1, sell_tkn, buy_tkn, sell_quantity=trade_size, buy_pool_id="stablepool2", sell_pool_id="stablepool")
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

    best_route = router.find_best_route( buy_tkn, sell_tkn)
    if best_route != ("stablepool", "omnipool"):
        raise ValueError(f"best route {best_route} != ('stablepool', 'omnipool')")

    new_router, new_agent = router.simulate_swap(agent1, buy_tkn, sell_tkn, sell_quantity=trade_size)
    new_router2, new_agent2 = router.simulate_swap_route(agent1, sell_tkn, buy_tkn, sell_quantity=trade_size, buy_pool_id="omnipool", sell_pool_id="stablepool")
    for tkn in new_agent.holdings:
        if new_agent.holdings[tkn] != new_agent2.holdings[tkn]:
            raise ValueError(f"new_agent holdings {new_agent.holdings[tkn]} != {new_agent2.holdings[tkn]}")
    for ex_id in new_router.exchanges:
        ex = new_router.exchanges[ex_id]
        for tkn in ex.liquidity:
            if ex.liquidity[tkn] != new_router2.exchanges[ex_id].liquidity[tkn]:
                raise ValueError(f"ex liquidity {ex.liquidity[tkn]} != {new_router2.exchanges[ex_id].liquidity[tkn]}")
