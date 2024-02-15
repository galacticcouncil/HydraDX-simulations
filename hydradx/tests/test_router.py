import pytest
from hypothesis import given, strategies as st, assume, settings, Verbosity

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.omnipool_router import OmnipoolRouter
from hydradx.model.amm.stableswap_amm import StableSwapPoolState

asset_quantity_strategy = st.floats(min_value=100000, max_value=10000000)


@given(st.lists(asset_quantity_strategy, min_size=11, max_size=11))
def test_price(assets: list[float]):
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

    router_price_hdx = router.price("HDX", "USDT", "omnipool", "omnipool")
    op_price_hdx = omnipool.price(omnipool, "HDX", "USDT")
    if router_price_hdx != op_price_hdx:
        raise ValueError(f"router price {router_price_hdx} != omnipool price {op_price_hdx}")

    router_price_lp = router.price("stablepool", "USDT", "omnipool", "omnipool")
    op_price_lp = omnipool.price(omnipool, "stablepool", "USDT")
    if router_price_lp != op_price_lp:
        raise ValueError(f"router price {router_price_lp} != omnipool price {op_price_lp}")

    router_price_inside_stablepool = router.price("stable1", "stable3", "stablepool", "stablepool")
    sp_price = stablepool.price("stable1", "stable3")
    if router_price_inside_stablepool != pytest.approx(sp_price, rel=1e-15):
        raise ValueError(f"router price {router_price_inside_stablepool} != stablepool price {sp_price}")

    router_price_outside_stablepool = router.price("stable1", "USDT", "stablepool", "omnipool")
    share_price = stablepool.share_price("stable1")
    op_price_share = omnipool.price(omnipool, "stablepool", "USDT")
    if router_price_outside_stablepool != op_price_share / share_price:
        raise ValueError(f"router price {router_price_outside_stablepool} != stablepool price {op_price_share / share_price}")

def test_price_example():
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

    router_price_outside_stablepool = router.price("DOT", "stable1", "omnipool", "stablepool")
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
    router.swap(agent1, "DOT", "USDT", buy_quantity=10000, buy_pool_id="omnipool", sell_pool_id="omnipool")
    omnipool2.swap(agent2, "DOT", "USDT", buy_quantity=10000)

    for token in omnipool.asset_list:
        if omnipool.liquidity[token] != omnipool2.liquidity[token]:
            raise ValueError(f"omnipool liquidity {omnipool.liquidity[token]} != {omnipool2.liquidity[token]}")
        if omnipool.lrna[token] != omnipool2.lrna[token]:
            raise ValueError(f"omnipool lrna {omnipool.lrna[token]} != {omnipool2.lrna[token]}")

    # test sell
    router.swap(agent1, "DOT", "USDT", sell_quantity=trade_size, buy_pool_id="omnipool", sell_pool_id="omnipool")
    omnipool2.swap(agent2, "DOT", "USDT", sell_quantity=trade_size)

    for token in omnipool.asset_list:
        if omnipool.liquidity[token] != omnipool2.liquidity[token]:
            raise ValueError(f"omnipool liquidity {omnipool.liquidity[token]} != {omnipool2.liquidity[token]}")
        if omnipool.lrna[token] != omnipool2.lrna[token]:
            raise ValueError(f"omnipool lrna {omnipool.lrna[token]} != {omnipool2.lrna[token]}")
