import pytest

from hydradx.model.modular_amm.omnipool_amm import *
import random


def test_market_construction():
    # noinspection SpellCheckingInspection
    lrna = amm.Asset(name='LRNA', price=1)
    hdx = amm.Asset(name='HDX', price=0.08)
    usd = amm.Asset(name='USD', price=1)
    doge = amm.Asset(name='DOGE', price=0.000001)
    eth = amm.Asset(name='ETH', price=4000.0)

    omnipool = OmniPool(
        lrna=lrna,
        tvl_cap_usd=1000000,
        lrna_fee=0.001,
        asset_fee=0.002,
        preferred_stablecoin=usd
    )
    omnipool.add_pool(hdx, 1000)
    omnipool.add_pool(usd, 1000)
    omnipool.add_pool(doge, 1000)
    omnipool.add_pool(eth, 1000)

    external_market = Market(omnipool.asset_list)

    agents = [
        OmnipoolAgent(name='LP')
        .add_asset(omnipool, doge, 1000),
        OmnipoolAgent(name='trader')
        .add_asset(omnipool, hdx, 1000)
        .add_asset(external_market, usd, 1000)
        .add_asset(external_market, eth, 1000),
        OmnipoolAgent(name='arbitrager')
        .add_asset(external_market, usd, 1000)
        .add_asset(omnipool, hdx, 1000)
    ]

    assert omnipool.B('HDX') == pytest.approx(1/3 * omnipool.S('HDX'))
