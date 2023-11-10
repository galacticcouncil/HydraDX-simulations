import random

from hypothesis import strategies as st

import hydradx.model.amm.omnipool_amm as oamm
import hydradx.model.amm.stableswap_amm as ssamm
from mpmath import mp, mpf
mp.dps = 50

asset_price_strategy = st.floats(min_value=0.0001, max_value=100000)
asset_price_bounded_strategy = st.floats(min_value=0.1, max_value=10)
asset_number_strategy = st.integers(min_value=3, max_value=5)
arb_precision_strategy = st.integers(min_value=1, max_value=5)
asset_quantity_strategy = st.floats(min_value=100, max_value=10000000)
asset_quantity_bounded_strategy = st.floats(min_value=1000000, max_value=10000000)
percentage_of_liquidity_strategy = st.floats(min_value=0.0000001, max_value=0.10)
reasonable_percentage_of_liquidity_strategy = st.floats(min_value=0.01, max_value=0.10)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)
amplification_strategy = st.floats(min_value=1, max_value=10000)


@st.composite
def reasonable_holdings(draw, token_count: int = 0):
    token_count = token_count or draw(asset_number_strategy)
    return [draw(asset_quantity_bounded_strategy) for _ in range(token_count)]


@st.composite
def reasonable_market(draw, token_count: int = 0):
    token_count = token_count or draw(asset_number_strategy)
    return [draw(asset_price_bounded_strategy) for _ in range(token_count)]


@st.composite
def reasonable_pct(draw, token_count: int = 0):
    token_count = token_count or draw(asset_number_strategy)
    return [draw(reasonable_percentage_of_liquidity_strategy) for _ in range(token_count)]


@st.composite
def reasonable_market_dict(draw, token_count: int = 0):
    price_list = draw(reasonable_market(token_count))
    price_dict = {'HDX': price_list[1], 'USD': 1.0}
    price_dict.update({
        ''.join(
            random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3)): price_list[i + 2] for i in
        range(token_count - 2)
    })
    return price_dict


@st.composite
def assets_reasonable_config(draw, token_count: int = 0):
    token_count = token_count or draw(asset_number_strategy)
    usd_price_lrna = draw(asset_price_bounded_strategy)
    return_dict = {
        'HDX': {
            'liquidity': draw(asset_quantity_bounded_strategy),
            'LRNA': draw(asset_quantity_bounded_strategy)
        },
        'USD': {
            'liquidity': draw(asset_quantity_bounded_strategy),
            'LRNA_price': usd_price_lrna
        }
    }
    return_dict.update({
        ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3)): {
            'liquidity': draw(asset_quantity_bounded_strategy),
            'LRNA': draw(asset_quantity_bounded_strategy)
        } for _ in range(token_count - 2)
    })
    return return_dict


@st.composite
def omnipool_reasonable_config(
        draw,
        asset_dict=None,
        token_count=0,
        lrna_fee=None,
        asset_fee=None,
        tvl_cap_usd=0,
        imbalance=None,
        remove_liquidity_volatility_threshold: float = 0
):
    asset_dict: dict = asset_dict or draw(assets_reasonable_config(token_count))

    test_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=tvl_cap_usd or float('inf'),
        asset_fee=draw(st.floats(min_value=0, max_value=0.1)) if asset_fee is None else asset_fee,
        lrna_fee=draw(st.floats(min_value=0, max_value=0.1)) if lrna_fee is None else lrna_fee,
        remove_liquidity_volatility_threshold=remove_liquidity_volatility_threshold,
        withdrawal_fee=True,
        min_withdrawal_fee=0.0001,
    )

    test_state.lrna_imbalance = -draw(asset_quantity_strategy) if imbalance is None else imbalance
    test_state.update()
    return test_state


@st.composite
def assets_config(draw, token_count: int = 0):
    token_count = token_count or draw(asset_number_strategy)
    usd_price_lrna = draw(asset_price_strategy)
    return_dict = {
        'HDX': {
            'liquidity': mpf(draw(asset_quantity_strategy)),
            'LRNA': mpf(draw(asset_quantity_strategy))
        },
        'USD': {
            'liquidity': draw(asset_quantity_strategy),
            'LRNA_price': usd_price_lrna
        }
    }
    return_dict.update({
        ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3)): {
            'liquidity': draw(asset_quantity_strategy),
            'LRNA': draw(asset_quantity_strategy)
        } for _ in range(token_count - 2)
    })
    return return_dict


@st.composite
def omnipool_config(
        draw,
        asset_dict=None,
        token_count=0,
        lrna_fee=None,
        asset_fee=None,
        tvl_cap_usd=0,
        sub_pools: dict = None,
        imbalance: float = None,
        withdrawal_fee=True
):
    asset_dict: dict = asset_dict or draw(assets_config(token_count))

    sub_pool_instances: dict['str', ssamm.StableSwapPoolState] = {}
    if sub_pools:
        for i, (name, pool) in enumerate(sub_pools.items()):
            base_token = list(asset_dict.keys())[i + 1]
            sub_pool_instance = draw(stableswap_config(
                asset_dict=pool['asset_dict'] if 'asset_dict' in pool else None,
                token_count=pool['token_count'] if 'token_count' in pool else None,
                amplification=pool['amplification'] if 'amplification' in pool else None,
                trade_fee=pool['trade_fee'] if 'trade_fee' in pool else None,
                base_token=base_token
            ))
            asset_dict.update({tkn: {
                'liquidity': sub_pool_instance.liquidity[tkn],
                'LRNA': (
                    asset_dict[base_token]['LRNA'] * sub_pool_instance.liquidity[tkn]
                    / asset_dict[base_token]['liquidity']
                    if 'LRNA' in asset_dict[base_token] else
                    asset_dict[base_token]['LRNA_price'] * sub_pool_instance.liquidity[tkn]
                )
            } for tkn in sub_pool_instance.asset_list})
            sub_pool_instances[name] = sub_pool_instance

    test_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=tvl_cap_usd or float('inf'),
        asset_fee=draw(st.floats(min_value=0, max_value=0.1)) if asset_fee is None else asset_fee,
        lrna_fee=draw(st.floats(min_value=0, max_value=0.1)) if lrna_fee is None else lrna_fee,
        imbalance=imbalance if imbalance is not None else draw(st.floats(min_value=-1, max_value=1)),
        withdrawal_fee=withdrawal_fee
    )

    for name, pool in sub_pool_instances.items():
        test_state.create_sub_pool(
            tkns_migrate=pool.asset_list,
            sub_pool_id=name,
            amplification=pool.amplification,
            trade_fee=pool.trade_fee
        )

    test_state.lrna_imbalance = -draw(asset_quantity_strategy)
    test_state.update()
    return test_state


@st.composite
def stableswap_config(
        draw,
        asset_dict=None,
        token_count: int = None,
        trade_fee: float = None,
        amplification: float = None,
        precision: float = 0.00001,
        unique_id: str = '',
        base_token: str = 'USD'
):
    token_count = token_count or draw(asset_number_strategy)
    asset_dict = asset_dict or {
        f"{base_token}-{'abcdefghijklmnopqrstuvwxyz'[i % 26]}{i // 26}": mpf(draw(asset_quantity_strategy))
        for i in range(token_count)
    }
    test_state = ssamm.StableSwapPoolState(
        tokens=asset_dict,
        amplification=draw(amplification_strategy) if amplification is None else amplification,
        precision=precision,
        trade_fee=draw(st.floats(min_value=0, max_value=0.1)) if trade_fee is None else trade_fee,
        unique_id=unique_id or '/'.join(asset_dict.keys())
    )
    return test_state
