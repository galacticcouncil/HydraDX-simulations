from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.oracle import Block, Oracle


def test_initialization_from_block():
    initial_state = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': 10000000 / 0.05, 'LRNA': 1000000},
            'USD': {'liquidity': 10000000, 'LRNA': 1000000},
            'DOT': {'liquidity': 10000000 / 5, 'LRNA': 1000000},
        },
        asset_fee=0.0025,
        lrna_fee=0.0005
    )
    first_block = Block(initial_state)
    oracle = Oracle(first_block, sma_equivalent_length=100)
    for tkn in ['HDX', 'USD', 'DOT']:
        assert oracle.liquidity[tkn] == initial_state.liquidity[tkn]
        assert oracle.volume_in[tkn] == 0
        assert oracle.volume_out[tkn] == 0
        assert oracle.price[tkn] == oamm.price(initial_state, tkn)


def test_initialization_from_last_values():
    last_values = {
        'liquidity': {'HDX': 5000000, 'USD': 500000, 'DOT': 100000},
        'volume_in': {'HDX': 10000, 'USD': 10000, 'DOT': 10000},
        'volume_out': {'HDX': 10000, 'USD': 10000, 'DOT': 10000},
        'price': {'HDX': 0.05, 'USD': 1, 'DOT': 5},
    }

    oracle = Oracle(last_values=last_values, sma_equivalent_length=100)

    for tkn in ['HDX', 'USD', 'DOT']:
        assert oracle.liquidity[tkn] == last_values['liquidity'][tkn]
        assert oracle.volume_in[tkn] == last_values['volume_in'][tkn]
        assert oracle.volume_out[tkn] == last_values['volume_out'][tkn]
        assert oracle.price[tkn] == last_values['price'][tkn]

# def test_price_update():
#     initial_state = oamm.OmnipoolState(
#         tokens={
#             'HDX': {'liquidity': 10000000 / 0.05, 'LRNA': 1000000},
#             'USD': {'liquidity': 10000000, 'LRNA': 1000000},
#         },
#         asset_fee=0.0025,
#         lrna_fee=0.0005
#     )
#     first_block = Block(initial_state)
#     oracle = Oracle(first_block, sma_equivalent_length=100)
#
#     oracle.update(second_block)
#     assert oracle.price["DAI"] == 1
#     assert oracle.price["USDC"] == 1
