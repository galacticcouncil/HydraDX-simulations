from hydradx.model.amm.oracle import Oracle, Block
from hydradx.model.amm.omnipool_amm import OmnipoolState

def test_oracle_multi_block_update():
    start_block = Block(
        OmnipoolState(
            tokens={
                'HDX': {'liquidity': 1000, 'LRNA': 1},
                'USD': {'liquidity': 10, 'LRNA': 1}
            }
        )
    )
    end_block = Block(
        OmnipoolState(
            tokens={
                'HDX': {'liquidity': 2000, 'LRNA': 2.2},
                'USD': {'liquidity': 11, 'LRNA': 0.99}
            }
        )
    )
    end_block.volume_in['HDX'] = 100
    end_block.volume_in['USD'] = 1
    end_block.volume_out['HDX'] = 50
    end_block.volume_out['USD'] = 0.5
    oracle_1 = Oracle(
        first_block=start_block,
        decay_factor=0.1
    )
    oracle_2 = Oracle(
        first_block=start_block,
        decay_factor=0.1
    )
    for i in range(10):
        end_block.time_step += 1
        oracle_1.update(end_block)
    oracle_2.update(end_block)

    for tkn in start_block.liquidity:
        assert oracle_1.liquidity[tkn] == oracle_2.liquidity[tkn]
        assert oracle_1.price[tkn] == oracle_2.price[tkn]
        assert oracle_1.volume_in[tkn] == oracle_2.volume_in[tkn]
        assert oracle_1.volume_out[tkn] == oracle_2.volume_out[tkn]
