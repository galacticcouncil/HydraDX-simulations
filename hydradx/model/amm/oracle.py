from .exchange import Exchange
from typing import Protocol, Callable

class ExchangeLike(Protocol):
    liquidity: dict
    asset_list: list
    price: dict or Callable[[str], float]
    time_step: int

class Block:
    def __init__(self, input_state: ExchangeLike):
        price = {tkn: input_state.price(tkn) for tkn in input_state.asset_list} \
            if isinstance(input_state, Exchange) else input_state.price
        self.liquidity = {tkn: input_state.liquidity[tkn] for tkn in input_state.asset_list}
        self.price = {tkn: price[tkn] for tkn in input_state.asset_list}
        self.volume_in = {
            tkn: input_state.volume_in if hasattr(input_state, 'volume_in') else 0.0 for tkn in input_state.asset_list
        }
        self.volume_out = {
            tkn: input_state.volume_out if hasattr(input_state, 'volume_out') else 0.0 for tkn in input_state.asset_list
        }
        self.withdrawals = {tkn: 0.0 for tkn in input_state.asset_list}
        self.lps = {tkn: 0.0 for tkn in input_state.asset_list}
        self.asset_list = input_state.asset_list.copy()
        self.time_step = input_state.time_step

    def copy(self):
        return Block(input_state=self)


class Oracle:
    def __init__(self, first_block: Block = None, decay_factor: float = 0, sma_equivalent_length: int = 0,
                 last_values: dict = None):
        if decay_factor:
            self.decay_factor = decay_factor
        elif sma_equivalent_length:
            self.decay_factor = 2 / (sma_equivalent_length + 1)
        else:
            raise ValueError('Either decay_factor or sma_equivalent_length must be specified')
        self.length = sma_equivalent_length or 2 / self.decay_factor - 1
        if last_values is not None:
            self.asset_list = list((last_values['liquidity']).keys())
            self.liquidity = {k: v for (k, v) in last_values['liquidity'].items()}
            self.price = {k: v for (k, v) in last_values['price'].items()}
            self.volume_in = {k: v for (k, v) in last_values['volume_in'].items()}
            self.volume_out = {k: v for (k, v) in last_values['volume_out'].items()}
            self.last_updated = (
                {k: v for (k, v) in last_values['last_updated'].items()}
                if 'last_updated' in last_values
                else {tkn: 0 for tkn in self.asset_list}
            )
            self.time_step = last_values['time_step'] if 'time_step' in last_values else 0
        elif first_block is not None:
            self.asset_list = [tkn for tkn in first_block.asset_list]
            self.liquidity = {k: v for (k, v) in first_block.liquidity.items()}
            self.price = {k: v for (k, v) in first_block.price.items()}
            self.volume_in = {tkn: first_block.volume_in[tkn] for tkn in self.asset_list}
            self.volume_out = {tkn: first_block.volume_out[tkn] for tkn in self.asset_list}
            self.time_step = first_block.time_step
            self.last_updated = {tkn: first_block.time_step for tkn in self.asset_list}
        else:
            raise ValueError('Either last_values or first_block must be specified')

    def add_asset(self, tkn: str, liquidity: float):
        self.liquidity[tkn] = liquidity
        self.volume_in[tkn] = 0.0
        self.volume_out[tkn] = 0.0
        self.asset_list.append(tkn)

    def update(self, block: Block, assets: list[str] = None):
        if assets is None:
            assets = self.asset_list
        update_steps = {tkn: max(block.time_step - self.last_updated[tkn], 0) for tkn in assets}
        update_factor = {tkn: (1 - self.decay_factor) ** update_steps[tkn] for tkn in assets}
        for tkn in assets:
            if update_steps[tkn] == 0:
                continue
            # we assume price and liquidity have stayed the same since last update
            self.liquidity[tkn] = update_factor[tkn] * self.liquidity[tkn] + (1 - update_factor[tkn]) * block.liquidity[tkn]
            self.price[tkn] = update_factor[tkn] * self.price[tkn] + (1 - update_factor[tkn]) * block.price[tkn]
            # we assume volume_in and volume_out have been 0 since last update
            self.volume_in[tkn] = update_factor[tkn] * self.volume_in[tkn] + self.decay_factor * block.volume_in[tkn]
            self.volume_out[tkn] = update_factor[tkn] * self.volume_out[tkn] + self.decay_factor * block.volume_out[tkn]
            self.last_updated[tkn] = block.time_step
        self.time_step = block.time_step
        return self

    def copy(self):
        new_oracle = Oracle(
            first_block=None,
            decay_factor=self.decay_factor,
            sma_equivalent_length=self.length,
            last_values={
                'liquidity': self.liquidity,
                'price': self.price,
                'volume_in': self.volume_in,
                'volume_out': self.volume_out,
                'last_updated': self.last_updated
            }
        )
        new_oracle.time_step = self.time_step
        return new_oracle

class OracleArchiveState:
    def __init__(self, oracle: Oracle):
        self.liquidity = {tkn: oracle.liquidity[tkn] for tkn in oracle.asset_list}
        self.price = {tkn: oracle.price[tkn] for tkn in oracle.asset_list}
        self.volume_in = {tkn: oracle.volume_in[tkn] for tkn in oracle.asset_list}
        self.volume_out = {tkn: oracle.volume_out[tkn] for tkn in oracle.asset_list}
        self.age = oracle.time_step
