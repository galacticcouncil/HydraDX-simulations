from .amm import AMM


class Block:
    def __init__(self, input_state: AMM):
        self.liquidity = {tkn: input_state.liquidity[tkn] for tkn in input_state.asset_list}
        self.price = {tkn: input_state.price(tkn) for tkn in input_state.asset_list}
        self.volume_in = {tkn: 0 for tkn in input_state.asset_list}
        self.volume_out = {tkn: 0 for tkn in input_state.asset_list}
        self.asset_list = input_state.asset_list.copy()


class Oracle:
    def __init__(self, first_block: Block, decay_factor: float = 0, sma_equivalent_length: int = 0):
        if decay_factor:
            self.decay_factor = decay_factor
        elif sma_equivalent_length:
            self.decay_factor = 2 / (sma_equivalent_length + 1)
        else:
            raise ValueError('Either decay_factor or sma_equivalent_length must be specified')
        self.length = sma_equivalent_length or 2 / self.decay_factor - 1
        self.asset_list = []
        self.liquidity = first_block.liquidity
        self.price = first_block.price
        self.volume_in = first_block.volume_in
        self.volume_out = first_block.volume_out
        self.age = 0

    def add_asset(self, tkn: str, liquidity: float):
        self.liquidity[tkn] = liquidity
        self.volume_in[tkn] = 0
        self.volume_out[tkn] = 0
        self.asset_list.append(tkn)

    def update_value(self, tkn: str, attribute: str, value: float):
        return (1 - self.decay_factor) * getattr(self, attribute)[tkn] + self.decay_factor * value

    def update(self, block: Block):
        self.age += 1
        for tkn in block.liquidity:
            self.liquidity[tkn] = self.update_value(
                tkn, 'liquidity', block.liquidity[tkn]
            ) if tkn in self.liquidity else block.liquidity[tkn]
            self.price[tkn] = self.update_value(
                tkn, 'price', block.price[tkn]
            ) if tkn in self.price else block.price[tkn]
            self.volume_in[tkn] = self.update_value(
                tkn, 'volume_in', block.volume_in[tkn]
            ) if tkn in self.volume_in else block.volume_in[tkn]
            self.volume_out[tkn] = self.update_value(
                tkn, 'volume_out', block.volume_out[tkn]
            ) if tkn in self.volume_out else block.volume_out[tkn]
        return self
