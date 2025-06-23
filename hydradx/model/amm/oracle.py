from .exchange import Exchange


class Block:
    def __init__(self, input_state: Exchange):
        self.liquidity = {tkn: input_state.liquidity[tkn] for tkn in input_state.asset_list}
        self.price = {tkn: input_state.price(tkn) for tkn in input_state.asset_list}
        self.volume_in = {tkn: 0 for tkn in input_state.asset_list}
        self.volume_out = {tkn: 0 for tkn in input_state.asset_list}
        self.withdrawals = {tkn: 0 for tkn in input_state.asset_list}
        self.lps = {tkn: 0 for tkn in input_state.asset_list}
        self.asset_list = input_state.asset_list.copy()
        self.time_step = input_state.time_step


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
        elif first_block is not None:
            self.asset_list = first_block.asset_list
            self.liquidity = first_block.liquidity
            self.price = first_block.price
            self.volume_in = first_block.volume_in
            self.volume_out = first_block.volume_out
        else:
            raise ValueError('Either last_values or first_block must be specified')
        self.age = 0
        self.last_updated = {tkn: first_block.time_step - 1 if first_block is not None else 0 for tkn in self.asset_list}

    def add_asset(self, tkn: str, liquidity: float):
        self.liquidity[tkn] = liquidity
        self.volume_in[tkn] = 0
        self.volume_out[tkn] = 0
        self.asset_list.append(tkn)

    def update(self, block: Block, assets: list[str] = None):
        if assets is None:
            assets = self.asset_list
        update_steps = {tkn: max(block.time_step - self.last_updated[tkn], 1) for tkn in assets}
        update_factor = {tkn: (1 - self.decay_factor) ** update_steps[tkn] for tkn in assets}
        for attr in ['liquidity', 'price', 'volume_in', 'volume_out']:
            block_attr = getattr(block, attr)
            self_attr = getattr(self, attr)
            for tkn in assets:
                self_attr[tkn] = (
                    update_factor[tkn] * self_attr.get(tkn, block_attr[tkn])
                    + (1 - update_factor[tkn]) * block_attr[tkn]
                )
        self.last_updated.update({tkn: block.time_step for tkn in assets})
        self.age = block.time_step
        return self


class OracleArchiveState:
    def __init__(self, oracle: Oracle):
        self.liquidity = {tkn: oracle.liquidity[tkn] for tkn in oracle.asset_list}
        self.price = {tkn: oracle.price[tkn] for tkn in oracle.asset_list}
        self.volume_in = {tkn: oracle.volume_in[tkn] for tkn in oracle.asset_list}
        self.volume_out = {tkn: oracle.volume_out[tkn] for tkn in oracle.asset_list}
        self.age = oracle.age
