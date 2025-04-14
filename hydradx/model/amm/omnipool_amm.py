import copy
from typing import Callable
import math

from .agents import Agent
from .exchange import Exchange
from .oracle import Oracle, Block, OracleArchiveState
from .stableswap_amm import StableSwapPoolState
from typing import Literal

class DynamicFee:

    def __init__(
        self,
        minimum: float = 0,
        maximum: float = float('inf'),
        amplification: float = 0,
        decay: float = 0,
        # ^^ if these four are provided, we can figure the rest out.
        current: dict[str: float] = None,
        liquidity: dict = None,
        net_volume: dict = None,
        last_updated: dict = None
    ):
        self.amplification = amplification
        self.decay = decay
        self.minimum = minimum
        self.maximum = maximum
        if current is None:
            self.current = {}
        else:
            self.current = current
        self.last_updated = last_updated or {tkn: 0 for tkn in self.current}
        self.liquidity_at_last_update = liquidity if liquidity is not None else {}
        self.volume_at_last_update = net_volume if net_volume is not None else {}

    def update(self, time_step: int, volume: dict, liquidity: dict):
        for tkn in self.current:
            if self.last_updated[tkn] == time_step:
                # update only when fee[tkn] has been accessed this block
                self.liquidity_at_last_update[tkn] = liquidity[tkn]
                self.volume_at_last_update[tkn] = volume[tkn]


class OmnipoolState(Exchange):
    unique_id: str = 'omnipool'

    def __init__(self,
                 tokens: dict[str: dict],
                 tvl_cap: float = float('inf'),
                 preferred_stablecoin: str = None,
                 asset_fee: dict or DynamicFee or float = 0,
                 lrna_fee: dict or DynamicFee or float = 0,
                 oracles: dict[str: int] = None,
                 trade_limit_per_block: float = float('inf'),
                 update_function: Callable = None,
                 last_oracle_values: dict = None,
                 max_withdrawal_per_block: float = 1,
                 max_lp_per_block: float = float('inf'),
                 remove_liquidity_volatility_threshold: float = 0,
                 withdrawal_fee: bool = True,
                 min_withdrawal_fee: float = 0.0001,
                 lrna_mint_pct: float = 1.0,
                 unique_id: str = 'omnipool',
                 lrna_fee_burn: float = 0.5,
                 lrna_fee_destination: Agent = None,
                 dynamic_fee_precision: int = 20
                 ):
        """
        tokens should be a dict in the form of [str: dict]
        the nested dict needs the following parameters:
        {
          'liquidity': float  # starting risk asset liquidity in the pool
          (
          'LRNA': float  # starting LRNA on the other side of the pool
          or
          'LRNA_price': float  # price of the asset denominated in LRNA
          )

          optional:
          'weight_cap': float  # maximum fraction of TVL that may be held in this pool
          'oracle': dict  {name: period}  # name of the oracle its period, i.e. how slowly it decays
        }
        """

        super().__init__()

        if 'HDX' not in tokens:
            raise ValueError('HDX not included in tokens.')
        if preferred_stablecoin is not None and preferred_stablecoin not in tokens:
            raise ValueError(f'{preferred_stablecoin} is preferred stablecoin, but not included in tokens.')

        self.asset_list: list[str] = []
        self.liquidity = {}
        self.lrna = {}
        self.shares = {}
        self.protocol_shares = {}
        self.weight_cap = {}
        self.tvl_cap = tvl_cap
        if preferred_stablecoin is None and "USD" in tokens:
            self.stablecoin = "USD"
        else:
            self.stablecoin = preferred_stablecoin
        self.fail = ''
        self.sub_pools = dict()  # require sub_pools to be added through create_sub_pool
        self.update_function = update_function
        self.max_withdrawal_per_block = max_withdrawal_per_block
        self.max_lp_per_block = max_lp_per_block
        self.remove_liquidity_volatility_threshold = remove_liquidity_volatility_threshold
        self.lrna_mint_pct = lrna_mint_pct
        self.withdrawal_fee = withdrawal_fee

        if withdrawal_fee:
            self.min_withdrawal_fee = min_withdrawal_fee

        for token, pool in tokens.items():
            assert pool['liquidity'], f'token {token} missing required parameter: liquidity'
            if 'LRNA' in pool:
                lrna = pool['LRNA']
            elif 'LRNA_price' in pool:
                lrna = pool['liquidity'] * pool['LRNA_price']
            else:
                raise ValueError("token {name} missing required parameter: ('LRNA' or 'LRNA_price)")
            self.add_token(
                token,
                liquidity=pool['liquidity'],
                lrna=lrna,
                shares=pool['shares'] if 'shares' in pool else pool['liquidity'],
                protocol_shares=pool['shares'] if 'shares' in pool else pool['liquidity'],
                weight_cap=pool['weight_cap'] if 'weight_cap' in pool else 1
            )

        self.oracles = {}

        if oracles is None or 'price' not in oracles:
            if last_oracle_values is None or 'price' not in last_oracle_values:
                self.oracles['price'] = Oracle(sma_equivalent_length=9, first_block=Block(self))
            else:
                self.oracles['price'] = Oracle(sma_equivalent_length=9, first_block=Block(self),
                                               last_values=last_oracle_values['price'])
        if last_oracle_values is not None and oracles is not None:
            self.oracles.update({
                name: Oracle(
                    sma_equivalent_length=period,
                    last_values=last_oracle_values[name] if name in last_oracle_values else None
                )
                for name, period in oracles.items()
            })
        elif oracles is not None:
            self.oracles.update({
                name: Oracle(sma_equivalent_length=period, first_block=Block(self))
                for name, period in oracles.items()
            })

        # trades per block cannot exceed this fraction of the pool's liquidity
        self.trade_limit_per_block = trade_limit_per_block

        self.time_step = 0
        self.asset_fee = asset_fee
        self.lrna_fee = lrna_fee
        self.lrna_fee_burn = lrna_fee_burn
        if lrna_fee_burn > 1 or lrna_fee_burn < 0:
            raise ValueError('lrna_fee_burn must be >= 0 and <= 1')
        if lrna_fee_destination is None:
            lrna_fee_destination = Agent(holdings={'LRNA': 0})
        self.lrna_fee_destination = lrna_fee_destination
        self.dynamic_fee_precision = dynamic_fee_precision

        self.current_block = Block(self)
        self.unique_id = unique_id

    def _create_dynamic_fee(self, value: DynamicFee or dict or float, fee_type: Literal['lrna', 'asset']) -> DynamicFee:
        raise_oracle = 'price'
        def get_last_volume():
            return {
                tkn: (self.oracles[raise_oracle].volume_in[tkn]
                - self.oracles[raise_oracle].volume_out[tkn])
                if fee_type == 'lrna' else
                (self.oracles[raise_oracle].volume_out[tkn]
                - self.oracles[raise_oracle].volume_in[tkn])
                for tkn in self.asset_list
            }
        if isinstance(value, DynamicFee):
            return_val = DynamicFee(
                amplification=value.amplification,
                decay=value.decay,
                minimum=value.minimum,
                maximum=value.maximum,
                current={tkn: value.current[tkn] if tkn in value.current else value.minimum for tkn in self.asset_list},
                liquidity={tkn: value.liquidity_at_last_update[tkn] if tkn in value.liquidity_at_last_update else self.liquidity[tkn] for tkn in self.asset_list},
                net_volume={tkn: value.volume_at_last_update[tkn] for tkn in self.asset_list} if value.volume_at_last_update else get_last_volume(),
                last_updated=value.last_updated
            )
            return return_val
        elif isinstance(value, dict):
            current = {
                tkn: value[tkn] if tkn in value else (
                    (self.last_lrna_fee[tkn] if tkn in self.last_lrna_fee else self._lrna_fee.minimum)
                    if fee_type == 'lrna' else
                    (self.last_fee[tkn] if tkn in self.last_fee else self._asset_fee.minimum)
                ) for tkn in self.asset_list}
            return DynamicFee(
                current=current,
                minimum=min(current.values()),
                maximum=max(current.values()),
                liquidity={tkn: self.liquidity[tkn] for tkn in self.liquidity},
                net_volume=get_last_volume()
            )
        else:  # value is a number
            return DynamicFee(
                current={tkn: value for tkn in self.asset_list},
                maximum=value,
                minimum=value,
                liquidity={tkn: self.liquidity[tkn] for tkn in self.liquidity},
                net_volume=get_last_volume()
            )

    @property
    def lrna_fee(self) -> Callable[[str], float]:
        return self._get_lrna_fee

    @lrna_fee.setter
    def lrna_fee(self, value: DynamicFee or dict or float):
        self._lrna_fee = self._create_dynamic_fee(value, 'lrna')

    def _get_lrna_fee(self, tkn):
        return self.compute_dynamic_fee(
            fee=self._lrna_fee,
            tkn=tkn
        )

    @property
    def last_lrna_fee(self):
        return self._lrna_fee.current

    @property
    def asset_fee(self) -> Callable[[str], float]:
        return self._get_asset_fee

    @asset_fee.setter
    def asset_fee(self, value):
        self._asset_fee = self._create_dynamic_fee(value, 'asset')

    def _get_asset_fee(self, tkn):
        return self.compute_dynamic_fee(
            fee=self._asset_fee,
            tkn=tkn
        )

    @property
    def last_fee(self):
        return self._asset_fee.current

    def compute_dynamic_fee(self, fee: DynamicFee, tkn: str) -> float:
        if fee.amplification == 0:
            fee.last_updated[tkn] = self.time_step
        if fee.last_updated[tkn] == self.time_step:
            # return the last fee if it's already been computed for this tkn and block
            return fee.current[tkn]

        # use this approximation to catch up to where we think the fee should be,
        # knowing there have been no trades until now
        num_blocks = int(self.time_step - fee.last_updated[tkn])
        m = min(self.dynamic_fee_precision, num_blocks)
        x = fee.amplification * fee.volume_at_last_update[tkn] / self.liquidity[tkn]
        j_sum = 0
        w = 1 - self.oracles['price'].decay_factor
        w_j = 1
        for j in range(m):
            j_sum += w_j / (
                    1 + (fee.liquidity_at_last_update[tkn] - self.liquidity[tkn]) / self.liquidity[tkn] * w_j
            )
            w_j *= w
        w_term = (math.pow(w, m + 1) - math.pow(w, num_blocks + 1)) / self.oracles['price'].decay_factor
        delta = x * (j_sum + w_term) - num_blocks * fee.decay
        fee_value = min(max(fee.current[tkn] + delta, fee.minimum), fee.maximum)
        fee.current[tkn] = fee_value
        fee.last_updated[tkn] = self.time_step
        return fee_value

    def add_token(
            self,
            tkn: str,
            liquidity: float,
            lrna: float,
            shares: float,
            protocol_shares: float = 0,
            weight_cap: float = 1
    ):
        self.asset_list.append(tkn)
        self.liquidity[tkn] = liquidity
        self.lrna[tkn] = lrna
        self.shares[tkn] = shares
        self.protocol_shares[tkn] = protocol_shares or shares
        self.weight_cap[tkn] = weight_cap
        if hasattr(self, '_lrna_fee'):
            if tkn not in self._lrna_fee.current: self._lrna_fee.current[tkn] = self._lrna_fee.minimum
            if tkn not in self._asset_fee.current: self._asset_fee.current[tkn] = self._asset_fee.minimum
            if tkn not in self._lrna_fee.liquidity_at_last_update: self._lrna_fee.liquidity_at_last_update[tkn] = liquidity
            if tkn not in self._asset_fee.liquidity_at_last_update: self._asset_fee.liquidity_at_last_update[tkn] = liquidity
            if tkn not in self._lrna_fee.volume_at_last_update: self._lrna_fee.volume_at_last_update[tkn] = 0
            if tkn not in self._asset_fee.volume_at_last_update: self._asset_fee.volume_at_last_update[tkn] = 0
            if tkn not in self._lrna_fee.last_updated: self._lrna_fee.last_updated[tkn] = 0
            if tkn not in self._asset_fee.last_updated: self._asset_fee.last_updated[tkn] = 0
        if hasattr(self, 'current_block'):
            self.current_block.price[tkn] = self.lrna[tkn] / self.liquidity[tkn]
            self.current_block.liquidity[tkn] = self.liquidity[tkn]
            self.current_block.lps[tkn] = 0
            self.current_block.withdrawals[tkn] = 0
            self.current_block.volume_in[tkn] = 0
            self.current_block.volume_out[tkn] = 0
        for oracle in self.oracles.values() if self.oracles else []:
            oracle.liquidity[tkn] = self.liquidity[tkn]
            oracle.price[tkn] = self.lrna[tkn] / self.liquidity[tkn]
            oracle.volume_in[tkn] = 0
            oracle.volume_out[tkn] = 0
        return self

    def remove_token(self, tkn: str):
        self.asset_list.remove(tkn)
        return self

    def update(self):
        # update oracles
        self.current_block.price['HDX'] = self.lrna['HDX'] / self.liquidity['HDX']

        for name, oracle in self.oracles.items():
            oracle.update(self.current_block)

        # update fees
        self._lrna_fee.update(
            time_step=self.time_step,
            volume={
                tkn: (self.oracles['price'].volume_in[tkn] - self.oracles['price'].volume_out[tkn])
                for tkn in self._lrna_fee.current
            },
            liquidity=self.oracles['price'].liquidity
        )
        self._asset_fee.update(
            time_step=self.time_step,
            volume={
                tkn: (self.oracles['price'].volume_out[tkn] - self.oracles['price'].volume_in[tkn])
                for tkn in self._asset_fee.current
            },
            liquidity=self.oracles['price'].liquidity
        )

        # update current block
        self.time_step += 1
        self.current_block = Block(self)

        self.fail = ''
        if self.update_function:
            self.update_function(self)

        return self

    @property
    def lrna_total(self):
        return sum(self.lrna.values())

    @property
    def total_value_locked(self):
        if self.stablecoin is None:
            raise ValueError('No stablecoin defined')
        return self.liquidity[self.stablecoin] * self.lrna_total / self.lrna[self.stablecoin]

    def sell_limit(self, tkn_buy: str, tkn_sell: str):
        return float('inf')

    def buy_limit(self, tkn_buy: str, tkn_sell: str):
        if tkn_buy not in self.liquidity:
            return 0
        return self.liquidity[tkn_buy] * (1 - self.asset_fee(tkn_buy))

    def copy(self):
        copy_state = copy.deepcopy(self)
        copy_state.fail = ''
        return copy_state

    def archive(self):
        return OmnipoolArchiveState(self)

    def __repr__(self):
        # don't go overboard with the precision here
        precision = 12
        lrna = {tkn: round(self.lrna[tkn], precision) for tkn in self.lrna}
        lrna_total = round(self.lrna_total, precision)
        liquidity = {tkn: round(self.liquidity[tkn], precision) for tkn in self.liquidity}
        weight_cap = {tkn: round(self.weight_cap[tkn], precision) for tkn in self.weight_cap}
        usd_prices = (
            {tkn: round(self.usd_price(tkn), precision) for tkn in self.asset_list}
            if self.stablecoin is not None
            else {tkn: 'N/A' for tkn in self.asset_list}
        )
        newline = '\n'
        return (
            f'Omnipool: {self.unique_id}\n'
            f'********************************\n'
            f'tvl cap: {self.tvl_cap}\n'
            f'lrna fee:\n\n'
            f'{newline.join([f"    {tkn}: {self.last_lrna_fee[tkn]}" for tkn in self.asset_list])}\n\n'
            f'asset fee:\n\n'
            f'{newline.join([f"    {tkn}: {self.last_fee[tkn]}" for tkn in self.asset_list])}\n\n'
            f'asset pools: (\n\n'
        ) + '\n'.join(
            [(
                    f'    *{tkn}*\n'
                    f'    asset quantity: {liquidity[tkn]}\n'
                    f'    lrna quantity: {lrna[tkn]}\n'
                    f'    USD price: {usd_prices[tkn]}\n' +
                    # f'    tvl: ${lrna[tkn] * liquidity[self.stablecoin] / lrna[self.stablecoin]}\n'
                    f'    weight: {lrna[tkn]}/{lrna_total} ({lrna[tkn] / lrna_total})\n'
                    f'    weight cap: {weight_cap[tkn]}\n'
                    f'    total shares: {self.shares[tkn]}\n'
                    f'    protocol shares: {self.protocol_shares[tkn]}\n'
            ) for tkn in self.asset_list]
        ) + '\n)\n' + f'sub pools: (\n\n    ' + ')\n(\n'.join(
            [
                '\n    '.join(pool_desc.split('\n'))
                for pool_desc in
                [repr(pool) for pool in self.sub_pools.values()]
            ]
        ) + '\n)\n' + f'oracles: (\n' + '\n\n'.join([
            f'    name: {name}\n    length: {oracle.length}\n'
            for name, oracle in self.oracles.items()
        ]) + f'\n)\n\nerror message: {self.fail or "None"}'

    def calculate_sell_from_buy(
            self,
            tkn_buy: str,
            tkn_sell: str,
            buy_quantity: float
    ):
        """
        Given a buy quantity, calculate the effective price, so we can execute it as a sell
        """
        asset_fee = self.asset_fee(tkn_buy)
        if buy_quantity >= self.liquidity[tkn_buy] * (1 - asset_fee):
            return float('inf')
        delta_Qj = self.lrna[tkn_buy] * buy_quantity / (
                self.liquidity[tkn_buy] * (1 - asset_fee) - buy_quantity)
        lrna_fee = self.lrna_fee(tkn_sell)
        # lrna_fee = self.last_lrna_fee[tkn_sell]
        delta_Qi = -delta_Qj / (1 - lrna_fee)
        if -delta_Qi >= self.lrna[tkn_sell]:
            return float('inf')
        delta_Ri = -self.liquidity[tkn_sell] * delta_Qi / (self.lrna[tkn_sell] + delta_Qi)
        return delta_Ri

    def calculate_buy_from_sell(
            self,
            tkn_buy: str,
            tkn_sell: str,
            sell_quantity: float
    ):
        """
        Given a sell quantity, calculate the effective price, so we can execute it as a buy
        """
        delta_Ri = sell_quantity
        delta_Qi = self.lrna[tkn_sell] * -delta_Ri / (self.liquidity[tkn_sell] + delta_Ri)
        asset_fee = self.asset_fee(tkn_buy)
        lrna_fee = self.lrna_fee(tkn_sell)
        delta_Qt = -delta_Qi * (1 - lrna_fee)
        delta_Rj = self.liquidity[tkn_buy] * -delta_Qt / (self.lrna[tkn_buy] + delta_Qt) * (1 - asset_fee)
        return -delta_Rj

    def buy_spot(self, tkn_buy: str, tkn_sell: str, fee: float = None):
        if fee is None:
            fee = {}
            if tkn_buy == 'LRNA':
                fee['asset'] = 0
            elif tkn_sell == 'LRNA':
                fee['lrna'] = 0
            elif tkn_sell not in self.asset_list:
                for pool in self.sub_pools.values():
                    if tkn_sell in pool.asset_list:
                        fee['lrna'] = pool.trade_fee
                        break
            if tkn_buy == 'LRNA':
                raise ValueError('cannot buy LRNA from Omnipool')
            elif tkn_buy not in self.asset_list:
                for pool in self.sub_pools.values():
                    if tkn_buy in pool.asset_list:
                        fee['asset'] = pool.trade_fee
                        break
            if 'lrna' not in fee:
                fee['lrna'] = self.lrna_fee(tkn_sell)
            if 'asset' not in fee:
                fee['asset'] = self.asset_fee(tkn_buy)
        elif not isinstance(fee, dict):
            fee = {
                'lrna': fee,
                'asset': fee
            }
        if tkn_buy not in self.asset_list:
            return 0
        elif tkn_sell not in self.asset_list + ['LRNA']:
            return 0
        else:
            return self.price(tkn_buy, tkn_sell) / (1 - fee['lrna']) / (1 - fee['asset'])

    def sell_spot(self, tkn_sell: str, tkn_buy: str, fee: float = None):
        if fee is None:
            fee = {}
            if tkn_buy == 'LRNA':
                fee['asset'] = 0
            elif tkn_sell == 'LRNA':
                fee['lrna'] = 0
            elif tkn_sell not in self.asset_list:
                for pool in self.sub_pools.values():
                    if tkn_sell in pool.asset_list:
                        fee['lrna'] = pool.trade_fee
                        break
            if tkn_buy == 'LRNA':
                raise ValueError('cannot buy LRNA from Omnipool')
            elif tkn_buy not in self.asset_list:
                for pool in self.sub_pools.values():
                    if tkn_buy in pool.asset_list:
                        fee['asset'] = pool.trade_fee
                        break
            if 'lrna' not in fee:
                fee['lrna'] = self.lrna_fee(tkn_sell)
            if 'asset' not in fee:
                fee['asset'] = self.asset_fee(tkn_buy)
        elif not isinstance(fee, dict):
            fee = {
                'lrna': fee,
                'asset': fee
            }
        if tkn_buy not in self.asset_list:
            return 0
        elif tkn_sell not in self.asset_list + ["LRNA"]:
            return 0
        else:
            return self.price(tkn_sell, tkn_buy) * (1 - fee['lrna']) * (1 - fee['asset'])

    def get_sub_pool(self, tkn: str):
        # if asset in not in omnipool, return the ID of the sub_pool where it can be found
        if tkn in self.asset_list:
            return ''
        else:
            for pool in self.sub_pools.values():
                if tkn in pool.asset_list:
                    return pool.unique_id

    def swap(
            self,
            agent: Agent,
            tkn_buy: str, tkn_sell: str,
            buy_quantity: float = 0,
            sell_quantity: float = 0
    ):
        """
        execute swap in place (modify and return self and agent)
        all swaps, LRNA, sub-pool, and asset swaps, are executed through this function
        """
        old_buy_liquidity = self.liquidity[tkn_buy] if tkn_buy in self.liquidity else 0
        old_sell_liquidity = self.liquidity[tkn_sell] if tkn_sell in self.liquidity else 0

        if tkn_buy not in self.asset_list + ['LRNA'] or tkn_sell not in self.asset_list + ['LRNA']:
            # note: this default routing behavior assumes that an asset will only exist in one place in the omnipool
            return_val = self.stable_swap(
                agent=agent,
                sub_pool_buy_id=self.get_sub_pool(tkn=tkn_buy),
                sub_pool_sell_id=self.get_sub_pool(tkn=tkn_sell),
                tkn_sell=tkn_sell, tkn_buy=tkn_buy,
                buy_quantity=buy_quantity,
                sell_quantity=sell_quantity
            )

        elif tkn_sell == 'LRNA':
            return_val = self._lrna_swap(agent, buy_quantity, -sell_quantity, tkn_buy)
        elif tkn_buy == 'LRNA':
            return_val = self._lrna_swap(agent, -sell_quantity, buy_quantity, tkn_sell)

        elif buy_quantity and not sell_quantity:
            # back into correct delta_Ri, then execute sell
            delta_Ri = self.calculate_sell_from_buy(tkn_buy, tkn_sell, buy_quantity)
            if delta_Ri < 0:
                return self.fail_transaction(f'insufficient LRNA in {tkn_sell}')
            if delta_Ri == float('inf'):
                return self.fail_transaction('not enough liquidity in sell pool to buy that much')
            # including both buy_quantity and sell_quantity potentially introduces a 'hack'
            # where you could include both and *not* have them match, but we're not worried about that
            # because this is not a production environment. Just don't do it.
            return self.swap(
                agent=agent,
                tkn_buy=tkn_buy,
                tkn_sell=tkn_sell,
                buy_quantity=buy_quantity,
                sell_quantity=delta_Ri
            )
        else:
            # basic Omnipool swap
            i = tkn_sell
            j = tkn_buy
            delta_Ri = sell_quantity
            if delta_Ri <= 0:
                return self.fail_transaction('sell amount must be greater than zero')
            if not agent.validate_holdings(i, delta_Ri):
                return self.fail_transaction(f"Agent doesn't have enough {i}")

            # get the fees we will be using
            asset_fee = self.asset_fee(tkn_buy)
            lrna_fee = self.lrna_fee(tkn_sell)
            min_lrna_fee = self._lrna_fee.minimum
            # also update both fees for each asset, because that's what they do in production
            self.asset_fee(tkn_sell)
            self.lrna_fee(tkn_buy)

            delta_Qi = self.lrna[tkn_sell] * -delta_Ri / (self.liquidity[tkn_sell] + delta_Ri)
            lrna_fee_total = -delta_Qi * lrna_fee
            lrna_fee_burn = self.lrna_fee_burn * lrna_fee_total
            fee_deposit = lrna_fee_total - lrna_fee_burn
            delta_Qt = -delta_Qi - lrna_fee_total
            delta_Qm = (self.lrna[tkn_buy] + delta_Qt) * delta_Qt * asset_fee / self.lrna[
                tkn_buy] * self.lrna_mint_pct
            delta_Qj = delta_Qt + delta_Qm
            delta_Rj = self.liquidity[tkn_buy] * -delta_Qt / (self.lrna[tkn_buy] + delta_Qt) * (1 - asset_fee)
            delta_QH = 0  # -lrna_fee * delta_Qi
            if self.lrna_fee_destination:
                self.lrna_fee_destination.holdings['LRNA'] += fee_deposit

            # per-block trade limits
            if (
                    -delta_Rj - self.current_block.volume_in[tkn_buy] + self.current_block.volume_out[tkn_buy]
                    > self.trade_limit_per_block * self.current_block.liquidity[tkn_buy]
            ):
                return self.fail_transaction(
                    f'{self.trade_limit_per_block * 100}% per block trade limit exceeded in {tkn_buy}.'
                )
            elif (
                    delta_Ri + self.current_block.volume_in[tkn_sell] - self.current_block.volume_out[tkn_sell]
                    > self.trade_limit_per_block * self.current_block.liquidity[tkn_sell]
            ):
                return self.fail_transaction(
                    f'{self.trade_limit_per_block * 100}% per block trade limit exceeded in {tkn_sell}.'
                )
            self.lrna[i] += delta_Qi
            self.lrna[j] += delta_Qj
            self.liquidity[i] += delta_Ri
            self.liquidity[j] += -buy_quantity or delta_Rj
            self.lrna['HDX'] += delta_QH

            agent.remove(i, delta_Ri)
            agent.add(j, buy_quantity or -delta_Rj)

            return_val = self

        # update oracle
        if tkn_buy in self.asset_list:
            buy_quantity = old_buy_liquidity - self.liquidity[tkn_buy]
            self.current_block.volume_out[tkn_buy] += buy_quantity
            self.current_block.price[tkn_buy] = self.lrna[tkn_buy] / self.liquidity[tkn_buy]
        if tkn_sell in self.asset_list:
            sell_quantity = self.liquidity[tkn_sell] - old_sell_liquidity
            self.current_block.volume_in[tkn_sell] += sell_quantity
            self.current_block.price[tkn_sell] = self.lrna[tkn_sell] / self.liquidity[tkn_sell]
        return return_val

    def _lrna_swap(
            self,
            agent: Agent,
            delta_ra: float = 0,
            delta_qa: float = 0,
            tkn: str = '',
            modify_imbalance: bool = True
    ):
        """
        Execute LRNA swap in place (modify and return)
        """
        asset_fee = self.asset_fee(tkn)
        lrna_fee = self.lrna_fee(tkn)

        if delta_qa < 0:
            # selling LRNA
            if not agent.validate_holdings('LRNA', -delta_qa):
                return self.fail_transaction('Agent has insufficient lrna')
            delta_ra = -self.liquidity[tkn] * delta_qa / (-delta_qa + self.lrna[tkn]) * (1 - asset_fee)

            delta_qm = asset_fee * (-delta_qa) / self.lrna[tkn] * (self.lrna[tkn] - delta_qa) * self.lrna_mint_pct
            delta_q = delta_qm - delta_qa

            self.lrna[tkn] += delta_q
            self.liquidity[tkn] += -delta_ra

        elif delta_ra > 0:
            # buying asset
            if -delta_ra + self.liquidity[tkn] <= 0:
                return self.fail_transaction('insufficient assets in pool')
            denom = (self.liquidity[tkn] * (1 - asset_fee) - delta_ra)
            delta_qa = -self.lrna[tkn] * delta_ra / denom
            delta_qm = -asset_fee * (1 - asset_fee) * (self.liquidity[tkn] / denom) * delta_qa * self.lrna_mint_pct
            delta_q = -delta_qa + delta_qm

            self.lrna[tkn] += delta_q
            self.liquidity[tkn] -= delta_ra

        # buying LRNA
        elif delta_qa > 0:
            # buying LRNA
            lrna_fee_total = delta_qa / (1 - lrna_fee) - delta_qa
            delta_qi = -delta_qa - lrna_fee_total
            lrna_fee_burn = self.lrna_fee_burn * lrna_fee_total
            fee_deposit = lrna_fee_total - lrna_fee_burn
            if delta_qi + self.lrna[tkn] <= 0:
                return self.fail_transaction('insufficient lrna in pool')
            delta_ra = -self.liquidity[tkn] * -delta_qi / (delta_qi + self.lrna[tkn])
            if not agent.validate_holdings(tkn, -delta_ra):
                return self.fail_transaction('Agent has insufficient assets')
            self.lrna[tkn] += delta_qi  # burn the LRNA fee
            self.liquidity[tkn] += -delta_ra
            if self.lrna_fee_destination:
                self.lrna_fee_destination.holdings['LRNA'] += fee_deposit

        elif delta_ra < 0:
            # selling asset
            if not agent.validate_holdings(tkn, delta_ra):
                return self.fail_transaction('agent has insufficient assets')
            delta_qi = self.lrna[tkn] * delta_ra / (self.liquidity[tkn] - delta_ra)
            lrna_fee_total = -delta_qi * lrna_fee
            lrna_fee_burn = lrna_fee_total * self.lrna_fee_burn
            fee_deposit = lrna_fee_total - lrna_fee_burn
            delta_qa = -delta_qi - lrna_fee_total
            self.lrna[tkn] += delta_qi
            self.liquidity[tkn] -= delta_ra
            if self.lrna_fee_destination:
                self.lrna_fee_destination.holdings['LRNA'] += fee_deposit

        else:
            return self.fail_transaction('All deltas are zero.')

        if delta_qa > 0:
            agent.add('LRNA', delta_qa)
            agent.remove(tkn, -delta_ra)
        else:
            agent.remove('LRNA', -delta_qa)
            agent.add(tkn, delta_ra)

        return self

    def stable_swap(
            self,
            agent: Agent,
            tkn_sell: str, tkn_buy: str,
            sub_pool_buy_id: str = "",
            sub_pool_sell_id: str = "",
            buy_quantity: float = 0,
            sell_quantity: float = 0
    ):
        if tkn_sell == 'LRNA':
            if buy_quantity:
                sub_pool = self.sub_pools[sub_pool_buy_id]
                # buy a specific quantity of a stableswap asset using LRNA
                shares_needed = sub_pool.calculate_withdrawal_shares(tkn_remove=tkn_buy, quantity=buy_quantity)
                self._lrna_swap(agent, delta_ra=shares_needed, tkn=sub_pool.unique_id)
                if self.fail:
                    # if the swap failed, the transaction failed.
                    return self.fail_transaction(self.fail)
                sub_pool.withdraw_asset(agent, buy_quantity, tkn_buy)
                return self
            elif sell_quantity:
                sub_pool = self.sub_pools[sub_pool_buy_id]
                agent_shares = agent.holdings[sub_pool.unique_id]
                self.swap(
                    agent=agent,
                    tkn_buy=sub_pool.unique_id, tkn_sell='LRNA',
                    sell_quantity=sell_quantity
                )
                if self.fail:
                    # if the swap failed, the transaction failed.
                    return self.fail_transaction(self.fail)
                delta_shares = agent.holdings[sub_pool.unique_id] - agent_shares
                sub_pool.remove_liquidity(agent, delta_shares, tkn_buy)
                return self

        elif sub_pool_sell_id and tkn_buy in self.asset_list:
            sub_pool: StableSwapPoolState = self.sub_pools[sub_pool_sell_id]
            if sell_quantity:
                # sell a stableswap asset for an omnipool asset
                agent_shares = agent.holdings[sub_pool.unique_id] if sub_pool.unique_id in agent.holdings else 0
                sub_pool.add_liquidity(agent, sell_quantity, tkn_sell)
                if self.fail:
                    # the transaction failed.
                    return self.fail_transaction(self.fail)
                delta_shares = agent.holdings[sub_pool.unique_id] - agent_shares
                self.swap(
                    agent=agent,
                    tkn_buy=tkn_buy,
                    tkn_sell=sub_pool.unique_id,
                    sell_quantity=delta_shares
                )
                return self
            elif buy_quantity:
                # buy an omnipool asset with a stableswap asset
                sell_shares = self.calculate_sell_from_buy(tkn_buy, sub_pool.unique_id, buy_quantity)
                if sell_shares < 0:
                    return self.fail_transaction("Not enough liquidity in the stableswap/LRNA pool.")
                sub_pool.buy_shares(agent, sell_shares, tkn_sell)
                if sub_pool.fail:
                    return self.fail_transaction(sub_pool.fail)
                self.swap(agent, tkn_buy, sub_pool.unique_id, buy_quantity)
                return self

        elif sub_pool_buy_id and tkn_sell in self.asset_list:
            sub_pool: StableSwapPoolState = self.sub_pools[sub_pool_buy_id]
            if buy_quantity:
                # buy a stableswap asset with an omnipool asset
                shares_traded = sub_pool.calculate_withdrawal_shares(tkn_buy, buy_quantity)

                # buy shares in the subpool
                self.swap(agent, tkn_buy=sub_pool.unique_id, tkn_sell=tkn_sell, buy_quantity=shares_traded)
                if self.fail:
                    # if the swap failed, the transaction failed.
                    return self.fail_transaction(self.fail)
                # withdraw the shares for the desired token
                sub_pool.withdraw_asset(agent, quantity=buy_quantity, tkn_remove=tkn_buy)
                if sub_pool.fail:
                    return self.fail_transaction(sub_pool.fail)
                return self
            elif sell_quantity:
                # sell an omnipool asset for a stableswap asset
                agent_shares = agent.holdings[sub_pool.unique_id] if sub_pool.unique_id in agent.holdings else 0
                self.swap(
                    agent=agent,
                    tkn_buy=sub_pool.unique_id,
                    tkn_sell=tkn_sell,
                    sell_quantity=sell_quantity
                )
                delta_shares = agent.holdings[sub_pool.unique_id] - agent_shares
                if self.fail:
                    return self.fail_transaction(self.fail)
                sub_pool.remove_liquidity(
                    agent=agent, shares_removed=delta_shares, tkn_remove=tkn_buy
                )
                return self
        elif sub_pool_buy_id and sub_pool_sell_id:
            # trade between two subpools
            pool_buy: StableSwapPoolState = self.sub_pools[sub_pool_buy_id]
            pool_sell: StableSwapPoolState = self.sub_pools[sub_pool_sell_id]
            if buy_quantity:
                # buy enough shares of tkn_sell to afford buy_quantity worth of tkn_buy
                shares_bought = pool_buy.calculate_withdrawal_shares(tkn_buy, buy_quantity)
                if shares_bought > pool_buy.liquidity[tkn_buy]:
                    return self.fail_transaction(f'Not enough liquidity in {pool_buy.unique_id}: {tkn_buy}.')
                shares_sold = self.calculate_sell_from_buy(
                    tkn_buy=pool_buy.unique_id,
                    tkn_sell=pool_sell.unique_id,
                    buy_quantity=shares_bought
                )
                pool_sell.buy_shares(
                    agent=agent, quantity=shares_sold,
                    tkn_add=tkn_sell
                )
                if pool_sell.fail:
                    return self.fail_transaction(pool_sell.fail)
                self.swap(
                    agent=agent,
                    tkn_buy=pool_buy.unique_id, tkn_sell=pool_sell.unique_id,
                    buy_quantity=shares_bought
                )
                if self.fail:
                    return self.fail_transaction(self.fail)
                pool_buy.withdraw_asset(
                    agent=agent, quantity=buy_quantity,
                    tkn_remove=tkn_buy, fail_on_overdraw=False
                )
                if pool_buy.fail:
                    return self.fail_transaction(pool_buy.fail)

                # if all three parts succeeded, then we're good!
                return self
            elif sell_quantity:
                agent_sell_holdings = agent.holdings[sub_pool_sell_id] if sub_pool_sell_id in agent.holdings else 0
                pool_sell.add_liquidity(
                    agent=agent, quantity=sell_quantity, tkn_add=tkn_sell
                )
                if pool_sell.fail:
                    return self.fail_transaction(pool_sell.fail)
                delta_sell_holdings = agent.holdings[sub_pool_sell_id] - agent_sell_holdings
                agent_buy_holdings = agent.holdings[sub_pool_buy_id] if sub_pool_buy_id in agent.holdings else 0
                self.swap(
                    agent=agent,
                    tkn_buy=pool_buy.unique_id, tkn_sell=pool_sell.unique_id,
                    sell_quantity=delta_sell_holdings
                )
                if self.fail:
                    return self.fail_transaction(self.fail)
                delta_buy_holdings = agent.holdings[sub_pool_buy_id] - agent_buy_holdings
                pool_buy.remove_liquidity(
                    agent=agent, shares_removed=delta_buy_holdings, tkn_remove=tkn_buy
                )
                if pool_buy.fail:
                    return self.fail_transaction(pool_buy.fail)
                return self
        else:
            raise ValueError('buy_quantity or sell_quantity must be specified.')

    def create_sub_pool(
            self,
            tkns_migrate: dict[str: float] or list[str],
            unique_id: str,
            amplification: float,
            trade_fee: float = 0
    ):
        if isinstance(tkns_migrate, list):
            tkns_migrate = {tkn: self.liquidity[tkn] for tkn in tkns_migrate}

        new_sub_pool = StableSwapPoolState(
            tokens=tkns_migrate,
            amplification=amplification,
            unique_id=unique_id,
            trade_fee=trade_fee
        )
        new_sub_pool.conversion_metrics = {
            tkn: {
                'price': self.lrna[tkn] / self.liquidity[tkn],
                'old_shares': self.shares[tkn] * tkns_migrate[tkn] / self.liquidity[tkn],
                'omnipool_shares': self.liquidity[tkn] * tkns_migrate[tkn] / self.liquidity[tkn],
                'subpool_shares': self.liquidity[tkn] * tkns_migrate[tkn] / self.liquidity[tkn]
            } for tkn in tkns_migrate
        }
        new_sub_pool.shares = sum(
            [self.liquidity[tkn] * tkns_migrate[tkn] / self.liquidity[tkn] for tkn in tkns_migrate])
        self.sub_pools[unique_id] = new_sub_pool
        self.add_token(
            unique_id,
            liquidity=sum([self.liquidity[tkn] * tkns_migrate[tkn] / self.liquidity[tkn] for tkn in tkns_migrate]),
            shares=sum([self.liquidity[tkn] * tkns_migrate[tkn] / self.liquidity[tkn] for tkn in tkns_migrate]),
            lrna=sum([self.lrna[tkn] * tkns_migrate[tkn] / self.liquidity[tkn] for tkn in tkns_migrate]),
            protocol_shares=sum([
                self.lrna[tkn] * tkns_migrate[tkn] / self.liquidity[tkn] * self.protocol_shares[tkn] / self.shares[tkn]
                for tkn in tkns_migrate
            ])
        )

        # remove assets from Omnipool
        for tkn in tkns_migrate:
            self.lrna[tkn] -= self.lrna[tkn] * tkns_migrate[tkn] / self.liquidity[tkn]
            self.liquidity[tkn] -= tkns_migrate[tkn]
            if self.liquidity[tkn] == 0:
                self.asset_list.remove(tkn)
        return self

    def migrate_asset(self, tkn_migrate: str, sub_pool_id: str):
        """
        Move an asset from the Omnipool into a stableswap subpool.
        """
        sub_pool: StableSwapPoolState = self.sub_pools[sub_pool_id]
        s = sub_pool.unique_id
        i = tkn_migrate
        if tkn_migrate in sub_pool.liquidity:
            raise AssertionError('Assets should only exist in one place in the Omnipool at a time.')
        sub_pool.liquidity[i] = self.liquidity[i]
        self.protocol_shares[s] += (
                self.shares[s] * self.lrna[i] / self.lrna[s] * self.protocol_shares[i] / self.shares[i]
        )

        sub_pool.conversion_metrics[i] = {
            'price': self.lrna[i] / self.lrna[s] * sub_pool.shares / self.liquidity[i],
            'old_shares': self.shares[i],
            'omnipool_shares': self.lrna[i] * self.shares[s] / self.lrna[s],
            'subpool_shares': self.lrna[i] * sub_pool.shares / self.lrna[s]
        }
        for j in range(len(sub_pool.peg)):
            assert sub_pool.peg[j] == 1  # non-1 peg not supported for subpools
            assert sub_pool.peg_target[j] == 1
        sub_pool.peg.append(1)
        sub_pool.peg_target.append(1)


        self.shares[s] += self.lrna[i] * self.shares[s] / self.lrna[s]
        self.liquidity[s] += self.lrna[i] * sub_pool.shares / self.lrna[s]
        sub_pool.shares += self.lrna[i] * sub_pool.shares / self.lrna[s]
        self.lrna[s] += self.lrna[i]

        # remove asset from omnipool and add it to subpool
        self.lrna[i] = 0
        self.liquidity[i] = 0
        self.asset_list.remove(i)
        sub_pool.asset_list.append(i)
        return self

    def migrate_lp(
            self,
            agent: Agent,
            sub_pool_id: str,
            tkn_migrate: str
    ):
        sub_pool = self.sub_pools[sub_pool_id]
        conversions = sub_pool.conversion_metrics[tkn_migrate]
        old_pool_id = (self.unique_id, tkn_migrate)
        old_share_price = agent.share_prices[old_pool_id]
        # TODO: maybe this is an edge case or not allowed, but what if the agent already has a share price locked in?
        # ex., maybe they have LPed into the new subpool after their asset was migrated,
        # but before they had migrated their own position
        agent.share_prices[sub_pool_id] = old_share_price / conversions['price']
        if sub_pool_id not in agent.holdings:
            agent.holdings[sub_pool_id] = 0
        agent.holdings[sub_pool_id] += (
                agent.holdings[old_pool_id] / conversions['old_shares'] * conversions['omnipool_shares']
        )
        self.liquidity[sub_pool_id] += (
                agent.holdings[old_pool_id] / conversions['old_shares'] * conversions['subpool_shares']
        )
        agent.holdings[old_pool_id] = 0

        return self

    def calculate_remove_liquidity(self, agent: Agent, quantity: float = None, tkn_remove: str = None,
                                   nft_id: str = None):
        """
        If quantity is specified and nft_id is specified, remove specified quantity of shares from specified position.
        If quantity is specified and nft_id is unspecified, remove specified quantity of shares from holdings.
        If quantity is unspecified and nft_id is specified, remove specified position.
        If quantity is unspecified and nft_id is unspecified, remove all liquidity.
        """

        if tkn_remove is None:
            if nft_id is None:
                raise AssertionError('tkn_remove must be specified if nft_id is not provided.')
            else:
                tkn_remove = agent.nfts[nft_id].tkn

        delta_qa, delta_r, delta_q, delta_s, delta_b, nft_ids = 0, 0, 0, 0, 0, []
        if quantity is not None:
            if nft_id is None:  # remove specified quantity of shares from holdings
                k = (self.unique_id, tkn_remove)
                delta_qa, delta_r, delta_q, delta_s, delta_b = self._calculate_remove_one_position(
                    quantity=quantity, tkn_remove=tkn_remove, share_price=agent.share_prices[k]
                )
            else:  # remove specified quantity of shares from specified position
                delta_qa, delta_r, delta_q, delta_s, delta_b = self._calculate_remove_one_position(
                    quantity=quantity, tkn_remove=tkn_remove, share_price=agent.nfts[nft_id].price
                )
                nft_ids = [nft_id]
        else:
            if nft_id is not None:  # remove specified position
                delta_qa, delta_r, delta_q, delta_s, delta_b = self._calculate_remove_one_position(
                    quantity=agent.nfts[nft_id].shares, tkn_remove=tkn_remove, share_price=agent.nfts[nft_id].price
                )
                nft_ids = [nft_id]
            else:  # remove all liquidity
                for nft_id in agent.nfts:
                    nft = agent.nfts[nft_id]
                    if isinstance(nft, OmnipoolLiquidityPosition):
                        if nft.pool_id == self.unique_id and nft.tkn == tkn_remove:
                            nft_ids.append(nft_id)
                            dqa, dr, dq, ds, db = self._calculate_remove_one_position(
                                quantity=nft.shares, tkn_remove=tkn_remove, share_price=nft.price
                            )
                            delta_qa += dqa
                            delta_r += dr
                            delta_q += dq
                            delta_s += ds
                            delta_b += db
                if (self.unique_id, tkn_remove) in agent.holdings:
                    dqa, dr, dq, ds, db = self._calculate_remove_one_position(
                        quantity=agent.holdings[(self.unique_id, tkn_remove)], tkn_remove=tkn_remove,
                        share_price=agent.share_prices[(self.unique_id, tkn_remove)]
                    )
                    delta_qa += dqa
                    delta_r += dr
                    delta_q += dq
                    delta_s += ds
                    delta_b += db
        return delta_qa, delta_r, delta_q, delta_s, delta_b, nft_ids

    def _calculate_remove_one_position(self, quantity, tkn_remove, share_price):
        """
        calculated the pool and agent deltas for removing liquidity from a sub pool
        return as a tuple in this order:
        delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l

        delta_qa (agent LRNA)
        delta_r (pool/agent liquidity)
        delta_q (pool LRNA)
        delta_s (pool shares)
        delta_b (protocol shares)
        delta_l (LRNA imbalance)
        """
        k = (self.unique_id, tkn_remove)

        quantity = -abs(quantity)
        assert quantity <= 0, f"delta_S cannot be positive: {quantity}"
        if tkn_remove not in self.asset_list:
            raise AssertionError(f"Invalid token name: {tkn_remove}")

        # determine if they should get some LRNA back as well as the asset they invested
        piq = self.lrna_price(tkn_remove)
        p0 = share_price
        mult = (piq - p0) / (piq + p0)

        # Share update
        delta_b = max(mult * quantity, 0)
        delta_s = quantity + delta_b

        # Token amounts update
        delta_q = self.lrna[tkn_remove] / self.shares[tkn_remove] * delta_s
        delta_r = delta_q / piq

        if piq > p0:  # prevents rounding errors
            delta_qa = -piq * (
                    2 * piq / (piq + p0) * quantity / self.shares[tkn_remove]
                    * self.liquidity[tkn_remove] - delta_r
            )
        else:
            delta_qa = 0

        if hasattr(self, 'withdrawal_fee') and self.withdrawal_fee > 0:
            # calculate withdraw fee
            diff = abs(self.oracles['price'].price[tkn_remove] - piq) / self.oracles['price'].price[tkn_remove]
            fee = max(min(diff, 1), self.min_withdrawal_fee)

            delta_r *= 1 - fee
            delta_qa *= 1 - fee
            delta_q *= 1 - fee

        # L update: LRNA fees to be burned before they will start to accumulate again

        return delta_qa, delta_r, delta_q, delta_s, delta_b

    def add_liquidity(
            self,
            agent: Agent = None,
            quantity: float = 0,
            tkn_add: str = '',
            nft_id: str = None
    ):
        """Compute new state after liquidity addition"""

        if quantity <= 0:
            return self.fail_transaction('Quantity must be non-negative.')

        delta_Q = self.lrna_price(tkn_add) * quantity

        if nft_id is None and (self.unique_id, tkn_add) in agent.holdings:
            return self.fail_transaction(
                'Agent already has liquidity in this pool. Try using nft_id input.'
            )

        if nft_id is not None and nft_id in agent.nfts:
            raise AssertionError('Agent already has an NFT with this ID.')

        if not agent.validate_holdings(tkn_add, quantity):
            return self.fail_transaction(
                f'Agent has insufficient funds ({agent.holdings[tkn_add]} < {quantity}).'
            )

        if (self.lrna[tkn_add] + delta_Q) / (self.lrna_total + delta_Q) > self.weight_cap[tkn_add]:
            return self.fail_transaction(
                'Transaction rejected because it would exceed the weight cap in pool[{i}].'
            )

        if self.tvl_cap < float('inf'):
            if (self.total_value_locked() + quantity * self.usd_price(tkn_add)) > self.tvl_cap:
                return self.fail_transaction('Transaction rejected because it would exceed the TVL cap.')

        # assert quantity > 0, f"delta_R must be positive: {quantity}"
        if tkn_add not in self.asset_list:
            for sub_pool in self.sub_pools.values():
                if tkn_add in sub_pool.asset_list:
                    old_agent_holdings = agent.get_holdings(sub_pool.unique_id)
                    sub_pool.add_liquidity(
                        agent=agent,
                        quantity=quantity,
                        tkn_add=tkn_add
                    )
                    # deposit into the Omnipool
                    return self.add_liquidity(
                        agent=agent,
                        quantity=agent.get_holdings(sub_pool.unique_id) - old_agent_holdings,
                        tkn_add=sub_pool.unique_id,
                        nft_id=nft_id
                    )
            raise AssertionError(f"invalid value for i: {tkn_add}")
        else:
            # enforce upper limit on liquidity addition
            if quantity > self.max_lp_per_block * self.liquidity[tkn_add] - self.current_block.lps[tkn_add]:
                return self.fail_transaction(
                    'Transaction rejected because it would exceed the max LP per block.'
                )

        # Share update
        if self.shares[tkn_add]:
            shares_added = (delta_Q / self.lrna[tkn_add]) * self.shares[tkn_add]
        else:
            shares_added = delta_Q
        self.shares[tkn_add] += shares_added

        # LRNA add (mint)
        self.lrna[tkn_add] += delta_Q

        # Token amounts update
        self.liquidity[tkn_add] += quantity

        # agent update
        agent.remove(tkn_add, quantity)
        if nft_id is None:
            k = (self.unique_id, tkn_add)
            # shares go to provisioning agent
            agent.holdings[k] = shares_added
            # set price at which liquidity was added
            agent.share_prices[k] = self.lrna_price(tkn_add)
            agent.delta_r[k] = quantity
        else:
            lp_position = OmnipoolLiquidityPosition(tkn_add, self.lrna_price(tkn_add), shares_added, quantity,
                                                    self.unique_id)
            agent.nfts[nft_id] = lp_position

        # update block
        self.current_block.lps[tkn_add] += quantity
        # update fees
        self.asset_fee(tkn_add)
        self.lrna_fee(tkn_add)

        return self

    def remove_liquidity(self, agent: Agent, quantity: float = None, tkn_remove: str = '', nft_id: str = None):
        """
        Remove liquidity from a sub pool.
        If quantity is specified and nft_id is specified, remove specified quantity of shares from specified position.
        If quantity is specified and nft_id is unspecified, remove specified quantity of shares from holdings.
        If quantity is unspecified and nft_id is specified, remove specified position.
        If quantity is unspecified and nft_id is unspecified, remove all liquidity.
        """

        k = (self.unique_id, tkn_remove)
        if nft_id is not None:
            if nft_id not in agent.nfts:
                return self.fail_transaction('Agent does not have liquidity position with specified nft_id.')
            tkn_remove = agent.nfts[nft_id].tkn

        if tkn_remove not in self.asset_list:
            for sub_pool in self.sub_pools.values():
                if tkn_remove in sub_pool.asset_list:
                    sub_pool.remove_liquidity(
                        agent, quantity, tkn_remove
                    )
                    if sub_pool.fail:
                        return self.fail_transaction(sub_pool.fail)
                    else:
                        return self

            raise AssertionError(f"invalid value for tkn_remove: {tkn_remove}")

        if quantity == 0:
            return self
        # if nft_id is None and not agent.is_holding(k):
        #     return self.fail_transaction('Agent does not have liquidity in this pool.')
        if nft_id is None:
            if quantity is not None and not agent.validate_holdings(k, quantity):
                return self.fail_transaction('Agent does not have enough liquidity in this pool.')
        else:
            if nft_id not in agent.nfts:
                return self.fail_transaction('Agent does not have liquidity position with specified nft_id.')
            elif agent.nfts[nft_id].pool_id != self.unique_id:
                return self.fail_transaction('Specified position is wrong pool.')
            elif agent.nfts[nft_id].tkn != tkn_remove:
                return self.fail_transaction('Specified position is wrong asset.')
            elif quantity is not None and agent.nfts[nft_id].shares < quantity:
                return self.fail_transaction('Agent does not have enough shares in specified position.')

        if self.remove_liquidity_volatility_threshold and self.remove_liquidity_volatility_threshold < float('inf'):
            if self.oracles['price']:
                volatility = abs(
                    self.oracles['price'].price[tkn_remove] / self.current_block.price[tkn_remove] - 1
                )
                if volatility > self.remove_liquidity_volatility_threshold:
                    return self.fail_transaction(
                        f"Withdrawal rejected because the oracle volatility is too high: {volatility} > "
                        f"{self.remove_liquidity_volatility_threshold}"
                    )

        val = self.calculate_remove_liquidity(agent, quantity, tkn_remove, nft_id)
        delta_qa, delta_r, delta_q, delta_s, delta_b, nft_ids = val[:6]

        max_remove = (
                self.max_withdrawal_per_block * self.shares[tkn_remove] - self.current_block.withdrawals[tkn_remove]
        )
        if abs(delta_s) > max_remove:
            return self.fail_transaction(
                f"Transaction rejected because it would exceed the withdrawal limit: {abs(delta_s)} > {max_remove}",
                agent
            )

        if delta_r + self.liquidity[tkn_remove] < 0:
            return self.fail_transaction('Cannot remove more liquidity than exists in the pool.')

        self.liquidity[tkn_remove] += delta_r
        self.shares[tkn_remove] += delta_s
        self.protocol_shares[tkn_remove] += delta_b
        self.lrna[tkn_remove] += delta_q

        # distribute tokens to agent
        if delta_qa > 0:
            if 'LRNA' not in agent.holdings:
                agent.holdings['LRNA'] = 0
            agent.holdings['LRNA'] += delta_qa
        if tkn_remove not in agent.holdings:
            agent.holdings[tkn_remove] = 0
        agent.holdings[tkn_remove] -= delta_r

        # remove lp position(s)
        if nft_id is None:
            if quantity is not None:
                agent.holdings[k] -= quantity
                if agent.holdings[k] == 0:
                    agent.share_prices[k] = 0
            else:
                if k in agent.holdings:
                    agent.holdings[k] = 0
                    agent.share_prices[k] = 0
                for nft_id in nft_ids:
                    del agent.nfts[nft_id]
        else:
            if quantity is not None:
                agent.nfts[nft_id].shares -= quantity
            if quantity is None or agent.nfts[nft_id].shares == 0:
                del agent.nfts[nft_id]

        self.current_block.withdrawals[tkn_remove] += abs(delta_s)
        # update fees
        self.asset_fee(tkn_remove)
        self.lrna_fee(tkn_remove)
        return self

    def price(self, tkn: str, denominator: str = '') -> float:
        """
        price of an asset i denominated in j, according to current market conditions in the omnipool
        """
        if tkn not in self.asset_list + ['LRNA']:
            raise ValueError(f'{tkn} is not in the Omnipool')
        elif tkn == denominator:
            return 1
        elif not denominator or denominator == 'LRNA':
            return self.lrna_price(tkn)
        elif denominator not in self.asset_list:
            raise ValueError(f'{denominator} is not in the Omnipool')
        elif tkn == 'LRNA':
            return 1 / self.lrna_price(denominator)
        elif self.liquidity[tkn] == 0:
            return 0
        return self.lrna[tkn] / self.liquidity[tkn] / self.lrna[denominator] * self.liquidity[denominator]

    def usd_price(self, tkn, usd_asset=None):
        if usd_asset is None:
            if self.stablecoin is None:
                raise ValueError('no stablecoin set or provided as argument')
            else:
                usd_asset = self.stablecoin
        if tkn == 'LRNA':
            return 1 / self.lrna_price(usd_asset)
        else:
            return self.lrna_price(tkn) / self.lrna_price(usd_asset)

    def lrna_price(self, tkn: str, fee: float = 0) -> float:
        """Price of i denominated in LRNA"""
        if tkn == "LRNA":
            return 1
        elif self.liquidity[tkn] == 0:
            return 0
        else:
            return (self.lrna[tkn] / self.liquidity[tkn]) * (1 - fee)

    def value_assets(self, assets: dict[str, float], equivalency_map: dict[str, str] = None,
                     numeraire: str = None) -> float:
        # assets is a dict of token: quantity
        # returns the value of the assets in USD
        if numeraire is None:
            if self.stablecoin is None:
                raise ValueError('no stablecoin set or provided as argument')
            else:
                numeraire = self.stablecoin
        if equivalency_map is None:
            equivalency_map = {}
        numeraire_synonyms = [numeraire]
        for eq in equivalency_map:
            if equivalency_map[eq] == 'USD':
                numeraire_synonyms.append(eq)
        value = 0
        for tkn in assets:
            equivalents = [tkn]
            if tkn in equivalency_map:
                equivalents += [equivalency_map[tkn]]
            for eq in equivalency_map:
                if equivalency_map[eq] in equivalents:
                    equivalents.append(eq)
            tkn_value = 0
            for numeraire_tkn in numeraire_synonyms:
                for eq in equivalents:
                    if self.sell_spot(eq, numeraire_tkn) > 0:
                        tkn_value += assets[tkn] * self.price(eq, numeraire_tkn)
                        break
                if tkn_value != 0:
                    break
            value += tkn_value
        return value

    def cash_out(self, agent: Agent, prices) -> float:
        """
        return the value of the agent's holdings if they withdraw all liquidity
        and then sell at current spot prices
        """

        delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l = 0, {}, {}, {}, {}, 0
        nft_ids = []

        for k in agent.holdings:
            if isinstance(k, tuple) and len(k) == 2 and k[0] == self.unique_id:  # LP shares of correct pool
                tkn = k[1]
                dqa, dr, dq, ds, db, ids = self.calculate_remove_liquidity(agent, tkn_remove=tkn)
                delta_qa += dqa
                delta_r[tkn] = dr + (delta_r[tkn] if tkn in delta_r else 0)
                delta_q[tkn] = dq + (delta_q[tkn] if tkn in delta_q else 0)
                delta_s[tkn] = ds + (delta_s[tkn] if tkn in delta_s else 0)
                delta_b[tkn] = db + (delta_b[tkn] if tkn in delta_b else 0)
                nft_ids += ids

        for nft_id in agent.nfts:
            if nft_id not in nft_ids and agent.nfts[nft_id].pool_id == self.unique_id:
                tkn = agent.nfts[nft_id].tkn
                dqa, dr, dq, ds, db, _ = self.calculate_remove_liquidity(agent, nft_id=nft_id)
                delta_qa += dqa
                delta_r[tkn] = dr + (delta_r[tkn] if tkn in delta_r else 0)
                delta_q[tkn] = dq + (delta_q[tkn] if tkn in delta_q else 0)
                delta_s[tkn] = ds + (delta_s[tkn] if tkn in delta_s else 0)
                delta_b[tkn] = db + (delta_b[tkn] if tkn in delta_b else 0)

        # agent_holdings = new_agent.holdings
        lrna_removed = {tkn: -delta_q[tkn] if tkn in delta_q else 0 for tkn in self.asset_list}
        liquidity_removed = {tkn: -delta_r[tkn] if tkn in delta_r else 0 for tkn in self.asset_list}

        # if 'LRNA' in prices:
        #     raise ValueError('LRNA price should not be given.')
        agent_lrna = delta_qa
        if 'LRNA' in agent.holdings:
            agent_lrna += agent.holdings['LRNA']

        if 'LRNA' not in prices and agent_lrna > 0:
            lrna_total = self.lrna_total - sum(lrna_removed.values())
            lrna_sells = {
                tkn: -(self.lrna[tkn] - lrna_removed[tkn]) / lrna_total * agent_lrna
                for tkn in self.asset_list
            }

            lrna_profits = dict()

            # sell LRNA optimally back to the pool
            for tkn, delta_qa in lrna_sells.items():
                asset_fee = (
                    self.asset_fee(tkn)
                    if hasattr(self, 'asset_fee')
                    else self.last_fee[tkn]
                )
                lrna_profits[tkn] = (
                        -(self.liquidity[tkn] - liquidity_removed[tkn]) * delta_qa
                        / (-delta_qa + self.lrna[tkn] - lrna_removed[tkn])
                        * (1 - asset_fee)
                )
                liquidity_removed[tkn] += lrna_profits[tkn]

        new_holdings = {tkn: agent.holdings[tkn] for tkn in agent.holdings}
        new_holdings['LRNA'] = agent_lrna
        for tkn in liquidity_removed:
            if tkn not in new_holdings:
                new_holdings[tkn] = 0
            new_holdings[tkn] += liquidity_removed[tkn]

        return value_assets(prices, new_holdings)


class OmnipoolLiquidityPosition:
    def __init__(self, tkn: str, price: float, shares: float, delta_r: float, pool_id: str = None):
        self.tkn = tkn
        self.price = price
        self.shares = shares
        self.delta_r = delta_r
        self.pool_id = pool_id

    def copy(self):
        return OmnipoolLiquidityPosition(self.tkn, self.price, self.shares, self.delta_r, self.pool_id)


class OmnipoolArchiveState:
    def __init__(self, state: OmnipoolState):
        self.asset_list = [tkn for tkn in state.asset_list]
        self.liquidity = {k: v for (k, v) in state.liquidity.items()}
        self.lrna = {k: v for (k, v) in state.lrna.items()}
        self.lrna_total = sum(state.lrna.values())
        self.shares = {k: v for (k, v) in state.shares.items()}
        self.protocol_shares = {k: v for (k, v) in state.protocol_shares.items()}
        self.fail = state.fail
        self.stablecoin = state.stablecoin
        # self.sub_pools = copy.deepcopy(self.sub_pools)
        self.oracles = {k: OracleArchiveState(v) for (k, v) in state.oracles.items()}
        self.unique_id = state.unique_id
        self.volume_in = state.current_block.volume_in
        self.volume_out = state.current_block.volume_out
        # record these for analysis later
        self.last_fee = {k: v for (k, v) in state.last_fee.items()}
        self.last_lrna_fee = {k: v for (k, v) in state.last_lrna_fee.items()}
        # borrow some methods from parent
        self.lrna_price = OmnipoolState.lrna_price
        self.price = OmnipoolState.price
        self.usd_price = OmnipoolState.usd_price


def asset_invariant(state: OmnipoolState, i: str) -> float:
    """Invariant for specific asset"""
    return state.liquidity[i] * state.lrna[i]


def swap_lrna_delta_Qi(state: OmnipoolState, delta_ri: float, i: str) -> float:
    return state.lrna[i] * (- delta_ri / (state.liquidity[i] + delta_ri))


def swap_lrna_delta_Ri(state: OmnipoolState, delta_qi: float, i: str) -> float:
    return state.liquidity[i] * (- delta_qi / (state.lrna[i] + delta_qi))


def weight_i(state: OmnipoolState, i: str) -> float:
    return state.lrna[i] / state.lrna_total


def simulate_swap(
        old_state: OmnipoolState,
        old_agent: Agent,
        tkn_buy: str,
        tkn_sell: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
) -> tuple[OmnipoolState, Agent]:
    """
    execute swap on a copy of old_state and old_agent, and return the copies
    """
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    new_state.swap(
        agent=new_agent,
        sell_quantity=sell_quantity,
        buy_quantity=buy_quantity,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
    )
    return new_state, new_agent


def simulate_migrate(
        old_state: OmnipoolState,
        tkn_migrate: str,
        sub_pool_id: str
) -> OmnipoolState:
    new_state = old_state.copy()
    return new_state.migrate_asset(tkn_migrate, sub_pool_id)


def simulate_add_liquidity(
        old_state: OmnipoolState,
        old_agent: Agent,
        quantity: float = 0,
        tkn_add: str = '',
        nft_id: str = None
) -> tuple[OmnipoolState, Agent]:
    """Copy state, then add liquidity and return new state"""
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    new_state.add_liquidity(new_agent, quantity, tkn_add, nft_id)
    return new_state, new_agent


def simulate_remove_liquidity(
        old_state: OmnipoolState,
        old_agent: Agent,
        quantity: float = None,
        tkn_remove: str = '',
        nft_id: str = None
) -> tuple[OmnipoolState, Agent]:
    """Compute new state after liquidity removal"""
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    new_state.remove_liquidity(new_agent, quantity, tkn_remove, nft_id)
    return new_state, new_agent


def value_assets(prices: dict, assets: dict) -> float:
    """
    return the value of the agent's assets if they were sold at current spot prices
    """
    return sum([
        assets[i] * prices[i] if i in prices else 0
        for i in assets.keys()
    ])
