import copy
from numbers import Number
from typing import Callable

from .agents import Agent
from .amm import AMM, FeeMechanism, basic_fee
from .oracle import Oracle, Block, OracleArchiveState
from .stableswap_amm import StableSwapPoolState


class OmnipoolState(AMM):
    unique_id: str = 'omnipool'

    def __init__(self,
                 tokens: dict[str: dict],
                 tvl_cap: float = float('inf'),
                 preferred_stablecoin: str = None,
                 asset_fee: dict or FeeMechanism or float = None,
                 lrna_fee: dict or FeeMechanism or float = None,
                 oracles: dict[str: int] = None,
                 trade_limit_per_block: float = float('inf'),
                 update_function: Callable = None,
                 last_asset_fee: dict or float = None,
                 last_lrna_fee: dict or float = None,
                 imbalance: float = 0.0,
                 last_oracle_values: dict = None,
                 max_withdrawal_per_block: float = 1,
                 max_lp_per_block: float = float('inf'),
                 remove_liquidity_volatility_threshold: float = 0,
                 withdrawal_fee: bool = True,
                 min_withdrawal_fee: float = 0.0001,
                 lrna_mint_pct: float = 0.0,
                 unique_id: str = 'omnipool'
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
        self.default_asset_fee = asset_fee if isinstance(asset_fee, Number) else 0.0
        self.default_lrna_fee = asset_fee if isinstance(asset_fee, Number) else 0.0
        self.lrna_imbalance = imbalance  # AKA "L"
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

        # trades per block cannot exceed this fraction of the pool's liquidity
        self.trade_limit_per_block = trade_limit_per_block

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
                shares=pool['liquidity'],
                protocol_shares=pool['liquidity'],
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

        self.asset_fee = self._get_fee(asset_fee)
        self.lrna_fee = self._get_fee(lrna_fee)

        self.time_step = 0
        self.current_block = Block(self)

        if isinstance(last_asset_fee, Number):
            self.last_fee = {tkn: last_asset_fee for tkn in self.asset_list}
        elif isinstance(last_asset_fee, dict):
            self.last_fee = {tkn: last_asset_fee[tkn] if tkn in last_asset_fee else 0 for tkn in self.asset_list}
        else:
            self.last_fee = {tkn: 0 for tkn in self.asset_list}

        if isinstance(last_lrna_fee, Number):
            self.last_lrna_fee = {tkn: last_lrna_fee for tkn in self.asset_list}
        elif isinstance(last_lrna_fee, dict):
            self.last_lrna_fee = {tkn: last_lrna_fee[tkn] if tkn in last_lrna_fee else 0 for tkn in self.asset_list}
        else:
            self.last_lrna_fee = {tkn: 0 for tkn in self.asset_list}

        self.unique_id = unique_id

    def __setattr__(self, key, value):
        # if key is a fee, make sure it's a dict[str: FeeMechanism]
        if key in ['lrna_fee', 'asset_fee']:
            super().__setattr__(key, self._get_fee(value))
        else:
            super().__setattr__(key, value)

    def _get_fee(self, value: dict or FeeMechanism or float) -> dict:

        if isinstance(value, dict):
            if set(value.keys()) != set(self.asset_list):
                # I do not believe we were handling this case correctly
                # we can extend this when it is a priority
                raise ValueError(f'fee dict keys must match asset list: {self.asset_list}')
            return ({
                tkn: (
                    value[tkn].assign(self, tkn)
                    if isinstance(fee, FeeMechanism)
                    else basic_fee(fee).assign(self, tkn)
                )
                for tkn, fee in value.items()
            })
        elif isinstance(value, FeeMechanism):
            return {tkn: copy.deepcopy(value).assign(self, tkn) for tkn in self.asset_list}
        else:
            return {tkn: basic_fee(value or 0).assign(self, tkn) for tkn in self.asset_list}

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
        if hasattr(self, 'asset_fee'):
            self.asset_fee[tkn] = basic_fee(self.default_asset_fee).assign(self, tkn)
            self.lrna_fee[tkn] = basic_fee(self.default_lrna_fee).assign(self, tkn)
            self.last_fee[tkn] = 0
            self.last_lrna_fee[tkn] = 0
        if hasattr(self, 'current_block'):
            self.current_block.price[tkn] = self.lrna[tkn] / self.liquidity[tkn]
            self.current_block.liquidity[tkn] = self.liquidity[tkn]
            self.current_block.lps[tkn] = 0
            self.current_block.withdrawals[tkn] = 0
            self.current_block.volume_in[tkn] = 0
            self.current_block.volume_out[tkn] = 0
        return self

    def remove_token(self, tkn: str):
        self.asset_list.remove(tkn)
        return self

    def update(self):
        # update oracles
        self.current_block.price['HDX'] = self.lrna['HDX'] / self.liquidity['HDX']

        for name, oracle in self.oracles.items():
            oracle.update(self.current_block)

        # update current block
        self.time_step += 1
        self.current_block = Block(self)

        # update fees
        self.last_fee = {tkn: self.asset_fee[tkn].compute() for tkn in self.asset_list}
        self.last_lrna_fee = {tkn: self.lrna_fee[tkn].compute() for tkn in self.asset_list}

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
        return self.liquidity[tkn_buy] * (1 - self.asset_fee[tkn_buy].compute())

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
            {tkn: round(usd_price(self, tkn), precision) for tkn in self.asset_list}
            if self.stablecoin is not None
            else {tkn: 'N/A' for tkn in self.asset_list}
        )
        newline = '\n'
        return (
            f'Omnipool: {self.unique_id}\n'
            f'********************************\n'
            f'tvl cap: {self.tvl_cap}\n'
            f'LRNA imbalance: {self.lrna_imbalance}\n'
            f'lrna fee:\n\n'
            f'{newline.join(["    " + tkn + ": " + self.lrna_fee[tkn].name for tkn in self.asset_list])}\n\n'
            f'asset fee:\n\n'
            f'{newline.join(["    " + tkn + ": " + self.asset_fee[tkn].name for tkn in self.asset_list])}\n\n'
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
        asset_fee = self.asset_fee[tkn_buy].compute()
        if buy_quantity >= self.liquidity[tkn_buy] * (1 - asset_fee):
            return float('inf')
        delta_Qj = self.lrna[tkn_buy] * buy_quantity / (
                self.liquidity[tkn_buy] * (1 - asset_fee) - buy_quantity)
        lrna_fee = self.lrna_fee[tkn_sell].compute()
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
        asset_fee = self.asset_fee[tkn_buy].compute()
        lrna_fee = self.lrna_fee[tkn_sell].compute()
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
                fee['lrna'] = self.lrna_fee[tkn_sell].compute()
            if 'asset' not in fee:
                fee['asset'] = self.asset_fee[tkn_buy].compute()
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
            return price(self, tkn_buy, tkn_sell) / (1 - fee['lrna']) / (1 - fee['asset'])

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
                fee['lrna'] = self.lrna_fee[tkn_sell].compute()
            if 'asset' not in fee:
                fee['asset'] = self.asset_fee[tkn_buy].compute()
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
            return price(self, tkn_sell, tkn_buy) * (1 - fee['lrna']) * (1 - fee['asset'])

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
            sell_quantity: float = 0,
            modify_imbalance: bool = True,  # this is a hack to avoid modifying the imbalance for arbitrager LRNA swaps,
            # since those would not actually be executed as LRNA swaps
            # note that we still apply the imbalance modification due to LRNA fee
            # collection, we just don't apply the imbalance modification from
            # the sale of LRNA back to the pool.
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
            return_val = self.lrna_swap(agent, buy_quantity, -sell_quantity, tkn_buy, modify_imbalance)
        elif tkn_buy == 'LRNA':
            return_val = self.lrna_swap(agent, -sell_quantity, buy_quantity, tkn_sell, modify_imbalance)
    
        elif buy_quantity and not sell_quantity:
            # back into correct delta_Ri, then execute sell
            delta_Ri = self.calculate_sell_from_buy(tkn_buy, tkn_sell, buy_quantity)
            if delta_Ri < 0:
                return self.fail_transaction(f'insufficient LRNA in {tkn_sell}', agent)
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
                return self.fail_transaction('sell amount must be greater than zero', agent)
            if delta_Ri > agent.holdings[i]:
                return self.fail_transaction(f"Agent doesn't have enough {i}", agent)

            delta_Qi = self.lrna[tkn_sell] * -delta_Ri / (self.liquidity[tkn_sell] + delta_Ri)
            asset_fee = self.asset_fee[tkn_buy].compute()
            lrna_fee = self.lrna_fee[tkn_sell].compute()
            delta_Qt = -delta_Qi * (1 - lrna_fee)
            delta_Qm = (self.lrna[tkn_buy] + delta_Qt) * delta_Qt * asset_fee / self.lrna[
                tkn_buy] * self.lrna_mint_pct
            delta_Qj = delta_Qt + delta_Qm
            delta_Rj = self.liquidity[tkn_buy] * -delta_Qt / (self.lrna[tkn_buy] + delta_Qt) * (1 - asset_fee)
            delta_L = min(-delta_Qi * lrna_fee, -self.lrna_imbalance)
            delta_QH = -lrna_fee * delta_Qi - delta_L

            if self.liquidity[i] + sell_quantity > 10 ** 12:
                return self.fail_transaction('Asset liquidity cannot exceed 10 ^ 12.', agent)
    
            # per-block trade limits
            if (
                    -delta_Rj - self.current_block.volume_in[tkn_buy] + self.current_block.volume_out[tkn_buy]
                    > self.trade_limit_per_block * self.current_block.liquidity[tkn_buy]
            ):
                return self.fail_transaction(
                    f'{self.trade_limit_per_block * 100}% per block trade limit exceeded in {tkn_buy}.', agent
                )
            elif (
                    delta_Ri + self.current_block.volume_in[tkn_sell] - self.current_block.volume_out[tkn_sell]
                    > self.trade_limit_per_block * self.current_block.liquidity[tkn_sell]
            ):
                return self.fail_transaction(
                    f'{self.trade_limit_per_block * 100}% per block trade limit exceeded in {tkn_sell}.', agent
                )
            self.lrna[i] += delta_Qi
            self.lrna[j] += delta_Qj
            self.liquidity[i] += delta_Ri
            self.liquidity[j] += -buy_quantity or delta_Rj
            self.lrna['HDX'] += delta_QH
            self.lrna_imbalance += delta_L
    
            if j not in agent.holdings:
                agent.holdings[j] = 0
            agent.holdings[i] -= delta_Ri
            agent.holdings[j] -= -buy_quantity or delta_Rj
    
            return_val = self
    
        # update oracle
        if tkn_buy in self.current_block.asset_list:
            buy_quantity = old_buy_liquidity - self.liquidity[tkn_buy]
            self.current_block.volume_out[tkn_buy] += buy_quantity
            self.current_block.price[tkn_buy] = self.lrna[tkn_buy] / self.liquidity[tkn_buy]
        if tkn_sell in self.current_block.asset_list:
            sell_quantity = self.liquidity[tkn_sell] - old_sell_liquidity
            self.current_block.volume_in[tkn_sell] += sell_quantity
            self.current_block.price[tkn_sell] = self.lrna[tkn_sell] / self.liquidity[tkn_sell]
        return return_val
    
    def lrna_swap(
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
    
        if delta_qa < 0:
            asset_fee = self.asset_fee[tkn].compute()
            if -delta_qa + self.lrna[tkn] <= 0:
                return self.fail_transaction('insufficient lrna in pool', agent)
            delta_ra = -self.liquidity[tkn] * delta_qa / (-delta_qa + self.lrna[tkn]) * (1 - asset_fee)

            delta_qm = asset_fee * (-delta_qa) / self.lrna[tkn] * (self.lrna[tkn] - delta_qa) * self.lrna_mint_pct
            delta_q = delta_qm - delta_qa

            if modify_imbalance:
                q = self.lrna_total
                self.lrna_imbalance += -delta_q * (q + self.lrna_imbalance) / (q + delta_q) - delta_q

            self.lrna[tkn] += delta_q
            self.liquidity[tkn] += -delta_ra
    
        elif delta_ra > 0:
            asset_fee = self.asset_fee[tkn].compute()
            if -delta_ra + self.liquidity[tkn] <= 0:
                return self.fail_transaction('insufficient assets in pool', agent)
            denom = (self.liquidity[tkn] * (1 - asset_fee) - delta_ra)
            delta_qa = -self.lrna[tkn] * delta_ra / denom
            delta_qm = -asset_fee * (1 - asset_fee) * (self.liquidity[tkn] / denom) * delta_qa * self.lrna_mint_pct
            delta_q = -delta_qa + delta_qm

            if modify_imbalance:
                q = self.lrna_total
                self.lrna_imbalance -= delta_q * (q + self.lrna_imbalance) / (q + delta_q) + delta_q

            self.lrna[tkn] += delta_q
            self.liquidity[tkn] -= delta_ra
    
        # buying LRNA
        elif delta_qa > 0:
            lrna_fee = self.lrna_fee[tkn].compute()
            delta_qi = -delta_qa / (1 - lrna_fee)
            if delta_qi + self.lrna[tkn] <= 0:
                return self.fail_transaction('insufficient lrna in pool', agent)
            delta_ra = -self.liquidity[tkn] * -delta_qi / (delta_qi + self.lrna[tkn])
            if agent.holdings[tkn] < -delta_ra:
                return self.fail_transaction('Agent has insufficient assets', agent)
            self.lrna[tkn] += delta_qi
            self.liquidity[tkn] += -delta_ra
            # if modify_imbalance:
            #     self.lrna_imbalance += - delta_qi * (q + l) / (q + delta_qi) - delta_qi
    
            # we assume, for now, that buying LRNA is only possible when modify_imbalance = False
            lrna_fee_amt = -(delta_qa + delta_qi)
            delta_l = min(-self.lrna_imbalance, lrna_fee_amt)
            self.lrna_imbalance += delta_l
            self.lrna["HDX"] += lrna_fee_amt - delta_l
    
        elif delta_ra < 0:
            lrna_fee = self.lrna_fee[tkn].compute()
            # delta_ri = -delta_ra
            if delta_ra > agent.holdings[tkn]:
                return self.fail_transaction('agent has insufficient assets', agent)
            delta_qi = self.lrna[tkn] * delta_ra / (self.liquidity[tkn] - delta_ra)
            delta_qa = -delta_qi * (1 - lrna_fee)
            self.lrna[tkn] += delta_qi
            self.liquidity[tkn] -= delta_ra
            # if modify_imbalance:
            #     self.lrna_imbalance += - delta_qi * (q + l) / (q + delta_qi) - delta_qi
    
            # we assume, for now, that buying LRNA is only possible when modify_imbalance = False
            lrna_fee_amt = -(delta_qa + delta_qi)
            delta_l = min(-self.lrna_imbalance, lrna_fee_amt)
            self.lrna_imbalance += delta_l
            self.lrna["HDX"] += lrna_fee_amt - delta_l
    
        else:
            return self.fail_transaction('All deltas are zero.', agent)
    
        if 'LRNA' not in agent.holdings:
            agent.holdings['LRNA'] = 0
        if tkn not in agent.holdings:
            agent.holdings[tkn] = 0

        agent.holdings['LRNA'] += delta_qa
        agent.holdings[tkn] += delta_ra
    
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
                self.lrna_swap(agent, delta_ra=shares_needed, tkn=sub_pool.unique_id)
                if self.fail:
                    # if the swap failed, the transaction failed.
                    return self.fail_transaction(self.fail, agent)
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
                    return self.fail_transaction(self.fail, agent)
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
                    return self.fail_transaction(self.fail, agent)
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
                    return self.fail_transaction("Not enough liquidity in the stableswap/LRNA pool.", agent)
                sub_pool.buy_shares(agent, sell_shares, tkn_sell)
                if sub_pool.fail:
                    return self.fail_transaction(sub_pool.fail, agent)
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
                    return self.fail_transaction(self.fail, agent)
                # withdraw the shares for the desired token
                sub_pool.withdraw_asset(agent, quantity=buy_quantity, tkn_remove=tkn_buy)
                if sub_pool.fail:
                    return self.fail_transaction(sub_pool.fail, agent)
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
                    return self.fail_transaction(self.fail, agent)
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
                    return self.fail_transaction(f'Not enough liquidity in {pool_buy.unique_id}: {tkn_buy}.', agent)
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
                    return self.fail_transaction(pool_sell.fail, agent)
                self.swap(
                    agent=agent,
                    tkn_buy=pool_buy.unique_id, tkn_sell=pool_sell.unique_id,
                    buy_quantity=shares_bought
                )
                if self.fail:
                    return self.fail_transaction(self.fail, agent)
                pool_buy.withdraw_asset(
                    agent=agent, quantity=buy_quantity,
                    tkn_remove=tkn_buy, fail_on_overdraw=False
                )
                if pool_buy.fail:
                    return self.fail_transaction(pool_buy.fail, agent)
    
                # if all three parts succeeded, then we're good!
                return self
            elif sell_quantity:
                agent_sell_holdings = agent.holdings[sub_pool_sell_id] if sub_pool_sell_id in agent.holdings else 0
                pool_sell.add_liquidity(
                    agent=agent, quantity=sell_quantity, tkn_add=tkn_sell
                )
                if pool_sell.fail:
                    return self.fail_transaction(pool_sell.fail, agent)
                delta_sell_holdings = agent.holdings[sub_pool_sell_id] - agent_sell_holdings
                agent_buy_holdings = agent.holdings[sub_pool_buy_id] if sub_pool_buy_id in agent.holdings else 0
                self.swap(
                    agent=agent,
                    tkn_buy=pool_buy.unique_id, tkn_sell=pool_sell.unique_id,
                    sell_quantity=delta_sell_holdings
                )
                if self.fail:
                    return self.fail_transaction(self.fail, agent)
                delta_buy_holdings = agent.holdings[sub_pool_buy_id] - agent_buy_holdings
                pool_buy.remove_liquidity(
                    agent=agent, shares_removed=delta_buy_holdings, tkn_remove=tkn_buy
                )
                if pool_buy.fail:
                    return self.fail_transaction(pool_buy.fail, agent)
                return self
        else:
            raise ValueError('buy_quantity or sell_quantity must be specified.')

    def create_sub_pool(
            self,
            tkns_migrate: dict[str: float] or list[str],
            unique_id: str,
            amplification: float,
            trade_fee: FeeMechanism or float = 0
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
        new_sub_pool.shares = sum([self.liquidity[tkn] * tkns_migrate[tkn] / self.liquidity[tkn] for tkn in tkns_migrate])
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

    def calculate_remove_liquidity(self, agent: Agent, quantity: float = None, tkn_remove: str = None, nft_id: str = None):
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

        delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l, nft_ids = 0, 0, 0, 0, 0, 0, []
        if quantity is not None:
            if nft_id is None:  # remove specified quantity of shares from holdings
                k = (self.unique_id, tkn_remove)
                return self._calculate_remove_one_position(
                    quantity=quantity, tkn_remove=tkn_remove, share_price=agent.share_prices[k]
                )
            else:  # remove specified quantity of shares from specified position
                delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l = self._calculate_remove_one_position(
                    quantity=quantity, tkn_remove=tkn_remove, share_price=agent.nfts[nft_id].price
                )
                nft_ids = [nft_id]
        else:
            if nft_id is not None:  # remove specified position
                delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l = self._calculate_remove_one_position(
                    quantity=agent.nfts[nft_id].shares, tkn_remove=tkn_remove, share_price=agent.nfts[nft_id].price
                )
                nft_ids = [nft_id]
            else:  # remove all liquidity
                for nft_id in agent.nfts:
                    nft = agent.nfts[nft_id]
                    if isinstance(nft, OmnipoolLiquidityPosition):
                        if nft.pool_id == self.unique_id and nft.tkn == tkn_remove:
                            nft_ids.append(nft_id)
                            dqa, dr, dq, ds, db, dl = self._calculate_remove_one_position(
                                quantity=nft.shares, tkn_remove=tkn_remove, share_price=nft.price
                            )
                            delta_qa += dqa
                            delta_r += dr
                            delta_q += dq
                            delta_s += ds
                            delta_b += db
                            delta_l += dl
                if (self.unique_id, tkn_remove) in agent.holdings:
                    dqa, dr, dq, ds, db, dl = self._calculate_remove_one_position(
                        quantity=agent.holdings[(self.unique_id, tkn_remove)], tkn_remove=tkn_remove,
                        share_price=agent.share_prices[(self.unique_id, tkn_remove)]
                    )
                    delta_qa += dqa
                    delta_r += dr
                    delta_q += dq
                    delta_s += ds
                    delta_b += db
                    delta_l += dl
        return delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l, nft_ids

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
        piq = lrna_price(self, tkn_remove)
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
        delta_l = delta_r * piq * self.lrna_imbalance / self.lrna_total
    
        return delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l
    
    def add_liquidity(
            self,
            agent: Agent = None,
            quantity: float = 0,
            tkn_add: str = '',
            nft_id: str = None
    ):
        """Compute new state after liquidity addition"""
    
        if quantity <= 0:
            return self.fail_transaction('Quantity must be non-negative.', agent)

        delta_Q = lrna_price(self, tkn_add) * quantity

        if nft_id is None and (self.unique_id, tkn_add) in agent.holdings:
            return self.fail_transaction(
                'Agent already has liquidity in this pool. Try using nft_id input.', agent
            )

        if nft_id is not None and nft_id in agent.nfts:
            raise AssertionError('Agent already has an NFT with this ID.')
    
        if agent.holdings[tkn_add] < quantity:
            return self.fail_transaction(
                f'Agent has insufficient funds ({agent.holdings[tkn_add]} < {quantity}).', agent
            )
    
        if (self.lrna[tkn_add] + delta_Q) / (self.lrna_total + delta_Q) > self.weight_cap[tkn_add]:
            return self.fail_transaction(
                'Transaction rejected because it would exceed the weight cap in pool[{i}].', agent
            )

        if self.tvl_cap < float('inf'):
            if (self.total_value_locked() + quantity * usd_price(self, tkn_add)) > self.tvl_cap:
                return self.fail_transaction('Transaction rejected because it would exceed the TVL cap.', agent)
    
        # assert quantity > 0, f"delta_R must be positive: {quantity}"
        if tkn_add not in self.asset_list:
            for sub_pool in self.sub_pools.values():
                if tkn_add in sub_pool.asset_list:
                    old_agent_holdings = (
                        agent.holdings[sub_pool.unique_id] if sub_pool.unique_id in agent.holdings else 0
                    )
                    sub_pool.add_liquidity(
                        agent=agent,
                        quantity=quantity,
                        tkn_add=tkn_add
                    )
                    # deposit into the Omnipool
                    return self.add_liquidity(
                        agent=agent,
                        quantity=agent.holdings[sub_pool.unique_id] - old_agent_holdings,
                        tkn_add=sub_pool.unique_id,
                        nft_id=nft_id
                    )
            raise AssertionError(f"invalid value for i: {tkn_add}")
        else:
            # enforce upper limit on liquidity addition
            if quantity > self.max_lp_per_block * self.liquidity[tkn_add] - self.current_block.lps[tkn_add]:
                return self.fail_transaction(
                    'Transaction rejected because it would exceed the max LP per block.', agent
                )
    
        # Share update
        if self.shares[tkn_add]:
            shares_added = (delta_Q / self.lrna[tkn_add]) * self.shares[tkn_add]
        else:
            shares_added = delta_Q
        self.shares[tkn_add] += shares_added

        # L update: LRNA fees to be burned before they will start to accumulate again
        delta_L = quantity * lrna_price(self, tkn_add) * self.lrna_imbalance / self.lrna_total
        self.lrna_imbalance += delta_L
    
        # LRNA add (mint)
        self.lrna[tkn_add] += delta_Q
    
        # Token amounts update
        self.liquidity[tkn_add] += quantity

        # agent update
        agent.holdings[tkn_add] -= quantity
        if nft_id is None:
            k = (self.unique_id, tkn_add)
            agent.holdings[k] = 0
            # shares go to provisioning agent
            agent.holdings[k] += shares_added
            # set price at which liquidity was added
            agent.share_prices[k] = lrna_price(self, tkn_add)
            agent.delta_r[k] = quantity
        else:
            lp_position = OmnipoolLiquidityPosition(tkn_add, lrna_price(self, tkn_add), shares_added, quantity, self.unique_id)
            agent.nfts[nft_id] = lp_position
    
        # update block
        self.current_block.lps[tkn_add] += quantity
    
        return self

    # def remove_all_liquidity(self, agent: Agent, tkn_remove: str):
    #     agent_assets = list(agent.holdings.keys())
    #     for k in agent_assets:
    #         if len(k) > 1 and k[1] == tkn_remove:
    #             k_split = k[0].split("_")
    #             if len(k_split) == 1:
    #                 self.remove_liquidity(agent, agent.holdings[k], tkn_remove)
    #             elif k[0].split("_")[0] == self.unique_id:
    #                 self.remove_liquidity(agent, agent.holdings[k], tkn_remove, int(k[0].split("_")[1]))
    #     return self
    
    def remove_liquidity(self, agent: Agent, quantity: float = None, tkn_remove: str = '', nft_id: str = None):
        """
        Remove liquidity from a sub pool.
        If quantity is specified and nft_id is specified, remove specified quantity of shares from specified position.
        If quantity is specified and nft_id is unspecified, remove specified quantity of shares from holdings.
        If quantity is unspecified and nft_id is specified, remove specified position.
        If quantity is unspecified and nft_id is unspecified, remove all liquidity.
        """

        # if i is None:
        #     k = (self.unique_id, tkn_remove)
        # else:
        #     k = (self.unique_id + "_" + str(i), tkn_remove)

        k = (self.unique_id, tkn_remove)
        if nft_id is not None:
            if nft_id not in agent.nfts:
                return self.fail_transaction('Agent does not have liquidity position with specified nft_id.', agent)
            tkn_remove = agent.nfts[nft_id].tkn

        if tkn_remove not in self.asset_list:
            for sub_pool in self.sub_pools.values():
                if tkn_remove in sub_pool.asset_list:
                    sub_pool.remove_liquidity(
                        agent, quantity, tkn_remove
                    )
                    if sub_pool.fail:
                        return self.fail_transaction(sub_pool.fail, agent)
                    else:
                        return self
    
            raise AssertionError(f"invalid value for tkn_remove: {tkn_remove}")

        if quantity == 0:
            return self
        # if nft_id is None and not agent.is_holding(k):
        #     return self.fail_transaction('Agent does not have liquidity in this pool.', agent)
        if nft_id is None:
            if quantity is not None and not agent.is_holding(k, quantity):
                return self.fail_transaction('Agent does not have enough liquidity in this pool.', agent)
        else:
            if nft_id not in agent.nfts:
                return self.fail_transaction('Agent does not have liquidity position with specified nft_id.', agent)
            elif agent.nfts[nft_id].pool_id != self.unique_id:
                return self.fail_transaction('Specified position is wrong pool.', agent)
            elif agent.nfts[nft_id].tkn != tkn_remove:
                return self.fail_transaction('Specified position is wrong asset.', agent)
            elif quantity is not None and agent.nfts[nft_id].shares < quantity:
                return self.fail_transaction('Agent does not have enough shares in specified position.', agent)
    
        if self.remove_liquidity_volatility_threshold and self.remove_liquidity_volatility_threshold < float('inf'):
            if self.oracles['price']:
                volatility = abs(
                    self.oracles['price'].price[tkn_remove] / self.current_block.price[tkn_remove] - 1
                )
                if volatility > self.remove_liquidity_volatility_threshold:
                    return self.fail_transaction(
                        f"Withdrawal rejected because the oracle volatility is too high: {volatility} > "
                        f"{self.remove_liquidity_volatility_threshold}", agent
                    )

        val = self.calculate_remove_liquidity(agent, quantity, tkn_remove, nft_id)
        delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l = val[:6]
        if len(val) == 7:
            nft_ids = val[6]

        max_remove = (
                self.max_withdrawal_per_block * self.shares[tkn_remove] - self.current_block.withdrawals[tkn_remove]
        )
        if abs(delta_s) > max_remove:
            return self.fail_transaction(
                f"Transaction rejected because it would exceed the withdrawal limit: {abs(delta_s)} > {max_remove}", agent
            )

        if delta_r + self.liquidity[tkn_remove] < 0:
            return self.fail_transaction('Cannot remove more liquidity than exists in the pool.', agent)
    
        self.liquidity[tkn_remove] += delta_r
        self.shares[tkn_remove] += delta_s
        self.protocol_shares[tkn_remove] += delta_b
        self.lrna[tkn_remove] += delta_q
        self.lrna_imbalance += delta_l

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
        return self

    def value_assets(self, assets: dict[str, float], equivalency_map: dict[str, str] = None, stablecoin: str = None) -> float:
        # assets is a dict of token: quantity
        # returns the value of the assets in USD
        if stablecoin is None:
            if self.stablecoin is None:
                raise ValueError('no stablecoin set or provided as argument')
            else:
                stablecoin = self.stablecoin
        if equivalency_map is None:
            equivalency_map = {}
        usd_synonyms = [stablecoin]
        for eq in equivalency_map:
            if equivalency_map[eq] == 'USD':
                usd_synonyms.append(eq)
        value = 0
        for tkn in assets:
            equivalents = [tkn]
            if tkn in equivalency_map:
                equivalents += [equivalency_map[tkn]]
            for eq in equivalency_map:
                if equivalency_map[eq] in equivalents:
                    equivalents.append(eq)
            tkn_value = 0
            for usd in usd_synonyms:
                for eq in equivalents:
                    if self.sell_spot(eq, usd) > 0:
                        tkn_value += assets[tkn] * price(self, eq, usd)
                        break
                if tkn_value != 0:
                    break
            value += tkn_value
        return value


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
        self.lrna_imbalance = state.lrna_imbalance
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


# Works with OmnipoolState *or* OmnipoolArchiveState
def price(state: OmnipoolState or OmnipoolArchiveState, tkn: str, denominator: str = '') -> float:
    """
    price of an asset i denominated in j, according to current market conditions in the omnipool
    """
    if tkn not in state.asset_list + ['LRNA']:
        return 0
    elif tkn == denominator:
        return 1
    elif not denominator or denominator == 'LRNA':
        return lrna_price(state, tkn)
    elif denominator not in state.asset_list:
        return 0
    elif tkn == 'LRNA':
        return 1 / lrna_price(state, denominator)
    elif state.liquidity[tkn] == 0:
        return 0
    return state.lrna[tkn] / state.liquidity[tkn] / state.lrna[denominator] * state.liquidity[denominator]


def usd_price(state: OmnipoolState or OmnipoolArchiveState, tkn, usd_asset=None):
    if usd_asset is None:
        if state.stablecoin is None:
            raise ValueError('no stablecoin set or provided as argument')
        else:
            usd_asset = state.stablecoin
    if tkn == 'LRNA':
        return 1 / state.lrna_price(state, usd_asset)
    else:
        return price(state, tkn) / price(state, usd_asset)


def lrna_price(state: OmnipoolState or OmnipoolArchiveState, tkn: str, fee: float = 0) -> float:
    """Price of i denominated in LRNA"""
    if tkn == "LRNA":
        return 1
    elif state.liquidity[tkn] == 0:
        return 0
    else:
        return (state.lrna[tkn] / state.liquidity[tkn]) * (1 - fee)


def asset_invariant(state: OmnipoolState, i: str) -> float:
    """Invariant for specific asset"""
    return state.liquidity[i] * state.lrna[i]


def swap_lrna_delta_Qi(state: OmnipoolState, delta_ri: float, i: str) -> float:
    return state.lrna[i] * (- delta_ri / (state.liquidity[i] + delta_ri))


def swap_lrna_delta_Ri(state: OmnipoolState, delta_qi: float, i: str) -> float:
    return state.liquidity[i] * (- delta_qi / (state.lrna[i] + delta_qi))


def weight_i(state: OmnipoolState, i: str) -> float:
    return state.lrna[i] / state.lrna_total


def simulate_swap_lrna(
        old_state: OmnipoolState,
        old_agent: Agent,
        delta_ra: float = 0,
        delta_qa: float = 0,
        tkn: str = '',
        modify_imbalance: bool = True
) -> tuple[OmnipoolState, Agent]:
    """Compute new state after LRNA swap"""

    new_state = old_state.copy()
    new_agent = old_agent.copy()

    new_state.lrna_swap(new_agent, delta_ra, delta_qa, tkn, modify_imbalance)
    return new_state, new_agent


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


OmnipoolArchiveState.usd_price = staticmethod(usd_price)
OmnipoolArchiveState.lrna_price = staticmethod(lrna_price)
OmnipoolArchiveState.price = staticmethod(price)

OmnipoolState.usd_price = staticmethod(usd_price)
OmnipoolState.lrna_price = staticmethod(lrna_price)
OmnipoolState.price = staticmethod(price)


# ===============================================================================
# fee mechanisms
# ===============================================================================
def slip_fee(slip_factor: float, minimum_fee: float = 0) -> FeeMechanism:
    def fee_function(
            exchange: AMM, tkn: str, delta_tkn: float
    ) -> float:
        return (slip_factor * abs(delta_tkn) / (exchange.liquidity[tkn] + delta_tkn)) + minimum_fee

    return FeeMechanism(fee_function, f"Slip fee (alpha={slip_factor}, min={minimum_fee}")


def dynamicadd_asset_fee(
        minimum: float = 0,
        amplification: float = 1,
        raise_oracle_name: str = 'short',
        decay: float = 0.001,
        fee_max: float = 0.5
) -> FeeMechanism:
    class Fee(FeeMechanism):
        def __init__(self):
            super().__init__(
                fee_function=self.fee_function,
                name=f'Dynamic fee (oracle={raise_oracle_name}, amplification={amplification}, min={minimum})'
            )
            self.amplification = amplification
            self.decay = decay
            self.minimum = minimum
            self.fee_max = fee_max
            self.raise_oracle_name = raise_oracle_name
            # force compute on first call
            self.time_step = -1

        def fee_function(self, exchange: OmnipoolState, tkn: str, delta_tkn: float = 0) -> float:
            if self.time_step == exchange.time_step:
                # return the last fee if it's already been computed for this tkn and block
                return exchange.last_fee[tkn]

            self.time_step = exchange.time_step
            raise_oracle: Oracle = exchange.oracles[raise_oracle_name]

            if raise_oracle.liquidity[tkn] != 0:
                x = (raise_oracle.volume_out[tkn] - raise_oracle.volume_in[tkn]) / raise_oracle.liquidity[tkn]
            else:
                x = 0

            fee_adj = amplification * x - decay

            previous_fee = exchange.last_fee[tkn]

            fee = min(max(previous_fee + fee_adj, minimum), fee_max)
            exchange.last_fee[tkn] = fee

            return fee

    return Fee()


def dynamicadd_lrna_fee(
        minimum: float = 0,
        amplification: float = 1,
        raise_oracle_name: str = 'short',
        decay: float = 0.001,
        fee_max: float = 0.5,
) -> FeeMechanism:
    class Fee(FeeMechanism):
        def __init__(self):
            super().__init__(
                fee_function=self.fee_function,
                name=f'Dynamic fee (oracle={raise_oracle_name}, amplification={amplification}, min={minimum})'
            )
            self.amplification = amplification
            self.decay = decay
            self.minimum = minimum
            self.fee_max = fee_max
            self.raise_oracle_name = raise_oracle_name
            self.time_step = -1

        def fee_function(
                self, exchange: OmnipoolState, tkn: str, delta_tkn: float = 0) -> float:
            if self.time_step == exchange.time_step:
                # return the last fee if it's already been computed for this tkn and block
                return exchange.last_lrna_fee[tkn]

            self.time_step = exchange.time_step
            raise_oracle: Oracle = exchange.oracles[raise_oracle_name]

            if raise_oracle.liquidity[tkn] != 0:
                x = (raise_oracle.volume_in[tkn] - raise_oracle.volume_out[tkn]) / raise_oracle.liquidity[tkn]
            else:
                x = 0

            fee_adj = amplification * x - decay

            previous_fee = exchange.last_lrna_fee[tkn]

            fee = min(max(previous_fee + fee_adj, minimum), fee_max)
            exchange.last_lrna_fee[tkn] = fee

            return fee

    return Fee()


def value_assets(prices: dict, assets: dict) -> float:
    """
    return the value of the agent's assets if they were sold at current spot prices
    """
    return sum([
        assets[i] * prices[i] if i in prices else 0
        for i in assets.keys()
    ])


def _turn_off_validations(omnipool: OmnipoolState) -> OmnipoolState:
    new_state = omnipool.copy()
    new_state.remove_liquidity_volatility_threshold = float('inf')
    new_state.max_withdrawal_per_block = float('inf')
    return new_state


def cash_out_omnipool(omnipool: OmnipoolState, agent: Agent, prices) -> float:
    """
    return the value of the agent's holdings if they withdraw all liquidity
    and then sell at current spot prices
    """

    new_state, new_agent = _turn_off_validations(omnipool), agent.copy()
    if 'LRNA' not in new_agent.holdings:
        new_agent.holdings['LRNA'] = 0
    for tkn in omnipool.asset_list:
        new_state, new_agent = simulate_remove_liquidity(new_state, new_agent, tkn_remove=tkn)

    agent_holdings = new_agent.holdings
    lrna_removed = {tkn: omnipool.lrna[tkn] - new_state.lrna[tkn] for tkn in omnipool.lrna}
    liquidity_removed = {tkn: omnipool.liquidity[tkn] - new_state.liquidity[tkn] for tkn in omnipool.liquidity}

    if 'LRNA' in prices:
        raise ValueError('LRNA price should not be given.')

    if 'LRNA' not in prices and agent_holdings['LRNA'] > 0:
        lrna_total = omnipool.lrna_total - sum(lrna_removed.values())
        lrna_sells = {
            tkn: -(omnipool.lrna[tkn] - lrna_removed[tkn]) / lrna_total * agent_holdings['LRNA']
            for tkn in omnipool.asset_list
        }

        agent_holdings['LRNA'] = 0
        lrna_profits = dict()

        # sell LRNA optimally back to the pool
        for tkn, delta_qa in lrna_sells.items():
            if tkn not in agent_holdings:
                agent_holdings[tkn] = 0
            asset_fee = (
                omnipool.asset_fee[tkn].compute()
                if hasattr(omnipool, 'asset_fee')
                else omnipool.last_fee[tkn]
            )
            lrna_profits[tkn] = (
                    -(omnipool.liquidity[tkn] - liquidity_removed[tkn]) * delta_qa
                    / (-delta_qa + omnipool.lrna[tkn] - lrna_removed[tkn])
                    * (1 - asset_fee)
            )
            agent_holdings[tkn] += lrna_profits[tkn]

    del agent_holdings['LRNA']

    for tkn in agent_holdings.keys():
        if agent_holdings[tkn] > 0 and tkn not in prices:
            raise ValueError(f'Agent has holdings in {tkn} but no price was given.')

    return value_assets(prices, agent_holdings)
