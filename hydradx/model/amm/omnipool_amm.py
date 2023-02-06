import copy
from numbers import Number
from typing import Callable

from . import stableswap_amm as stableswap
from .agents import Agent
from .amm import AMM, FeeMechanism, basic_fee
from .oracle import Oracle, Block
from .stableswap_amm import StableSwapPoolState


class OmnipoolState(AMM):
    unique_id: str = 'omnipool'

    def __init__(self,
                 tokens: dict[str: dict],
                 tvl_cap: float = float('inf'),
                 preferred_stablecoin: str = "USD",
                 asset_fee: dict or FeeMechanism or float = None,
                 lrna_fee: dict or FeeMechanism or float = None,
                 oracles: dict[str: int] = None,
                 trade_limit_per_block: float = float('inf'),
                 update_function: Callable = None,
                 last_asset_fee: dict or float = None,
                 last_lrna_fee: dict or float = None,
                 imbalance: float = 0.0,
                 last_oracle_values: dict = None,
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
        if preferred_stablecoin not in tokens:
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
        self.stablecoin = preferred_stablecoin
        self.fail = ''
        self.sub_pools = {}  # require sub_pools to be added through create_sub_pool
        self.update_function = update_function
        self.oracles = {
            name: Oracle(sma_equivalent_length=period, first_block=Block(self), last_values=last_oracle_values[name])
            for name, period in oracles.items()
        } if oracles else {}

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
            return {tkn: value.assign(self, tkn) for tkn in self.asset_list}
        else:
            return {tkn: basic_fee(value or 0).assign(self, tkn) for tkn in self.asset_list}

    def add_token(
            self,
            tkn: str,
            liquidity: float,
            lrna: float,
            shares: float,
            protocol_shares: float,
            weight_cap: float = 1
    ):
        self.asset_list.append(tkn)
        self.liquidity[tkn] = liquidity
        self.lrna[tkn] = lrna
        self.shares[tkn] = shares
        self.protocol_shares[tkn] = protocol_shares
        self.weight_cap[tkn] = weight_cap
        if hasattr(self, 'asset_fee'):
            self.asset_fee[tkn] = basic_fee(self.default_asset_fee).assign(self)
            self.lrna_fee[tkn] = basic_fee(self.default_lrna_fee).assign(self)
            self.last_fee[tkn] = 0
            self.last_lrna_fee[tkn] = 0

    def remove_token(self, tkn: str):
        self.asset_list.remove(tkn)

    def update(self):
        # update oracles
        for name, oracle in self.oracles.items():
            oracle.update(self.current_block)

        # self.last_fee = {tkn: self.asset_fee[tkn].compute(tkn) for tkn in self.asset_list}
        # self.last_lrna_fee = {tkn: self.lrna_fee[tkn].compute(tkn) for tkn in self.asset_list}

        # update current block
        self.time_step += 1
        self.current_block = Block(self)
        if self.update_function:
            self.update_function(self)
        return self

    @property
    def lrna_total(self):
        return sum(self.lrna.values())

    @property
    def total_value_locked(self):
        # base this just on the LRNA/USD exchange rate in the pool
        return self.liquidity[self.stablecoin] * self.lrna_total / self.lrna[self.stablecoin]

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
        prices = {tkn: round(usd_price(self, tkn), precision) for tkn in self.asset_list}
        newline = '\n'
        return (
            f'Omnipool: {self.unique_id}\n'
            f'********************************\n'
            f'tvl cap: {self.tvl_cap}\n'
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
                    f'    USD price: {prices[tkn]}\n' +
                    f'    tvl: ${lrna[tkn] * liquidity[self.stablecoin] / lrna[self.stablecoin]}\n'
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
        ) + '\n)' + f'\n\nerror message: {self.fail or "None"}'


class OmnipoolArchiveState:
    def __init__(self, state: OmnipoolState):
        self.asset_list = [tkn for tkn in state.asset_list]
        self.liquidity = {k: v for (k, v) in state.liquidity.items()}
        self.lrna = {k: v for (k, v) in state.lrna.items()}
        self.lrna_total = sum(self.lrna.values())
        self.shares = {k: v for (k, v) in state.shares.items()}
        self.protocol_shares = {k: v for (k, v) in state.protocol_shares.items()}
        self.lrna_imbalance = state.lrna_imbalance
        self.fail = state.fail
        self.stablecoin = state.stablecoin
        # self.sub_pools = copy.deepcopy(state.sub_pools)
        self.oracles = {k: v for (k, v) in state.oracles.items()}
        self.unique_id = state.unique_id

        # record these for analysis later
        self.last_fee = {k: v for (k, v) in state.last_fee.items()}
        self.last_lrna_fee = {k: v for (k, v) in state.last_lrna_fee.items()}


# Works with OmnipoolState *or* OmnipoolArchiveState
def price(state: OmnipoolState or OmnipoolArchiveState, tkn: str, denominator: str = '') -> float:
    """
    price of an asset i denominated in j, according to current market conditions in the omnipool
    """
    if state.liquidity[tkn] == 0:
        return 0
    elif not denominator:
        return lrna_price(state, tkn)
    return state.lrna[tkn] / state.liquidity[tkn] / state.lrna[denominator] * state.liquidity[denominator]


def usd_price(state: OmnipoolState or OmnipoolArchiveState, tkn):
    return price(state, tkn) / price(state, state.stablecoin)


def lrna_price(state: OmnipoolState or OmnipoolArchiveState, i: str, fee: float = 0) -> float:
    """Price of i denominated in LRNA"""
    if state.liquidity[i] == 0:
        return 0
    else:
        return (state.lrna[i] / state.liquidity[i]) * (1 - fee)


def calculate_sell_from_buy(
        state: OmnipoolState,
        tkn_buy: str,
        tkn_sell: str,
        buy_quantity: float
):
    """
    Given a buy quantity, calculate the effective price, so we can execute it as a sell
    """

    asset_fee = state.asset_fee[tkn_buy].compute(tkn=tkn_buy, delta_tkn=-buy_quantity)
    # asset_fee = state.last_fee[tkn_buy]
    delta_Qj = state.lrna[tkn_buy] * buy_quantity / (
            state.liquidity[tkn_buy] * (1 - asset_fee) - buy_quantity)
    lrna_fee = state.lrna_fee[tkn_sell].compute(tkn=tkn_sell, delta_tkn=(
            state.liquidity[tkn_buy] * delta_Qj /
            (state.lrna[tkn_buy] - delta_Qj)
    ))
    # lrna_fee = state.last_lrna_fee[tkn_sell]
    delta_Qi = -delta_Qj / (1 - lrna_fee)
    delta_Ri = -state.liquidity[tkn_sell] * delta_Qi / (state.lrna[tkn_sell] + delta_Qi)
    return delta_Ri


def get_sub_pool(state: OmnipoolState, tkn: str):
    # if asset in not in omnipool, return the ID of the sub_pool where it can be found
    if tkn in state.asset_list:
        return ''
    else:
        for pool in state.sub_pools.values():
            if tkn in pool.asset_list:
                return pool.unique_id


def execute_swap(
        state: OmnipoolState,
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
    old_buy_liquidity = state.liquidity[tkn_buy] if tkn_buy in state.liquidity else 0
    old_sell_liquidity = state.liquidity[tkn_sell] if tkn_sell in state.liquidity else 0

    if tkn_buy not in state.asset_list + ['LRNA'] or tkn_sell not in state.asset_list + ['LRNA']:
        # note: this default routing behavior assumes that an asset will only exist in one place in the omnipool
        return_val = execute_stable_swap(
            state=state,
            agent=agent,
            sub_pool_buy_id=get_sub_pool(state=state, tkn=tkn_buy),
            sub_pool_sell_id=get_sub_pool(state=state, tkn=tkn_sell),
            tkn_sell=tkn_sell, tkn_buy=tkn_buy,
            buy_quantity=buy_quantity,
            sell_quantity=sell_quantity
        )

    elif tkn_sell == 'LRNA':
        return_val = execute_lrna_swap(state, agent, buy_quantity, -sell_quantity, tkn_buy, modify_imbalance)
    elif tkn_buy == 'LRNA':
        return_val = execute_lrna_swap(state, agent, -sell_quantity, buy_quantity, tkn_sell, modify_imbalance)

    elif buy_quantity and not sell_quantity:
        # back into correct delta_Ri, then execute sell
        delta_Ri = calculate_sell_from_buy(state, tkn_buy, tkn_sell, buy_quantity)
        # including both buy_quantity and sell_quantity potentially introduces a 'hack'
        # where you could include both and *not* have them match, but we're not worried about that
        # because this is not a production environment. Just don't do it.
        return execute_swap(
            state=state,
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
            return state.fail_transaction('sell amount must be greater than zero', agent)

        delta_Qi = state.lrna[tkn_sell] * -delta_Ri / (state.liquidity[tkn_sell] + delta_Ri)
        asset_fee = state.asset_fee[tkn_sell].compute(tkn=tkn_sell, delta_tkn=sell_quantity)
        # asset_fee = state.last_fee[tkn_sell]
        lrna_fee = state.lrna_fee[tkn_buy].compute(
            tkn=tkn_buy,
            delta_tkn=(state.liquidity[tkn_buy] * sell_quantity
                       / (state.lrna[tkn_buy] + sell_quantity) * (1 - asset_fee))
        )
        # lrna_fee = state.last_lrna_fee[tkn_buy]

        delta_Qj = -delta_Qi * (1 - lrna_fee)
        delta_Rj = state.liquidity[tkn_buy] * -delta_Qj / (state.lrna[tkn_buy] + delta_Qj) * (1 - asset_fee)
        delta_L = min(-delta_Qi * lrna_fee, -state.lrna_imbalance)
        delta_QH = -lrna_fee * delta_Qi - delta_L

        if state.liquidity[i] + sell_quantity > 10 ** 12:
            return state.fail_transaction('Asset liquidity cannot exceed 10 ^ 12.', agent)

        if agent.holdings[i] < sell_quantity:
            return state.fail_transaction(f"Agent doesn't have enough {i}", agent)

        # per-block trade limits
        if (
                -delta_Rj - state.current_block.volume_in[tkn_buy] + state.current_block.volume_out[tkn_buy]
                > state.trade_limit_per_block * state.current_block.liquidity[tkn_buy]
        ):
            return state.fail_transaction(
                f'{state.trade_limit_per_block * 100}% per block trade limit exceeded in {tkn_buy}.', agent
            )
        elif (
                delta_Ri + state.current_block.volume_in[tkn_sell] - state.current_block.volume_out[tkn_sell]
                > state.trade_limit_per_block * state.current_block.liquidity[tkn_sell]
        ):
            return state.fail_transaction(
                f'{state.trade_limit_per_block * 100}% per block trade limit exceeded in {tkn_sell}.', agent
            )
        state.lrna[i] += delta_Qi
        state.lrna[j] += delta_Qj
        state.liquidity[i] += delta_Ri
        state.liquidity[j] += -buy_quantity or delta_Rj
        state.lrna['HDX'] += delta_QH
        state.lrna_imbalance += delta_L

        if j not in agent.holdings:
            agent.holdings[j] = 0
        agent.holdings[i] -= delta_Ri
        agent.holdings[j] -= -buy_quantity or delta_Rj

        return_val = state, agent

    # update oracle
    if tkn_buy in state.current_block.asset_list:
        buy_quantity = old_buy_liquidity - state.liquidity[tkn_buy]
        state.current_block.volume_out[tkn_buy] += buy_quantity
        # state.current_block.price[tkn_buy] = state.lrna[tkn_buy] / state.liquidity[tkn_buy]
    if tkn_sell in state.current_block.asset_list:
        sell_quantity = state.liquidity[tkn_sell] - old_sell_liquidity
        state.current_block.volume_in[tkn_sell] += sell_quantity
        # state.current_block.price[tkn_sell] = state.lrna[tkn_sell] / state.liquidity[tkn_sell]
    return return_val


def execute_lrna_swap(
        state: OmnipoolState,
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
        asset_fee = state.asset_fee[tkn].compute(
            tkn=tkn, delta_tkn=state.liquidity[tkn] * delta_qa / (delta_qa + state.lrna[tkn])
        )
        if -delta_qa + state.lrna[tkn] <= 0:
            return state.fail_transaction('insufficient lrna in pool', agent)
        delta_ra = -state.liquidity[tkn] * delta_qa / (-delta_qa + state.lrna[tkn]) * (1 - asset_fee)

        if modify_imbalance:
            q = state.lrna_total
            state.lrna_imbalance += delta_qa * (q + state.lrna_imbalance) / (q - delta_qa) + delta_qa

        state.lrna[tkn] += -delta_qa
        state.liquidity[tkn] += -delta_ra

    elif delta_ra > 0:
        asset_fee = state.asset_fee[tkn].compute(tkn=tkn, delta_tkn=delta_ra)
        if -delta_ra + state.liquidity[tkn] <= 0:
            return state.fail_transaction('insufficient assets in pool', agent)
        delta_qa = -state.lrna[tkn] * delta_ra / (state.liquidity[tkn] * (1 - asset_fee) - delta_ra)

        if modify_imbalance:
            q = state.lrna_total
            state.lrna_imbalance += delta_qa * (q + state.lrna_imbalance) / (q - delta_qa) + delta_qa

        state.lrna[tkn] += -delta_qa
        state.liquidity[tkn] += -delta_ra

    # buying LRNA
    elif delta_qa > 0:
        lrna_fee = state.lrna_fee[tkn].compute(
            tkn=tkn, delta_tkn=state.liquidity[tkn] * delta_qa / (delta_qa + state.lrna[tkn])
        )
        delta_qi = -delta_qa / (1 - lrna_fee)
        if delta_qi + state.lrna[tkn] <= 0:
            return state.fail_transaction('insufficient lrna in pool', agent)
        delta_ra = -state.liquidity[tkn] * -delta_qi / (delta_qi + state.lrna[tkn])
        state.lrna[tkn] += delta_qi
        state.liquidity[tkn] += -delta_ra
        # if modify_imbalance:
        #     state.lrna_imbalance += - delta_qi * (q + l) / (q + delta_qi) - delta_qi

        # we assume, for now, that buying LRNA is only possible when modify_imbalance = False
        lrna_fee_amt = -(delta_qa + delta_qi)
        delta_l = min(-state.lrna_imbalance, lrna_fee_amt)
        state.lrna_imbalance += delta_l
        state.lrna["HDX"] += lrna_fee_amt - delta_l

    elif delta_ra < 0:
        lrna_fee = state.lrna_fee[tkn].compute(tkn=tkn, delta_tkn=delta_ra)
        # delta_ri = -delta_ra
        if state.liquidity[tkn] - delta_ra <= 0:
            return state.fail_transaction('insufficient assets in pool', agent)
        delta_qi = state.lrna[tkn] * delta_ra / (state.liquidity[tkn] - delta_ra)
        delta_qa = -delta_qi * (1 - lrna_fee)
        state.lrna[tkn] += delta_qi
        state.liquidity[tkn] -= delta_ra
        # if modify_imbalance:
        #     state.lrna_imbalance += - delta_qi * (q + l) / (q + delta_qi) - delta_qi

        # we assume, for now, that buying LRNA is only possible when modify_imbalance = False
        lrna_fee_amt = -(delta_qa + delta_qi)
        delta_l = min(-state.lrna_imbalance, lrna_fee_amt)
        state.lrna_imbalance += delta_l
        state.lrna["HDX"] += lrna_fee_amt - delta_l

    else:
        return state.fail_transaction('All deltas are zero.', agent)

    agent.holdings['LRNA'] += delta_qa
    agent.holdings[tkn] += delta_ra

    return state, agent


def execute_stable_swap(
        state: OmnipoolState,
        agent: Agent,
        tkn_sell: str, tkn_buy: str,
        sub_pool_buy_id: str = "",
        sub_pool_sell_id: str = "",
        buy_quantity: float = 0,
        sell_quantity: float = 0
) -> tuple[AMM, Agent]:
    if tkn_sell == 'LRNA':
        if buy_quantity:
            sub_pool = state.sub_pools[sub_pool_buy_id]
            # buy a specific quantity of a stableswap asset using LRNA
            shares_needed = sub_pool.calculate_withdrawal_shares(tkn_remove=tkn_buy, quantity=buy_quantity)
            execute_lrna_swap(state, agent, delta_ra=shares_needed, tkn=sub_pool.unique_id)
            if state.fail:
                # if the swap failed, the transaction failed.
                return state.fail_transaction(state.fail, agent)
            stableswap.execute_withdraw_asset(sub_pool, agent, buy_quantity, tkn_buy)
            return state, agent
        elif sell_quantity:
            sub_pool = state.sub_pools[sub_pool_buy_id]
            agent_shares = agent.holdings[sub_pool.unique_id]
            execute_swap(
                state=state,
                agent=agent,
                tkn_buy=sub_pool.unique_id, tkn_sell='LRNA',
                sell_quantity=sell_quantity
            )
            if state.fail:
                # if the swap failed, the transaction failed.
                return state.fail_transaction(state.fail, agent)
            delta_shares = agent.holdings[sub_pool.unique_id] - agent_shares
            stableswap.execute_remove_liquidity(sub_pool, agent, delta_shares, tkn_buy)
            return state, agent

    elif sub_pool_sell_id and tkn_buy in state.asset_list:
        sub_pool: StableSwapPoolState = state.sub_pools[sub_pool_sell_id]
        if sell_quantity:
            # sell a stableswap asset for an omnipool asset
            agent_shares = agent.holdings[sub_pool.unique_id] if sub_pool.unique_id in agent.holdings else 0
            stableswap.execute_add_liquidity(sub_pool, agent, sell_quantity, tkn_sell)
            if state.fail:
                # the transaction failed.
                return state.fail_transaction(state.fail, agent)
            delta_shares = agent.holdings[sub_pool.unique_id] - agent_shares
            execute_swap(
                state=state,
                agent=agent,
                tkn_buy=tkn_buy,
                tkn_sell=sub_pool.unique_id,
                sell_quantity=delta_shares
            )
            return state, agent
        elif buy_quantity:
            # buy an omnipool asset with a stableswap asset
            sell_shares = calculate_sell_from_buy(state, tkn_buy, sub_pool.unique_id, buy_quantity)
            if sell_shares < 0:
                return state.fail_transaction("Not enough liquidity in the stableswap/LRNA pool.", agent)
            stableswap.execute_buy_shares(sub_pool, agent, sell_shares, tkn_sell)
            if sub_pool.fail:
                return state.fail_transaction(sub_pool.fail, agent)
            execute_swap(state, agent, tkn_buy, sub_pool.unique_id, buy_quantity)
            return state, agent

    elif sub_pool_buy_id and tkn_sell in state.asset_list:
        sub_pool: StableSwapPoolState = state.sub_pools[sub_pool_buy_id]
        if buy_quantity:
            # buy a stableswap asset with an omnipool asset
            shares_traded = sub_pool.calculate_withdrawal_shares(tkn_buy, buy_quantity)

            # buy shares in the subpool
            execute_swap(state, agent, tkn_buy=sub_pool.unique_id, tkn_sell=tkn_sell, buy_quantity=shares_traded)
            if state.fail:
                # if the swap failed, the transaction failed.
                return state.fail_transaction(state.fail, agent)
            # withdraw the shares for the desired token
            stableswap.execute_withdraw_asset(sub_pool, agent, quantity=buy_quantity, tkn_remove=tkn_buy)
            if sub_pool.fail:
                return state.fail_transaction(sub_pool.fail, agent)
            return state, agent
        elif sell_quantity:
            # sell an omnipool asset for a stableswap asset
            agent_shares = agent.holdings[sub_pool.unique_id] if sub_pool.unique_id in agent.holdings else 0
            execute_swap(
                state=state,
                agent=agent,
                tkn_buy=sub_pool.unique_id,
                tkn_sell=tkn_sell,
                sell_quantity=sell_quantity
            )
            delta_shares = agent.holdings[sub_pool.unique_id] - agent_shares
            if state.fail:
                return state.fail_transaction(state.fail, agent)
            stableswap.execute_remove_liquidity(
                state=sub_pool, agent=agent, shares_removed=delta_shares, tkn_remove=tkn_buy
            )
            return state, agent
    elif sub_pool_buy_id and sub_pool_sell_id:
        # trade between two subpools
        pool_buy: StableSwapPoolState = state.sub_pools[sub_pool_buy_id]
        pool_sell: StableSwapPoolState = state.sub_pools[sub_pool_sell_id]
        if buy_quantity:
            # buy enough shares of tkn_sell to afford buy_quantity worth of tkn_buy
            shares_bought = pool_buy.calculate_withdrawal_shares(tkn_buy, buy_quantity)
            if shares_bought > pool_buy.liquidity[tkn_buy]:
                return state.fail_transaction(f'Not enough liquidity in {pool_buy.unique_id}: {tkn_buy}.', agent)
            shares_sold = calculate_sell_from_buy(
                state=state,
                tkn_buy=pool_buy.unique_id,
                tkn_sell=pool_sell.unique_id,
                buy_quantity=shares_bought
            )
            stableswap.execute_buy_shares(
                state=pool_sell,
                agent=agent, quantity=shares_sold,
                tkn_add=tkn_sell
            )
            if pool_sell.fail:
                return state.fail_transaction(pool_sell.fail, agent)
            execute_swap(
                state=state,
                agent=agent,
                tkn_buy=pool_buy.unique_id, tkn_sell=pool_sell.unique_id,
                buy_quantity=shares_bought
            )
            if state.fail:
                return state.fail_transaction(state.fail, agent)
            stableswap.execute_withdraw_asset(
                state=pool_buy,
                agent=agent, quantity=buy_quantity,
                tkn_remove=tkn_buy, fail_on_overdraw=False
            )
            if pool_buy.fail:
                return state.fail_transaction(pool_buy.fail, agent)

            # if all three parts succeeded, then we're good!
            return state, agent
        elif sell_quantity:
            agent_sell_holdings = agent.holdings[sub_pool_sell_id] if sub_pool_sell_id in agent.holdings else 0
            stableswap.execute_add_liquidity(
                state=pool_sell,
                agent=agent, quantity=sell_quantity, tkn_add=tkn_sell
            )
            if pool_sell.fail:
                return state.fail_transaction(pool_sell.fail, agent)
            delta_sell_holdings = agent.holdings[sub_pool_sell_id] - agent_sell_holdings
            agent_buy_holdings = agent.holdings[sub_pool_buy_id] if sub_pool_buy_id in agent.holdings else 0
            execute_swap(
                state=state,
                agent=agent,
                tkn_buy=pool_buy.unique_id, tkn_sell=pool_sell.unique_id,
                sell_quantity=delta_sell_holdings
            )
            if state.fail:
                return state.fail_transaction(state.fail, agent)
            delta_buy_holdings = agent.holdings[sub_pool_buy_id] - agent_buy_holdings
            stableswap.execute_remove_liquidity(
                state=pool_buy, agent=agent, shares_removed=delta_buy_holdings, tkn_remove=tkn_buy
            )
            if pool_buy.fail:
                return state.fail_transaction(pool_buy.fail, agent)
            return state, agent
    else:
        raise ValueError('buy_quantity or sell_quantity must be specified.')


def execute_create_sub_pool(
        state: OmnipoolState,
        tkns_migrate: list[str],
        sub_pool_id: str,
        amplification: float,
        trade_fee: FeeMechanism or float = 0
):
    new_sub_pool = StableSwapPoolState(
        tokens={tkn: state.liquidity[tkn] for tkn in tkns_migrate},
        amplification=amplification,
        unique_id=sub_pool_id,
        trade_fee=trade_fee
    )
    new_sub_pool.conversion_metrics = {
        tkn: {
            'price': state.lrna[tkn] / state.liquidity[tkn],
            'old_shares': state.shares[tkn],
            'omnipool_shares': state.lrna[tkn],
            'subpool_shares': state.lrna[tkn]
        } for tkn in tkns_migrate
    }
    new_sub_pool.shares = sum([state.lrna[tkn] for tkn in tkns_migrate])
    state.sub_pools[sub_pool_id] = new_sub_pool
    state.add_token(
        sub_pool_id,
        liquidity=sum([state.lrna[tkn] for tkn in tkns_migrate]),
        shares=sum([state.lrna[tkn] for tkn in tkns_migrate]),
        lrna=sum([state.lrna[tkn] for tkn in tkns_migrate]),
        protocol_shares=sum([
            state.lrna[tkn] * state.protocol_shares[tkn] / state.shares[tkn] for tkn in tkns_migrate
        ])
    )

    # remove assets from Omnipool
    for tkn in tkns_migrate:
        state.liquidity[tkn] = 0
        state.lrna[tkn] = 0
        state.asset_list.remove(tkn)
    return state


def execute_migrate_asset(state: OmnipoolState, tkn_migrate: str, sub_pool_id: str):
    """
    Move an asset from the Omnipool into a stableswap subpool.
    """
    sub_pool: StableSwapPoolState = state.sub_pools[sub_pool_id]
    s = sub_pool.unique_id
    i = tkn_migrate
    if tkn_migrate in sub_pool.liquidity:
        raise AssertionError('Assets should only exist in one place in the Omnipool at a time.')
    sub_pool.liquidity[i] = state.liquidity[i]
    state.protocol_shares[s] += (
            state.shares[s] * state.lrna[i] / state.lrna[s] * state.protocol_shares[i] / state.shares[i]
    )

    sub_pool.conversion_metrics[i] = {
        'price': state.lrna[i] / state.lrna[s] * sub_pool.shares / state.liquidity[i],
        'old_shares': state.shares[i],
        'omnipool_shares': state.lrna[i] * state.shares[s] / state.lrna[s],
        'subpool_shares': state.lrna[i] * sub_pool.shares / state.lrna[s]
    }

    state.shares[s] += state.lrna[i] * state.shares[s] / state.lrna[s]
    state.liquidity[s] += state.lrna[i] * sub_pool.shares / state.lrna[s]
    sub_pool.shares += state.lrna[i] * sub_pool.shares / state.lrna[s]
    state.lrna[s] += state.lrna[i]

    # remove asset from omnipool and add it to subpool
    state.lrna[i] = 0
    state.liquidity[i] = 0
    state.asset_list.remove(i)
    sub_pool.asset_list.append(i)
    return state


def execute_migrate_lp(
        state: OmnipoolState,
        agent: Agent,
        sub_pool_id: str,
        tkn_migrate: str
):
    sub_pool = state.sub_pools[sub_pool_id]
    conversions = sub_pool.conversion_metrics[tkn_migrate]
    old_pool_id = (state.unique_id, tkn_migrate)
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
    state.liquidity[sub_pool_id] += (
            agent.holdings[old_pool_id] / conversions['old_shares'] * conversions['subpool_shares']
    )
    agent.holdings[old_pool_id] = 0

    return state, agent


def calculate_remove_liquidity(state: OmnipoolState, agent: Agent, quantity: float, tkn_remove: str):
    """
    calculated the pool and agent deltas for removing liquidity from a sub pool
    return as a tuple in this order:
    delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l

    delta_qa (agent LRNA)
    delta_r (pool liquidity)
    delta_q (pool LRNA)
    delta_s (pool shares)
    delta_b (protocol shares)
    delta_l (LRNA imbalance)
    """
    quantity = -abs(quantity)
    assert quantity <= 0, f"delta_S cannot be positive: {quantity}"
    # assert tkn_remove in state.asset_list, f"invalid token name: {tkn_remove}"
    if tkn_remove not in state.asset_list:
        return 0, 0, 0, 0, 0, 0

    if (state.unique_id, tkn_remove) not in agent.share_prices:
        return 0, 0, 0, 0, 0, 0

    # determine if they should get some LRNA back as well as the asset they invested
    piq = lrna_price(state, tkn_remove)
    p0 = agent.share_prices[(state.unique_id, tkn_remove)]
    mult = (piq - p0) / (piq + p0)

    # Share update
    delta_b = max(mult * quantity, 0)
    delta_s = quantity + delta_b

    # Token amounts update
    delta_r = state.liquidity[tkn_remove] * max((quantity + delta_b) / state.shares[tkn_remove], -1)

    if piq >= p0:  # prevents rounding errors
        if 'LRNA' not in agent.holdings:
            agent.holdings['LRNA'] = 0
        delta_qa = -piq * (
                2 * piq / (piq + p0) * quantity / state.shares[tkn_remove]
                * state.liquidity[tkn_remove] - delta_r
        )
    else:
        delta_qa = 0

    # LRNA burn
    delta_q = lrna_price(state, tkn_remove) * delta_r

    # L update: LRNA fees to be burned before they will start to accumulate again
    delta_l = (
            delta_r * state.lrna[tkn_remove] / state.liquidity[tkn_remove]
            * state.lrna_imbalance / state.lrna_total
    )

    return delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l


def execute_add_liquidity(
        state: OmnipoolState,
        agent: Agent = None,
        quantity: float = 0,
        tkn_add: str = ''
) -> tuple[OmnipoolState, Agent]:
    """Compute new state after liquidity addition"""

    delta_Q = lrna_price(state, tkn_add) * quantity
    if not (state.unique_id, tkn_add) in agent.holdings:
        agent.holdings[(state.unique_id, tkn_add)] = 0

    quantity = quantity or state.trade_limit_per_block * state.liquidity[tkn_add]
    if agent.holdings[tkn_add] < quantity:
        return state.fail_transaction(f'Agent has insufficient funds ({agent.holdings[tkn_add]} < {quantity}).', agent)

    if (state.lrna[tkn_add] + delta_Q) / (state.lrna_total + delta_Q) > state.weight_cap[tkn_add]:
        return state.fail_transaction(
            'Transaction rejected because it would exceed the weight cap in pool[{i}].', agent
        )

    if (state.total_value_locked + quantity * usd_price(state, tkn_add)) > state.tvl_cap:
        return state.fail_transaction('Transaction rejected because it would exceed the TVL cap.', agent)

    if (state.liquidity[tkn_add] + quantity) > 10 ** 12:
        # TODO: this may not actually exist as part of the protocol
        return state.fail_transaction('Asset liquidity cannot exceed 10 ^ 12.', agent)

    # assert quantity > 0, f"delta_R must be positive: {quantity}"
    if tkn_add not in state.asset_list:
        for sub_pool in state.sub_pools.values():
            if tkn_add in sub_pool.asset_list:
                old_agent_holdings = agent.holdings[sub_pool.unique_id] if sub_pool.unique_id in agent.holdings else 0
                stableswap.execute_add_liquidity(
                    state=sub_pool,
                    agent=agent,
                    quantity=quantity,
                    tkn_add=tkn_add
                )
                # deposit into the Omnipool
                return execute_add_liquidity(
                    state, agent,
                    quantity=agent.holdings[sub_pool.unique_id] - old_agent_holdings,
                    tkn_add=sub_pool.unique_id
                )
        raise AssertionError(f"invalid value for i: {tkn_add}")

    # Share update
    if state.shares[tkn_add]:
        shares_added = ((state.liquidity[tkn_add] + quantity) / state.liquidity[tkn_add] - 1) * state.shares[tkn_add]
    else:
        shares_added = quantity
    state.shares[tkn_add] += shares_added

    # shares go to provisioning agent
    agent.holdings[(state.unique_id, tkn_add)] += shares_added

    # L update: LRNA fees to be burned before they will start to accumulate again
    delta_L = (
            quantity * state.lrna[tkn_add] / state.liquidity[tkn_add]
            * state.lrna_imbalance / state.lrna_total
    )
    state.lrna_imbalance += delta_L

    # LRNA add (mint)
    state.lrna[tkn_add] += delta_Q

    # Token amounts update
    state.liquidity[tkn_add] += quantity
    agent.holdings[tkn_add] -= quantity

    # set price at which liquidity was added
    agent.share_prices[(state.unique_id, tkn_add)] = lrna_price(state, tkn_add)

    return state, agent


def execute_remove_liquidity(state: OmnipoolState, agent: Agent, quantity: float, tkn_remove: str):
    """
    Remove liquidity from a sub pool.
    """

    if quantity == 0:
        return state, agent

    if tkn_remove not in state.asset_list:
        for sub_pool in state.sub_pools.values():
            if tkn_remove in sub_pool.asset_list:
                stableswap.execute_remove_liquidity(
                    sub_pool, agent, quantity, tkn_remove
                )
                if sub_pool.fail:
                    return state.fail_transaction(sub_pool.fail, agent)
                else:
                    return state, agent

        raise AssertionError(f"invalid value for i: {tkn_remove}")

    quantity = abs(quantity)
    delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l = calculate_remove_liquidity(
        state, agent, quantity, tkn_remove
    )
    if delta_r + state.liquidity[tkn_remove] < 0:
        return state.fail_transaction('Cannot remove more liquidity than exists in the pool.', agent)
    elif quantity > agent.holdings[(state.unique_id, tkn_remove)]:
        return state.fail_transaction('Cannot remove more liquidity than the agent has invested.', agent)

    state.liquidity[tkn_remove] += delta_r
    state.shares[tkn_remove] += delta_s
    state.protocol_shares[tkn_remove] += delta_b
    state.lrna[tkn_remove] += delta_q
    state.lrna_imbalance += delta_l
    agent.holdings['LRNA'] += delta_qa
    agent.holdings[(state.unique_id, tkn_remove)] -= quantity
    agent.holdings[tkn_remove] -= delta_r
    return state, agent


def asset_invariant(state: OmnipoolState, i: str) -> float:
    """Invariant for specific asset"""
    return state.liquidity[i] * state.lrna[i]


def swap_lrna_delta_Qi(state: OmnipoolState, delta_ri: float, i: str) -> float:
    return state.lrna[i] * (- delta_ri / (state.liquidity[i] + delta_ri))


def swap_lrna_delta_Ri(state: OmnipoolState, delta_qi: float, i: str) -> float:
    return state.liquidity[i] * (- delta_qi / (state.lrna[i] + delta_qi))


def weight_i(state: OmnipoolState, i: str) -> float:
    return state.lrna[i] / state.lrna_total


def swap_lrna(
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

    return execute_lrna_swap(new_state, new_agent, delta_ra, delta_qa, tkn, modify_imbalance)


def swap(
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

    execute_swap(
        state=new_state,
        agent=new_agent,
        sell_quantity=sell_quantity,
        buy_quantity=buy_quantity,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
    )

    return new_state, new_agent


def migrate(
        old_state: OmnipoolState,
        tkn_migrate: str,
        sub_pool_id: str
) -> OmnipoolState:
    return execute_migrate_asset(old_state.copy(), tkn_migrate, sub_pool_id)


def add_liquidity(
        old_state: OmnipoolState,
        old_agent: Agent,
        quantity: float = 0,
        tkn_add: str = ''
) -> tuple[OmnipoolState, Agent]:
    """Copy state, then added liquidity and return new state"""
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    return execute_add_liquidity(new_state, new_agent, quantity, tkn_add)


def remove_liquidity(
        old_state: OmnipoolState,
        old_agent: Agent,
        quantity: float,
        tkn_remove: str
) -> tuple[OmnipoolState, Agent]:
    """Compute new state after liquidity removal"""
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    return execute_remove_liquidity(new_state, new_agent, quantity, tkn_remove)


OmnipoolState.swap = staticmethod(swap)
OmnipoolState.execute_swap = staticmethod(execute_swap)
OmnipoolState.add_liquidity = staticmethod(add_liquidity)
OmnipoolState.execute_add_liquidity = staticmethod(execute_add_liquidity)
OmnipoolState.remove_liquidity = staticmethod(remove_liquidity)
OmnipoolState.usd_price = staticmethod(usd_price)
OmnipoolState.lrna_price = staticmethod(lrna_price)
OmnipoolState.price = staticmethod(price)

OmnipoolArchiveState.usd_price = staticmethod(usd_price)
OmnipoolArchiveState.lrna_price = staticmethod(lrna_price)
OmnipoolArchiveState.price = staticmethod(price)


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
    class Fee:
        def __init__(self):
            # force compute on first call
            self.time_step = -1

        def fee_function(
                self, exchange: OmnipoolState, tkn: str, delta_tkn: float = 0
        ) -> float:

            if self.time_step == exchange.time_step:
                # since fee is the same regardless of delta_tkn, just return the last fee
                # if it's already been computed for this block
                return exchange.last_fee[tkn]
            else:
                self.time_step = exchange.time_step

            raise_oracle: Oracle = exchange.oracles[raise_oracle_name]

            if raise_oracle.liquidity[tkn] != 0:
                x = (raise_oracle.volume_out[tkn] - raise_oracle.volume_in[tkn]) / exchange.liquidity[tkn]
            else:
                x = 0

            fee_adj = amplification * x - decay

            previous_fee = exchange.last_fee[tkn]

            fee = min(max(previous_fee + fee_adj, minimum), fee_max)
            exchange.last_fee[tkn] = fee

            return fee

    return FeeMechanism(
        fee_function=Fee().fee_function,
        name=f'Dynamic fee (oracle={raise_oracle_name}, amplification={amplification}, min={minimum})'
    )


def dynamicadd_lrna_fee(
        minimum: float = 0,
        amplification: float = 1,
        raise_oracle_name: str = 'short',
        decay: float = 0.001,
        fee_max: float = 0.5,
) -> FeeMechanism:
    class Fee:
        def __init__(self):
            self.time_step = -1

        def fee_function(
                self, exchange: OmnipoolState, tkn: str, delta_tkn: float = 0
        ) -> float:
            if not hasattr(exchange, 'last_lrna_fee'):
                exchange.last_lrna_fee = {tkn: 0 for tkn in exchange.asset_list}

            raise_oracle: Oracle = exchange.oracles[raise_oracle_name]

            if raise_oracle.liquidity[tkn] != 0:
                x = (raise_oracle.volume_in[tkn] - raise_oracle.volume_out[tkn]) / exchange.liquidity[tkn]
            else:
                x = 0

            fee_adj = amplification * x - decay

            previous_fee = exchange.last_lrna_fee[tkn]

            fee = min(max(previous_fee + fee_adj, minimum), fee_max)
            exchange.last_lrna_fee[tkn] = fee

            return fee

    return FeeMechanism(
        fee_function=Fee().fee_function,
        name=f'Dynamic LRNA fee (oracle={raise_oracle_name}, amplification={amplification}, min={minimum})'
    )


def value_assets(prices: dict, assets: dict) -> float:
    """
    return the value of the agent's assets if they were sold at current spot prices
    """
    return sum([
        assets[i] * prices[i] if i in prices else 0
        for i in assets.keys()
    ])


def cash_out_omnipool(omnipool: OmnipoolState, agent: Agent, prices) -> float:
    """
    return the value of the agent's holdings if they withdraw all liquidity
    and then sell at current spot prices
    """
    if 'LRNA' not in agent.holdings:
        agent.holdings['LRNA'] = 0
    withdraw_holdings = {tkn: agent.holdings[tkn] for tkn in list(agent.holdings.keys())}

    for key in agent.holdings.keys():
        # shares.keys might just be the pool name, or it might be a tuple (pool, token)
        if isinstance(key, tuple):
            tkn = key[1]
            withdraw_holdings[key] = 0
        else:
            tkn = key
        # optimized for omnipool, no copy operations
        delta_qa, delta_r, delta_q, \
            delta_s, delta_b, delta_l = calculate_remove_liquidity(
            omnipool,
            agent,
            agent.holdings[key],
            tkn_remove=tkn
        )
        withdraw_holdings['LRNA'] += delta_qa
        if tkn not in withdraw_holdings:
            withdraw_holdings[tkn] = 0
        withdraw_holdings[tkn] -= delta_r

    return value_assets(prices, withdraw_holdings)
