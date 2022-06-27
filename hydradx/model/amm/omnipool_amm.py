import copy
import string
from .agents import Agent
from .global_state import AMM


class OmnipoolState(AMM):
    def __init__(self,
                 tokens: dict[str: dict],
                 tvl_cap: float = float('inf'),
                 preferred_stablecoin: str = "USD",
                 asset_fee: float = 0,
                 lrna_fee: float = 0,
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
        }
        """
        # TODO: consider what these fields should actually be called, in light of compatibility with Basilisk

        super().__init__()

        if 'HDX' not in tokens:
            raise ValueError('HDX not included in tokens.')

        self.asset_list: list[str] = []
        self.liquidity = {}
        self.lrna = {}
        self.shares = {}
        self.protocol_shares = {}
        self.tvl = {}
        self.weight_cap = {}
        for token, pool in tokens.items():
            assert pool['liquidity'], f'token {token} missing required parameter: liquidity'
            if not ('LRNA' in pool or 'LRNA_price' in pool):
                raise ValueError("token {name} missing required parameter: ('LRNA' or 'LRNA_price)")
            self.asset_list.append(token)
            self.liquidity[token] = (pool['liquidity'])
            self.shares[token] = (pool['liquidity'])
            self.protocol_shares[token] = (pool['liquidity'])
            self.weight_cap[token] = (pool['weight_cap'] if 'weight_cap' in pool else 1)
            if 'LRNA' in pool:
                self.lrna[token] = (pool['LRNA'])
            else:
                self.lrna[token] = (pool['LRNA_price'] * pool['liquidity'])

        self.asset_fee = asset_fee
        self.lrna_fee = lrna_fee
        self.lrna_imbalance = 0  # AKA "L"
        self.tvl_cap = tvl_cap
        self.stablecoin = preferred_stablecoin
        self.fail = ''

        # count TVL for each pool once all values are set
        for token in self.asset_list:
            self.tvl[token] = (
                self.lrna[token] * self.liquidity[self.stablecoin] / self.lrna[self.stablecoin]
            )

    def price(self, i: str):
        return self.lrna[i] / self.liquidity[i]

    @property
    def lrna_total(self):
        return sum(self.lrna.values())

    @property
    def tvl_total(self):
        return sum(self.tvl.values())

    def copy(self):
        copy_state = copy.deepcopy(self)
        copy_state.fail = ''
        return copy_state

    def __repr__(self):
        return (
            f'Omnipool\n'
            f'tvl cap: {self.tvl_cap}\n'
            f'lrna fee: {self.lrna_fee}\n'
            f'asset fee: {self.asset_fee}\n'
            f'asset pools: (\n'
        ) + ')\n(\n'.join(
            [(
                f'    {token}\n'
                f'    asset quantity: {self.liquidity[token]}\n'
                f'    lrna quantity: {self.lrna[token]}\n'
                f'    price: {price_i(self, token)}\n'
                f'    tvl: {self.tvl[token]}\n'
                f'    weight: {self.tvl[token]}/{self.tvl_total} ({self.tvl[token] / self.tvl_total})\n'
                f'    weight cap: {self.weight_cap[token]}\n'
                f'    total shares: {self.shares[token]}\n'
                f'    protocol shares: {self.protocol_shares[token]}\n'
            ) for token in self.asset_list]
        ) + '\n)'


def asset_invariant(state: OmnipoolState, i: str) -> float:
    """Invariant for specific asset"""
    return state.liquidity[i] * state.lrna[i]


def swap_lrna_delta_Qi(state: OmnipoolState, delta_ri: float, i: str) -> float:
    return state.lrna[i] * (- delta_ri / (state.liquidity[i] + delta_ri))


def swap_lrna_delta_Ri(state: OmnipoolState, delta_qi: float, i: str) -> float:
    return state.liquidity[i] * (- delta_qi / (state.lrna[i] + delta_qi))


def weight_i(state: OmnipoolState, i: str) -> float:
    return state.lrna[i] / state.lrna_total


def price_i(state: OmnipoolState, i: str, fee: float = 0) -> float:
    """Price of i denominated in LRNA"""
    if state.liquidity[i] == 0:
        return 0
    else:
        return (state.lrna[i] / state.liquidity[i]) * (1 - fee)


def swap_lrna(
        old_state: OmnipoolState,
        old_agent: Agent,
        delta_ra: float = 0,
        delta_qa: float = 0,
        tkn: str = ''
) -> tuple[OmnipoolState, Agent]:
    """Compute new state after LRNA swap"""

    new_state = old_state.copy()
    new_agent = old_agent.copy()

    if delta_qa < 0:
        delta_Q = -delta_qa
        delta_R = old_state.liquidity[tkn] * -delta_Q / (delta_Q + old_state.lrna[tkn]) * (1 - old_state.asset_fee)
        delta_L = -delta_Q * (1 + (1 - old_state.asset_fee) * old_state.lrna[tkn] / (old_state.lrna[tkn] + delta_Q))
        delta_ra = -delta_R
    elif delta_ra > 0:
        delta_R = -delta_ra
        delta_Q = old_state.lrna[tkn] * -delta_R / (old_state.liquidity[tkn] * (1 - old_state.asset_fee) + delta_R)
        delta_L = -delta_Q * (1 + (1 - old_state.asset_fee) * old_state.lrna[tkn] / (old_state.lrna[tkn] + delta_Q))
        delta_qa = -delta_Q
    else:
        # print(f'Invalid swap (delta_Qa {delta_Qa}, delta_Ra {delta_Ra}')
        return AMM.fail(old_state, old_agent)

    if delta_qa + old_agent.holdings['LRNA'] < 0:
        # agent doesn't have enough lrna
        return AMM.fail(old_state, old_agent)
    elif delta_ra + old_agent.holdings[tkn] < 0:
        # agent doesn't have enough asset[i]
        return AMM.fail(old_state, old_agent)
    elif delta_R + old_state.liquidity[tkn] <= 0:
        # insufficient assets in pool, transaction fails
        return AMM.fail(old_state, old_agent)
    elif delta_Q + old_state.lrna[tkn] <= 0:
        # insufficient lrna in pool, transaction fails
        return AMM.fail(old_state, old_agent)

    new_agent.holdings['LRNA'] += delta_qa
    new_agent.holdings[tkn] += delta_ra
    new_state.lrna[tkn] += delta_Q
    new_state.liquidity[tkn] += delta_R
    new_state.lrna_imbalance += delta_L

    return new_state, new_agent


def swap_assets_direct(
        old_state: OmnipoolState,
        old_agent: Agent,
        delta_token: float,
        tkn_buy: str,
        tkn_sell: str
) -> tuple[OmnipoolState, Agent]:
    i = tkn_sell
    j = tkn_buy
    delta_Ri = delta_token
    assert delta_Ri > 0, 'sell amount must be greater than zero'

    delta_Qi = old_state.lrna[i] * -delta_Ri / (old_state.liquidity[i] + delta_Ri)
    delta_Qj = -delta_Qi * (1 - old_state.lrna_fee)
    delta_Rj = old_state.liquidity[j] * -delta_Qj / (old_state.lrna[j] + delta_Qj) * (1 - old_state.asset_fee)
    delta_L = min(-delta_Qi * old_state.lrna_fee, -old_state.lrna_imbalance)
    delta_QH = -old_state.lrna_fee * delta_Qi - delta_L

    new_state = old_state.copy()
    new_state.lrna[i] += delta_Qi
    new_state.lrna[j] += delta_Qj
    new_state.liquidity[i] += delta_Ri
    new_state.liquidity[j] += delta_Rj
    new_state.lrna['HDX'] += delta_QH
    new_state.lrna_imbalance += delta_L

    new_agent = old_agent.copy()
    new_agent.holdings[i] -= delta_Ri
    new_agent.holdings[j] -= delta_Rj

    return new_state, new_agent


def swap(
        old_state: OmnipoolState,
        old_agent: Agent,
        tkn_buy: str,
        tkn_sell: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
) -> tuple[OmnipoolState, Agent]:

    if tkn_sell == 'LRNA' or tkn_buy == 'LRNA':

        if tkn_sell == 'LRNA':
            delta_qa = sell_quantity or -buy_quantity
            delta_ra = buy_quantity or -sell_quantity
            tkn = tkn_buy

        else:  # tkn_buy == 'LRNA'
            delta_qa = sell_quantity or -buy_quantity
            delta_ra = buy_quantity or -sell_quantity
            tkn = tkn_sell

        swap_lrna(
            old_state=old_state,
            old_agent=old_agent,
            delta_ra=delta_ra,
            delta_qa=delta_qa,
            tkn=tkn
        )

    elif sell_quantity != 0:

        new_state, new_agents = swap_assets_direct(
            old_state=old_state,
            old_agent=old_agent,
            delta_token=sell_quantity,
            tkn_buy=tkn_buy,
            tkn_sell=tkn_sell,
        )

        return new_state, new_agents
    elif buy_quantity != 0:
        # back into correct delta_Ri, then execute sell
        delta_Qj = -old_state.lrna[tkn_buy] * buy_quantity / (
                old_state.liquidity[tkn_buy] * (1 - old_state.asset_fee) + buy_quantity)
        delta_Qi = -delta_Qj/(1 - old_state.lrna_fee)
        delta_Ri = -old_state.liquidity[tkn_sell] * delta_Qi / (old_state.lrna[tkn_sell] + delta_Qi)
        return swap(
            old_state=old_state,
            old_agent=old_agent,
            tkn_buy=tkn_buy,
            tkn_sell=tkn_sell,
            sell_quantity=delta_Ri
        )

    else:
        raise


def add_liquidity(
        old_state: OmnipoolState,
        old_agent: Agent = None,
        delta_r: float = 0,
        tkn: str = ''
) -> tuple[OmnipoolState, Agent]:
    """Compute new state after liquidity addition"""

    assert delta_r > 0, f"delta_R must be positive: {delta_r}"
    assert tkn in old_state.asset_list, f"invalid value for i: {tkn}"

    new_state = old_state.copy()
    new_agent = old_agent.copy()

    # Token amounts update
    new_state.liquidity[tkn] += delta_r

    if old_agent:
        new_agent.holdings[tkn] -= delta_r
        if new_agent.holdings[tkn] < 0:
            # print('Transaction rejected because agent has insufficient funds.')
            # print(f'agent {LP_id}, asset {new_state["token_list"][i]}, amount {delta_R}')
            return AMM.fail(old_state, old_agent)

    # Share update
    if new_state.shares[tkn]:
        new_state.shares[tkn] *= new_state.liquidity[tkn] / old_state.liquidity[tkn]
    else:
        new_state.shares[tkn] = new_state.liquidity[tkn]

    if old_agent:
        # shares go to provisioning agent
        new_agent.shares[tkn] += new_state.shares[tkn] - old_state.shares[tkn]
    else:
        # shares go to protocol
        new_state.protocol_shares[tkn] += new_state.shares[tkn] - old_state.shares[tkn]

    # LRNA add (mint)
    delta_Q = price_i(old_state, tkn) * delta_r
    new_state.lrna[tkn] += delta_Q

    # L update: LRNA fees to be burned before they will start to accumulate again
    delta_L = delta_r * old_state.lrna[tkn] / old_state.liquidity[tkn] * old_state.lrna_imbalance / old_state.lrna_total
    new_state.lrna_imbalance += delta_L

    # T update: TVL soft cap
    usd = new_state.stablecoin
    delta_t = new_state.lrna[tkn] * new_state.liquidity[usd] / new_state.lrna[usd] - new_state.tvl[tkn]
    new_state.tvl[tkn] += delta_t

    if new_state.lrna[tkn] / new_state.lrna_total > new_state.weight_cap[tkn]:
        # print(f'Transaction rejected because it would exceed the weight cap in pool[{i}].')
        # print(f'agent {LP_id}, asset {new_state["token_list"][i]}, amount {delta_R}')
        return AMM.fail(old_state, old_agent)

    if new_state.tvl_total > new_state.tvl_cap:
        # print('Transaction rejected because it would exceed the TVL cap.')
        # print(f'agent {LP_id}, asset {new_state["token_list"][i]}, amount {delta_R}')
        return AMM.fail(old_state, old_agent)

    # set price at which liquidity was added
    if old_agent:
        new_agent.share_prices[tkn] = new_state.price(tkn)

    return new_state, new_agent


def remove_liquidity(
        old_state: OmnipoolState,
        old_agent: Agent,
        delta_s: float,
        tkn: str
) -> tuple[OmnipoolState, Agent]:
    """Compute new state after liquidity removal"""
    assert delta_s <= 0, f"delta_S cannot be positive: {delta_s}"
    assert tkn in old_state.asset_list, f"invalid token name: {tkn}"

    new_state = old_state.copy()
    new_agent = old_agent.copy()

    if delta_s == 0:
        return new_state, new_agent

    piq = price_i(old_state, tkn)
    p0 = new_agent.share_prices[tkn]
    mult = (piq - p0) / (piq + p0)

    # Share update
    delta_B = max(mult * delta_s, 0)
    new_state.protocol_shares[tkn] += delta_B
    new_state.shares[tkn] += delta_s + delta_B
    new_agent.shares[tkn] += delta_s

    # Token amounts update
    delta_R = old_state.liquidity[tkn] * max((delta_s + delta_B) / old_state.shares[tkn], -1)
    new_state.liquidity[tkn] += delta_R
    new_agent.holdings[tkn] -= delta_R
    if piq >= p0:  # prevents rounding errors
        new_agent.holdings['LRNA'] -= piq * (
                2 * piq / (piq + p0) * delta_s / old_state.shares[tkn] * old_state.liquidity[tkn] - delta_R)

    # LRNA burn
    delta_Q = price_i(old_state, tkn) * delta_R
    new_state.lrna[tkn] += delta_Q

    # L update: LRNA fees to be burned before they will start to accumulate again
    delta_L = delta_R * old_state.lrna[tkn] / old_state.liquidity[tkn] * old_state.lrna_imbalance / old_state.lrna_total
    new_state.lrna_imbalance += delta_L

    return new_state, new_agent


OmnipoolState.swap = staticmethod(swap)
OmnipoolState.add_liquidity = staticmethod(add_liquidity)
OmnipoolState.remove_liquidity = staticmethod(remove_liquidity)
