import copy
import string


class OmnipoolState:
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
        return copy.deepcopy(self)

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
        old_agents: dict,
        trader_id: string,
        delta_ra: float = 0,
        delta_qa: float = 0,
        i: str = '',
        fee_assets: float = 0,
        fee_lrna: float = 0
) -> tuple[OmnipoolState, dict]:
    """Compute new state after LRNA swap"""

    new_state = old_state.copy()
    new_agents = copy.deepcopy(old_agents)

    if delta_qa < 0:
        delta_Q = -delta_qa
        delta_R = old_state.liquidity[i] * -delta_Q / (delta_Q + old_state.lrna[i]) * (1 - fee_assets)
        delta_L = -delta_Q * (1 + (1 - fee_assets) * old_state.lrna[i] / (old_state.lrna[i] + delta_Q))
        delta_ra = -delta_R
    elif delta_ra > 0:
        delta_R = -delta_ra
        delta_Q = old_state.lrna[i] * -delta_R / (old_state.liquidity[i] * (1 - fee_assets) + delta_R)
        delta_L = -delta_Q * (1 + (1 - fee_assets) * old_state.lrna[i] / (old_state.lrna[i] + delta_Q))
        delta_qa = -delta_Q
    else:
        # print(f'Invalid swap (delta_Qa {delta_Qa}, delta_Ra {delta_Ra}')
        return fail(old_state, old_agents)

    if delta_qa + old_agents[trader_id]['q'] < 0:
        # agent doesn't have enough lrna
        return fail(old_state, old_agents)
    elif delta_ra + old_agents[trader_id]['r'][i] < 0:
        # agent doesn't have enough asset[i]
        return fail(old_state, old_agents)
    elif delta_R + old_state.liquidity[i] <= 0:
        # insufficient assets in pool, transaction fails
        return fail(old_state, old_agents)
    elif delta_Q + old_state.lrna[i] <= 0:
        # insufficient lrna in pool, transaction fails
        return fail(old_state, old_agents)

    new_agents[trader_id]['q'] += delta_qa
    new_agents[trader_id]['r'][i] += delta_ra
    new_state.lrna[i] += delta_Q
    new_state.liquidity[i] += delta_R
    new_state.lrna_imbalance += delta_L

    return new_state, new_agents


def swap_assets_direct(
        old_state: OmnipoolState,
        old_agents: dict,
        trader_id: string,
        delta_token: float,
        tkn_buy: str,
        tkn_sell: str,
        fee_assets: float = 0,
        fee_lrna: float = 0
) -> tuple[OmnipoolState, dict]:
    i = tkn_sell
    j = tkn_buy
    delta_Ri = delta_token
    assert delta_Ri > 0, 'sell amount must be greater than zero'

    delta_Qi = old_state.lrna[i] * -delta_Ri / (old_state.liquidity[i] + delta_Ri)
    delta_Qj = -delta_Qi * (1 - fee_lrna)
    delta_Rj = old_state.liquidity[j] * -delta_Qj / (old_state.lrna[j] + delta_Qj) * (1 - fee_assets)
    delta_L = min(-delta_Qi * fee_lrna, -old_state.lrna_imbalance)
    delta_QH = -fee_lrna * delta_Qi - delta_L

    new_state = old_state.copy()
    new_state.lrna[i] += delta_Qi
    new_state.lrna[j] += delta_Qj
    new_state.liquidity[i] += delta_Ri
    new_state.liquidity[j] += delta_Rj
    new_state.lrna['HDX'] += delta_QH
    new_state.lrna_imbalance += delta_L

    new_agents = copy.deepcopy(old_agents)
    new_agents[trader_id]['r'][i] -= delta_Ri
    new_agents[trader_id]['r'][j] -= delta_Rj

    return new_state, new_agents


def swap_assets(
        old_state: OmnipoolState,
        old_agents: dict,
        trader_id: string,
        trade_type: string,
        delta_token: float,
        tkn_buy: str,
        tkn_sell: str,
        fee_assets: float = 0,
        fee_lrna: float = 0
) -> tuple[OmnipoolState, dict]:
    if trade_type == 'sell':

        new_state, new_agents = swap_assets_direct(
            old_state=old_state,
            old_agents=old_agents,
            trader_id=trader_id,
            delta_token=delta_token,
            tkn_buy=tkn_buy,
            tkn_sell=tkn_sell,
            fee_assets=fee_assets,
            fee_lrna=fee_lrna
        )

        return new_state, new_agents
    elif trade_type == 'buy':
        # back into correct delta_Ri, then execute sell
        delta_Qj = -old_state.lrna[tkn_buy] * delta_token / (old_state.liquidity[tkn_buy] * (1 - fee_assets) + delta_token)
        delta_Qi = -delta_Qj/(1 - fee_lrna)
        delta_Ri = -old_state.liquidity[tkn_sell] * delta_Qi / (old_state.lrna[tkn_sell] + delta_Qi)
        return swap_assets(old_state, old_agents, trader_id, 'sell', delta_Ri, tkn_buy, tkn_sell, fee_assets, fee_lrna)

    else:
        raise


def add_risk_liquidity(
        old_state: OmnipoolState,
        old_agents: dict,
        lp_id: string,
        delta_r: float,
        i: str
) -> tuple[OmnipoolState, dict]:
    """Compute new state after liquidity addition"""

    assert delta_r > 0, f"delta_R must be positive: {delta_r}"
    assert i in old_state.asset_list, f"invalid value for i: {i}"

    new_state = old_state.copy()
    new_agents = copy.deepcopy(old_agents)

    # Token amounts update
    new_state.liquidity[i] += delta_r
    if lp_id:
        new_agents[lp_id]['r'][i] -= delta_r
        if new_agents[lp_id]['r'][i] < 0:
            # print('Transaction rejected because agent has insufficient funds.')
            # print(f'agent {LP_id}, asset {new_state["token_list"][i]}, amount {delta_R}')
            return fail(old_state, old_agents)

    # Share update
    if new_state.shares[i]:
        new_state.shares[i] *= new_state.liquidity[i] / old_state.liquidity[i]
    else:
        new_state.shares[i] = new_state.liquidity[i]

    if lp_id:
        # shares go to provisioning agent
        new_agents[lp_id]['s'][i] += new_state.shares[i] - old_state.shares[i]
    else:
        # shares go to protocol
        new_state.protocol_shares[i] += new_state.shares[i] - old_state.shares[i]

    # LRNA add (mint)
    delta_Q = price_i(old_state, i) * delta_r
    new_state.lrna[i] += delta_Q

    # L update: LRNA fees to be burned before they will start to accumulate again
    delta_L = delta_r * old_state.lrna[i] / old_state.liquidity[i] * old_state.lrna_imbalance / old_state.lrna_total
    new_state.lrna_imbalance += delta_L

    # T update: TVL soft cap
    usd = new_state.stablecoin
    delta_t = new_state.lrna[i] * new_state.liquidity[usd] / new_state.lrna[usd] - new_state.tvl[i]
    new_state.tvl[i] += delta_t

    if new_state.lrna[i] / new_state.lrna_total > new_state.weight_cap[i]:
        # print(f'Transaction rejected because it would exceed the weight cap in pool[{i}].')
        # print(f'agent {LP_id}, asset {new_state["token_list"][i]}, amount {delta_R}')
        return fail(old_state, old_agents)

    if new_state.tvl_total > new_state.tvl_cap:
        # print('Transaction rejected because it would exceed the TVL cap.')
        # print(f'agent {LP_id}, asset {new_state["token_list"][i]}, amount {delta_R}')
        return fail(old_state, old_agents)

    # set price at which liquidity was added
    if lp_id:
        new_agents[lp_id]['p'][i] = price_i(new_state, i)

    return new_state, new_agents


def remove_risk_liquidity(
        old_state: OmnipoolState,
        old_agents: dict,
        lp_id: string,
        delta_s: float,
        i: str
) -> tuple[OmnipoolState, dict]:
    """Compute new state after liquidity removal"""
    assert delta_s <= 0, f"delta_S cannot be positive: {delta_s}"
    assert i in old_state.asset_list, f"invalid value for i: {i}"

    new_state = copy.deepcopy(old_state)
    new_agents = copy.deepcopy(old_agents)

    if delta_s == 0:
        return new_state, new_agents

    piq = price_i(old_state, i)
    p0 = new_agents[lp_id]['p'][i]
    mult = (piq - p0) / (piq + p0)

    # Share update
    delta_B = max(mult * delta_s, 0)
    new_state.protocol_shares[i] += delta_B
    new_state.shares[i] += delta_s + delta_B
    new_agents[lp_id]['s'][i] += delta_s

    # Token amounts update
    delta_R = old_state.liquidity[i] * max((delta_s + delta_B) / old_state.shares[i], -1)
    new_state.liquidity[i] += delta_R
    new_agents[lp_id]['r'][i] -= delta_R
    if piq >= p0:  # prevents rounding errors
        new_agents[lp_id]['q'] -= piq * (
                2 * piq / (piq + p0) * delta_s / old_state.shares[i] * old_state.liquidity[i] - delta_R)

    # LRNA burn
    delta_Q = price_i(old_state, i) * delta_R
    new_state.lrna[i] += delta_Q

    # L update: LRNA fees to be burned before they will start to accumulate again
    delta_L = delta_R * old_state.lrna[i] / old_state.liquidity[i] * old_state.lrna_imbalance / old_state.lrna_total
    new_state.lrna_imbalance += delta_L

    return new_state, new_agents


def fail(old_state: OmnipoolState, old_agents: dict) -> tuple[OmnipoolState, dict]:
    return old_state.copy(), copy.deepcopy(old_agents)
