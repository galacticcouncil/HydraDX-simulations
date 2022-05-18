import copy
import string


class MarketState:
    def __init__(self,
                 asset_fee: float,
                 lrna_fee: float,
                 tvl_cap: float,
                 preferred_stablecoin: str,
                 tokens: dict[str: dict]
                 ):
        """
        tokens should be a dict in the form of [str: dict]
        the nested dict needs the following parameters:
        {
          'liquidity': float  # starting liquidity in the pool
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

        self.asset_list = []
        for name, token in tokens.items:
            assert token['liquidity'], f'token {name} missing required parameter: liquidity'
            if not ('LRNA' in token or 'LRNA_price' in token):
                raise ValueError("token {name} missing required parameter: ('LRNA' or 'LRNA_price)")
            token['shares'] = token['liquidity']
            token['protocol_shares'] = token['liquidity']
            token['weight_cap'] = token['weight_cap'] if 'weight_cap' in token else 1
            self.asset_list.append(name)

        self.pools = tokens

        self.asset_fee = asset_fee
        self.lrna_fee = lrna_fee
        self.lrna_imbalance = 0  # AKA "L"
        self.tvl_cap = tvl_cap
        self.preferred_stablecoin = preferred_stablecoin

        # count TVL for each pool once all values are set
        for token in self.pools.values():
            token['TVL'] = self.R(token) * self.P(token) / self.P(self.preferred_stablecoin)

    def B(self, token: str):
        """ number of shares in pool[token] which are owned by the protocol """
        return self.pools[token]['protocol_shares']

    @property
    def C(self):
        """ overall (soft) cap on total value locked, denominated in preferred_stablecoin """
        return self.tvl_cap

    def O(self, token: str) -> float:
        """ fraction of total TVL that can be stored in pool[token] """
        return self.pools[token]['weight_cap']

    def P(self, token: str) -> float:
        """ the price of asset[token] in LRNA """
        return self.Q(token) / self.R(token)

    def Q(self, token: str) -> float:
        """ the quantity of LRNA in pool[token] """
        return self.pools[token]['LRNA']

    def R(self, token: str) -> float:
        """ quantity of asset in pool[token] """
        return self.pools[token]['liquidity']

    def S(self, token: str) -> float:
        """ total shares existing in pool[token] """
        return self.pools[token]['shares']

    def T(self, token: str) -> float:
        """ total value locked in pool[token], denominated in preferred_stablecoin """
        return self.pools[token]['TVL']

    def add_delta_q(self, token: str, quantity: float):
        """ change the amount of LRNA in pool[token] """
        self.pools[token]['LRNA'] += quantity

    def add_delta_r(self, token: str, quantity: float):
        """ change the amount of asset in pool[token] """
        self.pools[token]['liquidity'] += quantity

    def add_delta_s(self, token: str, quantity: float):
        """ change the number of shares in pool[token] """
        self.pools[token]['shares'] += quantity

    def add_delta_l(self, quantity: float):
        """ change the LRNA imbalance in the exchange """
        self.lrna_imbalance += quantity

    def add_delta_b(self, token: str, quantity: float):
        """ change shares in pool[token] owned by the protocol """
        self.pools[token]['protocol_shares'] += quantity

    def add_delta_t(self, token: str, quantity: float):
        """ change shares in pool[token] owned by the protocol """
        self.pools[token]['TVL'] += quantity

    @property
    def L(self) -> float:
        """ the current LRNA imbalance in the pool """
        return self.lrna_imbalance

    @L.setter
    def L(self, value):
        self.lrna_imbalance = value

    def T_total(self) -> float:
        """ total value locked in all pools combined, denominated in preferred_stablecoin """
        return sum(self.T(token) for token in self.pools)

    def Q_total(self) -> float:
        """ total LRNA in all pools combined, denominated in preferred_stablecoin """
        return sum(self.Q(token) for token in self.pools)

    def copy(self):
        return copy.deepcopy(self)


def asset_invariant(state: MarketState, i: str) -> float:
    """Invariant for specific asset"""
    return state.R(i) * state.Q(i)


def swap_lrna_delta_Qi(state: MarketState, delta_ri: float, i: str) -> float:
    return state.Q(i) * (- delta_ri / (state.R(i) + delta_ri))


def swap_lrna_delta_Ri(state: MarketState, delta_qi: float, i: str) -> float:
    return state.R(i) * (- delta_qi / (state.Q(i) + delta_qi))


def weight_i(state: MarketState, i: str) -> float:
    return state.Q(i) / state.Q_total()


def price_i(state: MarketState, i: str, fee: float = 0) -> float:
    """Price of i denominated in LRNA"""
    if state.R(i) == 0:
        return 0
    else:
        return (state.Q(i) / state.R(i)) * (1 - fee)


def swap_lrna(
        old_state: MarketState,
        old_agents: dict,
        trader_id: string,
        delta_ra: float,
        delta_qa: float,
        i: str,
        fee_assets: float = 0,
        fee_lrna: float = 0
) -> tuple:
    """Compute new state after LRNA swap"""

    if delta_ra >= old_state.R(i) * (1 - fee_assets):
        # insufficient assets in pool, transaction fails
        return old_state, old_agents

    new_state = old_state.copy()
    new_agents = copy.deepcopy(old_agents)

    if delta_qa < 0:
        delta_Q = -delta_qa
        delta_R = old_state.R(i) * -delta_Q / (delta_Q + old_state.Q(i)) * (1 - fee_assets)
        delta_L = -delta_Q * (1 + (1 - fee_assets) * old_state.Q(i) / (old_state.Q(i) + delta_Q))
        delta_ra = -delta_R
    elif delta_ra > 0:
        delta_R = -delta_ra
        delta_Q = old_state.Q(i) * -delta_R / (old_state.R(i) * (1 - fee_assets) + delta_R)
        delta_L = -delta_Q * (1 + (1 - fee_assets) * old_state.Q(i) / (old_state.Q(i) + delta_Q))
        delta_qa = -delta_Q
    else:
        # print(f'Invalid swap (delta_Qa {delta_Qa}, delta_Ra {delta_Ra}')
        return old_state, old_agents

    if delta_qa + old_agents[trader_id]['q'] < 0:
        # agent doesn't have enough lrna
        return old_state, old_agents
    elif delta_ra + old_agents[trader_id]['r'][i] < 0:
        # agent doesn't have enough asset[i]
        return old_state, old_agents

    new_agents[trader_id]['q'] += delta_qa
    new_agents[trader_id]['r'][i] += delta_ra
    new_state.add_delta_q(i, delta_Q)
    new_state.add_delta_r(i, delta_R)
    new_state.add_delta_l(delta_L)

    return new_state, new_agents


def swap_assets_direct(
        old_state: MarketState,
        old_agents: dict,
        trader_id: string,
        delta_token: float,
        i_buy: str,
        i_sell: str,
        fee_assets: float = 0,
        fee_lrna: float = 0
) -> tuple:
    i = i_sell
    j = i_buy
    delta_Ri = delta_token
    assert delta_Ri > 0, 'sell amount must be greater than zero'

    delta_Qi = old_state.Q(i) * -delta_Ri / (old_state.R(i) + delta_Ri)
    delta_Qj = -delta_Qi * (1 - fee_lrna)
    delta_Rj = old_state.R(j) * -delta_Qj / (old_state.Q(j) + delta_Qj) * (1 - fee_assets)
    delta_L = min(-delta_Qi * fee_lrna, -old_state.L)
    delta_QH = -fee_lrna * delta_Qi - delta_L

    new_state = old_state.copy()
    new_state.add_delta_q(i, delta_Qi)
    new_state.add_delta_q(j, delta_Qj)
    new_state.add_delta_r(i, delta_Ri)
    new_state.add_delta_r(j, delta_Rj)
    new_state.add_delta_q('HDX', delta_QH)
    new_state.add_delta_l(delta_L)

    new_agents = copy.deepcopy(old_agents)
    new_agents[trader_id]['r'][i] -= delta_Ri
    new_agents[trader_id]['r'][j] -= delta_Rj

    return new_state, new_agents


def swap_assets(
        old_state: MarketState,
        old_agents: dict,
        trader_id: string,
        trade_type: string,
        delta_token: float,
        i_buy: str,
        i_sell: str,
        fee_assets: float = 0,
        fee_lrna: float = 0
) -> tuple:
    if trade_type == 'sell':

        new_state, new_agents = swap_assets_direct(
            old_state=old_state,
            old_agents=old_agents,
            trader_id=trader_id,
            delta_token=delta_token,
            i_buy=i_buy,
            i_sell=i_sell,
            fee_assets=fee_assets,
            fee_lrna=fee_lrna
        )

        return new_state, new_agents
    elif trade_type == 'buy':
        # back into correct delta_Ri, then execute sell
        delta_Qj = -old_state.Q(i_buy) * delta_token / (old_state.R(i_buy)*(1 - fee_assets) + delta_token)
        delta_Qi = -delta_Qj/(1 - fee_lrna)
        delta_Ri = -old_state.R(i_sell) * delta_Qi / (old_state.Q(i_sell) + delta_Qi)
        return swap_assets(old_state, old_agents, trader_id, 'sell', delta_Ri, i_buy, i_sell, fee_assets, fee_lrna)

    else:
        raise


def add_risk_liquidity(
        old_state: MarketState,
        old_agents: dict,
        lp_id: string,
        delta_r: float,
        i: str
) -> tuple:
    """Compute new state after liquidity addition"""

    assert delta_r > 0, "delta_R must be positive: " + str(delta_r)
    assert i in old_state.asset_list, "invalid value for i: " + str(i)

    new_state = old_state.copy()
    new_agents = copy.deepcopy(old_agents)

    # Token amounts update
    new_state.add_delta_r(i, delta_r)
    if lp_id:
        new_agents[lp_id]['r'][i] -= delta_r
        if new_agents[lp_id]['r'][i] < 0:
            # print('Transaction rejected because agent has insufficient funds.')
            # print(f'agent {LP_id}, asset {new_state["token_list"][i]}, amount {delta_R}')
            return old_state, old_agents

    # Share update
    new_state.add_delta_s(i, new_state.S(i) * new_state.R(i) / old_state.R(i) or 1)

    if lp_id:
        # shares go to provisioning agent
        new_agents[lp_id]['s'][i] += new_state.S(i) - old_state.S(i)
    else:
        # shares go to protocol
        new_state.add_delta_b(i, new_state.S(i) - old_state.S(i))

    # LRNA add (mint)
    delta_Q = price_i(old_state, i) * delta_r
    new_state.add_delta_q(i, delta_Q)

    # L update: LRNA fees to be burned before they will start to accumulate again
    delta_L = delta_r * old_state.Q(i) / old_state.R(i) * old_state.L / old_state.Q_total()
    new_state.add_delta_l(delta_L)

    # T update: TVL soft cap
    stable_index = new_state.preferred_stablecoin
    delta_t = new_state.Q(i) * new_state.R(stable_index)/new_state.Q(stable_index) - new_state.T(i)
    new_state.add_delta_t(i, delta_t)

    if new_state.Q(i) / new_state.Q_total() > new_state.O(i):
        # print(f'Transaction rejected because it would exceed the weight cap in pool[{i}].')
        # print(f'agent {LP_id}, asset {new_state["token_list"][i]}, amount {delta_R}')
        return old_state, old_agents

    if new_state.T_total() > new_state.tvl_cap:
        # print('Transaction rejected because it would exceed the TVL cap.')
        # print(f'agent {LP_id}, asset {new_state["token_list"][i]}, amount {delta_R}')
        return old_state, old_agents

    # set price at which liquidity was added
    if lp_id:
        new_agents[lp_id]['p'][i] = price_i(new_state, i)

    return new_state, new_agents


def remove_risk_liquidity(
        old_state: MarketState,
        old_agents: dict,
        lp_id: string,
        delta_s: float,
        i: str
) -> tuple:
    """Compute new state after liquidity removal"""
    assert delta_s <= 0, "delta_S cannot be positive: " + str(delta_s)
    assert i in old_state.asset_list, "invalid value for i: " + str(i)

    new_state = copy.deepcopy(old_state)
    new_agents = copy.deepcopy(old_agents)

    if delta_s == 0:
        return new_state, new_agents

    piq = price_i(old_state, i)
    p0 = new_agents[lp_id]['p'][i]
    mult = (piq - p0) / (piq + p0)

    # Share update
    delta_B = max(mult * delta_s, 0)
    new_state.add_delta_b(i, delta_B)
    new_state.add_delta_s(i, delta_s + delta_B)
    new_agents[lp_id]['s'][i] += delta_s

    # Token amounts update
    delta_R = old_state.R(i) * max((delta_s + delta_B) / old_state.S(i), -1)
    new_state.add_delta_r(i, delta_R)
    new_agents[lp_id]['r'][i] -= delta_R
    if piq >= p0:  # prevents rounding errors
        new_agents[lp_id]['q'] -= piq * (
                2 * piq / (piq + p0) * delta_s / old_state.S(i) * old_state.R(i) - delta_R)

    # LRNA burn
    delta_Q = price_i(old_state, i) * delta_R
    new_state.add_delta_q(i, delta_Q)

    # L update: LRNA fees to be burned before they will start to accumulate again
    delta_L = delta_R * old_state.Q(i)/old_state.R(i) * old_state.L/old_state.Q_total()
    new_state.add_delta_l(delta_L)

    return new_state, new_agents
