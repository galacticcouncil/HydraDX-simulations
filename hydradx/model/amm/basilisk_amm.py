import copy


class BasiliskPoolState:
    unique_ids = {''}

    def __init__(self, tokens: dict[str: float], trade_fee: float = 0):
        self.trade_fee = trade_fee
        self.liquidity = dict()
        self.asset_list = []

        for token, quantity in tokens.items():
            self.asset_list.append(token)
            self.liquidity[token] = quantity

        self.shares = self.liquidity[self.asset_list[0]]

        self.unique_id = ''
        i = 0
        while self.unique_id in BasiliskPoolState.unique_ids:
            self.unique_id = '-'.join(self.asset_list + [str(i)])
            i += 1

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return (
            f'BasiliskPool\n'
            f'trade fee: {self.trade_fee}\n'
            f'tokens: (\n'
        ) + ')\n(\n'.join(
            [(
                f'    {token}\n'
                f'    quantity: {self.liquidity[token]}\n'
                f'    weight: {self.liquidity[token] / sum(self.liquidity.values())}\n'
            ) for token in self.asset_list]
        ) + '\n)'


def add_risk_liquidity(
        old_state: BasiliskPoolState,
        old_agents: dict,
        lp_id: str,
        quantity: float,
        tkn_add: str
) -> tuple[BasiliskPoolState, dict]:
    new_agents = copy.deepcopy(old_agents)
    new_state = old_state.copy()

    lp = new_agents[lp_id]
    delta_r = {}
    for token in old_state.asset_list:
        delta_r[tkn_add] = quantity * new_state.liquidity[token] / new_state.liquidity[tkn_add]
        lp['r'][token] -= delta_r[tkn_add]
        new_state.liquidity[tkn_add] += delta_r

        if lp['r'][token] < 0:
            # fail
            return fail(old_state, old_agents)

    new_shares = new_state.liquidity[tkn_add] / old_state.liquidity[tkn_add] - 1
    new_state.shares += new_shares
    lp['s'][new_state.unique_id] += new_shares
    return new_state, new_agents


def remove_liquidity(
        old_state: BasiliskPoolState,
        old_agents: dict,
        lp_id: str,
        quantity: float,
        tkn: str
) -> tuple[BasiliskPoolState, dict]:
    quantity = abs(quantity)
    new_state, new_agents = add_risk_liquidity(
        old_state, old_agents, lp_id, -quantity, tkn
    )  # does this work? test
    if min(new_state.liquidity) < 0:
        return fail(old_state, old_agents)

    return new_state, new_agents


def swap(
        old_state: BasiliskPoolState,
        old_agents: dict,
        trader_id: str,
        buy_quantity: float,
        sell_quantity: float,
        tkn_sell: str,
        tkn_buy: str
):
    new_agents = copy.deepcopy(old_agents)
    new_state = old_state.copy()
    trader = new_agents[trader_id]

    invariant = 1
    for token in new_state.asset_list:
        invariant *= new_state.liquidity[token]

    if buy_quantity > 0:
        new_state.liquidity[tkn_buy] -= buy_quantity
        sell_quantity = invariant / new_state.liquidity[tkn_buy]
        trader['r'][tkn_sell] -= sell_quantity * new_state.trade_fee

    elif sell_quantity > 0:
        new_state.liquidity[tkn_sell] += sell_quantity
        buy_quantity = invariant / new_state.liquidity[tkn_sell]
        trader['r'][tkn_buy] -= buy_quantity * new_state.trade_fee

    else:
        return fail(old_state, old_agents)

    trader['r'][tkn_buy] -= buy_quantity
    trader['r'][tkn_sell] -= sell_quantity
    new_state.liquidity[tkn_buy] -= buy_quantity
    new_state.liquidity[tkn_sell] += sell_quantity

    if new_state.liquidity[tkn_buy] < 0:
        return fail(old_state, old_agents)

    if trader['r'][tkn_sell] < 0:
        return fail(old_state, old_agents)

    return new_state, new_agents


def fail(old_state: BasiliskPoolState, old_agents: dict) -> tuple[BasiliskPoolState, dict]:
    return old_state.copy(), copy.deepcopy(old_agents)
