import copy

from .amm import AMM
from .agents import Agent
from mpmath import mp, mpf


class StableSwapPoolState(AMM):
    unique_id: str = 'stableswap'

    def __init__(
            self,
            tokens: dict,
            amplification: float,
            precision: float = 0.0001,
            trade_fee: float = 0,
            unique_id: str = ''
    ):
        """
        Tokens should be in the form of:
        {
            token1: quantity,
            token2: quantity
        }
        There can be up to five.
        """
        super().__init__()
        if len(tokens.keys()) > 5:
            raise ValueError('Too many tokens (limit 5)')

        self.amplification = amplification
        self.precision = precision
        self.liquidity = dict()
        self.asset_list: list[str] = []
        self.trade_fee = trade_fee
        if unique_id:
            self.unique_id = unique_id

        for token, quantity in tokens.items():
            self.asset_list.append(token)
            self.liquidity[token] = quantity

        self.shares = self.calculate_d()
        self.conversion_metrics = {}

    @property
    def ann(self) -> float:
        return self.amplification * len(self.asset_list) ** len(self.asset_list)

    @property
    def n_coins(self) -> int:
        return len(self.asset_list)

    @property
    def d(self) -> float:
        return self.calculate_d()

    def has_converged(self, v0, v1) -> bool:
        diff = abs(v0 - v1)
        if (v1 <= v0 and diff < self.precision) or (v1 > v0 and diff <= self.precision):
            return True
        return False

    def calculate_d(self, reserves=(), max_iterations=128) -> float:
        reserves = reserves or self.liquidity.values()
        xp_sorted = sorted(reserves)
        s = sum(xp_sorted)
        if s == 0:
            return 0

        d = s
        for i in range(max_iterations):

            d_p = d
            for x in xp_sorted:
                d_p *= d / (x * self.n_coins)

            d_prev = d
            d = (self.ann * s + d_p * self.n_coins) * d / ((self.ann - 1) * d + (self.n_coins + 1) * d_p)

            if self.has_converged(d_prev, d):
                return d

    """
    Given a value for D and the balances of all tokens except 1, calculate what the balance of the final token should be
    """

    def calculate_y(self, reserves: list, d: float, max_iterations=128):

        # get all the balances except tkn_out and sort them from low to high
        balances = sorted(reserves)

        s = sum(balances)
        c = d
        for reserve in balances:
            c *= d / reserve / self.n_coins
        c *= d / self.ann / self.n_coins

        b = s + d / self.ann
        y = d

        for _ in range(max_iterations):
            y_prev = y
            y = (y ** 2 + c) / (2 * y + b - d)

            if self.has_converged(y_prev, y):
                return y

        return y

    # price is denominated in the first asset
    @property
    def spot_price(self):
        x, y = self.liquidity.values()
        return self.price_at_balance([x, y], self.d)

    # price is denominated in the first asset
    def price_at_balance(self, balances: list, d: float):
        x, y = balances
        c = self.amplification * self.n_coins ** (2 * self.n_coins)
        return (x / y) * (c * x * y ** 2 + d ** 3) / (c * x ** 2 * y + d ** 3)

    def modified_balances(self, delta: dict = None, omit: list = ()):
        balances = copy.copy(self.liquidity)
        if delta:
            for tkn, value in delta.items():
                balances[tkn] += value
        if omit:
            for tkn in omit:
                balances.pop(tkn)
        return list(balances.values())

    def calculate_withdrawal_shares(self, tkn_remove, quantity):
        updated_d = self.calculate_d(self.modified_balances(delta={tkn_remove: -quantity}))
        return self.shares * (1 - updated_d / self.d) / (1 - self.trade_fee)

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        # round to given precision
        precision = 10
        liquidity = {tkn: round(self.liquidity[tkn], precision) for tkn in self.asset_list}
        shares = round(self.shares, precision)
        return (
                   f'Stable Swap Pool: {self.unique_id}\n'
                   f'********************************\n'
                   f'trade fee: {self.trade_fee}\n'
                   f'shares: {shares}\n'
                   f'amplification constant: {self.amplification}\n'
                   f'tokens: (\n\n'
               ) + '\n'.join(
            [(
                f'    {token}\n'
                f'    quantity: {liquidity[token]}\n'
                f'    weight: {liquidity[token] / sum(liquidity.values())}\n'
                + (
                    f'    conversion metrics:\n'
                    f'        price: {self.conversion_metrics[token]["price"]}\n'
                    f'        old shares: {self.conversion_metrics[token]["old_shares"]}\n'
                    f'        Omnipool shares: {self.conversion_metrics[token]["omnipool_shares"]}\n'
                    f'        subpool shares: {self.conversion_metrics[token]["subpool_shares"]}\n'
                    if token in self.conversion_metrics else ""
                )
            ) for token in self.asset_list]
        ) + '\n)\n' + (
                   f'error message:{self.fail or "none"}'
               )


def execute_swap(
        state: StableSwapPoolState,
        agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
):
    if buy_quantity:
        reserves = state.modified_balances(delta={tkn_buy: -buy_quantity}, omit=[tkn_sell])
        sell_quantity = (state.calculate_y(reserves, state.d) - state.liquidity[tkn_sell]) / (1 - state.trade_fee)
    elif sell_quantity:
        reserves = state.modified_balances(delta={tkn_sell: sell_quantity}, omit=[tkn_buy])
        buy_quantity = (state.liquidity[tkn_buy] - state.calculate_y(reserves, state.d)) * (1 - state.trade_fee)

    if agent.holdings[tkn_sell] < sell_quantity:
        return state.fail_transaction('Agent has insufficient funds.', agent)
    elif state.liquidity[tkn_buy] <= buy_quantity:
        return state.fail_transaction('Pool has insufficient liquidity.', agent)

    new_agent = agent  # .copy()
    if tkn_buy not in new_agent.holdings:
        new_agent.holdings[tkn_buy] = 0
    new_agent.holdings[tkn_buy] += buy_quantity
    new_agent.holdings[tkn_sell] -= sell_quantity
    state.liquidity[tkn_buy] -= buy_quantity
    state.liquidity[tkn_sell] += sell_quantity

    return state, new_agent
#
#
# def execute_remove_liquidity(
#         state: StableSwapPoolState,
#         agent: Agent,
#         shares_removed: float,
#         tkn_remove: str
# ):
#     if shares_removed > agent.holdings[state.unique_id]:
#         raise ValueError('Agent tried to remove more shares than it owns.')
#     elif shares_removed <= 0:
#         raise ValueError('Withdraw quantity must be > 0.')
#
#     share_fraction = shares_removed / state.shares
#
#     updated_d = state.d * (1 - share_fraction * (1 - state.trade_fee))
#     delta_tkn = state.calculate_y(
#         state.modified_balances(delta={}, omit=[tkn_remove]),
#         updated_d
#     ) - state.liquidity[tkn_remove]
#
#     if delta_tkn >= state.liquidity[tkn_remove]:
#         return state.fail_transaction(f'Not enough liquidity in {tkn_remove}.', agent)
#
#     if tkn_remove not in agent.holdings:
#         agent.holdings[tkn_remove] = 0
#
#     state.shares -= shares_removed
#     agent.holdings[state.unique_id] -= shares_removed
#     state.liquidity[tkn_remove] += delta_tkn
#     agent.holdings[tkn_remove] -= delta_tkn  # agent is receiving funds, because delta_tkn is a negative number
#     return state, agent


def execute_remove_uniform(
        state: StableSwapPoolState,
        agent: Agent,
        shares_removed: float
):
    if shares_removed > agent.holdings[state.unique_id]:
        raise ValueError('Agent tried to remove more shares than it owns.')
    elif shares_removed <= 0:
        raise ValueError('Withdraw quantity must be > 0.')

    share_fraction = shares_removed / state.shares

    delta_tkns = {}
    for tkn in state.asset_list:
        delta_tkns[tkn] = share_fraction * state.liquidity[tkn]  # delta_tkn is positive

        if delta_tkns[tkn] >= state.liquidity[tkn]:
            return state.fail_transaction(f'Not enough liquidity in {tkn}.', agent)

        if tkn not in agent.holdings:
            agent.holdings[tkn] = 0

    state.shares -= shares_removed
    agent.holdings[state.unique_id] -= shares_removed

    for tkn in state.asset_list:
        state.liquidity[tkn] -= delta_tkns[tkn]
        agent.holdings[tkn] += delta_tkns[tkn]  # agent is receiving funds, because delta_tkn is a negative number
    return state, agent


def execute_withdraw_asset(
        state: StableSwapPoolState,
        agent: Agent,
        quantity: float,
        tkn_remove: str,
        fail_on_overdraw: bool = True
):
    """
    Calculate a withdrawal based on the asset quantity rather than the share quantity
    """
    if quantity >= state.liquidity[tkn_remove]:
        return state.fail_transaction(f'Not enough liquidity in {tkn_remove}.', agent)
    if quantity <= 0:
        raise ValueError('Withdraw quantity must be > 0.')

    shares_removed = state.calculate_withdrawal_shares(tkn_remove, quantity)

    if shares_removed > agent.holdings[state.unique_id]:
        if fail_on_overdraw:
            return state.fail_transaction('Agent tried to remove more shares than it owns.', agent)
        else:
            # just round down
            shares_removed = agent.holdings[state.unique_id]

    if tkn_remove not in agent.holdings:
        agent.holdings[tkn_remove] = 0

    agent.holdings[state.unique_id] -= shares_removed
    state.shares -= shares_removed
    state.liquidity[tkn_remove] -= quantity
    agent.holdings[tkn_remove] += quantity
    return state, agent


def execute_remove_liquidity(
        state: StableSwapPoolState,
        agent: Agent,
        shares_removed: float,
        tkn_remove: str,
):
    # First, need to calculate
    # * Get current D
    # * Solve Eqn against y_i for D - _token_amount

    if shares_removed > agent.holdings[state.unique_id]:
        return state.fail_transaction('Agent has insufficient funds.', agent)
    elif shares_removed <= 0:
        return state.fail_transaction('Withdraw quantity must be > 0.', agent)

    _fee = state.trade_fee

    initial_d = state.calculate_d()
    reduced_d = initial_d - shares_removed * initial_d / state.shares

    xp_reduced = copy.copy(state.liquidity)
    xp_reduced.pop(tkn_remove)

    reduced_y = state.calculate_y(state.modified_balances(omit=[tkn_remove]), reduced_d)
    asset_reserve = state.liquidity[tkn_remove]

    for tkn in state.asset_list:
        if tkn == tkn_remove:
            dx_expected = state.liquidity[tkn] * reduced_d / initial_d - reduced_y
            asset_reserve -= _fee * dx_expected
        else:
            dx_expected = state.liquidity[tkn] - state.liquidity[tkn] * reduced_d / initial_d
            xp_reduced[tkn] -= _fee * dx_expected

    dy = asset_reserve - state.calculate_y(list(xp_reduced.values()), reduced_d)

    agent.holdings[state.unique_id] -= shares_removed
    state.shares -= shares_removed
    state.liquidity[tkn_remove] -= dy
    if tkn_remove not in agent.holdings:
        agent.holdings[tkn_remove] = 0
    agent.holdings[tkn_remove] += dy
    return state, agent


def execute_add_liquidity(
        state: StableSwapPoolState,
        agent: Agent,
        quantity: float,
        tkn_add: str
):
    if quantity <= 0:
        return state.fail_transaction('Add quantity must be > 0.', agent)
    # elif state.unique_id in agent.holdings:
    #     return state.fail_transaction('Agent already has shares.', agent)

    initial_d = state.d

    updated_d = state.calculate_d(state.modified_balances(delta={tkn_add: quantity}))

    if updated_d < initial_d:
        return state.fail_transaction('invariant decreased for some reason', agent)
    if agent.holdings[tkn_add] < quantity:
        return state.fail_transaction(f"Agent doesn't have enough {tkn_add}.", agent)

    state.liquidity[tkn_add] += quantity
    agent.holdings[tkn_add] -= quantity

    if state.shares == 0:
        agent.holdings[state.unique_id] = updated_d
        state.shares = updated_d

    elif state.shares < 0:
        return state.fail_transaction('Shares cannot go below 0.', agent)
        # why would this possibly happen?

    else:
        d_diff = updated_d - initial_d
        share_amount = state.shares * d_diff / initial_d
        state.shares += share_amount
        if state.unique_id not in agent.holdings:
            agent.holdings[state.unique_id] = 0
        agent.holdings[state.unique_id] += share_amount
        agent.share_prices[state.unique_id] = quantity / share_amount

    return state, agent


def execute_buy_shares(
        state: StableSwapPoolState,
        agent: Agent,
        quantity: float,
        tkn_add: str,
        fail_overdraft: bool = True
):
    initial_d = state.d
    updated_d = initial_d * (state.shares + quantity) / state.shares
    delta_tkn = state.calculate_y(
        state.modified_balances(omit=[tkn_add]),
        d=updated_d
    ) - state.liquidity[tkn_add]

    if delta_tkn > agent.holdings[tkn_add]:
        if fail_overdraft:
            return state.fail_transaction(f"Agent doesn't have enough {tkn_add}.", agent)
        else:
            # instead of failing, just round down
            delta_tkn = agent.holdings[tkn_add]
            return execute_add_liquidity(state, agent, delta_tkn, tkn_add)

    state.liquidity[tkn_add] += delta_tkn
    agent.holdings[tkn_add] -= delta_tkn
    state.shares += quantity
    if state.unique_id not in agent.holdings:
        agent.holdings[state.unique_id] = 0
    agent.holdings[state.unique_id] += quantity
    return state, agent


def swap(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
):
    return execute_swap(old_state.copy(), old_agent.copy(), tkn_sell, tkn_buy, buy_quantity, sell_quantity)


def add_liquidity(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        quantity: float,  # quantity of asset to be added
        tkn_add: str
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return execute_add_liquidity(new_state, new_agent, quantity, tkn_add)


def remove_liquidity(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        quantity: float,  # in this case, quantity refers to a number of shares, not quantity of asset
        tkn_remove: str
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return execute_remove_liquidity(new_state, new_agent, quantity, tkn_remove)


StableSwapPoolState.add_liquidity = staticmethod(add_liquidity)
StableSwapPoolState.execute_add_liquidity = staticmethod(execute_add_liquidity)
StableSwapPoolState.remove_liquidity = staticmethod(remove_liquidity)
StableSwapPoolState.execute_remove_liquidity = staticmethod(execute_remove_liquidity)
StableSwapPoolState.swap = staticmethod(swap)
StableSwapPoolState.execute_swap = staticmethod(execute_swap)
StableSwapPoolState.execute_remove_uniform = staticmethod(execute_remove_uniform)
