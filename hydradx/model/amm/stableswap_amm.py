import copy

from .amm import AMM
from .agents import Agent


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


    def spot_price(self, i: int = 1):
        balances = list(self.liquidity.values())
        if i == 0:  # price of the numeraire is always 1
            return 1
        return self.price_at_balance(balances, self.d, i)

    def price_at_balance(self, balances: list, d: float, i: int = 1):
        c = self.amplification * self.n_coins ** (2 * self.n_coins)
        p = 1
        for x in balances:
            p *= x
        x = balances[0]
        y = balances[i]
        n = len(balances)
        return (x / y) * (c * y * p + d ** (n + 1)) / (c * x * p + d ** (n + 1))

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

    def swap(
            self,
            agent: Agent,
            tkn_sell: str,
            tkn_buy: str,
            buy_quantity: float = 0,
            sell_quantity: float = 0
    ):
        if buy_quantity:
            reserves = self.modified_balances(delta={tkn_buy: -buy_quantity}, omit=[tkn_sell])
            sell_quantity = (self.calculate_y(reserves, self.d) - self.liquidity[tkn_sell]) / (1 - self.trade_fee)
        elif sell_quantity:
            reserves = self.modified_balances(delta={tkn_sell: sell_quantity}, omit=[tkn_buy])
            buy_quantity = (self.liquidity[tkn_buy] - self.calculate_y(reserves, self.d)) * (1 - self.trade_fee)

        if agent.holdings[tkn_sell] < sell_quantity:
            return self.fail_transaction('Agent has insufficient funds.', agent)
        elif self.liquidity[tkn_buy] <= buy_quantity:
            return self.fail_transaction('Pool has insufficient liquidity.', agent)

        new_agent = agent  # .copy()
        if tkn_buy not in new_agent.holdings:
            new_agent.holdings[tkn_buy] = 0
        new_agent.holdings[tkn_buy] += buy_quantity
        new_agent.holdings[tkn_sell] -= sell_quantity
        self.liquidity[tkn_buy] -= buy_quantity
        self.liquidity[tkn_sell] += sell_quantity

        return self

    def withdraw_asset(
            self,
            agent: Agent,
            quantity: float,
            tkn_remove: str,
            fail_on_overdraw: bool = True
    ):
        """
        Calculate a withdrawal based on the asset quantity rather than the share quantity
        """
        if quantity >= self.liquidity[tkn_remove]:
            return self.fail_transaction(f'Not enough liquidity in {tkn_remove}.', agent)
        if quantity <= 0:
            raise ValueError('Withdraw quantity must be > 0.')

        shares_removed = self.calculate_withdrawal_shares(tkn_remove, quantity)

        if shares_removed > agent.holdings[self.unique_id]:
            if fail_on_overdraw:
                return self.fail_transaction('Agent tried to remove more shares than it owns.', agent)
            else:
                # just round down
                shares_removed = agent.holdings[self.unique_id]

        if tkn_remove not in agent.holdings:
            agent.holdings[tkn_remove] = 0

        agent.holdings[self.unique_id] -= shares_removed
        self.shares -= shares_removed
        self.liquidity[tkn_remove] -= quantity
        agent.holdings[tkn_remove] += quantity
        return self

    def remove_liquidity(
            self,
            agent: Agent,
            shares_removed: float,
            tkn_remove: str,
    ):
        # First, need to calculate
        # * Get current D
        # * Solve Eqn against y_i for D - _token_amount

        if shares_removed > agent.holdings[self.unique_id]:
            return self.fail_transaction('Agent has insufficient funds.', agent)
        elif shares_removed <= 0:
            return self.fail_transaction('Withdraw quantity must be > 0.', agent)

        _fee = self.trade_fee
        _fee *= self.n_coins / 4 / (self.n_coins - 1)

        initial_d = self.calculate_d()
        reduced_d = initial_d - shares_removed * initial_d / self.shares

        xp_reduced = copy.copy(self.liquidity)
        xp_reduced.pop(tkn_remove)

        reduced_y = self.calculate_y(self.modified_balances(omit=[tkn_remove]), reduced_d)
        asset_reserve = self.liquidity[tkn_remove]

        for tkn in self.asset_list:
            if tkn == tkn_remove:
                dx_expected = self.liquidity[tkn] * reduced_d / initial_d - reduced_y
                asset_reserve -= _fee * dx_expected
            else:
                dx_expected = self.liquidity[tkn] - self.liquidity[tkn] * reduced_d / initial_d
                xp_reduced[tkn] -= _fee * dx_expected

        dy = asset_reserve - self.calculate_y(list(xp_reduced.values()), reduced_d)

        agent.holdings[self.unique_id] -= shares_removed
        self.shares -= shares_removed
        self.liquidity[tkn_remove] -= dy
        if tkn_remove not in agent.holdings:
            agent.holdings[tkn_remove] = 0
        agent.holdings[tkn_remove] += dy
        return self

    def add_liquidity(
            self,
            agent: Agent,
            quantity: float,
            tkn_add: str
    ):
        initial_d = self.d

        updated_d = self.calculate_d(self.modified_balances(delta={tkn_add: quantity}))

        if updated_d < initial_d:
            return self.fail_transaction('invariant decreased for some reason', agent)
        if agent.holdings[tkn_add] < quantity:
            return self.fail_transaction(f"Agent doesn't have enough {tkn_add}.", agent)

        self.liquidity[tkn_add] += quantity
        agent.holdings[tkn_add] -= quantity

        if self.shares == 0:
            agent.holdings[self.unique_id] = updated_d
            self.shares = updated_d

        elif self.shares < 0:
            return self.fail_transaction('Shares cannot go below 0.', agent)
            # why would this possibly happen?

        else:
            d_diff = updated_d - initial_d
            share_amount = self.shares * d_diff / initial_d
            self.shares += share_amount
            if self.unique_id not in agent.holdings:
                agent.holdings[self.unique_id] = 0
            agent.holdings[self.unique_id] += share_amount
            agent.share_prices[self.unique_id] = quantity / share_amount

        return self

    def buy_shares(
            self,
            agent: Agent,
            quantity: float,
            tkn_add: str,
            fail_overdraft: bool = True
    ):
        initial_d = self.d
        updated_d = initial_d * (self.shares + quantity) / self.shares
        delta_tkn = self.calculate_y(
            self.modified_balances(omit=[tkn_add]),
            d=updated_d
        ) - self.liquidity[tkn_add]

        if delta_tkn > agent.holdings[tkn_add]:
            if fail_overdraft:
                return self.fail_transaction(f"Agent doesn't have enough {tkn_add}.", agent)
            else:
                # instead of failing, just round down
                delta_tkn = agent.holdings[tkn_add]
                return self.add_liquidity(agent, delta_tkn, tkn_add)

        self.liquidity[tkn_add] += delta_tkn
        agent.holdings[tkn_add] -= delta_tkn
        self.shares += quantity
        if self.unique_id not in agent.holdings:
            agent.holdings[self.unique_id] = 0
        agent.holdings[self.unique_id] += quantity
        return self

    def remove_uniform(
            self,
            agent: Agent,
            shares_removed: float
    ):
        if shares_removed > agent.holdings[self.unique_id]:
            raise ValueError('Agent tried to remove more shares than it owns.')
        elif shares_removed <= 0:
            raise ValueError('Withdraw quantity must be > 0.')

        share_fraction = shares_removed / self.shares

        delta_tkns = {}
        for tkn in self.asset_list:
            delta_tkns[tkn] = share_fraction * self.liquidity[tkn]  # delta_tkn is positive

            if delta_tkns[tkn] >= self.liquidity[tkn]:
                return self.fail_transaction(f'Not enough liquidity in {tkn}.', agent)

            if tkn not in agent.holdings:
                agent.holdings[tkn] = 0

        self.shares -= shares_removed
        agent.holdings[self.unique_id] -= shares_removed

        for tkn in self.asset_list:
            self.liquidity[tkn] -= delta_tkns[tkn]
            agent.holdings[tkn] += delta_tkns[tkn]  # agent is receiving funds, because delta_tkn is a negative number
        return self


def simulate_swap(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return new_state.swap(new_agent, tkn_sell, tkn_buy, buy_quantity, sell_quantity), new_agent


def simulate_add_liquidity(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        quantity: float,  # quantity of asset to be added
        tkn_add: str
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return new_state.add_liquidity(new_agent, quantity, tkn_add), new_agent


def simulate_remove_liquidity(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        quantity: float,  # in this case, quantity refers to a number of shares, not quantity of asset
        tkn_remove: str
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return new_state.remove_liquidity(new_agent, quantity, tkn_remove), new_agent


def simulate_buy_shares(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        quantity: float,
        tkn_add: str,
        fail_overdraft: bool = True
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return new_state.buy_shares(
        agent=new_agent,
        quantity=quantity,
        tkn_add=tkn_add,
        fail_overdraft=fail_overdraft
    ), new_agent
