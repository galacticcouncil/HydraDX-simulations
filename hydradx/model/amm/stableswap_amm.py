from .global_state import AMM
from .agents import Agent
from mpmath import mpf, mp
mp.dps = 50

# N_COINS = 2  # I think we cannot currently go higher than this
# ann means how concentrated the liquidity is;
# the higher the number, the less the price changes as the pool moves away from balance


class StableSwapPoolState(AMM):
    def __init__(self, tokens: dict, amplification: float, precision: float = 1, trade_fee: float = 0):
        """
        Tokens should be in the form of:
        {
            token1: quantity,
            token2: quantity
        }
        There should only be two.
        """
        super().__init__()
        self.amplification = amplification
        self.precision = precision
        self.liquidity = dict()
        self.asset_list: list[str] = []
        self.trade_fee = trade_fee

        for token, quantity in tokens.items():
            self.asset_list.append(token)
            self.liquidity[token] = mpf(quantity)

        self.shares = self.calculate_d()
        self.d = self.calculate_d()

    @property
    def ann(self):
        return self.amplification * len(self.asset_list) ** len(self.asset_list)

    def has_converged(self, v0, v1) -> bool:
        diff = abs(v0 - v1)
        if (v1 <= v0 and diff < self.precision) or (v1 > v0 and diff <= self.precision):
            return True
        return False

    def calculate_d(self, reserves=(), max_iterations=128):
        n_coins = len(self.asset_list)
        reserves = reserves or self.liquidity.values()
        xp_sorted = sorted(reserves)
        s = sum(xp_sorted)
        if s == 0:
            return 0

        d = s
        for i in range(max_iterations):

            d_p = d
            for x in xp_sorted:
                d_p *= d / (x * n_coins)

            d_prev = d
            d = (self.ann * s + d_p * n_coins) * d / ((self.ann - 1) * d + (n_coins + 1) * d_p) + 2

            if self.has_converged(d_prev, d):
                return d

    def calculate_y(self, reserve, d, max_iterations=128):
        s = reserve
        c = d
        c *= d / (2 * reserve)
        c *= d / (self.ann * len(self.liquidity.keys()))

        b = s + d / self.ann
        y = d
        for i in range(max_iterations):
            y_prev = y
            y = (y ** 2 + c) / (2 * y + b - d)
            if self.has_converged(y_prev, y):
                return y

    # Calculate new amount of reserve OUT given amount to be added to the pool
    def calculate_y_given_in(
        self,
        amount: float,
        tkn_in: str,
    ) -> float:
        new_reserve_in = self.liquidity[tkn_in] + amount
        d = self.calculate_d()
        return self.calculate_y(new_reserve_in, d)

    # Calculate new amount of reserve IN given amount to be withdrawn from the pool
    def calculate_y_given_out(
            self,
            amount: float,
            tkn_out: str
    ) -> float:
        new_reserve_out = self.liquidity[tkn_out] - amount
        d = self.calculate_d()
        return self.calculate_y(new_reserve_out, d)

    def calculate_out_given_in(
        self,
        tkn_in: str,
        tkn_out: str,
        amount_in: float
    ):
        new_reserve_out = self.calculate_y_given_in(amount_in, tkn_in)
        return self.liquidity[tkn_out] - new_reserve_out

    def calculate_in_given_out(
            self,
            tkn_in: str,
            tkn_out: str,
            amount_out: float
    ):
        new_reserve_in = self.calculate_y_given_out(amount_out, tkn_out)
        return new_reserve_in - self.liquidity[tkn_in]

    def spot_price(self):
        x, y = self.liquidity.values()
        d = self.d
        return (x / y) * (self.ann * x * y ** 2 + d ** 3) / (self.ann * x ** 2 * y + d ** 3)

    def execute_swap(
        self,
        old_agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
    ):
        d = self.calculate_d()
        if buy_quantity:
            sell_quantity = (self.calculate_y(self.liquidity[tkn_buy] - buy_quantity, d)
                             - self.liquidity[tkn_sell]) * (1 + self.trade_fee)
        elif sell_quantity:
            buy_quantity = (self.liquidity[tkn_buy] -
                            self.calculate_y(self.liquidity[tkn_sell] + sell_quantity, d)) * (1 - self.trade_fee)

        if old_agent.holdings[tkn_sell] - sell_quantity < 0:
            return self.fail_transaction('Agent has insufficient funds.'), old_agent
        elif self.liquidity[tkn_buy] <= buy_quantity:
            return self.fail_transaction('Pool has insufficient liquidity.'), old_agent

        new_agent = old_agent  # .copy()
        new_agent.holdings[tkn_buy] += buy_quantity
        new_agent.holdings[tkn_sell] -= sell_quantity
        self.liquidity[tkn_buy] -= buy_quantity
        self.liquidity[tkn_sell] += sell_quantity

        return self, new_agent

    def __repr__(self):
        return (
            f'Stable Swap Pool\n'
            f'base trade fee: {self.trade_fee}\n'
            f'shares: {self.shares}\n'
            f'tokens: (\n'
        ) + ')\n(\n'.join(
            [(
                f'    {token}\n'
                f'    quantity: {self.liquidity[token]}\n'
                f'    weight: {self.liquidity[token] / sum(self.liquidity.values())}\n'
            ) for token in self.asset_list]
        ) + '\n)\n' + (
            f'error message:{self.fail or "none"}'
        )


def swap(
    old_state: StableSwapPoolState,
    old_agent: Agent,
    tkn_sell: str,
    tkn_buy: str,
    buy_quantity: float = 0,
    sell_quantity: float = 0
):
    return old_state.copy().execute_swap(old_agent.copy(), tkn_sell, tkn_buy, buy_quantity, sell_quantity)


def add_liquidity(
    old_state: StableSwapPoolState,
    old_agent: Agent,
    quantity: float,
    tkn_add: str
):
    initial_d = old_state.calculate_d()
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    for token in old_state.asset_list:
        delta_r = quantity * old_state.liquidity[token] / old_state.liquidity[tkn_add]
        new_agent.holdings[token] -= delta_r
        new_state.liquidity[token] += delta_r

    updated_d = new_state.calculate_d()

    if updated_d < initial_d:
        return None

    if old_state.shares == 0:
        new_agent.shares[new_state.unique_id] = updated_d
        new_state.shares = updated_d

    else:
        d_diff = updated_d - initial_d
        share_amount = old_state.shares * d_diff / initial_d
        new_state.shares += share_amount
        if new_state.unique_id not in new_agent.shares:
            new_agent.shares[new_state.unique_id] = 0
        new_agent.shares[new_state.unique_id] += share_amount

    return new_state, new_agent


def remove_liquidity(
    old_state: StableSwapPoolState,
    old_agent: Agent,
    quantity: float,
    tkn_remove: str
):
    if quantity > old_agent.shares[old_state.unique_id]:
        raise ValueError('Agent tried to remove more shares than it owns.')

    share_fraction = quantity / old_state.shares
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    new_state.shares -= quantity
    new_agent.shares[old_state.unique_id] -= quantity
    for tkn in new_state.asset_list:
        new_agent.holdings[tkn] += old_state.liquidity[tkn] * share_fraction
        new_state.liquidity[tkn] -= old_state.liquidity[tkn] * share_fraction
    return new_state, new_agent


StableSwapPoolState.add_liquidity = staticmethod(add_liquidity)
StableSwapPoolState.remove_liquidity = staticmethod(remove_liquidity)
StableSwapPoolState.swap = staticmethod(swap)
