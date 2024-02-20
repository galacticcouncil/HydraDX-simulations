import copy

from .agents import Agent
from .amm import AMM


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
        self.amp_change_step = 0
        self.target_amp_block = 0
        self.time_step = 0
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
        return self.amplification * self.n_coins

    @property
    def n_coins(self) -> int:
        return len(self.asset_list)

    @property
    def d(self) -> float:
        return self.calculate_d()

    def fail_transaction(self, error: str, **kwargs):
        self.fail = error
        return self

    def update(self):
        self.time_step += 1
        if self.target_amp_block >= self.time_step:
            self.amplification += self.amp_change_step

    def set_amplification(self, amplification: float, duration: float):
        self.target_amp_block = self.time_step + duration
        self.amp_change_step = (amplification - self.amplification) / duration

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

    # price is denominated in the first asset by default
    def spot_price(self, i: int = 1):
        """
        return the price of TKN denominated in NUMÉRAIRE
        """
        balances = list(self.liquidity.values())
        if i == 0:  # price of the numeraire is always 1
            return 1
        return self.price_at_balance(balances, self.d, i)

    def sell_spot(self, tkn_sell, tkn_buy: str, fee: float = None):
        if fee is None:
            fee = self.trade_fee
        if tkn_buy not in self.liquidity or tkn_sell not in self.liquidity:
            return 0
        else:
            return self.price(tkn_sell, tkn_buy) * (1 - fee)

    def buy_spot(self, tkn_buy: str, tkn_sell, fee: float = None):
        if fee is None:
            fee = self.trade_fee
        if tkn_buy not in self.liquidity or tkn_sell not in self.liquidity:
            return 0
        else:
            return self.price(tkn_buy, tkn_sell) / (1 - fee)

    def sell_limit(self, tkn_buy, tkn_sell):
        return self.liquidity[tkn_buy]

    def buy_limit(self, tkn_buy, tkn_sell):
        return self.liquidity[tkn_buy]

    def calculate_buy_from_sell(self, tkn_buy, tkn_sell, sell_quantity):
        reserves = self.modified_balances(delta={tkn_sell: sell_quantity}, omit=[tkn_buy])
        return (self.liquidity[tkn_buy] - self.calculate_y(reserves, self.d)) * (1 - self.trade_fee)

    def calculate_sell_from_buy(self, tkn_buy, tkn_sell, buy_quantity):
        reserves = self.modified_balances(delta={tkn_buy: -buy_quantity}, omit=[tkn_sell])
        return (self.calculate_y(reserves, self.d) - self.liquidity[tkn_sell]) / (1 - self.trade_fee)

    def price(self, tkn, denomination: str = ''):
        """
        return the price of TKN denominated in NUMÉRAIRE
        """
        if tkn == denomination:
            return 1
        if tkn not in self.liquidity or denomination not in self.liquidity:
            return 0
        i = list(self.liquidity.keys()).index(tkn)
        j = list(self.liquidity.keys()).index(denomination)
        return self.price_at_balance(
            balances=list(self.liquidity.values()),
            d=self.d,
            i=i, j=j
        )

    def price_at_balance(self, balances: list, d: float, i: int = 1, j: int = 0):
        n = self.n_coins
        ann = self.ann

        c = d
        sorted_bal = sorted(balances)
        for x in sorted_bal:
            c = c * d / (n * x)

        xi = balances[i]
        xj = balances[j]

        p = xj * (ann * xi + c) / (ann * xj + c) / xi

        return p

    def share_price(self, numeraire: str = ''):
        i = 0 if numeraire == '' else list(self.liquidity.keys()).index(numeraire)
        d = self.calculate_d()
        s = self.shares
        a = self.amplification
        n = self.n_coins

        c = d
        sorted_liq = sorted(self.liquidity.values())
        for x in sorted_liq:
            c = c * d / (n * x)
        xi = self.liquidity[self.asset_list[i]]
        ann = self.ann
        p = (d * xi * ann + xi * (n + 1) * c - xi * d) / (xi * ann + c) / s
        return p

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
            return self.fail_transaction('Agent has insufficient funds.')
        elif self.liquidity[tkn_buy] <= buy_quantity:
            return self.fail_transaction('Pool has insufficient liquidity.')

        if tkn_buy not in agent.holdings:
            agent.holdings[tkn_buy] = 0
        agent.holdings[tkn_buy] += buy_quantity
        agent.holdings[tkn_sell] -= sell_quantity
        self.liquidity[tkn_buy] -= buy_quantity
        self.liquidity[tkn_sell] += sell_quantity

        return self

    def swap_one(
            self,
            agent: Agent,
            quantity: float,
            tkn_sell: str = '',
            tkn_buy: str = '',
    ):
        """
        This can be used when you want to change the price of one asset without changing the price of the others.
        Specify one asset to buy or sell, and the quantity of each of the *other* assets to sell or buy.
        The quantity of the specified asset to trade will be determined.
        Caution: this will only work correctly if the pool is initially balanced (spot prices equal on all assets).
        """
        if tkn_sell and tkn_buy:
            raise ValueError('Cannot specify both buy and sell quantities.')

        if tkn_buy:
            tkns_sell = list(filter(lambda t: t != tkn_buy, self.asset_list))
            for tkn in tkns_sell:
                if tkn not in agent.holdings:
                    self.fail_transaction(f'Agent does not have any {tkn}.')
            if min([agent.holdings[tkn] for tkn in tkns_sell]) < quantity:
                return self.fail_transaction('Agent has insufficient funds.')

            sell_quantity = quantity
            buy_quantity = (self.liquidity[tkn_buy] - self.calculate_y(
                self.modified_balances(delta={tkn: quantity for tkn in tkns_sell}, omit=[tkn_buy]),
                self.d
            )) * (1 - self.trade_fee)

            if self.liquidity[tkn_buy] < buy_quantity:
                return self.fail_transaction('Pool has insufficient liquidity.')

            for tkn in tkns_sell:
                self.liquidity[tkn] += sell_quantity
                agent.holdings[tkn] -= sell_quantity
            self.liquidity[tkn_buy] -= buy_quantity
            agent.holdings[tkn_buy] += buy_quantity

        elif tkn_sell:
            tkns_buy = list(filter(lambda t: t != tkn_sell, self.asset_list))
            buy_quantity = quantity

            if min([self.liquidity[tkn] for tkn in tkns_buy]) < buy_quantity:
                return self.fail_transaction('Pool has insufficient liquidity.')

            sell_quantity = (self.calculate_y(
                self.modified_balances(delta={tkn: -quantity for tkn in tkns_buy}, omit=[tkn_sell]),
                self.d
            ) - self.liquidity[tkn_sell]) / (1 - self.trade_fee)
            if agent.holdings[tkn_sell] < sell_quantity:
                return self.fail_transaction(f'Agent has insufficient funds. {agent.holdings[tkn_sell]} < {quantity}')
            for tkn in tkns_buy:
                self.liquidity[tkn] -= buy_quantity
                if tkn not in agent.holdings:
                    agent.holdings[tkn] = 0
                agent.holdings[tkn] += buy_quantity
            self.liquidity[tkn_sell] += sell_quantity
            agent.holdings[tkn_sell] -= sell_quantity

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
            return self.fail_transaction(f'Not enough liquidity in {tkn_remove}.')
        if quantity <= 0:
            raise ValueError('Withdraw quantity must be > 0.')

        shares_removed = self.calculate_withdrawal_shares(tkn_remove, quantity)

        if shares_removed > agent.holdings[self.unique_id]:
            if fail_on_overdraw:
                return self.fail_transaction('Agent tried to remove more shares than it owns.')
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
            return self.fail_transaction('Agent has insufficient funds.')
        elif shares_removed <= 0:
            return self.fail_transaction('Withdraw quantity must be > 0.')

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
        updated_reserves = {
            tkn: self.liquidity[tkn] + (quantity if tkn == tkn_add else 0) for tkn in self.asset_list
        }
        initial_d = self.calculate_d()
        updated_d = self.calculate_d(tuple(updated_reserves.values()))
        if updated_d < initial_d:
            return self.fail_transaction('invariant decreased for some reason')
        if agent.holdings[tkn_add] < quantity:
            return self.fail_transaction(f"Agent doesn't have enough {tkn_add}.")

        fixed_fee = self.trade_fee
        fee = fixed_fee * self.n_coins / (4 * (self.n_coins - 1))

        d0, d1 = initial_d, updated_d

        adjusted_balances = (
            [
                updated_reserves[tkn] -
                abs(updated_reserves[tkn] - d1 * self.liquidity[tkn] / d0) * fee
                for tkn in self.asset_list
            ]
            if self.shares > 0 else updated_reserves
        )

        adjusted_d = self.calculate_d(adjusted_balances)
        if self.shares == 0:
            shares_return = updated_d
        else:
            d_diff = adjusted_d - initial_d
            shares_return = self.shares * d_diff / initial_d

        if self.unique_id not in agent.holdings:
            agent.holdings[self.unique_id] = 0
        agent.holdings[self.unique_id] += shares_return
        self.shares += shares_return
        self.liquidity[tkn_add] += quantity
        agent.holdings[tkn_add] -= quantity
        return self

    def buy_shares(
            self,
            agent: Agent,
            quantity: float,
            tkn_add: str,
            fail_overdraft: bool = True
    ):

        initial_d = self.d
        d1 = initial_d + initial_d * quantity / self.shares

        xp = self.modified_balances(omit=[tkn_add])
        y = self.calculate_y(xp, d1)

        fee = self.trade_fee * self.n_coins / (4 * (self.n_coins - 1))
        reserves_reduced = []
        asset_reserve = 0
        for tkn, balance in self.liquidity.items():
            dx_expected = (
                    balance * d1 / initial_d - balance
            ) if tkn != tkn_add else (
                    y - balance * d1 / initial_d
            )
            reduced_balance = balance - fee * dx_expected
            if tkn == tkn_add:
                asset_reserve = reduced_balance
            else:
                reserves_reduced.append(reduced_balance)

        y1 = self.calculate_y(reserves_reduced, d1)
        dy = y1 - asset_reserve
        dy_0 = y - asset_reserve
        fee_amount = dy - dy_0
        delta_tkn = dy + fee_amount

        if delta_tkn > agent.holdings[tkn_add]:
            if fail_overdraft:
                return self.fail_transaction(f"Agent doesn't have enough {tkn_add}.")
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
                return self.fail_transaction(f'Not enough liquidity in {tkn}.')

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
