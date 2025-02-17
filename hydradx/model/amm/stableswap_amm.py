import copy

from .agents import Agent
from .exchange import Exchange


class StableSwapPoolState(Exchange):
    unique_id: str = 'stableswap'

    def __init__(
            self,
            tokens: dict,
            amplification: float,
            precision: float = 0.0001,
            trade_fee: float = 0,
            unique_id: str = '',
            spot_price_precision: float = 1e-07,
            shares: float = 0,
            peg: float or list = None,
            peg_target: float or list = None,
            max_peg_update: float = float('inf')
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
        self.peg_target_updated_at = 0
        self.precision = precision
        self.spot_price_precision = spot_price_precision
        self.liquidity = dict()
        self.asset_list: list[str] = []
        self.trade_fee = trade_fee
        if unique_id:
            self.unique_id = unique_id

        for token, quantity in tokens.items():
            self.asset_list.append(token)
            self.liquidity[token] = quantity

        self.set_peg(peg)
        self.set_peg_target(peg_target)
        self.max_peg_update = max_peg_update

        self.shares = shares or self.calculate_d()
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

    def set_peg(self, peg=None):
        if peg is None:
            self.peg = [1] * len(self.asset_list)
        elif isinstance(peg, (int, float)):  # peg is price of second asset denominated in first asset
            assert len(self.asset_list) == 2
            self.peg = [1, peg]
        else:  # peg is list of prices of all assets except the first, denominate in first asset
            assert len(peg) == len(self.asset_list) - 1
            self.peg = [1] + peg

    def set_peg_target(self, peg_target=None):
        if peg_target is None:
            self.peg_target = [p for p in self.peg]
        elif isinstance(peg_target, (int, float)):
            assert len(self.asset_list) == 2
            self.peg_target = [1, peg_target]
        else:
            assert len(peg_target) == len(self.asset_list) - 1
            self.peg_target = [1] + peg_target

        assert len(self.peg) == len(self.peg_target)
        self.peg_target_updated_at = self.time_step

    def set_amplification(self, amplification: float, duration: float):
        self.target_amp_block = self.time_step + duration
        self.amp_change_step = (amplification - self.amplification) / duration

    def has_converged(self, v0, v1) -> bool:
        diff = abs(v0 - v1)
        if (v1 <= v0 and diff < self.precision) or (v1 > v0 and diff <= self.precision):
            return True
        return False

    def get_adjusted_liquidity(self, balances=None):
        if balances is None:
            balances = self.liquidity
        adjusted_liquidity = {}
        for tkn in balances:
            i = self.asset_list.index(tkn)
            adjusted_liquidity[tkn] = balances[tkn] * self.peg[i]
        return adjusted_liquidity

    def calculate_d(self, reserves=(), max_iterations=128) -> float:
        reserves = reserves or list(self.liquidity.values())
        xp_sorted = sorted([reserves[i] * self.peg[i] for i in range(self.n_coins)])
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

    def calculate_y(self, reserves: list, d: float, tkn_omit: str, max_iterations=128):

        # get all the balances except tkn_out and sort them from low to high
        pegs = [self.peg[i] for i, tkn in enumerate(self.asset_list) if tkn != tkn_omit]
        balances = sorted([reserves[i] * pegs[i] for i in range(self.n_coins - 1)])

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
                return y / self.peg[self.asset_list.index(tkn_omit)]

        return y / self.peg[self.asset_list.index(tkn_omit)]

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
        if tkn_buy not in self.liquidity or tkn_sell not in self.liquidity:
            return 0
        new_state = self.copy()
        fee_min = new_state._update_peg()
        if fee is None:
            fee = 0
        fee = max(fee, fee_min)
        return new_state.price(tkn_sell, tkn_buy) * (1 - fee)

    def buy_spot(self, tkn_buy: str, tkn_sell, fee: float = None):
        if tkn_buy not in self.liquidity or tkn_sell not in self.liquidity:
            return 0
        new_state = self.copy()
        fee_min = new_state._update_peg()
        if fee is None:
            fee = 0
        fee = max(fee, fee_min)
        return new_state.price(tkn_buy, tkn_sell) / (1 - fee)

    def sell_limit(self, tkn_buy, tkn_sell):  # TODO: fix this
        return self.liquidity[tkn_buy]

    def buy_limit(self, tkn_buy, tkn_sell):
        if tkn_buy not in self.liquidity:
            return 0
        return self.liquidity[tkn_buy]

    def calculate_buy_from_sell(self, tkn_buy, tkn_sell, sell_quantity):
        fee = self.calculate_fee()
        reserves = self.modified_balances(delta={tkn_sell: sell_quantity}, omit=[tkn_buy])
        return (self.liquidity[tkn_buy] - self.calculate_y(reserves, self.d, tkn_buy)) * (1 - fee)

    def calculate_sell_from_buy(self, tkn_buy, tkn_sell, buy_quantity):
        fee = self.calculate_fee()
        reserves = self.modified_balances(delta={tkn_buy: -buy_quantity}, omit=[tkn_sell])
        return (self.calculate_y(reserves, self.d, tkn_sell) - self.liquidity[tkn_sell]) / (1 - fee)

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
        adj_balances = [balances[k] * self.peg[k] for k in range(n)]
        sorted_bal = sorted(adj_balances)
        for x in sorted_bal:
            c = c * d / (n * x)

        xi = adj_balances[i]
        xj = adj_balances[j]

        p = xj * (ann * xi + c) / (ann * xj + c) / xi

        p_adj = p * self.peg[i] / self.peg[j]

        return p_adj

    def share_price(self, numeraire: str = ''):
        i = 0 if numeraire == '' else list(self.liquidity.keys()).index(numeraire)
        d = self.calculate_d()
        s = self.shares
        a = self.amplification
        n = self.n_coins

        c = d
        sorted_liq = sorted([self.liquidity[self.asset_list[i]] * self.peg[i] for i in range(n)])
        for x in sorted_liq:
            c = c * d / (n * x)
        xi = self.liquidity[self.asset_list[i]] * self.peg[i]
        ann = self.ann
        p = (d * xi * ann + xi * (n + 1) * c - xi * d) / (xi * ann + c) / s
        return p / self.peg[i]

    def modified_balances(self, delta: dict = None, omit: list = ()):
        balances = copy.copy(self.liquidity)
        if delta:
            for tkn, value in delta.items():
                balances[tkn] += value
        if omit:
            for tkn in omit:
                balances.pop(tkn)
        return list(balances.values())

    def calculate_withdrawal_shares(self, tkn_remove, quantity, fee = None):
        if fee is None:
            fee = self.calculate_fee()
        updated_d = self.calculate_d(self.modified_balances(delta={tkn_remove: -quantity}))
        return self.shares * (1 - updated_d / self.d) / (1 - fee)

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
            f'peg: {self.peg}\n'
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

    def _calculate_peg_deltas(self):
        peg_deltas = []
        block_ct = max([self.time_step - self.peg_target_updated_at, 1])
        for i in range(len(self.peg)):
            peg_diff = self.peg_target[i] - self.peg[i]
            max_peg_move = self.max_peg_update * self.peg[i] * block_ct
            if abs(peg_diff) <= max_peg_move:
                peg_deltas.append(peg_diff)
            elif self.peg[i] < self.peg_target[i]:
                peg_deltas.append(max_peg_move)
            else:
                peg_deltas.append(-max_peg_move)
        return peg_deltas

    def _calculate_fee_from_peg_deltas(self, peg_deltas):
        block_ct = max([self.time_step - self.peg_target_updated_at, 1])
        peg_relative_changes = [peg_deltas[i] / block_ct / self.peg[i] for i in range(len(self.peg))]
        peg_diff_per_block = (1 + max(peg_relative_changes)) / (1 + min(peg_relative_changes)) - 1
        return max(2 * peg_diff_per_block, self.trade_fee)

    def calculate_fee(self):
        peg_deltas = self._calculate_peg_deltas()
        return self._calculate_fee_from_peg_deltas(peg_deltas)

    def _update_peg(self) -> float:
        peg_deltas = self._calculate_peg_deltas()
        fee = self._calculate_fee_from_peg_deltas(peg_deltas)
        for i in range(len(self.peg)):
            self.peg[i] += peg_deltas[i]
        return fee

    def swap(
            self,
            agent: Agent,
            tkn_sell: str,
            tkn_buy: str,
            buy_quantity: float = 0,
            sell_quantity: float = 0
    ):
        fee = self._update_peg()

        if buy_quantity:
            reserves = self.modified_balances(delta={tkn_buy: -buy_quantity}, omit=[tkn_sell])
            sell_quantity = (self.calculate_y(reserves, self.d, tkn_sell) - self.liquidity[tkn_sell]) / (1 - fee)
        elif sell_quantity:
            reserves = self.modified_balances(delta={tkn_sell: sell_quantity}, omit=[tkn_buy])
            buy_quantity = (self.liquidity[tkn_buy] - self.calculate_y(reserves, self.d, tkn_buy)) * (1 - fee)

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
            tkn_buy: str = ''
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
            fee = self._update_peg()
            tkns_sell = list(filter(lambda t: t != tkn_buy, self.asset_list))
            for tkn in tkns_sell:
                if tkn not in agent.holdings:
                    self.fail_transaction(f'Agent does not have any {tkn}.')
            if min([agent.holdings[tkn] for tkn in tkns_sell]) < quantity:
                return self.fail_transaction('Agent has insufficient funds.')

            sell_quantity = quantity
            buy_quantity = (self.liquidity[tkn_buy] - self.calculate_y(
                self.modified_balances(delta={tkn: quantity for tkn in tkns_sell}, omit=[tkn_buy]),
                self.d,
                tkn_buy
            )) * (1 - fee)

            if self.liquidity[tkn_buy] < buy_quantity:
                return self.fail_transaction('Pool has insufficient liquidity.')

            for tkn in tkns_sell:
                self.liquidity[tkn] += sell_quantity
                agent.holdings[tkn] -= sell_quantity
            self.liquidity[tkn_buy] -= buy_quantity
            agent.holdings[tkn_buy] += buy_quantity

        elif tkn_sell:
            fee = self._update_peg()
            tkns_buy = list(filter(lambda t: t != tkn_sell, self.asset_list))
            buy_quantity = quantity

            if min([self.liquidity[tkn] for tkn in tkns_buy]) < buy_quantity:
                return self.fail_transaction('Pool has insufficient liquidity.')

            sell_quantity = (self.calculate_y(
                self.modified_balances(delta={tkn: -quantity for tkn in tkns_buy}, omit=[tkn_sell]),
                self.d,
                tkn_sell
            ) - self.liquidity[tkn_sell]) / (1 - fee)
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

        fee = self._update_peg()
        shares_removed = self.calculate_withdrawal_shares(tkn_remove, quantity, fee)

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
            tkn_remove: str
    ):
        # First, need to calculate
        # * Get current D
        # * Solve Eqn against y_i for D - _token_amount

        if shares_removed > agent.holdings[self.unique_id]:
            return self.fail_transaction('Agent has insufficient funds.')
        elif shares_removed <= 0:
            return self.fail_transaction('Withdraw quantity must be > 0.')

        _fee = self._update_peg()
        _fee *= self.n_coins / 4 / (self.n_coins - 1)

        initial_d = self.calculate_d()
        reduced_d = initial_d - shares_removed * initial_d / self.shares

        xp_reduced = copy.copy(self.liquidity)
        xp_reduced.pop(tkn_remove)

        reduced_y = self.calculate_y(self.modified_balances(omit=[tkn_remove]), reduced_d, tkn_remove)
        asset_reserve = self.liquidity[tkn_remove]

        for tkn in self.asset_list:
            if tkn == tkn_remove:
                dx_expected = self.liquidity[tkn] * reduced_d / initial_d - reduced_y
                asset_reserve -= _fee * dx_expected
                assert asset_reserve > 0
            else:
                dx_expected = self.liquidity[tkn] - self.liquidity[tkn] * reduced_d / initial_d
                xp_reduced[tkn] -= _fee * dx_expected
                assert xp_reduced[tkn] > 0

        dy = asset_reserve - self.calculate_y(list(xp_reduced.values()), reduced_d, tkn_remove)

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
        fee = self._update_peg()
        updated_reserves = {
            tkn: self.liquidity[tkn] + (quantity if tkn == tkn_add else 0) for tkn in self.asset_list
        }
        initial_d = self.calculate_d()
        updated_d = self.calculate_d(tuple(updated_reserves.values()))
        if updated_d < initial_d:
            return self.fail_transaction('invariant decreased for some reason')
        if agent.holdings[tkn_add] < quantity:
            return self.fail_transaction(f"Agent doesn't have enough {tkn_add}.")

        fixed_fee = fee
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

    def add_liquidity_spot(self, tkn_add: str, precision: float = None):
        """Calculates spot price of adding liquidity as shares denominated in liquidity"""
        if precision is None: precision = self.spot_price_precision
        trade_size = self.liquidity[tkn_add] * precision
        agent = Agent({tkn_add: trade_size})
        new_state, new_agent = simulate_add_liquidity(self, agent, trade_size, tkn_add)
        return trade_size / new_agent.holdings[self.unique_id]

    def buy_shares_spot(self, tkn_add: str, precision: float = None):
        """Calculates spot price of buying shares as shares denominated in liquidity"""
        if precision is None: precision = self.spot_price_precision
        trade_size = self.liquidity[tkn_add] * precision
        share_price = self.share_price(tkn_add)
        init_tkn_add = share_price * trade_size * 2
        agent = Agent({tkn_add: init_tkn_add})
        new_state, new_agent = simulate_buy_shares(self, agent, trade_size, tkn_add)
        return (init_tkn_add - new_agent.holdings[tkn_add]) / trade_size

    def remove_liquidity_spot(self, tkn_remove: str, precision: float = None):
        """Calculates spot price of removing liquidity as shares denominated in liquidity"""
        if precision is None: precision = self.spot_price_precision
        trade_size = self.liquidity[tkn_remove] * precision
        agent = Agent({self.unique_id: trade_size})
        new_state, new_agent = simulate_remove_liquidity(self, agent, trade_size, tkn_remove)
        return new_agent.holdings[tkn_remove] / trade_size

    def withdraw_asset_spot(self, tkn_remove: str, precision: float = None):
        """Calculates spot price of withdrawing asset as shares denominated in liquidity"""
        if precision is None: precision = self.spot_price_precision
        trade_size = self.liquidity[tkn_remove] * precision
        delta_shares = self.calculate_withdrawal_shares(tkn_remove, trade_size)
        return trade_size / delta_shares

    def buy_shares(
            self,
            agent: Agent,
            quantity: float,
            tkn_add: str,
            fail_overdraft: bool = True
    ):

        trade_fee = self._update_peg()
        initial_d = self.d
        d1 = initial_d + initial_d * quantity / self.shares

        xp = self.modified_balances(omit=[tkn_add])
        y = self.calculate_y(xp, d1, tkn_add)

        fee = trade_fee * self.n_coins / (4 * (self.n_coins - 1))
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

        y1 = self.calculate_y(reserves_reduced, d1, tkn_add)
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

        self._update_peg()
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


def simulate_remove_uniform(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        shares_removed: float
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return new_state.remove_uniform(new_agent, shares_removed), new_agent


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
