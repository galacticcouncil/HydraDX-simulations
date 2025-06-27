import math

from .agents import Agent
from .exchange import Exchange

class XykState(Exchange):
    unique_id: str = 'xyk'

    def __init__(
            self,
            tokens: dict[str, float],
            trade_fee: float = 0,
            unique_id: str = '',
            shares: float = 0
    ):

        """
        Tokens should be in the form of:
        {
            token1: quantity,
            token2: quantity
        }
        There must be exactly two.
        """
        super().__init__()
        if len(tokens.keys()) != 2:
            raise ValueError('Need exactly two tokens for XYK AMM')

        self.time_step = 0
        self.liquidity = dict()
        self.asset_list: list[str] = []
        self.trade_fee = trade_fee
        if unique_id:
            self.unique_id = unique_id

        for token, quantity in tokens.items():
            self.asset_list.append(token)
            self.liquidity[token] = quantity

        self.shares = shares or self.calculate_k()

    def fail_transaction(self, error: str, **kwargs):
        self.fail = error
        return self

    def update(self):
        self.time_step += 1

    def calculate_k(self) -> float:
        return math.sqrt(self.liquidity[self.asset_list[0]] * self.liquidity[self.asset_list[1]])

    def sell_spot(self, tkn_sell, tkn_buy: str, fee: float = None):
        if tkn_buy not in self.liquidity or tkn_sell not in self.liquidity:
            return 0
        if fee is None:
            fee = self.trade_fee
        return self.price(tkn_sell, tkn_buy) * (1 - fee)

    def buy_spot(self, tkn_buy: str, tkn_sell, fee: float = None):
        if tkn_buy not in self.liquidity or tkn_sell not in self.liquidity:
            return 0
        if fee is None:
            fee = self.trade_fee
        return self.price(tkn_buy, tkn_sell) / (1 - fee)

    def sell_limit(self, tkn_buy, tkn_sell):
        if tkn_sell not in self.liquidity:
            return 0
        return float("inf")

    def buy_limit(self, tkn_buy, tkn_sell):
        if tkn_buy not in self.liquidity:
            return 0
        return self.liquidity[tkn_buy]

    def calculate_buy_from_sell(self, tkn_buy, tkn_sell, sell_quantity):
        x, y = self.liquidity[tkn_sell], self.liquidity[tkn_buy]
        return y * (- sell_quantity / (x + sell_quantity)) * (1 - self.trade_fee)

    def calculate_sell_from_buy(self, tkn_buy, tkn_sell, buy_quantity):
        x, y = self.liquidity[tkn_sell], self.liquidity[tkn_buy]
        return x * (- buy_quantity / (buy_quantity + y * (1 - self.trade_fee)))

    def price(self, tkn, denomination: str = ''):
        """
        return the price of TKN denominated in NUMÉRAIRE
        """
        if tkn not in self.liquidity or denomination not in self.liquidity:
            return 0
        return self.liquidity[denomination] / self.liquidity[tkn] if tkn != denomination else 1

    def share_price(self, numeraire: str = ''):
        return 2 * self.liquidity[numeraire] / self.shares

    def copy(self):
        new_pool = XykState(
            {k: v for k,v in self.liquidity.items()},
            trade_fee=self.trade_fee,
            unique_id=self.unique_id,
            shares=self.shares
        )
        new_pool.time_step = self.time_step
        return new_pool

    def __repr__(self):
        # round to given precision
        precision = 10
        liquidity = {tkn: round(self.liquidity[tkn], precision) for tkn in self.asset_list}
        shares = round(self.shares, precision)
        return (
            f'XYK Pool: {self.unique_id}\n'
            f'********************************\n'
            f'trade fee: {self.trade_fee}\n'
            f'shares: {shares}\n'
            f'tokens: (\n\n'
        ) + '\n'.join(
            [(
                    f'    {token}\n'
                    f'    quantity: {liquidity[token]}\n'
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
            if buy_quantity > self.liquidity[tkn_buy]:
                return self.fail_transaction('Pool has insufficient liquidity.')
            sell_quantity = self.calculate_sell_from_buy(tkn_buy, tkn_sell, buy_quantity)
        elif sell_quantity:
            buy_quantity = self.calculate_buy_from_sell(tkn_buy, tkn_sell, sell_quantity)

        if not agent.validate_holdings(tkn_sell, sell_quantity):
            return self.fail_transaction('Agent has insufficient funds.')

        agent.remove(tkn_sell, sell_quantity)
        agent.add(tkn_buy, buy_quantity)
        self.liquidity[tkn_buy] -= buy_quantity
        self.liquidity[tkn_sell] += sell_quantity

        return self


def simulate_swap(
        old_state: XykState,
        old_agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return new_state.swap(new_agent, tkn_sell, tkn_buy, buy_quantity, sell_quantity), new_agent
