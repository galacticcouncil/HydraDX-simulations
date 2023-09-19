import math
from .amm import AMM, FeeMechanism, basic_fee
from .agents import Agent
# when checking i.e. liquidity < 0, how many zeroes do we need to see before it's close enough?
precision_level = 20


class ConstantProductPoolState(AMM):
    def __init__(
            self,
            tokens: dict[str: float],
            trade_fee: FeeMechanism or float = 0,
            unique_id=''
    ):
        """
        Tokens should be in the form of:
        {
            token1: quantity,
            token2: quantity
        }
        There should only be two.
        """
        super().__init__()
        self.trade_fee: FeeMechanism = trade_fee.assign(self) if isinstance(trade_fee, FeeMechanism)\
            else basic_fee(trade_fee).assign(self)
        self.liquidity = dict()
        self.asset_list: list[str] = []

        for token, quantity in tokens.items():
            self.asset_list.append(token)
            self.liquidity[token] = quantity

        self.shares = self.liquidity[self.asset_list[0]]

        self.unique_id = unique_id

    @staticmethod
    def thorchain_fee() -> FeeMechanism:
        def fee_function(exchange: AMM, tkn: str, delta_tkn: float):
            return delta_tkn / (delta_tkn + exchange.liquidity[tkn])

        return FeeMechanism(fee_function=fee_function, name='Thorchain fee')

    @staticmethod
    def custom_slip_fee(slip_factor: float, minimum: float = 0) -> FeeMechanism:
        def fee_function(
            exchange: AMM, tkn: str, delta_tkn: float
        ) -> float:
            fee = (slip_factor * delta_tkn
                   / (delta_tkn + exchange.liquidity[tkn])) + minimum

            return fee
        return FeeMechanism(fee_function=fee_function, name=f'slip fee {slip_factor * 100}% slippage + {minimum}')

    @property
    def invariant(self):
        return math.prod(self.liquidity.values())

    def __repr__(self):
        precision = 12
        return (
            f'Constant Product Pool\n'
            f'base trade fee: {self.trade_fee.name}\n'
            f'shares: {self.shares}\n'
            f'tokens: (\n'
        ) + ')\n(\n'.join(
            [(
                f'    {token}\n'
                f'    quantity: {round(self.liquidity[token], precision)}\n'
                f'    weight: {round(self.liquidity[token] / sum(self.liquidity.values()), precision)}\n'
            ) for token in self.asset_list]
        ) + '\n)'

    def swap(
        self,
        agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
    ):
    
        if not (tkn_buy in self.asset_list and tkn_sell in self.asset_list):
            return self.fail_transaction('Invalid token name.', agent)
    
        # turn a negative buy into a sell and vice versa
        if buy_quantity < 0:
            sell_quantity = -buy_quantity
            buy_quantity = 0
            t = tkn_sell
            tkn_sell = tkn_buy
            tkn_buy = t
        elif sell_quantity < 0:
            buy_quantity = -sell_quantity
            sell_quantity = 0
            t = tkn_sell
            tkn_sell = tkn_buy
            tkn_buy = t
    
        if sell_quantity != 0:
            # when amount to be paid in is specified, calculate payout
            buy_quantity = sell_quantity * self.liquidity[tkn_buy] / (
                    self.liquidity[tkn_sell] + sell_quantity)
            if math.isnan(buy_quantity):
                buy_quantity = sell_quantity  # this allows infinite liquidity for testing
            trade_fee = self.trade_fee.compute(tkn=tkn_sell, delta_tkn=sell_quantity)
            buy_quantity *= 1 - trade_fee
    
        elif buy_quantity != 0:
            # calculate input price from a given payout
            sell_quantity = buy_quantity * self.liquidity[tkn_sell] / (self.liquidity[tkn_buy] - buy_quantity)
            if math.isnan(sell_quantity):
                sell_quantity = buy_quantity  # this allows infinite liquidity for testing
            trade_fee = self.trade_fee.compute(tkn=tkn_sell, delta_tkn=sell_quantity)
            sell_quantity /= 1 - trade_fee
    
        else:
            return self.fail_transaction('Must specify buy quantity or sell quantity.', agent)
    
        if self.liquidity[tkn_sell] + sell_quantity <= 0 or self.liquidity[tkn_buy] - buy_quantity <= 0:
            return self.fail_transaction('Not enough liquidity in the pool.', agent)
    
        if agent.holdings[tkn_sell] - sell_quantity < 0 or agent.holdings[tkn_buy] + buy_quantity < 0:
            return self.fail_transaction('Agent has insufficient holdings.', agent)
    
        agent.holdings[tkn_buy] += buy_quantity
        agent.holdings[tkn_sell] -= sell_quantity
        self.liquidity[tkn_sell] += sell_quantity
        self.liquidity[tkn_buy] -= buy_quantity
    
        return self, agent
    
    def add_liquidity(
            self,
            agent: Agent,
            quantity: float,
            tkn_add: str
    ):
        if self.unique_id not in agent.holdings:
            agent.holdings[self.unique_id] = 0
    
        delta_r = {}
        for tkn in self.asset_list:
            delta_r[tkn] = quantity * self.liquidity[tkn] / self.liquidity[tkn_add]
    
            if agent.holdings[tkn] - delta_r[tkn] < 0:
                # fail
                return self.fail_transaction(f'Agent has insufficient funds ({tkn}).', agent)
    
        for tkn in self.asset_list:
            agent.holdings[tkn] -= delta_r[tkn]
            self.liquidity[tkn] += delta_r[tkn]
    
        new_shares = (self.liquidity[tkn_add] / (self.liquidity[tkn_add] - quantity) - 1) * self.shares
        self.shares += new_shares
    
        agent.holdings[self.unique_id] += new_shares
        if agent.holdings[self.unique_id] > 0:
            agent.share_prices[self.unique_id] = (
                self.liquidity[self.asset_list[1]] / self.liquidity[self.asset_list[0]]
            )
        return self, agent

    def remove_liquidity(
            self,
            agent: Agent,
            quantity: float,
            tkn_remove: str = ''
    ):

        if quantity < 0:
            return self.fail_transaction('Cannot remove negative liquidity.', agent)
        if quantity > agent.holdings[self.unique_id]:
            return self.fail_transaction('Tried to remove more shares than agent owns.', agent)
        if tkn_remove not in self.asset_list:
            # withdraw some of each
            tkns = self.asset_list
            withdraw_fraction = quantity / self.shares
            for tkn in tkns:
                withdraw_quantity = self.liquidity[tkn] * withdraw_fraction
                self.liquidity[tkn] -= withdraw_quantity
                agent.holdings[tkn] += withdraw_quantity
            agent.holdings[tkn_remove] -= quantity
        else:
            withdraw_quantity = abs(quantity) / self.shares * self.liquidity[tkn_remove]
            self.add_liquidity(
                agent, -withdraw_quantity, tkn_remove
            )
        return self


def simulate_add_liquidity(
    old_state: ConstantProductPoolState,
    old_agent: Agent,
    quantity: float,
    tkn_add: str
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    new_state.add_liquidity(
        agent=new_agent,
        quantity=quantity,
        tkn_add=tkn_add
    )
    return new_state, new_agent


def simulate_remove_liquidity(
    old_state: ConstantProductPoolState,
    old_agent: Agent,
    quantity: float,
    tkn_remove: str
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    new_state.remove_liquidity(
        agent=new_agent,
        quantity=quantity,
        tkn_remove=tkn_remove
    )
    return new_state, new_agent


def simulate_swap(
        old_state: ConstantProductPoolState,
        old_agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0

) -> tuple[ConstantProductPoolState, Agent]:
    new_agent = old_agent.copy()
    new_state = old_state.copy()
    new_state.swap(
        new_agent, tkn_sell, tkn_buy, buy_quantity, sell_quantity
    )
    return new_state, new_agent
# 
# 
# ConstantProductPoolState.swap = staticmethod(simulate_swap)
# ConstantProductPoolState.execute_swap = staticmethod(execute_swap)
# ConstantProductPoolState.execute_add_liquidity = staticmethod(execute_add_liquidity)
# ConstantProductPoolState.add_liquidity = staticmethod(simulate_add_liquidity)
# ConstantProductPoolState.remove_liquidity = staticmethod(simulate_remove_liquidity)
