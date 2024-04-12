from hydradx.model.amm.agents import Agent


class OTC:
    def __init__(self, buy_asset, sell_asset, sell_amount, buy_asset_price, partially_fillable=True):
        self.buy_asset = buy_asset
        self.sell_asset = sell_asset
        self.buy_amount = 0
        self.sell_amount = sell_amount
        self.price = buy_asset_price
        self.partially_fillable = partially_fillable

    def __repr__(self):
        return f"OTC({self.buy_asset}, {self.sell_asset}, {self.sell_amount}, {self.price}, {self.partially_fillable})"

    def copy(self):
        return OTC(self.buy_asset, self.sell_asset, self.sell_amount, self.price, self.partially_fillable)

    # numeraire is asset OTC order is selling
    def sell(self, agent: Agent, sell_quantity: float) -> None:
        '''sell_amount is the amount of the buy_asset that the agent is selling for the sell_asset.'''
        assert sell_quantity >= 0
        if sell_quantity * self.price > self.sell_amount:
            raise ValueError("Sell amount exceeds OTC sell amount.")
        self.buy_amount += sell_quantity
        self.sell_amount -= sell_quantity * self.price
        agent.holdings[self.buy_asset] -= sell_quantity
        agent.holdings[self.sell_asset] += sell_quantity * self.price

    def buy(self, agent: Agent, buy_quantity: float) -> None:
        '''buy_amount is the amount of the sell_asset that the agent is buying with the buy_asset.'''
        assert buy_quantity >= 0
        if buy_quantity > self.sell_amount:
            raise
        self.buy_amount += buy_quantity / self.price
        self.sell_amount -= buy_quantity
        agent.holdings[self.buy_asset] -= buy_quantity / self.price
        agent.holdings[self.sell_asset] += buy_quantity

    def validate(self) -> bool:
        if self.buy_amount < 0 or self.sell_amount < 0:
            return False
        if self.price <= 0 or self.price == float('inf'):
            return False
        if self.buy_asset == self.sell_asset:
            return False
        return True
