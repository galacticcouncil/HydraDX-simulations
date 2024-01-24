from hydradx.model.amm.agents import Agent


class OTC:
    def __init__(self, buy_asset, sell_asset, buy_amount, price, partially_fillable=True):
        self.buy_asset = buy_asset
        self.sell_asset = sell_asset
        self.buy_amount = buy_amount
        self.sell_amount = 0
        self.price = price
        self.partially_fillable = partially_fillable

    def __repr__(self):
        return f"OTC({self.buy_asset}, {self.sell_asset}, {self.buy_amount}, {self.price}, {self.partially_fillable})"

    def copy(self):
        return OTC(self.buy_asset, self.sell_asset, self.buy_amount, self.price, self.partially_fillable)


# numeraire is asset OTC order is selling
def sell_to_otc(otc: OTC, agent: Agent, sell_to_otc_amount: float) -> None:
    if sell_to_otc_amount > otc.buy_amount:
        raise
    otc.buy_amount -= sell_to_otc_amount
    otc.sell_amount += sell_to_otc_amount * otc.price
    agent.holdings[otc.buy_asset] -= sell_to_otc_amount
    agent.holdings[otc.sell_asset] += sell_to_otc_amount * otc.price
