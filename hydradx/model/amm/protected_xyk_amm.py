import math
from .amm import AMM, FeeMechanism, basic_fee
from .agents import Agent


class ProtectedXYKState(AMM):
    def __init__(
            self,
            stable_asset: str,
            stable_asset_quantity: float,
            stable_asset_virtual_quantity: float,
            volatile_asset: str,
            volatile_asset_quantity: float,
            break_price: float,
            trade_fee: float = 0,
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
        self.trade_fee: FeeMechanism = basic_fee(trade_fee).assign(self)

        self.asset_list = [stable_asset, volatile_asset]
        self.liquidity = [stable_asset_quantity, volatile_asset_quantity]
        self.a = stable_asset_virtual_quantity
        self.p = break_price

        self.shares = stable_asset_quantity

        self.unique_id = unique_id

    def get_stable_asset(self):
        return self.asset_list[0]

    def get_volatile_asset(self):
        return self.asset_list[1]

    def get_stable_asset_quantity(self):
        return self.liquidity[0]

    def get_volatile_asset_quantity(self):
        return self.liquidity[1]

    def get_invariant_one(self):
        return (self.liquidity[0] + self.a) * self.liquidity[1]

    def get_invariant_two(self):
        return self.liquidity[0] * (self.liquidity[1] - self.a / self.p)

    def price_one(self):
        return (self.liquidity[0] + self.a) / self.liquidity[1]

    def price_two(self):
        return self.liquidity[0] / (self.liquidity[1] - self.a / self.p)

    def price(self, tkn: str, denomination: str = '') -> float:
        """
        Calculate the spot price of the volatile asset denominated in the stable asset.
        """
        spot_price = self.price_one() if self.price_one() >= self.p else self.price_two()
        if denomination == self.asset_list[0]:  # stable asset
            return spot_price
        else:  # volatile asset
            return 1 / spot_price


def price(state: ProtectedXYKState):
    """
    Calculate the spot price of the volatile asset denominated in the stable asset.
    """
    return state.price(state.asset_list[1], state.asset_list[0])


def calculate_reserve_at_intersection(
        state: ProtectedXYKState,
        i: int
):
    """
    Calculate the reserve at a given price.
    :param state: the state of the pool
    :param price: the price, with volatile asset denominated in stable asset
    :param i: the index of the asset to calculate the reserve for
    :return: the reserve
    """

    # calculate spot price
    spot_price = price(state)

    if spot_price == state.p:
        return state.liquidity[i]
    elif spot_price > state.p:  # spot price > price
        new_y = math.sqrt(state.p * state.liquidity[1] * (state.liquidity[0] + state.a)) - state.a
    else:  # spot price < price
        new_y = math.sqrt((state.p * state.liquidity[1] - state.a) * state.liquidity[0])

    if i == 0:
        return new_y
    else:
        return (new_y + state.a) / state.p

# def execute_swap(
#         state: ProtectedXYKState,
#         agent: Agent,
#         tkn_sell: str,
#         tkn_buy: str,
#         buy_quantity: float = 0,
#         sell_quantity: float = 0
# ):
#     if not (tkn_buy in state.asset_list and tkn_sell in state.asset_list):
#         return state.fail_transaction('Invalid token name.', agent)
#
#     # turn a negative buy into a sell and vice versa
#     if buy_quantity < 0:
#         sell_quantity = -buy_quantity
#         buy_quantity = 0
#         t = tkn_sell
#         tkn_sell = tkn_buy
#         tkn_buy = t
#     elif sell_quantity < 0:
#         buy_quantity = -sell_quantity
#         sell_quantity = 0
#         t = tkn_sell
#         tkn_sell = tkn_buy
#         tkn_buy = t
#
#     if sell_quantity != 0:
#         # when amount to be paid in is specified, calculate payout
#         buy_quantity = sell_quantity * state.liquidity[tkn_buy] / (
#                 state.liquidity[tkn_sell] + sell_quantity)
#         if math.isnan(buy_quantity):
#             buy_quantity = sell_quantity  # this allows infinite liquidity for testing
#         trade_fee = state.trade_fee.compute(tkn=tkn_sell, delta_tkn=sell_quantity)
#         buy_quantity *= 1 - trade_fee
#
#     elif buy_quantity != 0:
#         # calculate input price from a given payout
#         sell_quantity = buy_quantity * state.liquidity[tkn_sell] / (state.liquidity[tkn_buy] - buy_quantity)
#         if math.isnan(sell_quantity):
#             sell_quantity = buy_quantity  # this allows infinite liquidity for testing
#         trade_fee = state.trade_fee.compute(tkn=tkn_sell, delta_tkn=sell_quantity)
#         sell_quantity /= 1 - trade_fee
#
#     else:
#         return state.fail_transaction('Must specify buy quantity or sell quantity.', agent)
#
#     if state.liquidity[tkn_sell] + sell_quantity <= 0 or state.liquidity[tkn_buy] - buy_quantity <= 0:
#         return state.fail_transaction('Not enough liquidity in the pool.', agent)
#
#     if agent.holdings[tkn_sell] - sell_quantity < 0 or agent.holdings[tkn_buy] + buy_quantity < 0:
#         return state.fail_transaction('Agent has insufficient holdings.', agent)
#
#     agent.holdings[tkn_buy] += buy_quantity
#     agent.holdings[tkn_sell] -= sell_quantity
#     state.liquidity[tkn_sell] += sell_quantity
#     state.liquidity[tkn_buy] -= buy_quantity
#
#     return state, agent
#
#
# ProtectedXYKPoolState.execute_swap = staticmethod(execute_swap)
