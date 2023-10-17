import copy
from .agents import Agent
from .amm import AMM


class OrderBook:
    def __init__(self, bids: list[list], asks: list[list[float]]):
        """
        bids and asks are in the form of (price: float, quantity: float) tuples
        could add an ID in there later
        """
        self.bids = bids
        self.asks = asks

    def copy(self):
        return copy.deepcopy(self)


# faster
OrderBook.copy = lambda self: OrderBook(
    bids=[list(i for i in bid) for bid in self.bids],
    asks=[list(i for i in ask) for ask in self.asks],
)


class CentralizedMarket(AMM):
    def __init__(
            self,
            order_book: dict[tuple[str, str], OrderBook],
            asset_list: list[str] = None,
            trade_fee: float = 0
    ):
        super().__init__()
        self.order_book = order_book
        if asset_list:
            self.asset_list = asset_list
        else:
            # using a dict instead of a set here to preserve order (python 3.7+)
            self.asset_list = list({tkn: 0 for pair in self.order_book for tkn in pair}.keys())
        if 'USD' not in self.asset_list:
            self.asset_list = ['USD', *self.asset_list]
        self.trade_fee = trade_fee

    def swap(
        self,
        agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
    ):
        if sell_quantity > agent.holdings[tkn_sell]:
            raise AssertionError('Agent does not have enough holdings to execute trade.')
        if tkn_sell not in self.asset_list or tkn_buy not in self.asset_list:
            raise AssertionError('Asset not found in centralized market.')

        if self.asset_list.index(tkn_sell) < self.asset_list.index(tkn_buy):
            base = tkn_buy
            quote = tkn_sell
        else:
            base = tkn_sell
            quote = tkn_buy

        if (base, quote) not in self.order_book:
            if (quote, base) in self.order_book:
                base, quote = quote, base
            else:
                return self.fail_transaction('Order book not found.')

        # make sure asks are sorted by price ascending and bids are sorted by price descending
        self.order_book[(base, quote)].asks.sort(key=lambda x: x[0])
        self.order_book[(base, quote)].bids.sort(key=lambda x: x[0], reverse=True)

        if sell_quantity > 0:
            if tkn_sell == quote:
                sell_tkns_remaining = sell_quantity
                tkns_bought = 0

                for bid in self.order_book[(base, quote)].bids:
                    if sell_tkns_remaining <= 0:
                        break
                    if bid[0] * bid[1] >= sell_tkns_remaining:
                        # this bid can fill the entire remaining order
                        tkns_bought += sell_tkns_remaining / bid[0]
                        bid[1] -= sell_tkns_remaining / bid[0]
                        sell_tkns_remaining = 0
                    else:
                        # this bid can partially fill the order
                        tkns_bought += bid[1]
                        sell_tkns_remaining -= bid[0] * bid[1]
                        bid[1] = 0

                agent.holdings[tkn_sell] -= sell_quantity - sell_tkns_remaining
                agent.holdings[tkn_buy] += tkns_bought
                self.order_book[(base, quote)].bids = [bid for bid in self.order_book[(base, quote)].bids if bid[1] > 0]
            else:
                sell_tkns_remaining = sell_quantity
                tkns_bought = 0

                for ask in self.order_book[(base, quote)].asks:
                    if sell_tkns_remaining <= 0:
                        break
                    if ask[1] >= sell_tkns_remaining:
                        tkns_bought += ask[0] * sell_tkns_remaining
                        ask[1] -= sell_tkns_remaining
                        sell_tkns_remaining = 0
                    else:
                        tkns_bought += ask[0] * ask[1]
                        sell_tkns_remaining -= ask[1]
                        ask[1] = 0

                agent.holdings[tkn_sell] -= sell_quantity - sell_tkns_remaining
                agent.holdings[tkn_buy] += tkns_bought
                self.order_book[(base, quote)].asks = [ask for ask in self.order_book[(base, quote)].asks if ask[1] > 0]

        elif buy_quantity > 0:
            if tkn_buy == quote:
                buy_tkns_remaining = buy_quantity
                tkns_sold = 0

                for ask in self.order_book[(base, quote)].asks:
                    if buy_tkns_remaining <= 0:
                        break
                    if ask[0] * ask[1] >= buy_tkns_remaining:
                        tkns_sold += buy_tkns_remaining / ask[0]
                        ask[1] -= buy_tkns_remaining / ask[0]
                        buy_tkns_remaining = 0
                    else:
                        tkns_sold += ask[1]
                        buy_tkns_remaining -= ask[0] * ask[1]
                        ask[1] = 0

                agent.holdings[tkn_buy] += buy_quantity - buy_tkns_remaining
                agent.holdings[tkn_sell] -= tkns_sold
                self.order_book[(base, quote)].asks = [ask for ask in self.order_book[(base, quote)].asks if ask[1] > 0]

            else:
                buy_tkns_remaining = buy_quantity
                tkns_sold = 0

                for bid in self.order_book[(base, quote)].bids:
                    if buy_tkns_remaining <= 0:
                        break
                    if bid[1] >= buy_tkns_remaining:
                        tkns_sold += bid[0] * buy_tkns_remaining
                        bid[1] -= buy_tkns_remaining
                        buy_tkns_remaining = 0
                    else:
                        tkns_sold += bid[0] * bid[1]
                        buy_tkns_remaining -= bid[1]
                        bid[1] = 0

                agent.holdings[tkn_buy] += buy_quantity - buy_tkns_remaining
                agent.holdings[tkn_sell] -= tkns_sold
                self.order_book[(base, quote)].bids = [bid for bid in self.order_book[(base, quote)].bids if bid[1] > 0]

        return self

    def fail_transaction(self, error: str, **kwargs):
        self.fail = error
        return self

    def copy(self):
        return copy.deepcopy(self)

    def price(self, tkn: str, numeraire: str = 'USD') -> float:
        if (tkn, numeraire) in self.order_book:
            base = tkn
            quote = numeraire
        elif (numeraire, tkn) in self.order_book:
            base = numeraire
            quote = tkn
        else:
            return 0

        if tkn == base:
            return sorted(self.order_book[(base, quote)].bids, reverse=True)[0][0]
        else:
            return 1 / sorted(self.order_book[(base, quote)].asks)[0][0]


# faster
CentralizedMarket.copy = lambda self: CentralizedMarket(
    asset_list=[tkn for tkn in self.asset_list],
    order_book={pair: book.copy() for pair, book in self.order_book.items()}
)
