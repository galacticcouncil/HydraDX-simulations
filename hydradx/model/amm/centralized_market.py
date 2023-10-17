import copy
from agents import Agent
from amm import AMM


class OrderBook:
    def __init__(self, bids: list[tuple], asks: list[tuple]):
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
    bids=[tuple(i for i in bid) for bid in self.bids],
    asks=[tuple(i for i in ask) for ask in self.asks],
)


class CentralizedMarket(AMM):
    def __init__(self, order_book: dict[tuple[str, str], OrderBook], asset_list: list[str] = None):
        super().__init__()
        self.order_book = order_book
        if asset_list:
            self.asset_list = asset_list
        else:
            # using a dict instead of a set here to preserve order (python 3.7+)
            self.asset_list = list({tkn: 0 for pair in self.order_book for tkn in pair}.keys())
        if 'USD' not in self.asset_list:
            self.asset_list = ['USD', *self.asset_list]

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
            quote = tkn_sell
            base = tkn_buy
        else:
            quote = tkn_buy
            base = tkn_sell

        if (base, quote) not in self.order_book:
            return self.fail_transaction('Order book not found.')

        # make sure bids are sorted by price ascending and asks are sorted by price descending
        self.order_book[(base, quote)].bids.sort(key=lambda x: x[0])
        self.order_book[(base, quote)].asks.sort(key=lambda x: x[0], reverse=True)

        if sell_quantity > 0:
            if tkn_sell == base:
                orders = self.order_book[(base, quote)].bids
            else:
                orders = self.order_book[(base, quote)].asks

            sell_tkns_remaining = sell_quantity
            tkns_bought = 0

            for bid in orders:
                if sell_tkns_remaining <= 0:
                    break
                if bid[1] >= sell_tkns_remaining:
                    # this bid can fill the entire remaining order
                    tkns_bought += bid[0] * sell_tkns_remaining
                    bid[1] -= sell_tkns_remaining
                    sell_tkns_remaining = 0
                else:
                    # this bid can partially fill the order
                    tkns_bought += bid[0] * bid[1]
                    sell_tkns_remaining -= bid[1]
                    orders.remove(bid)

            agent.holdings[tkn_sell] -= sell_quantity - sell_tkns_remaining
            agent.holdings[tkn_buy] += tkns_bought

        elif buy_quantity > 0:
            if tkn_buy == base:
                orders = self.order_book[(base, quote)].asks
            else:
                orders = self.order_book[(base, quote)].bids

            buy_tkns_remaining = buy_quantity
            tkns_sold = 0

            for ask in orders:
                if buy_tkns_remaining <= 0:
                    break
                if ask[1] >= buy_tkns_remaining:
                    # this ask can fill the entire remaining order
                    tkns_sold += ask[0] * buy_tkns_remaining
                    ask[1] -= buy_tkns_remaining
                    buy_tkns_remaining = 0
                else:
                    # this ask can partially fill the order
                    tkns_sold += ask[0] * ask[1]
                    buy_tkns_remaining -= ask[1]
                    orders.remove(ask)

            agent.holdings[tkn_buy] -= buy_quantity - buy_tkns_remaining
            agent.holdings[tkn_sell] += tkns_sold

        return self

    def fail_transaction(self, error: str, **kwargs):
        self.fail = error
        return self

    def copy(self):
        return copy.deepcopy(self)

    def price(self, tkn: str, numeraire: str = 'USD') -> float:
        if self.asset_list.index(tkn) > self.asset_list.index(numeraire):
            if (numeraire, tkn) in self.order_book:
                # return the lowest available bid
                return list(sorted(filter(
                    lambda bid: bid[0] == numeraire and bid[1] == tkn, self.order_book[(numeraire, tkn)].bids
                ), key=lambda bid: bid[1]))[0][1]

            else:
                return 0
        else:
            if (tkn, numeraire) in self.order_book:
                # return the highest available ask
                return list(sorted(filter(
                    lambda ask: ask[0] == tkn and ask[1] == numeraire, self.order_book[(tkn, numeraire)].asks
                ), key=lambda ask: ask[1], reverse=True))[0][1]

            else:
                return 0


# faster
CentralizedMarket.copy = lambda self: CentralizedMarket(
    asset_list=[tkn for tkn in self.asset_list],
    order_book={pair: book.copy() for pair, book in self.order_book.items()}
)
