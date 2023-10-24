import copy
from .agents import Agent
from .amm import AMM
import bisect


class SortedList(list):
    def __init__(self, iterable=None, reverse=False):
        super().__init__()
        self.reverse = reverse
        if iterable is not None:
            self.extend(sorted(iterable, reverse=self.reverse))

    def append(self, item):
        if self.reverse:
            index = bisect.bisect_left([x for x in reversed(self)], item)
            super().insert(len(self) - index, item)
        else:
            index = bisect.bisect_right(self, item)
            super().insert(index, item)

    def extend(self, iterable):
        for item in iterable:
            self.append(item)

    def insert(self, index, item):
        self.append(item)

    def remove(self, item):
        index = bisect.bisect_left(self, item) if not self.reverse else bisect.bisect_left([x for x in reversed(self)], item)
        index = len(self) - index - 1 if self.reverse else index
        if index < len(self) and self[index] == item:
            super().pop(index)
        else:
            raise ValueError(f"{item} not in list")

    def pop(self, index=-1):
        if 0 <= index < len(self):
            return super().pop(index)
        else:
            raise IndexError("pop index out of range")

    @property
    def reversed(self):
        return SortedList(self, not self.reverse)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return SortedList(super().__getitem__(key), self.reverse)
        return super().__getitem__(key)


class OrderBook:
    def __init__(self, bids: list[list], asks: list[list[float]]):
        """
        bids and asks are in the form of (price: float, quantity: float) tuples
        could add an ID in there later
        """
        self.bids = SortedList(bids, reverse=True)
        self.asks = SortedList(asks)

    def copy(self):
        return copy.deepcopy(self)


# faster
OrderBook.copy = lambda self: OrderBook(
    bids=[bid.copy() for bid in self.bids],
    asks=[ask.copy() for ask in self.asks],
)


class CentralizedMarket(AMM):
    def __init__(
            self,
            order_book: dict[tuple[str, str], OrderBook],
            asset_list: list[str] = None,
            trade_fee: float = 0
    ):
        """
        This is an 'AMM' even though it's not, because it's a convenient way to
        interface with the rest of the codebase.
        order_book is a dict of (base: str, quote: str) tuples to OrderBook objects.
        The lists in the OrderBook are SortedLists which stay sorted by price
        """
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

        remove_bids = 0
        remove_asks = 0
        if sell_quantity > 0:
            sell_tkns_remaining = sell_quantity
            tkns_bought = 0

            if tkn_sell == base:
                for bid in self.order_book[(base, quote)].bids:
                    if bid[1] >= sell_tkns_remaining:
                        # this bid can fill the entire remaining order
                        tkns_bought += bid[0] * sell_tkns_remaining
                        bid[1] -= sell_tkns_remaining
                        sell_tkns_remaining = 0
                    else:
                        # this bid can partially fill the order
                        tkns_bought += bid[1] * bid[0]
                        sell_tkns_remaining -= bid[1]
                        bid[1] = 0
                    if bid[1] == 0:
                        remove_bids += 1
                    if sell_tkns_remaining <= 0:
                        break
            else:
                for ask in self.order_book[(base, quote)].asks:
                    if ask[0] * ask[1] >= sell_tkns_remaining:
                        tkns_bought += sell_tkns_remaining / ask[0]
                        ask[1] -= sell_tkns_remaining / ask[0]
                        sell_tkns_remaining = 0
                    else:
                        tkns_bought += ask[1]
                        sell_tkns_remaining -= ask[1] * ask[0]
                        ask[1] = 0
                    if ask[1] == 0:
                        remove_asks += 1
                    if sell_tkns_remaining <= 0:
                        break

            agent.holdings[tkn_sell] -= sell_quantity - sell_tkns_remaining
            agent.holdings[tkn_buy] += tkns_bought * (1 - self.trade_fee)

        elif buy_quantity > 0:
            buy_tkns_remaining = buy_quantity
            tkns_sold = 0

            if tkn_buy == base:
                for ask in self.order_book[(base, quote)].asks:
                    if ask[1] >= buy_tkns_remaining:
                        tkns_sold += buy_tkns_remaining * ask[0]
                        ask[1] -= buy_tkns_remaining
                        buy_tkns_remaining = 0
                    else:
                        tkns_sold += ask[0] * ask[1]
                        buy_tkns_remaining -= ask[1]
                        ask[1] = 0
                    if ask[1] == 0:
                        remove_asks += 1
                    if buy_tkns_remaining <= 0:
                        break
            else:
                for bid in self.order_book[(base, quote)].bids:
                    if bid[0] * bid[1] >= buy_tkns_remaining:
                        tkns_sold += buy_tkns_remaining / bid[0]
                        bid[1] -= buy_tkns_remaining / bid[0]
                        buy_tkns_remaining = 0
                    else:
                        tkns_sold += bid[1]
                        buy_tkns_remaining -= bid[0] * bid[1]
                        bid[1] = 0
                    if bid[1] == 0:
                        remove_bids += 1
                    if buy_tkns_remaining <= 0:
                        break

            agent.holdings[tkn_buy] += buy_quantity - buy_tkns_remaining
            agent.holdings[tkn_sell] -= tkns_sold * (1 + self.trade_fee)

        # remove these afterward, so we don't mess up the iteration
        self.order_book[(base, quote)].bids = self.order_book[(base, quote)].bids[remove_bids:]
        self.order_book[(base, quote)].asks = self.order_book[(base, quote)].asks[remove_asks:]

        return self

    def fail_transaction(self, error: str, **kwargs):
        self.fail = error
        return self

    def copy(self):
        return copy.deepcopy(self)

    def buy_spot(self, tkn: str, numeraire: str = 'USD') -> float:
        if tkn == numeraire:
            return 1
        elif (tkn, numeraire) in self.order_book:
            return self.order_book[(tkn, numeraire)].asks[0][0] * (1 + self.trade_fee)
        else:
            return 0

    def sell_spot(self, tkn: str, numeraire: str = 'USD') -> float:
        if tkn == numeraire:
            return 1
        elif (tkn, numeraire) in self.order_book:
            return self.order_book[(tkn, numeraire)].bids[0][0] * (1 - self.trade_fee)
        else:
            return 0


# faster
CentralizedMarket.copy = lambda self: CentralizedMarket(
    asset_list=[tkn for tkn in self.asset_list],
    order_book={pair: book.copy() for pair, book in self.order_book.items()},
    trade_fee=self.trade_fee
)
