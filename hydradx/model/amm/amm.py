from .agents import Agent
import copy


class AMM:
    unique_id: str
    asset_list: list[str]
    liquidity: dict[str: float]

    def __init__(self):
        self.fail = ''

    def copy(self):
        copy_self = copy.deepcopy(self)
        copy_self.fail = ''
        return copy_self

    def swap(
        self,
        old_agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
    ):
        return self.copy(), old_agent.copy()

    def add_liquidity(
        self,
        old_agent: Agent,
        quantity: float,
        tkn_add: str
    ):
        return self.copy(), old_agent.copy()

    def remove_liquidity(
        self,
        old_agent: Agent,
        quantity: float,
        tkn_remove: str
    ):
        return self.copy(), old_agent.copy()

    def fail_transaction(self, error: str = 'fail'):
        failed_state = self.copy()
        failed_state.fail = error
        return failed_state
