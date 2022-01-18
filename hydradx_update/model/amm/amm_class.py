import copy
import string

from ..amm import omnipool_amm as oamm
from ..amm.omnipool_amm import swap_hdx_delta_Qi, swap_hdx_delta_Ri, swap_hdx, swap_assets


class AMM(): #args Event
    """
    AMM class parent
    """          
    def __init__ (self):
        """
        Initiate AMM Class instance
        """       

        self.state = {

                }
        
class OAMM(AMM): #args Event
    """
    AMM class parent
    """          
    def __init__ (self):
        """
        Initiate OAMM Child Class instance
        """       

        self.state = {
            super().__init__()
                }
        
    def initialize_shares(self, token_counts, init_d=None, agent_d=None) -> dict:
        if agent_d is None:
            agent_d = {}
        if init_d is None:
            init_d = {}

        n = len(token_counts['R'])
        state = copy.deepcopy(token_counts)
        state['S'] = copy.deepcopy(state['R'])
        state['A'] = [0]*n

        agent_shares = [sum([agent_d[agent_id]['s'][i] for agent_id in agent_d]) for i in range(n)]
        state['B'] = [state['S'][i] - agent_shares[i] for i in range(n)]

        state['D'] = 0
        state['T'] = init_d['T'] if 'T' in init_d else None
        state['H'] = init_d['H'] if 'H' in init_d else None

        return state
    
    def swap_hdx(self,
        old_state: dict,
        old_agents: dict,
        trader_id: string,
        delta_R: float,
        delta_Q: float,
        i: int,
        fee: float = 0
                        ) -> tuple:
        """Compute new state after HDX swap"""

        new_state = copy.deepcopy(old_state)
        new_agents = copy.deepcopy(old_agents)

        if delta_Q == 0 and delta_R != 0:
            delta_Q = swap_hdx_delta_Qi(old_state, delta_R, i)
        elif delta_R == 0 and delta_Q != 0:
            delta_R = swap_hdx_delta_Ri(old_state, delta_Q, i)
        else:
            return new_state, new_agents

        # Token amounts update
        # Fee is taken from the "out" leg
        if delta_Q < 0:
            new_state['R'][i] += delta_R
            new_state['Q'][i] += delta_Q
            new_agents[trader_id]['r'][i] -= delta_R
            new_agents[trader_id]['q'] -= delta_Q * (1 - fee)

            new_state['D'] -= delta_Q * fee
        else:
            new_state['R'][i] += delta_R
            new_state['Q'][i] += delta_Q
            new_agents[trader_id]['r'][i] -= delta_R * (1 - fee)
            new_agents[trader_id]['q'] -= delta_Q

            # distribute fees
            new_state['A'][i] -= delta_R * fee * (new_state['B'][i] / new_state['S'][i])
            for agent_id in new_agents:
                agent = new_agents[agent_id]
                agent['r'][i] -= delta_R * fee * (agent['s'][i] / new_state['S'][i])

        return new_state, new_agents
    
    
    def swap_assets(self,
        old_state: dict,
        old_agents: dict,
        trader_id: string,
        delta_sell: float,
        i_buy: int,
        i_sell: int,
        fee_assets: float = 0,
        fee_HDX: float = 0
        ) -> tuple:
        # swap asset in for HDX
        first_state, first_agents = swap_hdx(old_state, old_agents, trader_id, delta_sell, 0, i_sell, fee_HDX)
        # swap HDX in for asset
        delta_Q = first_agents[trader_id]['q'] - old_agents[trader_id]['q']
        return swap_hdx(first_state, first_agents, trader_id, 0, delta_Q, i_buy, fee_assets)
        
class PAMM(AMM): #args Event
    """
    AMM class parent
    """          
    def __init__ (self):
        """
        Initiate OAMM Child Class instance
        """       

        self.state = {
            super().__init__()
                }

