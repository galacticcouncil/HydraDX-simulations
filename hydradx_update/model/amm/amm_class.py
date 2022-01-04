import copy
import string

from ..amm import omnipool_amm as oamm

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

