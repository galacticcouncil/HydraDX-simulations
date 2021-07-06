print("running file: asset_utils.py")
import numpy as np

class V2_Asset(): #args
    """
    Asset class is tracking risk assets in a hydra omipool. 
    This includes the shares and weights of those assets.
    Method for adding new asset yet not in existence in the pool
    """  
    def __init__(self, asset_id, reserve, share, price):
        """
        Asset class initialized with 1 risk asset
        """        
        self.pool = {
                        # 'asset_id' : asset_id,
                        asset_id :{
                        'R': reserve,
                        'S': share,
                        'W': share,
                        'P': price,
                        'dP': 0,   # delta price
                        }}

    def add_new_asset(self, asset_id, reserve, share, price):
        """
        Asset class initialized with 1 risk asset
        """        
        # if price != 'unknown':
        self.pool.update({
                        # 'asset_id' : asset_id,
                        asset_id :{
                        'R': reserve,
                        'S': share,
                        'W': share,
                        'P': price}})
        # else calc price

    def add_liquidity_pool(self, asset_id, delta_R):
        """
        Liquidity added to the pool for one specific risk asset; spec 6-28-21
        - R increased by delta_R
        - S increased to Si * (Ri + delta_R) / Ri
        - no weight changes
        - Y increased acc to spec
        """ 
        
        Si = pool.get_share(asset_id) 
        Ci = pool.get_coefficient(asset_id)
        Ri = self.pool[key]['R']
        Y = prev_state['Y']
        for key in self.pool.keys():
            # print(self.pool.items()) 
            if key == asset_id:
                self.pool[key]['R'] += delta_R
                self.pool[key]['S'] = Si * (Ri + delta_R) / Ri
                Ci_plus = Ci * ((Ri + delta_R) / Ri) ** (a+1)
                #self.pool[key]['W'] += delta_W
                self.pool[key]['Y'] = ((Y ** (-a)) - Ci * (Ri ** (-a)) + Ci_plus * ((Ri + delta_R) ** (-a))) ** (- (1 / a))

    def remove_liquidity_pool(self, asset_id, delta_R):
        """
        Liquidity removed from the pool for one specific risk asset; spec 6-28-21; 
        same as add_liquidity pool; BUT delta_R is still assumed as positive, therefore the sign changes
        - R decreased by delta_R
        - S decreased to Si * (Ri - delta_R) / Ri
        """
        Si = pool.get_share(asset_id) 
        Ci = pool.get_coefficient(asset_id)
        Ri = self.pool[key]['R']
        Y = prev_state['Y']
        for key in self.pool.keys():
            # print(self.pool.items()) 
            if key == asset_id:
                self.pool[key]['R'] -= delta_R
                self.pool[key]['S'] = Si * (Ri - delta_R) / Ri
                Ci_plus = Ci * ((Ri - delta_R) / Ri) ** (a+1)
                #self.pool[key]['W'] += delta_W
                self.pool[key]['Y'] = ((Y ** (-a)) - Ci * (Ri ** (-a)) + Ci_plus * ((Ri - delta_R) ** (-a))) ** (- (1 / a))


    def q_to_r_pool(self, asset_id, delta_R):
        """
        In a 'q to r' swap the update of the pool variable for the traded asset:
        - R decreased by delta_R
        """ 
        for key in self.pool.keys():
            # print(self.pool.items()) 
            if key == asset_id:
                self.pool[key]['R'] -= delta_R

    def r_to_q_pool(self, asset_id, delta_R):
        """
        In a 'r to q' swap the update of the pool variable for the traded asset:
        - R increased by delta_R
        """
        for key in self.pool.keys():
            # print(self.pool.items()) 
            if key == asset_id:
                self.pool[key]['R'] += delta_R

    def get_price(self, asset_id):
        """
        returns price P of one specific asset from the pool variable
        """
        for key in self.pool.keys():
            # print(self.pool.items()) 
            if key == asset_id:
                return(self.pool[key]['P'])

    def get_reserve(self, asset_id):
        """
        returns reserve R of one specific asset from the pool variable
        """
        for key in self.pool.keys():
            # print(self.pool.items()) 
            if key == asset_id:
                return(self.pool[key]['R'])

    def get_share(self, asset_id):
        """
        returns share S of one specific asset from the pool variable
        """
        for key in self.pool.keys():
            # print(self.pool.items()) 
            if key == asset_id:
                return(self.pool[key]['S'])

    def get_weight(self, asset_id):
        """
        returns weight W of one specific asset from the pool variable
        """
        for key in self.pool.keys():
            # print(self.pool.items()) 
            if key == asset_id:
                return(self.pool[key]['W'])

    def get_coefficient(self, asset_id):
        """
        returns coefficient C of one specific asset from the pool variable
        """
        for key in self.pool.keys():
            # print(self.pool.items()) 
            if key == asset_id:
                return(self.pool[key]['C'])

    def update_price(self, Q, Wq):
        """
        updates price P of one specific asset from the pool variable and change in price dP for a Q and Sq according to
        P = (Q/Sq)/(R/S)
        dP = P - P
        """        
        for key in self.pool.keys():
            R = self.pool[key]['R']
            S = self.pool[key]['S']
            P = self.pool[key]['P']
            W = self.pool[key]['W']

            self.pool[key]['P'] = (Q/Wq)/(R/W)
            self.pool[key]['dP'] = self.pool[key]['P'] - P

    def update_price_q_i(self, Ki, Q, Sq):
        """
        updates price according to mechanism specification from 3-3-21
        """  
        for key in self.pool.keys():
            R = self.pool[key]['R']
            S = self.pool[key]['S']
            P = self.pool[key]['P']

        ###############################################################################################
        ########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
            first_term = Ki / R
            second_term_fraction = (Q / Sq) / (R / S)
            price_q_i = first_term - second_term_fraction * np.log(R)
        ########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
        ###############################################################################################

            self.pool[key]['P'] = price_q_i

    def update_price_a(self, a, Q, Wq):
        """
        updates price according to mechanism specification from 4-1-21
        """  
        for key in self.pool.keys():
            R = self.pool[key]['R']
            S = self.pool[key]['S']
            P = self.pool[key]['P']
            W = self.pool[key]['W']

            # spot_price = ((Q / Wq)**a) / (R / W)
      
            # self.pool[key]['P'] = spot_price
            self.pool[key]['P'] = ((Q / Wq)**a) / (R / W)

            # print('POOL Class ', ' A value = ', a, ' Spot Price ',spot_price,'Class Price = ', self.pool['i']['P'])
            # print('POOL Class ', ' A value = ', a, 'Asset ', key, ' Class Price = ', self.pool['i']['P'])
        
        # print('POOL Class ', ' A value = ', a, ' R over W= ', R/W,'Q over Wq = ', Q/Wq)
            
    def update_weight(self):
        """
        updates weight of specific asset from the pool variable according to
        W = S
        """ 
        for key in self.pool.keys():
            S = self.pool[key]['S']
            self.pool[key]['W'] = S

## Balance Asset Share Swap
    def swap_share_pool(self, asset_id, delta_S):
        """
        updates share of specific asset from the pool variable for a swap according to
        S = S + delta_S
        """ 
        for key in self.pool.keys():
            # print(self.pool.items()) 
            if key == asset_id:
                self.pool[key]['S'] += delta_S
    
    def swap_weight_pool(self, asset_id, delta_W):
        """
        updates weight of specific asset from the pool variable for a swap according to
        W = W + delta_W
        """ 
        for key in self.pool.keys():
            # print(self.pool.items()) 
            if key == asset_id:
                self.pool[key]['W'] += delta_W

    def __str__(self):
        """
        Print all attributes of an event
        """
        return str(self.__class__) + ": " + str(self.__dict__)     
print("end of file: asset_utils.py")
