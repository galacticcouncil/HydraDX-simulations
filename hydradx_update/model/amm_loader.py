from .amm import amm
from .amm import amm_class



# Initialize AMMS
def init_amms(params, step, history, prev_state, policy_input):
    '''
    Import modules List of AMMS within 1 configuration
    Import particular module for 1 active AMM per configuration
    '''
    # List of AMMs within 1 configuration
    # for a in params['amm']:
    #     globals()[a] = a
    #     from .amm import a
    #     a.AMM_param_test(a)
        
    # particular module for 1 active AMM per configuration
        
    # exec("%s = %d" % (params['amm'],amm_A))
    
    # globals()[params['amm']] = amm_A
    # # from .amm import a
    # a.AMM_param_test(a)
        
    return 'AMM_list', params['amm']

def amm_load_parent(params, step, history, prev_state, policy_input):
    '''
    Load parent class
    '''
       
    return 'AMM_parent', amm_class.AMM()

def amm_load_child(params, step, history, prev_state, policy_input):
    '''
    Load child class
    '''
    method_to_call = getattr(amm_class, params['amm']) 
    print(method_to_call)  
    return 'AMM_child', method_to_call()
