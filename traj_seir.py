import numpy as np
import random
import time 

# parameters config 
Track_open = False   
SIGMA = 0.2
GAMMA = 0.1
InitialE0 = 0
InitialI0 = 10
InitialR0= 0
DetectionProb = 0.879

##### simulation code ########
def epidemic_step(S0, E0, I0, R0, beta, anchor=2, gamma=GAMMA, sigma=SIGMA, interval=1.0/288):
    N = S0 + E0 + I0 + R0
    slot = 1/anchor  
    for i in range(anchor):   
        S = S0 + (-beta*I0*S0/N)*interval*slot
        E = E0 + (beta*I0*S0/N  - sigma*E0)*interval*slot
        I = I0 + (sigma*E0 - gamma*I0)*interval*slot
        R = R0 + (gamma * I0)*interval*slot
        N = S + E + I + R
        S0, E0, I0, R0 = S, E, I, R
    return S, E, I, R


def gird_epidemic_simualtion(grid_state,beta,state_matrix):
    
    # DO EPIDEMIC SIMULATION ON LOCAL GRID 
    S0,  E0, I0, R0 = len(grid_state[0]), len(grid_state[1]), len(grid_state[2]), len(grid_state[3])
    if  I0 + E0 ==0 :
        return 0,0,0, []
    S, E, I, R = epidemic_step(S0, E0, I0, R0, beta)
    Rinc = R - R0 
    Iinc = I - I0 + Rinc
    Einc = E - E0 + Iinc
    nR = min(I0, int(np.floor(Rinc) + (random.random() < (Rinc % 1)))) if Rinc > 0 else 0
    nI = min(E0,int(np.floor(Iinc) + (random.random() < (Iinc % 1)))) if Iinc > 0 else 0
    nE = min(S0,int(np.floor(Einc) + (random.random() < (Einc % 1)))) if Einc > 0 else 0
    who_e = []
    if nE:
        who_e = random.sample(grid_state[0], nE)
        state_matrix[who_e] = 1
    if nI: 
        who_i = random.sample(grid_state[1], nI)
        state_matrix[who_i] = 2
    if nR: 
        who_r = random.sample(grid_state[2], nR)
        state_matrix[who_r] = 3

    return nE, nI, nR, who_e

def global_epidemic(hids, hid2beta, hids_state,state_matrix,detection_hids, if_detection, infection_track, t):
    dE,dI,dR = 0,0,0
    # Spreading first 
    for hid in hids: 
        nE, nI, nR, who_e = gird_epidemic_simualtion(hids_state[hid],hid2beta[hid][t],state_matrix)
        if Track_open and nE!=0:
            for uid in who_e:
                # Record time, place, infected person, list of possible sources of infection
                infection_track.append({"uid": uid, "time":t,"hid":hid,"source":hids_state[hid][2]}) 
        dE += nE
        dI += nI
        dR += nR
    # Detection Second 
    if if_detection != True:
        return dE,dI,dR
    p_detect = DetectionProb
    for hid in detection_hids:
        i_set = hids_state[hid][2]
        detected = [user for user in i_set if random.uniform(0, 1) < p_detect]
        state_matrix[detected] = 4
    return dE,dI,dR


# Entry functions
def epimob_simulation(hid2beta,traj_data,hids,start_dt_idx,end_dt_idx,dt_hids):
    '''
    :hid2beta: mapping hid to spatial-temporal vary beta.
               using hid2beta[hid][t] to get the beta at grid g at time slot t.
    :traj_data: the grid-mapped trajectory data used for simulation. 
                the shape is (p_count, T).
                T indicates the total number of time slot and p_count indicates the number of individuals.
                using traj_data[uid][t] to get the position of individual-uid at time interval t.
    :hids: all the grids that appeared in the traj_data            
    :start_dt_idx: the start time slot of the detection/screening
    :end_dt_idx: the end time slot of of the detection/screening
    :dt_hids: the targeted grids for screening.
    '''
    st = time.time()
    ################## Simulation Initialization ######################
    uids = list(range(len(traj_data)))  # Generate a list of user ids  
    T = len(traj_data[0])
    # Initialize the user health status list, 0 for S, 1 for E, 2 for I, 3 for R
    state_matrix = np.array([ 0 for uid in uids]) 
    # Randomly select E0, I0, RO  people to be exposed, infected and recovered, respectively.
    E0,I0,R0= InitialE0,InitialI0,InitialR0  
    infected_ids = random.sample(uids, I0)  
    state_matrix[infected_ids] = 2 
    # Initialize the state information of each grid 
    hids_state = { hid: [[],[],[],[]] for hid in hids} 
    # Results Initialization
    simulation_res = dict({})
    simulation_res["curve"] = []
    infection_track = [{"time":0,"hid":traj_data[uid][0],"uid":uid, "source":[]} for uid in infected_ids] 
    e_count,i_count,r_count = E0,I0,R0


    ################### Simualtion #######################
    for t in range(T):
        if_detection = False
        if t >= start_dt_idx and t <= end_dt_idx:
            if_detection = True 
        # Initialize the grid information and calculate the id information of each category of people in each grid at time t
        for hid in hids:
            hids_state[hid]=[[],[],[],[]]
        for uid in uids:
            this_state = state_matrix[uid]
            if this_state == 4: # 4 represents the person who has been isolated due to screening
                continue
            grid_idx = traj_data[uid][t]
            hids_state[grid_idx][this_state].append(uid)
        # Simulate the propagation at time slot t and return the change in the number of E,I,R
        dE,dI,dR = global_epidemic(hids,hid2beta,hids_state,state_matrix,dt_hids,if_detection, infection_track, t)
        # Update the number of E, I, R and store them to the curve.
        e_count = e_count + dE - dI
        i_count = i_count + dI - dR
        r_count = r_count + dR 
        print(t, ">>>", e_count, i_count, r_count)
        simulation_res["curve"].append([e_count,i_count,r_count])

    ################### Result saving ######################
    et = time.time()
    simulation_res["time_cost"] = et-st
    simulation_res["state_matrix"] = state_matrix
    simulation_res['infection_track'] = infection_track
    print("total infection", simulation_res["curve"][-1])

    return simulation_res
