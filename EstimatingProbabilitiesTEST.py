import numpy as np
from copy import copy
def simulate_episode(init_prob_matrix, n_steps_max, alpha):
    prob_matrix=init_prob_matrix.copy() #store matrix
    n_nodes=prob_matrix.shape[0] #number
    #only one node is activated at the beginning of the episode
    #how do we choose the distribution? A user has alpha_i probability of seeing the i_th product
    #So I think that what we need is that the initial node is drawn from a discrete distribution
    #the 𝛼 ratios will be realizations of independent Dirichlet random variables.
    #I assume the parameter of the dirichlet to be all ones, since there's no prior assumption
    
    elements_idx = range(0,6)
    param = np.ones(6)
    alpha=np.random.dirichlet(np.ones(6),1)
    #I chose size=1 because we will have 6 alphas for every episode, oteherwise it's possible to generate this alpha
    #before the simulation of the episode
    #np.ones can be changed in order to give more weight to the competitor
   
    initial_active_node_idx=np.random.choice(elements_idx, p=alpha.reshape(-1)) #possiamo sostitirla con una multinomial di size =1
    initial_active_node = np.diag(np.ones(6))[initial_active_node_idx]
    #prendo l'n-esima riga della matrice diagonale, da rivedere la sintassi
    print(initial_active_node)
    history = np.array([initial_active_nodes]) #from a numpy to an array
    active_nodes=initial_active_nodes
    newly_active_nodes=active_nodes
    #I think that these 3 lines can be left the same
    t=0
    #
    while(t<n_steps_max and np.sum(newly_active_nodes)>0): #while finisce quando 
        p=(prob_matrix.T*active_nodes).T
        activated_edges=p>np.random.rand(p.shape[0], p.shape[1])
        prob_matrix=prob_matrix*((p!=0)==activated_edges)
        newly_active_nodes=(np.sum(activated_edges,axis=0)>0)*(1-active_nodes)
        active_nodes=np.array(active_nodes+newly_active_nodes)
        history = np.concatenate((history, [newly_active_nodes]), axis=0)
        t+=1
    return history

#credit assignment APPROACH
#I think that the whole function can be left without any

def estimate_probabilities(dataset, node_index, n_nodes):
    estimated_prob= np.ones(n_nodes)*1.0/(n_nodes-1)
    credits=np.zeros(n_nodes)
    occurr_v_active = np.zeros(n_nodes)
    n_episodes=len(dataset)
    for episode in dataset:
        idx_w_active=np.argwhere(episode[:,node_index]==1).reshape(-1)
        if len(idx_w_active)>0 and idx_w_active>0:
            active_nodes_in_prev_step=episode[idx_w_active-1,:].reshape(-1)
            credits+=active_nodes_in_prev_step/np.sum(active_nodes_in_prev_step)
        for v in range(0,n_nodes):
            if (v!=node_index):
                idx_v_active = np.argwhere (episode[:, v]==1).reshape(-1)
                if len(idx_v_active)>0 and (idx_v_active<idx_w_active or len(idx_w_active)==0):
                    occurr_v_active[v]+=1
    estimated_prob=credits/occurr_v_active
    estimated_prob=np.nan_to_num(estimated_prob)
    return estimated_prob

n_nodes= 5 #5 nodes
n_episodes = 1000 #this is also the number of clients
prob_matrix = np.random.uniform(0.0,0.1,(n_nodes, n_nodes)) #quale è la nostra probability matrix
node_index=4
dataset = []

for e in range (0, n_episodes):
    dataset.append(simulate_episode(init_prob_matrix=prob_matrix, n_steps_max=10))

estimated_prob= estimate_probabilities(dataset=dataset, node_index=node_index, n_nodes=n_nodes)

print("True P Matrix: ", prob_matrix[:,4])
print("Estimated P Matrix: ", estimated_prob)
