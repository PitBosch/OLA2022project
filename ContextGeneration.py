from os import environ
from shutil import which
from Environment import *
from Greedy_optimizer import *
import numpy as np
import pandas as pd
class ContextGeneration():
    #What we need to do is to evaluate every possible partition of the space of the features, 
    #and for every one of these we need to evaluate whether partitioning is better than not doing that
    def __init__(self, env: Environment):
        # Real environment
        self.env=env
        self.confidence=0.95
        #confidence for lower confidence bound
        self.feature=None
        #feature selezionata
        #self.contextvalue=np.array() #un vettore di grandezza 3 0:non splitto 1:splitto feature 1 2:splitto feature 2
        self.pmatrix=np.array([[0,0],[0,0]]) #la prima pmatrix che passo è una matrice di o di zeri o di uno 
        # sample the features 
        #non so se tenerlo qua o passarlo ogni volta ma
        #mi sa meglio passarlo ogni volta

    def run(self, simulinfo, graph_weights, user_index, feat_prob_mat):
        #compute /pass the value from  the environment
        #reward history 
        self.explore()
        group_list=feature_matrix_to_list(self.pmatrix)
        conversion_rates=simulinfo #-> ne ho 1 per ogni gruppo
        alpha_ratios=simulinfo #ne ho 1 per ogni gruppo
        n_prod=simulinfo #ne ho 1 per ogni gruppo
        #feat_prob cosa sono effettivamente?
        generate_feat_prob_matrix(feat_prob)
        #la feature prob matrix la posso ricavare ancora da simulinfo?
        #cosa devo fare con user_index?
        for group in group_list:
            cvarray.append(Greedy_optimizer.run(conversion_rates, alpha_ratios, n_prod, graph_weights, user_index, group_list, feat_prob_mat)['expected_reward'])
            if len(group)>1:

        ###DOMANDA###
        #MA GREEDY OPTIMIZER MI RIESCE A TROVARE GLI EXPECTED REWARDS DEGLI SPLIT? IO HO SOLO I CONV RATE
        
        
        
        return self.pmatrix

    def explore(self, simulinfo, feat_prob_mat,branch=None, depth=0): #DEPTH DIVENTA LIST
        """cvarray: è il vettore dei context value che ci servono quindi quello del branch 0, 
                    del branch 1 e quello non splittato che corrisponde ai reward
         branch: è il branch su cui continuo, devo capire come aggiornare la matrice in base a questa info"""
        group_list=feature_matrix_to_list(self.pmatrix)
        for group in group_list:
            bertot=Greedy_optimizer.run(conversion_rates=simulinfo["conversion_rates"], alpha_ratios=alpha_ratios, n_prod=n_prod, group, feat_prob_mat)['expected_reward']
            #best expected reward for the pmatrix
            #initially group_list=[[00,01,10,11]] so bertot will be of len=1
            I CAN GENERATE THE TWO POSSIBILE MATRICES -> THEN DECIDE 
        if branch is None:
            self.pmatrix[0,:] = max(self.pmatrix)+1
        else:
            self.pmatrix[0,branch] = max(self.pmatrix)+1

        group1=feature_matrix_to_list(self.pmatrix)

        cvarray[0]=Greedy_optimizer.run(conversion_rates, alpha_ratios, n_prod, graph_weights, user_index, group1, feat_prob_mat)['expected_reward']
        if branch is None:
            self.pmatrix[:,0] = max(self.pmatrix)+1
        else:
            self.pmatrix[branch,0] = max(self.pmatrix)+1

       self.feature=self.split(cvarray,bertot)
        if self.feature!=-1 and depth<2: # se anche la profondità è 2 allora finisco
                #pmatrix is a np matrix
                

                newcvarray=np.array(contextvalue(probs,rews), contextvalue(probs,rews), rewtot) #calcolo il nuovo array dei context value
                self.explore( cvarray, 0, depth+1)
                self.explore( cvarray, 1, depth+1) #memorizzo la feature scelta: esempio scelgo la feature 0 giovani/vecchi
        return 1 

    def generatecvarray(onversion_rates, alpha_ratios, n_prod, graph_weights, user_index, group_list, feat_prob_mat):
        for group in group_list:
            bertot=Greedy_optimizer.run(conversion_rates, alpha_ratios, n_prod, graph_weights, user_index, group, feat_prob_mat)['expected_reward'])
            if len(group)>1:
                #devo capire come implementare lo split anche qui
                ber1=
                ber2=
        return np.array(bertot, ber1, ber2)


    def contextvalue(self, probs, rews):
        return np.dot(self.lcb(probs), self.lcb(rews))

    def lcb(self, data): #vale sia per i reward che per le probabilities
        return data.mean()-np.sqrt(np.log(self.confidence)/(2*data.count()))

    def split(cvarray,rewtot): #lista di funzioni su cui posso fare split [0,1] [0] [1]
        #devo gestire la depth = 2
        if (max(cvarray[0],cvarray[1])>=rewtot) :
            if cvarray[0]>cvarray[1]:
                return 0
            else:
                return 1
        return -1

    

        """"    #compute the lower bound of the 7 reward that I need
        rewards=df_rewards
        rew=rewards[:,0].mean()
        rew00=rewards[which(rewards[1]==0),0].mean()))
        rew01=rewards[which(rewards[1]==1),0].mean()-np.sqrt(np.log(self.confidence)/(2*rewards[which(rewards[1]==1),0].count()))
        rew10=rewards[which(rewards[1]==2),0].mean()-np.sqrt(np.log(self.confidence)/(2*rewards[which(rewards[1]==2),0].count()))
        rew11=rewards[which(rewards[1]==3),0].mean()-np.sqrt(np.log(self.confidence)/(2*rewards[which(rewards[1]==3),0].count()))
        f1rew0=rewards[which(rewards[1]==0 or rewards[1]==1),0].mean()-np.sqrt(np.log(self.confidence)/(2*rewards[which(rewards[1]==0 or rewards[1]==1),0].count()))
        f1rew1=rewards[which(rewards[1]==2 or rewards[1]==3),0].mean()-np.sqrt(np.log(self.confidence)/(2*rewards[which(rewards[1]==2 or rewards[1]==3),0].count()))
        f2rew0=rewards[which(rewards[1]==0 or rewards[1]==2),0].mean()-np.sqrt(np.log(self.confidence)/(2*rewards[which(rewards[1]==0 or rewards[1]==2),0].count()))
        f2rew1=rewards[which(rewards[1]==1 or rewards[1]==3),0].mean()-np.sqrt(np.log(self.confidence)/(2*rewards[which(rewards[1]==1 or rewards[1]==3),0].count()))
        #da cambiare la struttura dati
        self.lcb_rewards=np.array(rew00, rew01, rew10, rew11, f1rew0, f1rew1, f2rew0, f2rew1, rew)
                np.dot(lcb_probability-np.sqrt(np.log(self.confidence)/(2*rewards[which(rewards[1]==0),0].count(), lcb_
        #devo calcolare 3 context values 
        if depth==1:
            lcb_rew=self.lcb_rewards
            f1_lcb_reward_aftersplit=np.array(lcb_rew[4],lcb_rew[5])
            f2_lcb_reward_aftersplit=np.array(lcb_rew[6],lcb_rew[7])
            f1_p_split=np.array(0.5,0.5) #da sistemare l'lcb della probabilità 
            f2_p_split=np.array(0.5,0.5)
            return np.array(np.dot(f1_lcb_reward_aftersplit, f1_p_split), np.dot(f2_lcb_reward_aftersplit, f2_p_split))
        if depth==2 

                if splitto==0:
                #split the first feature
                    if branch is None:
                        pmatrix[self.feature,:] = max(pmatrix[self.feature,])+1
                    else:
                        pmatrix[self.feature,branch] = max(pmatrix[self.feature,])+1

                else: 
                    if branch is None:
                        pmatrix[self.feature,:] = max(pmatrix[self.feature,])+1
                    else:
                        pmatrix[self.feature,branch] = max(pmatrix[self.feature,])+1
"""
