from asyncio.proactor_events import BaseProactorEventLoop
from msilib.schema import MsiDigitalCertificate
from operator import truediv
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
        self.group_list=[[[0,0],[0,1],[1,0],[1,1]]]
        #feature selezionata
        #self.contextvalue=np.array() #un vettore di grandezza 3 0:non splitto 1:splitto feature 1 2:splitto feature 2
        self.list_matrix=[np.array([[0,0],[0,0]]),np.array([[0,0],[0,0]]),np.array([[0,0],[0,0]])]#la prima pmatrix che passo è una matrice di o di zeri o di uno 
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
    #feat prob max rimane così comìè sempre o cambia?
    def split(self, simulinfo, feat_prob_mat, spfeature_list=[0,1]):
        """"""
        group_list=feature_matrix_to_list(self.list_matrix[2])
        if len(spfeature_list)==2:
            bertot=Greedy_optimizer.run(conversion_rates=simulinfo["conversion_rates"], alpha_ratios=simulinfo["alpha_ratios"], n_prod=simulinfo["n_prod"], group_list=groups, feat_prob_mat=feat_prob_mat)['expected_reward'] #DEPTH DIVENTA LIST
             #bestexpreward without
            
            #generate matrix -> la salvo
            #matrix to list
            if 0 in spfeature_list:
                gr0=feature_matrix_to_list(self.list_matrix[0]) 
                ber00=Greedy_optimizer.run(conversion_rates=simulinfo["conversion_rates"], alpha_ratios=simulinfo["alpha_ratios"], n_prod=simulinfo["n_prod"], group_list=gr0[0], feat_prob_mat=feat_prob_mat)['expected_reward'] #bestexpreward without
                ber01=Greedy_optimizer.run(conversion_rates=simulinfo["conversion_rates"], alpha_ratios=simulinfo["alpha_ratios"], n_prod=simulinfo["n_prod"], group_list=gr0[1], feat_prob_mat=feat_prob_mat)['expected_reward'] #bestexpreward without
                update_split_matrix(0)
            else:
                ber00, ber01=0,0
    
            
            if 1 in spfeature_list:
                gr1=feature_matrix_to_list(self.list_matrix[1])
                ber10=Greedy_optimizer.run(conversion_rates=simulinfo["conversion_rates"], alpha_ratios=simulinfo["alpha_ratios"], n_prod=simulinfo["n_prod"], group_list=gr1[0], feat_prob_mat=feat_prob_mat)['expected_reward'] #bestexpreward without
                ber11=Greedy_optimizer.run(conversion_rates=simulinfo["conversion_rates"], alpha_ratios=simulinfo["alpha_ratios"], n_prod=simulinfo["n_prod"], group_list=gr1[1], feat_prob_mat=feat_prob_mat)['expected_reward'] #bestexpreward without
                update_split_matrix(1)
            else:
                ber00,ber11=0,0

            
            cv1=ber00+ber01
            cv2=ber10+ber11
            feature_splitted=split(cv1,cv2,bertot)



            if feature_splitted==-1:
                return self.list_matrix[2]
            spfeature_list.remove('feature_splitted') #la seconda volta questa deve essere rimossa solo dopo aver fatto entrambi gli split
            self.list_matrix[2]=self.list_matrix[feature_splitted]

            return explore(simulinfo, feat_prob_mat=)
        
        if spfeature_list[0]==0:
            bertot=ber00
            gr=feature_matrix_to_list(self.list_matrix[spfeature_list])
            #generate matrix -> la salvo
            #matrix to list
            ber0=Greedy_optimizer.run(conversion_rates=simulinfo["conversion_rates"], alpha_ratios=simulinfo["alpha_ratios"], n_prod=simulinfo["n_prod"], group_list=gr[0], feat_prob_mat=feat_prob_mat)['expected_reward'] #bestexpreward without
            ber1=Greedy_optimizer.run(conversion_rates=simulinfo["conversion_rates"], alpha_ratios=simulinfo["alpha_ratios"], n_prod=simulinfo["n_prod"], group_list=gr[1], feat_prob_mat=feat_prob_mat)['expected_reward'] #bestexpreward without
            #generate matrix -> la salvo
            update_split_matrix(0)
            cv1=ber0+ber1
            feature_splitted=split(cv1,cv2,bertot)
            if feature_splitted==-1:
                return self.list_matrix[2]
            spfeature_list.remove('feature_splitted')
            self.list_matrix[2]=self.list_matrix[feature_splitted]



        if branch==0:
            #ho deciso su quale feature splittare ora devo selezionare il gruppo giusto
            self.list_matrix[2]
            update_split
        spfeature_list.remove('1')

        if self.feature!=-1 and depth<2: # se anche la profondità è 2 allora finisco
                #pmatrix is a np matrix
                

                newcvarray=np.array(contextvalue(probs,rews), contextvalue(probs,rews), rewtot) #calcolo il nuovo array dei context value
                self.explore( cvarray, 0, depth+1)
                self.explore( cvarray, 1, depth+1) #memorizzo la feature scelta: esempio scelgo la feature 0 giovani/vecchi
        return 1 

    def update_split_matrix(self, feature_to_split):
            if feature_to_split is None:
                self.list_matrix[feature_to_split][0,:] = max(self.list_matrix[feature_to_split])+1
            else:
                self.list_matrix[feature_to_split][0,feature_to_split] = max(self.list_matrix[feature_to_split])+1
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

    def split(cv0, cv1, rewtot): #lista di funzioni su cui posso fare split [0,1] [0] [1]
        #devo gestire la depth = 2
        if (max(cv0,cv1)>=rewtot) :
            if cv0>cv1:
                return 0
            else:
                return 1
        return -1

    def partition(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i : i+size]

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
