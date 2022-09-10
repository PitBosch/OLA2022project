from curses import KEY_A1
from importlib.util import spec_from_file_location
from Environment import *
from Greedy_optimizer import *
import numpy as np
from Step7_TS import *
class ContextGeneration():
    #What we need to do is to evaluate every possible partition of the space of the features, 
    #and for every one of these we need to evaluate whether partitioning is better than not doing that
    def __init__(self, env: Environment):
        self.env=env
        # Real environment
        self.confidence=0.95
        #confidence for lower confidence bound
        self.bertot=0
        #best expected reward initialized to 0 but that has to be fixed
        self.list_matrix=[np.array([[0,0],[0,0]]),np.array([[0,0],[0,0]]),np.array([[0,0],[0,0]])]#la prima pmatrix che passo è una matrice di o di zeri o di uno 
        #matrix that helps us saving our 

    def run(self, simulinfo):
        group_list=feature_matrix_to_list(self.list_matrix[2])
        n_userstot=0
        print("miao")
        #for keys in simulinfo["n_users"].keys():
       #     n_userstot+=simulinfo["n_users"]
        feat_prob=np.array([0.25,0.25])
        feat_prob_mat=generate_feat_prob_matrix(feat_prob) #NON HO CAPITO STA LINEA
        info=self.get_group_k_info(simulinfo, self.list_matrix[2],0)
        #ho messo user index 01 poi ci penso
        #self.bertot=Greedy_optimizer(self.env).run(conversion_rates=info[0][0], alphas_ratio=info[1][0], n_prod=info[2][0], user_index=1, group_list=group_list[0], feat_prob_mat=feat_prob_mat)['expected_reward']
        CR_list = [info[0]]*1
        alpha_list = [info[1]]*1 
        n_prod_list = [info[2]]*1 
        self.bertot=Greedy_optimizer(self.env).run(conversion_rates=CR_list, alphas_ratio=alpha_list, n_prod=n_prod_list, group_list=group_list[0], feat_prob_mat=feat_prob_mat)['expected_reward']
        #Find the best expected reward without split
        spfeature_list=[0,1]
        #initialize the list of features that have to be splitted
        lista=self.split0(simulinfo,feat_prob_mat)
        #split the first big group into two chunks
        fsp=lista[2]
        #the splitted feature is equal to this result
        if fsp==-1:
        #If the split is not made return the matrix
            return self.list_matrix[2]
        spfeature_list.remove(fsp)
        #Remove from the feature list the feature on which we have splitted
        self.split1(simulinfo, feat_prob_mat, spfeature_list,0, lista[0])
        #split on the first branch
        self.split1(simulinfo, feat_prob_mat, spfeature_list,1, lista[1])
        #split on the second branch
        return self.list_matrix[2]
        #return the final matrix

    def split0(self, simulinfo, feat_prob_mat):
        """function that split the tree in the first step"""
        p=feat_prob_mat
        self.update_split_matrix(0)
        gr0=feature_matrix_to_list(self.list_matrix[0])
        info=[self.get_group_k_info(simulinfo, 0),self.get_group_k_info(simulinfo, 1)]
        #not sure about 0 for every feature splitted
        #QUI INFO è SEMPRE SBAGLIATO PERCHè DEVO PASSARE UNA LISSTA
        ber00=Greedy_optimizer.run(conversion_rates=info[0][0], alpha_ratios=info[0][1], n_prod=info[0][2], group_list=gr0[0], feat_prob_mat=feat_prob_mat)['expected_reward'] #bestexpreward without
        ber01=Greedy_optimizer.run(conversion_rates=info[1][0], alpha_ratios=info[1][1], n_prod=info[1][2], group_list=gr0[1], feat_prob_mat=feat_prob_mat)['expected_reward'] #bestexpreward without
        self.update_split_matrix(1)
        info1=self.get_group_k_info(simulinfo,1)
        gr1=feature_matrix_to_list(self.list_matrix[1])
        ber10=Greedy_optimizer.run(conversion_rates=info1[0][0], alpha_ratios=info1[0][1], n_prod=info1[0][2], group_list=gr1[0], feat_prob_mat=feat_prob_mat)['expected_reward'] #bestexpreward without
        ber11=Greedy_optimizer.run(conversion_rates=info1[1][0], alpha_ratios=info1[1][1], n_prod=info1[1][2],  group_list=gr1[1], feat_prob_mat=feat_prob_mat)['expected_reward'] #bestexpreward without
        #run the greedy optimizer in order to find the ber
        cv1=self.contextvalue(np.array(ber00,ber01), np.array(p[0,0],p[0,1]))
        #calcolo context value feature 1 -> posso farla diventare una funzione prendendo anche quello sopra
        cv2=self.contextvalue(np.array(ber10,ber11), np.array(p[1,0],p[1,1]))
        #calcolo context value feature 2
        feature_splitted=self.compare(cv1,cv2,self.bertot)
        #compara i 3 context value
        if feature_splitted!=-1:
            self.list_matrix[0],self.list_matrix[1],self.list_matrix[2]=self.list_matrix[feature_splitted]
        #if there's no split update all the matrix in the matrix list
            if feature_splitted == 0:
                return [ber00, ber01, feature_splitted]
            elif feature_splitted==1:
                return [ber10, ber11, feature_splitted]
        return [0,0,feature_splitted] 
        #restituisco su quale feature ho splittato e il nuovo bertot, sulla matrice ho già fatto l'update


    def split1(self, simulinfo, feat_prob, spfeature_list,branch, ber):
        """function that split the tree in the second step"""
        
        #convert the matrix to group_list
        #
        #select only the groups that have more than 1 couples
        self.update_split_matrix(spfeature_list[0], branch) #splitto sulla feature che mi manca sul branch 0
        #CONTROLLA CON CARTA E PENNA BENE SE FUNZIONA L'UPDATE
        feat_prob_mat=generate_feat_prob_matrix(feat_prob)
        group_list=feature_matrix_to_list(self.list_matrix[spfeature_list[0]])  
        #convert the matrix to a group list
        gr1=list(filter(lambda x: len(x) == 1, group_list))
        #get only the groups that have length one (this works only for the second step not for the third)
        gr=list(filter(lambda x: x[abs(1-spfeature_list[0])]==branch, gr1))
        #get only the groups that have in the position of the feature splitted the actual branch
        #qua basta convertire le couple in stringhe ed è fatta
        key0=str(gr[0])
        key1=str(gr[1])
        ber0=Greedy_optimizer.run(conversion_rates=simulinfo["conversion_rates"][key0], alpha_ratios=simulinfo["alpha_ratios"][key0], n_prod=simulinfo["n_prod"][key0], group_list=gr[0], feat_prob_mat=feat_prob_mat)['expected_reward'] #bestexpreward without
        ber1=Greedy_optimizer.run(conversion_rates=simulinfo["conversion_rates"][key1], alpha_ratios=simulinfo["alpha_ratios"][key1], n_prod=simulinfo["n_prod"][key1], group_list=gr[1], feat_prob_mat=feat_prob_mat)['expected_reward'] #bestexpreward without
        #run the greedy optimizer to find the best expected reward
        p0=0.25
        p1=0.25

        cv1=self.contextvalue(np.array(ber0,ber1), np.array(p0,p1))
        #The context value of the first feature is computed as the sum of the other two
        feature_splitted=self.compare(cv1,0,ber)
        if feature_splitted!=-1:
            self.list_matrix[2]=self.list_matrix[feature_splitted]
        return feature_splitted #restituisco su quale feature ho splittato e il nuovo bertot, sulla matrice ho già fatto l'update
        

#DOMANDE: LOW CONFIDENCE BOUND AND PROBABILITIES

    def update_split_matrix(self, feature_to_split, branch=0):
            if feature_to_split is None:
                self.list_matrix[feature_to_split][branch,:] = max(self.list_matrix[feature_to_split])+1
            else:
                self.list_matrix[feature_to_split][branch,feature_to_split] = max(self.list_matrix[feature_to_split])+1
            return 1

    def contextvalue(self, probs, rews):
        return np.dot(self.lcb(probs), self.lcb(rews))

    def lcb(self, data): #vale sia per i reward che per le probabilities
        return data.mean()-np.sqrt(np.log(self.confidence)/(2*data.count()))

    def compare(cv0, cv1, rewtot): #lista di funzioni su cui posso fare split [0,1] [0] [1]
        #devo gestire la depth = 2
        if (max(cv0,cv1)>=rewtot) :
            if cv0>cv1:
                return 0
            else:
                return 1
        return -1

    def get_group_k_info(self, simul_history, matrix, k):
        a = np.ones((5,4))
        b = np.ones((5,4))
        #initialize a, b to a 5x4
        beta_alpha = np.ones((2,5))
        #initialize beta_alpha to a matrix 2x5 of ones
        n_prod_data = np.ones((2,5))
        #initialize n_prod_data to a matrix 2x5 of ones
        i_list,j_list = np.where(matrix == k)
        #find the i and j of the couple that belongs to the k-th group
        for k in range(len(i_list)):
            #loop through the list
            feat1 = i_list[k]
            feat2 = j_list[k]
            feat_key = str(feat1)+str(feat2)
            a += simul_history[feat_key]['CR_bought']
            b += simul_history[feat_key]['CR_seen'] - simul_history[feat_key]['CR_bought']
            beta_alpha += simul_history[feat_key]['initial_prod']
            n_prod_data += simul_history[feat_key]['n_prod_sold']
        beta_CR = [a, b]
        
        return [beta_CR, beta_alpha, n_prod_data]
    

   