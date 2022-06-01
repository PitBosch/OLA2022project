import Learner
import Environment as env
import UserCat as uc
import Product as pr
import numpy as np


#potrei creare le 3 categorie
#EV=Esperti
#IG=Inesperti Giovani
#IV=Inesperti Vecchi LO mettiamo dopo per chiarezza ora voglio solo provare

users=[]
products=[]

nameofproduct= [ #name of products
    "Calabazas",
    "Hinojo",
    "Sesamo",
    "Girasol",
    "Amapola"
]

prices=[[4., 5, 6 ,7],
    [9., 10, 11, 12],
    [18., 19, 20, 21],
    [24., 25, 26, 27],
    [33., 34, 35, 36]]
#1-2 di delta, Con sovrapposizione

cost=[2,4.5,9,14,17]

#sarebbe interessante anche prendere da file il tutto così da cambiare tutto più facilmente
#calcolo i margini dai cost mi sembra più sensato e anche più veloce se dobbiamo cambiare conitnuamente

cost2=np.tile(np.array([cost]).transpose(), (1, 4))
margins=np.array(prices)-cost2

Secondary_dict={
    "Calabazas": [0,1],
    "Hinojo": [4,3],
    "Sesamo": [0,4],
    "Girasol": [0,2],
    "Amapola": [1,2]
}
res_price_params={
    "shape": 1,
    "scale": 1,
}
probabilities=np.random.uniform(0.0,0.1,(5, 5))
#matrix generata da una uniform / 5 because it is the number of product
alphas=[1/10,1/10,1/10,1/10,1/10,1/2]
#per ora li generiamo così, tutti uguali -> devo generare 3 diversi vettori alpha

poisson_lambda=0.7
#=valore atteso del numero di prodotti acquistati (specifico per prodotto)...non dipende dal
#prodotto oltre che dallo user che dal tipo di user che
p_users=[1/3,1/3,1/3] #probabilità di essere un tipo di utente-> da cambiare

lambda_q=0.5 #just my idea of lambda
#possiamo stimarlo con i dati passati provenienti dal sito -> vino tot è stato comprato 15 volte

#proviamo a pensare, ha senso vederlo come coppia? categoria-prodotto? Avrei 3 categorie *5 prodotti-> 15 lambda diversi




for i in range (5):
    products.append(pr.Product(prices[i], i, nameofproduct[i],margins[i]))



for i in range(3):
    users.append(uc.UserCat(alphas, res_price_params, poisson_lambda, probabilities))

Env = env.Environment(users, products,  lambda_q, Secondary_dict,p_users)

n_users=1000
#supponiamo per il test che la combinazione dei prezzi non cambi mai

price_comb=[4.,9,18,24,33] #questa è la soluzione più semplice.

#idealmente qui ci sarà un for dove simulo tutti i giorni e dove price_comb cambia a ogni giorno in base ai diversi algoritmi)

Env.simulate_day(users_number=n_users,users_probs=p_users, price_combination=price_comb) #also p_users will be discarded

#for i in range(0,5): #questo perchè ho 5 prodotti
 #   pr.Product(prices[i], 1, "Calabazas", margins: [float])


#dobbiamo generare due array user e product


#n_arms dovrebbe essere 4 su 5 diversi MAB il reward è dato dalle
