# OLA2022project
 Project of the course Data Intelligence Applications with Professor Nicola Gatti in Politecnico di Milano A.Y. 2021/2022

### Results for the two cases of fully connected and not fully connected graph weights are presented repspectively in:
- Fully_Connected.ipynb
- NOT_Fully_Con.ipynb

#### Classes organization:
- Basic Classes to build the environment:
 * Product : store the information of a specific product (prices, cost, margins)
 * UserCat : define a category of user, with its parameters for reservation price, graph weights, alpha ratios and number of product sold per tipe of product
 * Environment : define the environment structure using UserCat and Product classes. In this are defined the methods to compute the expected reward, both theoretical and with uncertain parameters, simulate the interactio of the users with PEPAS e-commerce and apply the abrupt changes. In Environment.py are contained also a group of support functions usefull both for Environment definition and functioning and for the algorithms implemented.
- Steps Algorithms:
 * For each step we have a .py file defining the class of the needed algorithm. Thompson Sampling and UCB1 learners have a different base class (respectively ucb_learner.py and Learner.py)
 * Step 6: Cumulative Sum, the adopted change detection algorithm, is coded in Cusum.py
 * Step 7: Context Generation algorithm is implemented in ContextGeneration class, moreover ad hoc learners for the groups identified by the context generation are defined in ucb_context.py and TS_context.py
