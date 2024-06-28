#%%

import layer as ly
import numpy as np



# # %% Linear regression example 1

# data2 = np.random.normal(size=(100,2))
# data2 = data2.dot(np.diag([3,4]))
# output2 = data2.dot(np.array([3,2]).T) + 10


# # %%

# l1 = ly.Layer(2,1,ly.id_act,ly.sq_cost)

# #%%

# while True:
#     o1 = l1.output(data2)
#     print(l1.cost_eval(o1.T,output2))
#     l1.W -= 0.001*l1.w_grad()
#     l1.b -= 0.001*l1.b_grad()
#     print(l1.W)
#     print(l1.b)

# # %%

#%% Logistic regression example 1

gendata1 = np.random.normal(size=(4,100)) + np.array([[1],[2],[2],[2]])
gendata2 = np.random.normal(size=(4,100)) + np.array([[-1],[-2],[2],[2]])
gendata3 = np.random.normal(size=(4,100)) + np.array([[1],[2],[-2],[-2]])
gendata4 = np.random.normal(size=(4,100)) + np.array([[1],[-2],[2],[-2]])
gendata = np.concatenate([gendata1,gendata2,gendata3,gendata4], axis=1)
gendata = gendata.T





# %%

response = [100*[1,0,0,0] + 100*[0,1,0,0] + 100*[0,0,1,0]+100*[0,0,0,1]]
response = np.array(response)
response = response.reshape((400,4))

# %%

l2 = ly.Layer(4,4,ly.softmax_act,ly.cross_entropy_cost)

# %%

l2.act_fun.fun(gendata)

# %%

cost_old = 0
cost = 0

while cost_old == 0 or np.abs(cost_old - cost)/cost_old > 10E-5:
    o1 = l2.output(gendata)
    cost_old = cost
    cost = l2.cost_eval(o1,response)
    l2.W -= 0.001*l2.w_grad()
    l2.b -= 0.001*l2.b_grad()
    print(l2.W)
    print(l2.b)
    print(cost)
# %%
np.argmax(o1, axis=1)