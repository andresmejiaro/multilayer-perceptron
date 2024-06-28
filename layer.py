#%%

import numpy as np


class Act_Fun:
    def __init__(self,fun,df=None):
        self.fun=fun
        if df is None:
            self.df = lambda x: (fun(x+0.00001)-fun(x))/0.00001
        else:
            self.df = df


class Cost_Fun:
    def __init__(self,fun,df=None):
        self.fun=fun
        if df is None:
            self.df = lambda x,y: (fun(x+0.00001,y)-fun(x,y))/0.00001
        else:
            self.df = df



class Layer:
    def __init__(self, n_inputs,n_outputs, act_fun, cost_fun = None):
        if not isinstance(n_inputs,int) or not isinstance(n_outputs, int):
            raise ValueError ("number of input or outputs is not an integer")
        if not n_inputs > 0 or not n_outputs > 0:
            raise ValueError("number of inputs or outputs must be positive")
        if not isinstance(act_fun,Act_Fun):
            raise ValueError("Activation should be in class Act_Fun")
        if not cost_fun is None and not isinstance(cost_fun,Cost_Fun):
            raise ValueError("cost should be of class Cost_Fun")
        
        self.cost_fun = cost_fun
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        self.W = np.random.normal(size=(n_outputs, n_inputs))
        self.b = np.abs(np.random.normal(size=(n_outputs,)))       
        self.act_fun = act_fun


    def output (self, input_data):
        if input_data.shape[1] != self.n_inputs:
            raise ValueError("input size does not match number of inputs in this layer")
        self.a_minus_one = input_data        
        self.z = self.a_minus_one.dot(self.W.T)+ self.b
        self.a = self.act_fun.fun(self.z)
        return self.a
    
    def act_derivative(self):
        return self.act_fun.df(self.z)
    
    ## Each layer calculates and stores delta l-1
    def delta(self, delta_next = None, W_next = None):
        if delta_next is None or W_next is None and self.cost_fun is None:
            raise ValueError("Not enough info for gradient")
        if not self.cost_fun is None:
            return self.delta_
        self.delta_ = delta_next.dot(W_next)*self.act_fun.df(self.z)
        return self.delta_
  

    def w_grad(self):
        return self.delta_.dot(self.a_minus_one)/self.a_minus_one.shape[0]
    
    def b_grad(self):
        return np.mean(self.delta_,axis=1)
    
    def cost_eval(self, output, observed_data):
        self.delta_ = self.cost_fun.df(output,observed_data)
        return np.mean(self.cost_fun.fun(output,observed_data))
    

    


 
   



class Network:
    def __init__(self, input_size) -> None:
        self.last_size =  input_size
        self.layers = []
        pass

    def layer_append(self, layer):
        self.layers.append(layer)

    def output(self, input_data):
        out = input_data
   
    
    def cost_eval(self, input_data, observed_data):
        return self.layers[-1].cost_eval(self.output(input_data),observed_data)
    
    def train(input, output):
        pass
   


# %%

## some activations

#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_der(x):
    return np.exp(-x) / (1 + np.exp(-x))**2


sigmoid_act = Act_Fun(sigmoid,sigmoid_der)


#%%
def softmax(x):
    tr = np.exp(x)
    tr = tr/tr.sum(axis = 1,keepdims=True)
    return tr


softmax_act = Act_Fun(softmax,None)

# %%

def id(x):
    return x

def one(x):
    return 1

id_act = Act_Fun(id, one)


## some costs

#%%
sq_cost = Cost_Fun(lambda x,y: (x-y)**2, lambda x,y : 2*(x-y))

def cross_entr_cost(predicted,real):
    print(predicted)
    a = np.log(predicted)
    return -1/real.shape[1]* np.sum(a*real)

def cross_entr_der(predicted, real):
        return (predicted -real).T

cross_entropy_cost = Cost_Fun(cross_entr_cost,cross_entr_der)

# %%
