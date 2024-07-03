#%%

import numpy as np
from tqdm import tqdm
import time
import random
import plotly.express as px
import pandas as pd
import copy

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
        self.RMSProp_wt=0


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
        if (delta_next is None or W_next is None) and self.cost_fun is None:
            raise ValueError("Not enough info for gradient")
        if not self.cost_fun is None:
            return self.delta_
        self.delta_ = delta_next.dot(W_next)*self.act_fun.df(self.z)
        return self.delta_
  

    def w_grad(self):
        return self.delta_.T.dot(self.a_minus_one)/self.a_minus_one.shape[0]
    
    def b_grad(self):
        return np.mean(self.delta_.T,axis=1)
    
    def cost_eval(self, output, observed_data, validation = False):
        if not validation:
            self.delta_ = self.cost_fun.df(output,observed_data)
        return np.mean(self.cost_fun.fun(output,observed_data))
    
    def layer_update(self, training_method, learning_rate, gamma = 0):
        training_methods = {"GD": self.GD,"RMSProp":self.RMSProp}
        if not training_method in training_methods.keys():
            print("Training method unknown defaulting to gradient descent")
            training_method = "GD"
        training_methods[training_method](learning_rate, gamma)

    
    def GD(self, learning_rate, gamma):
        self.W -= learning_rate*self.w_grad()
        self.b -= learning_rate*self.b_grad()

    def RMSProp(self, learning_rate, gamma = 0.1):
        self.RMSProp_wt = gamma*self.RMSProp_wt + (1-gamma)*(np.sum(self.w_grad()**2) + np.sum(self.b_grad()**2))
        self.W -= learning_rate*self.w_grad()/np.sqrt(self.RMSProp_wt + 0.001)
        self.b -= learning_rate*self.b_grad()/np.sqrt(self.RMSProp_wt + 0.001)


 
   



class Network:
    def __init__(self, input_size, max_epoch = 10000) -> None:
        self.max_epoch = max_epoch
        self.input_size =  input_size
        self.cost_historic = []
        self.cost_val_historic = []
        self.layers = []
        

    def layer_append(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError("layer must be of layer class")
        if len(self.layers) > 0 and self.layers[-1].n_outputs != layer.n_inputs:
            raise ValueError("Layer number of inputs must be equal to last layer number of outputs")
        if len(self.layers) == 0 and self.input_size != layer.n_inputs:
            raise ValueError("Layer number of inputs must be equal to data number of inputs")
        self.layers.append(layer)

    def output(self, input_data):
        intermediate = input_data
        for layer in self.layers:
            intermediate = layer.output(intermediate)
        return intermediate
   
    
    def cost_eval(self, input_data, observed_data, validation = False):
        return self.layers[-1].cost_eval(self.output(input_data),observed_data, validation)
    

    def train_break(self, val_input):
      if len(self.cost_historic) < 2:
        return False
      if val_input is None:
        return  np.abs(self.cost_old - self.cost)/self.cost_old < 10E-20
      else:
        w = np.abs(self.cost_val_historic[-2] - self.cost_val_historic[-1])/self.cost_val_historic[-1] < 10E-20 or self.cost_val_historic[-1] > 1.1*np.min(self.cost_val_historic) 
        return w

    def train(self,input, observed,training_method = "GD", 
              learning_rate = 0.001, gamma = 0, val_input=None, 
              val_observed =None,batch_size = -1):
        self.cost_old = 0
        self.cost = 0
        if batch_size == -1:
            batch_size = input.shape[0]
        
        for i in range(self.max_epoch): 
            
            if self.train_break(val_input):
                break
            progress_bar = tqdm(total=input.shape[0]//batch_size + (input.shape[0]%batch_size > 0) ,
                                bar_format='{desc}|{percentage:3.0f}%|{bar}|Batch: {n_fmt}/{total_fmt}')
            orrange = list(range(input.shape[0]))
            random.shuffle(orrange)
            j = 0
            self.min_cost = float("Inf")
            while j < input.shape[0]:
                max_range = min(j+batch_size, input.shape[0])
                input_local = input[orrange[j:max_range],:]
                o1 = self.output(input_local)
                self.cost_old = self.cost
                observed_local = observed[orrange[j:max_range]]
                self.cost = self.layers[-1].cost_eval(o1,observed_local)
                delta_next = None
                W_next = None
                for lay in reversed(self.layers):
                    delta_next = lay.delta(delta_next, W_next)
                    W_next = lay.W
                for lay in self.layers:
                    lay.layer_update(training_method, learning_rate, gamma)        
                if val_input is None:
                    progress_bar.set_description(f"Cost: {cost:.4f}")
                else:
                    o2 = self.output(val_input)
                    self.val_cost= self.layers[-1].cost_eval(o2,val_observed, validation = True)
                    #time.sleep(0.1)
                    progress_bar.set_description(f"Epoch: {i} Cost: {self.cost:.4f} Validation Cost: {self.val_cost:.4f}")
                if not val_input is None:
                    self.cost_val_historic.append(self.val_cost)
                if val_input is None:
                    if self.cost < self.min_cost:
                        self.min_cost = self.cost
                        self.Final_Model = copy.deepcopy(self.layers)
                else: 
                    if self.val_cost < self.min_cost:
                        self.min_cost = self.val_cost
                        self.Final_Model = copy.deepcopy(self.layers)
                progress_bar.update(1)
                j += batch_size
                o1 = self.output(input)
                cost = self.layers[-1].cost_eval(o1,observed)
                self.cost_historic.append(cost)

        if not val_input is None:
            plot_df = pd.DataFrame({"x": list(range(len(self.cost_historic)))*2 ,
                                    "y":self.cost_historic+self.cost_val_historic,
                                    "color": ["Training Cost"]*len(self.cost_historic) + 
                                    ["Validation Cost"]*len(self.cost_historic)})
            plot = px.line(plot_df, x="x",y = "y", color="color")
        else:
            plot = px.line(x=list(range(len(self.cost_historic))),y = self.cost_historic)
        plot.show()
        return self.Final_Model
        
 


# %%

## some activations

#%% sigmoid


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_der(x):
    return np.exp(-x) / (1 + np.exp(-x))**2


sigmoid_act = Act_Fun(sigmoid,sigmoid_der)


#%% softmax act
def softmax(x):
    tr = np.exp(x)
    tr = tr/tr.sum(axis = 1,keepdims=True)
    return tr


softmax_act = Act_Fun(softmax,None)

# %% identity

def id(x):
    return x

def one(x):
    return 1

id_act = Act_Fun(id, one)
# %% leaky relu

def l_relu(x):
    return np.where(x < 0, 0.01* x, x)
    

def d_l_relu(x):
    return np.where(x < 0, 0.01, 1)
    
leaky_relu = Act_Fun(l_relu, d_l_relu)

# %% leaky relu

def relu(x):
    return np.where(x < 0,0, x)
    

def d_relu(x):
    return np.where(x < 0, 0, 1)
    
relu_act = Act_Fun(relu, d_relu)



## some costs

#%%
sq_cost = Cost_Fun(lambda x,y: (x-y)**2/np.shape[0], lambda x,y : 2*(x-y))

def cross_entr_cost(predicted,real):
    a = np.log(predicted)
    return -1/real.shape[0]* np.sum(a*real)

def cross_entr_der(predicted, real):
        return (predicted -real)

cross_entropy_cost = Cost_Fun(cross_entr_cost,cross_entr_der)

# %%
