# %%

import numpy as np
from tqdm import tqdm
import random
import plotly.express as px
import pandas as pd
import copy
from sklearn.metrics import accuracy_score, f1_score
import joblib



class Act_Fun:
    def __init__(self, fun, df=None):
        self.fun = fun
        if df is None:
            self.df = lambda x: (fun(x+0.00001)-fun(x))/0.00001
        else:
            self.df = df


class Cost_Fun:
    def __init__(self, fun, df=None):
        self.fun = fun
        if df is None:
            self.df = lambda x, y: (fun(x+0.00001, y)-fun(x, y))/0.00001
        else:
            self.df = df


class Layer:
    def __init__(self, n_inputs, n_outputs, act_fun, cost_fun=None):
        if not isinstance(n_inputs, int) or not isinstance(n_outputs, int):
            raise ValueError("number of input or outputs is not an integer")
        if not n_inputs > 0 or not n_outputs > 0:
            raise ValueError("number of inputs or outputs must be positive")
        if not isinstance(act_fun, Act_Fun):
            raise ValueError("Activation should be in class Act_Fun")
        if not cost_fun is None and not isinstance(cost_fun, Cost_Fun):
            raise ValueError("cost should be of class Cost_Fun")

        self.cost_fun = cost_fun
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.W = np.random.normal(size=(n_outputs, n_inputs),scale=10E-5)
        self.b = np.abs(np.random.normal(size=(n_outputs,),scale=10E-5))
        self.act_fun = act_fun
        self.RMSProp_wt_W = 0
        self.RMSProp_wt_b = 0
        self.old_W_grad = 0
        self.old_b_grad = 0

    def output(self, input_data):
        if input_data.shape[1] != self.n_inputs:
            raise ValueError(
                "input size does not match number of inputs in this layer")
        self.a_minus_one = input_data
        self.z = self.a_minus_one.dot(self.W.T) + self.b
        self.a = self.act_fun.fun(self.z)
        return self.a

    def act_derivative(self):
        return self.act_fun.df(self.z)

    # Each layer calculates and stores delta l-1
    def delta(self, delta_next=None, W_next=None):
        if (delta_next is None or W_next is None) and self.cost_fun is None:
            raise ValueError("Not enough info for gradient")
        if not self.cost_fun is None:
            return self.delta_
        self.delta_ = delta_next.dot(W_next)*self.act_fun.df(self.z)
        return self.delta_

    def w_grad(self):
        return self.delta_.T.dot(self.a_minus_one)/self.a_minus_one.shape[0]

    def b_grad(self):
        return np.mean(self.delta_.T, axis=1)

    def cost_eval(self, output, observed_data, validation=False):
        if not validation:
            self.delta_ = self.cost_fun.df(output, observed_data)
        return np.mean(self.cost_fun.fun(output, observed_data))

    def layer_update(self, training_method, learning_rate, gamma=0):
        training_methods = {"GD": self.GD, "RMSProp": self.RMSProp,"MO":self.Momentum}
        if not training_method in training_methods.keys():
            print("Training method unknown defaulting to gradient descent")
            training_method = "GD"
        training_methods[training_method](learning_rate, gamma)

    def GD(self, learning_rate, gamma):
        self.W -= +learning_rate*(self.w_grad() + 0.001*self.W)
        self.b -= +learning_rate*(self.b_grad() + 0.001*self.b)

    def RMSProp(self, learning_rate, gamma=0.1):
        self.RMSProp_wt_W = gamma*self.RMSProp_wt_W + \
            (1-gamma)*(np.sum(self.w_grad()**2))
        self.RMSProp_wt_b = gamma*self.RMSProp_wt_b + \
            (1-gamma)*(np.sum(self.b_grad()**2))
        self.W -= learning_rate*(self.w_grad()/np.sqrt(self.RMSProp_wt_W + 0.001) + 0.00*self.W)
        self.b -= learning_rate*(self.b_grad()/np.sqrt(self.RMSProp_wt_b + 0.001) + 0.00*self.b)

    def Momentum(self,learning_rate, gamma):
        W_grad = self.w_grad()
        b_grad = self.b_grad()
        self.W -= +learning_rate*((1-gamma)*W_grad +gamma*self.old_W_grad )
        self.b -= +learning_rate*((1-gamma)*b_grad +gamma*self.old_b_grad)
        self.old_W_grad = W_grad
        self.old_b_grad = b_grad

class Network:
    def __init__(self, input_size, max_epoch=10000) -> None:
        self.max_epoch = max_epoch
        self.input_size = input_size
        self.cost_historic = []
        self.cost_val_historic = []
        self.layers = []
        self.patience = self.max_epoch
        self.early_stopping_trigger = 0

    def layer_append(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError("layer must be of layer class")
        if len(self.layers) > 0 and self.layers[-1].n_outputs != layer.n_inputs:
            raise ValueError(
                "Layer number of inputs must be equal to last layer number of outputs")
        if len(self.layers) == 0 and self.input_size != layer.n_inputs:
            raise ValueError(
                "Layer number of inputs must be equal to data number of inputs")
        self.layers.append(layer)

    def output(self, input_data):
        intermediate = input_data
        for layer in self.layers:
            intermediate = layer.output(intermediate)
        return intermediate

    def cost_eval(self, input_data, observed_data, validation=False):
        predicted = self.output(input_data)
        return self.layers[-1].cost_eval(predicted, observed_data, validation)

    def train_break(self, val_input):
        if len(self.cost_historic) < 2:
            return False
        if val_input is None:
            return np.abs(self.cost_old - self.cost)/self.cost_old < 10E-20
        else:
            return self.val_cost < 0.05  
        
    def _train_startprint(self, input, val_input):
        print(f"X train shape {input.shape}")
        if not val_input is None:
            print(f"X validation shape {val_input.shape}")

    def _train_tqdm_update(self, progress_bar, val_input, val_observed, i):
        if val_input is None:
            progress_bar.set_description(
                f"Epoch: {i} Cost: ({self.cost:.4f}) Acc: ({self.ac_s:.4f}) F1: ({self.f1_s:.4f})")
        else:
            o2 = self.output(val_input)
            self.val_cost = self.layers[-1].cost_eval(
                o2, val_observed, validation=True)
            progress_bar.set_description(
                f"Epoch: {i} Cost: ({self.cost:.4f},{self.val_cost:.4f}) )")
        progress_bar.update(1)

    def _train_backprop(self, training_method, learning_rate, gamma):
        delta_next = None
        W_next = None
        for lay in reversed(self.layers):
            delta_next = lay.delta(delta_next, W_next)
            W_next = lay.W
        for lay in self.layers:
            lay.layer_update(training_method, learning_rate, gamma)

    def _train_initialize_costs(self):
        self.cost_old = 0
        self.cost = 0
        self.min_cost = float("Inf")
        self.val_cost = float("Inf")
        

    def _train_model_keep(self, val_input):
        if val_input is None and self.val_cost < self.min_cost:
            self.min_cost = self.val_cost
            self.Final_Model = copy.deepcopy(self.layers)
        elif (not val_input is None) and self.cost < self.min_cost:
            self.min_cost = self.cost
            self.Final_Model = copy.deepcopy(self.layers)

    def _train_metrics_update(self, input, observed, val_input, val_observed):
        
        o1 = self.output(input)
        self.cost = self.layers[-1].cost_eval(o1, observed)
        if not val_input is None:
            o2 = self.output(val_input)
            self.val_cost = self.layers[-1].cost_eval(o2, val_observed)
            
            

    def _train_save_epoch_historic(self, val_input):
        if not val_input is None:
            self.cost_val_historic.append(self.val_cost)
        self.cost_historic.append(self.cost)
        

    def _train_model_plots(self, val_input):
        if not val_input is None:
            plot_df = pd.DataFrame({"x": list(range(len(self.cost_historic)))*2,
                                    "y": self.cost_historic+self.cost_val_historic,
                                    "color": ["Training Cost"]*len(self.cost_historic) +
                                    ["Validation Cost"]*len(self.cost_historic)})
            plot = px.line(plot_df, x="x", y="y", color="color")
            plot2_df = pd.DataFrame({"x": list(range(len(self.accuracy_historic)))*2,
                                    "y": self.accuracy_historic+self.accuracy_val_historic,
                                     "color": ["Training Accuracy"]*len(self.accuracy_historic) +
                                     ["Validation Accuracy"]*len(self.accuracy_val_historic)})
            plot2 = px.line(plot2_df, x="x", y="y", color="color")
            plot3_df = pd.DataFrame({"x": list(range(len(self.f1_historic)))*2,
                                    "y": self.f1_historic+self.f1_val_historic,
                                     "color": ["Training F1"]*len(self.f1_historic) +
                                     ["Validation F1"]*len(self.f1_val_historic)})
            plot3 = px.line(plot3_df, x="x", y="y", color="color")
        else:
            plot = px.line(x=list(range(len(self.cost_historic))),
                           y=self.cost_historic)
            plot2 = px.line(
                x=list(range(len(self.accuracy_historic))), y=self.cost_historic)
            plot3 = px.line(x=list(range(len(self.f1_historic))),
                            y=self.cost_historic)
        plot.show()
        plot2.show()
        plot3.show()

    def train(self, input, observed, training_method="GD",
              learning_rate=0.001, gamma=0, val_input=None,
              val_observed=None, batch_size=-1):
        self._train_initialize_costs()
        self._train_startprint(input,val_input)
        if batch_size == -1:
            batch_size = input.shape[0]
        for i in range(self.max_epoch):
            if self.train_break(val_input):
                break
            progress_bar = tqdm(total=input.shape[0]//batch_size + (input.shape[0] % batch_size > 0),
                                bar_format='{desc}|{percentage:3.0f}%|{bar}|Batch: {n_fmt}/{total_fmt}')
            orrange = list(range(input.shape[0]))
            random.shuffle(orrange)
            j = 0
            while j < input.shape[0]:
                self.cost_old = self.cost
                max_range = min(j+batch_size, input.shape[0])
                input_local = input[orrange[j:max_range], :]
                o1 = self.output(input_local)
                observed_local = observed[orrange[j:max_range]]
                self.cost = self.layers[-1].cost_eval(o1, observed_local)
                self._train_backprop(training_method, learning_rate, gamma)
                self._train_metrics_update(
                    input, observed, val_input, val_observed)
                self._train_tqdm_update(
                    progress_bar, val_input, val_observed, i)
                j += batch_size
            self._train_model_keep(val_input)
            self._train_save_epoch_historic(val_input)
            if i % 10 == 0:
                joblib.dump([[j.W,j.b] for j in self.layers],f"{i}.joblib")
        self._train_model_plots(val_input)
        return self.Final_Model


# %%

# some activations

# %% sigmoid


def sigmoid(x):
    x = x - np.max(x)
    y = np.exp(x) 
    return y / (1 + y)


def sigmoid_der(x):
    z = sigmoid(x)
    return z*(1-z)


sigmoid_act = Act_Fun(sigmoid, sigmoid_der)


# %% softmax act
def softmax(x):
    tr = x - np.max(x)
    tr = np.exp(tr)
    tr = tr/np.sum(tr)
    return tr


softmax_act = Act_Fun(softmax, None)

# %% identity


def id(x):
    return x


def one(x):
    return 1


id_act = Act_Fun(id, one)
# %% leaky relu


def l_relu(x):
    return np.where(x < 0, 0.001 * x, x)


def d_l_relu(x):
    return np.where(x < 0, 0.001, 1)


leaky_relu = Act_Fun(l_relu, d_l_relu)

# %% leaky relu


def relu(x):
    return np.where(x < 0, 0, x)


def d_relu(x):
    return np.where(x < 0, 0, 1)


relu_act = Act_Fun(relu, d_relu)


# some costs

# %%
sq_cost = Cost_Fun(lambda x, y: (x-y)**2/np.shape[0], lambda x, y: 2*(x-y))


def cross_entr_cost(predicted, real):
    a = np.log(predicted)
    return -1/real.shape[0] * np.sum(a*real)


def cross_entr_der(predicted, real):
    return -(real - predicted)


cross_entropy_cost = Cost_Fun(cross_entr_cost, cross_entr_der)

# %%
