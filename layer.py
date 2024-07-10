# %%

import numpy as np
from tqdm import tqdm
import random
import plotly.express as px
import pandas as pd
import copy
from sklearn.metrics import accuracy_score, f1_score


class Act_Fun:
    def __init__(self, fun, df=None, id = "XX"):
        self.id = id
        self.fun = fun
        if df is None:
            self.df = lambda x: (fun(x+0.00001)-fun(x))/0.00001
        else:
            self.df = df


class Cost_Fun:
    def __init__(self, fun,  df=None,id = "xx" ):
        self.fun = fun
        self.id = id
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
        self.l2_reg = 0.01

        if act_fun.id == "Relu" or act_fun.id == "LeakyRelu":
            self.W = np.random.normal(loc = 0, scale=np.sqrt(2/n_inputs),size=(n_outputs, n_inputs))
        else:
            self.W = np.random.normal(loc = 0, scale=np.sqrt(2/(n_inputs+n_outputs)),size=(n_outputs, n_inputs))
        self.b = np.zeros(shape=(n_outputs,))
        
        self.act_fun = act_fun
        self.RMSProp_wt_W = 0
        self.RMSProp_wt_b = 0
        self.old_W_grad = 0
        self.old_b_grad = 0
        self.l2_reg = 0

    def output(self, input_data, state_save = True):
        if input_data.shape[1] != self.n_inputs:
            raise ValueError(
                "input size does not match number of inputs in this layer")
        a_minus_one = input_data
        z = a_minus_one.dot(self.W.T) + self.b        
        a = self.act_fun.fun(z)
        if state_save:
            self.a_minus_one = a_minus_one
            self.z = z
            self.a = a
        return a

     # Each layer calculates and stores delta l-1
    def delta(self, delta_next=None, W_next=None):
        if (delta_next is None or W_next is None) and self.cost_fun is None:
            raise ValueError("Not enough info for gradient")
        if not self.cost_fun is None:
            return self.delta_
        self.delta_ = delta_next.dot(W_next)*self.act_fun.df(self.z)
        return self.delta_

    def w_grad(self):
        a =self.delta_.T.dot(self.a_minus_one)/self.a_minus_one.shape[0]
        return a

    def b_grad(self):
        
        return np.mean(self.delta_.T, axis=1)

    def cost_eval(self, output, observed_data, validation=True):
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
        self.W = self.W - learning_rate*(self.w_grad()) - self.l2_reg*self.W
        self.b = self.b - learning_rate*(self.b_grad()) - self.l2_reg*self.b
 

    def RMSProp(self, learning_rate, gamma=0.1):
        self.RMSProp_wt_W = gamma*self.RMSProp_wt_W + \
            (1-gamma)*(np.sum(self.w_grad()**2))
        self.RMSProp_wt_b = gamma*self.RMSProp_wt_b + \
            (1-gamma)*(np.sum(np.sum(self.b_grad()**2)))
        self.W = self.W - learning_rate*(self.w_grad()/np.sqrt(self.RMSProp_wt_W + 10e-6)) - self.l2_reg*self.W
        self.b = self.b - learning_rate*(self.b_grad()/np.sqrt(self.RMSProp_wt_b + 10e-6)) - self.l2_reg*self.b

    # def Momentum(self,learning_rate, gamma):
    #     W_grad = self.w_grad()
    #     b_grad = self.b_grad()
    #     self.W = self.W - learning_rate*((1-gamma)*W_grad +gamma*self.old_W_grad)  + self.l2_reg *self.W
    #     self.b = self.b  - learning_rate*((1-gamma)*b_grad +gamma*self.old_b_grad) + self.l2_reg * self.b
    #     self.old_W_grad = W_grad
    #     self.old_b_grad = b_grad

    def Momentum(self, learning_rate, gamma):
        W_grad = self.w_grad()
        b_grad = self.b_grad()
        
        # Momentum update
        W_update = (1 - gamma) * W_grad + gamma * self.old_W_grad
        b_update = (1 - gamma) * b_grad + gamma * self.old_b_grad
        
        # Apply updates with learning rate and regularization
        self.W = self.W - learning_rate * (W_update + self.l2_reg * self.W)
        self.b = self.b - learning_rate * (b_update + self.l2_reg * self.b)
        
        # Save current gradients as old gradients for next iteration
        self.old_W_grad = W_grad
        self.old_b_grad = b_grad


class Network:
    def __init__(self, input_size, max_epoch=10000) -> None:
        self.max_epoch = max_epoch
        self.input_size = input_size
        self.cost_historic = []
        self.cost_val_historic = []
        self.accuracy_historic = []
        self.accuracy_val_historic = []
        self.f1_historic = []
        self.f1_val_historic = []
        self.layers = []
        self.patience = self.max_epoch
        self.Final_Model = None
        self.l2_reg = 0

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

    def output(self, input_data, state_save = True):
        intermediate = input_data
        for layer in self.layers:
            intermediate = layer.output(intermediate,state_save)
        return intermediate

    def cost_eval(self, input_data, observed_data, validation=True):
        predicted = self.output(input_data, state_save= not validation)
        return self.layers[-1].cost_eval(predicted, observed_data, validation)

    def train_break(self, val_input, stop):
        if len(self.cost_historic) < 2:
            return False
        if val_input is None:
            return np.abs(self.cost_old - self.cost)/self.cost_old < 10E-20
        else:
            if stop == "cost_val_hard":
                return self.val_cost < 0.05  
            if stop == "cost_val_relative":
                return self.cost_val_historic[-1]/self.cost_val_historic[-2] - 1 < 0.0001  
            return self.val_ac_s > 0.95
        
    def _train_startprint(self, input, val_input):
        print(f"X train shape {input.shape}")
        if not val_input is None:
            print(f"X validation shape {val_input.shape}")

    def _train_tqdm_update(self, progress_bar, val_input, val_observed, i):
        if val_input is None:
            progress_bar.set_description(
                f"Epoch: {i} Cost: ({self.cost:.4f})  Acc: ({self.ac_s:.4f}) F1: ({self.f1_s:.4f})")
        else:
            progress_bar.set_description(
                f"Epoch: {i} Cost:({self.cost:.4f},{self.val_cost:.4f}) Acc:({self.ac_s:.2f},{self.val_ac_s:.2f}) F1:({self.f1_s:.2f},{self.val_f1_s:.2f})")
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
        self.ac_s = 0
        self.val_ac_s = 0
        self.f1_s = 0
        self.val_f1_s = 0

    def _train_model_keep(self, val_input):
        if val_input is None and self.val_cost < self.min_cost:
            self.min_cost = self.val_cost
            self.Final_Model = copy.deepcopy(self.layers)
        elif (not val_input is None) and self.cost < self.min_cost:
            self.min_cost = self.cost
            self.Final_Model = copy.deepcopy(self.layers)

    def weight_cost_l2(self):
        a = [(np.sum(lay.W**2) + np.sum(lay.b**2)) for lay in self.layers]
        b = [(lay.W.shape[0]*lay.W.shape[1] + lay.b.shape[0]) for lay in self.layers]
        cost = np.sum(a)/np.sum(b)

        # for lay in self.layers:
        #     lay.l2_reg = 0*np.sqrt(cost)/(100+np.sqrt(cost))
        return cost
                


    def _train_metrics_update(self, input, observed, val_input, val_observed):
        
        o1 = self.output(input,False)
        self.cost = self.layers[-1].cost_eval(o1, observed, validation = True)
        labels_pred = np.argmax(o1, axis=1)
        labels_observed = np.argmax(observed, axis=1)
        self.ac_s = accuracy_score(labels_observed, labels_pred)
        try:
            self.f1_s = f1_score(labels_observed, labels_pred)
        except:
            self.f1_s = -1
        self.weight_cost= self.weight_cost_l2()
        if not val_input is None:
            o2 = self.output(val_input,False)
            self.val_cost = self.layers[-1].cost_eval(o2, val_observed, validation = True)
            labels_val_pred = np.argmax(o2, axis=1)
            labels_val_observed = np.argmax(val_observed, axis=1)
            self.val_ac_s = accuracy_score(
                labels_val_observed, labels_val_pred)
            try:
                self.val_f1_s = f1_score(labels_val_observed, labels_val_pred)
            except:
                self.val_f1_s = -1

    def _train_save_epoch_historic(self, val_input):
        if not val_input is None:
            self.cost_val_historic.append(self.val_cost)
            self.accuracy_val_historic.append(self.val_ac_s)
            self.f1_val_historic.append(self.val_f1_s)
        self.cost_historic.append(self.cost)
        self.accuracy_historic.append(self.ac_s)
        self.f1_historic.append(self.f1_s)

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
    
    def _train_initialize_reg(self, l2_reg):
        for lay in self.layers:
            lay.l2_reg = l2_reg

    def train(self, input, observed, training_method="GD",
              learning_rate=0.001, gamma=0, val_input=None,
              val_observed=None, batch_size=-1,l2_reg = 0,
              stop_method = "cost_val_reative"):
        self._train_initialize_costs()
        self._train_initialize_reg(l2_reg)
        self._train_startprint(input,val_input)
        if batch_size == -1:
            batch_size = input.shape[0]
        for i in range(self.max_epoch):
            if self.train_break(val_input, stop_method):
                break
            orrange = list(range(input.shape[0]))
            random.shuffle(orrange)
            j = 0
            progress_bar = tqdm(total=input.shape[0]//batch_size + (input.shape[0] % batch_size > 0),
                                bar_format='{desc}|{percentage:3.0f}%|{bar}|Batch: {n_fmt}/{total_fmt}')
            while j < input.shape[0]:
                self.cost_old = self.cost
                max_range = min(j+batch_size, input.shape[0])
                input_local = input[orrange[j:max_range], :]
                o1 = self.output(input_local,state_save=True)
                observed_local = observed[orrange[j:max_range]]
                self.cost = self.layers[-1].cost_eval(o1, observed_local, validation = False)
                self._train_backprop(training_method, learning_rate, gamma)
                self._train_metrics_update(
                    input, observed, val_input, val_observed)
                self._train_tqdm_update(
                    progress_bar, val_input, val_observed, i)
                j += batch_size
            self._train_model_keep(val_input)
            self._train_save_epoch_historic(val_input)
        self._train_model_plots(val_input)
        if self.Final_Model is not None:
            return self.Final_Model
        else:
            return self.layers
        
    
    @staticmethod
    def create_network(sizes, activation, cost, epochs=100000):
        red = Network(input_size=sizes[0], max_epoch=epochs)
        for i in range(len(sizes)):
            if i == 0:
                continue
            if i == len(sizes) - 1:
                red.layer_append(
                    Layer(sizes[i-1], sizes[i], softmax_act, cost))
            else:
                red.layer_append(Layer(sizes[i-1], sizes[i], activation))
        return red


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

sigmoid_act = Act_Fun(sigmoid, sigmoid_der, id = "Sigmoid")


# %% softmax act
def softmax(x):
    tr = np.exp(x-np.max(x))
    tr_sum = np.sum(tr,axis=1, keepdims=True)
    tr = tr/tr_sum
    return tr


softmax_act = Act_Fun(softmax, None, id = "Softmax")

# %% identity


def id(x):
    return x


def one(x):
    return 1


id_act = Act_Fun(id, one, id = "Identity")
# %% leaky relu


def l_relu(x):
    return np.where(x < 0, 0.01 * x, x)


def d_l_relu(x):
    w = np.where(x < 0, 0.01, 1)
    return w    

leaky_relu = Act_Fun(l_relu, d_l_relu, id = "LeakyRelu")

# %% leaky relu


def relu(x):
    return np.where(x < 0, 0, x)


def d_relu(x):
    return np.where(x < 0, 0, 1)


relu_act = Act_Fun(relu, d_relu, id = "Relu")


# some costs

# %%
sq_cost = Cost_Fun(lambda x, y: (x-y)**2/np.shape[0], lambda x, y: 2*(x-y), id = "MeanSq")


def cross_entr_cost(predicted, real):
    epsilon = 1e-12
    predicted = np.clip(predicted,epsilon, 1 - epsilon)
    a = np.log(predicted)
    return -1/real.shape[0] * np.sum(a*real)


def cross_entr_der(predicted, real):
    epsilon = 1e-12
    return (predicted - real)


cross_entropy_cost = Cost_Fun(cross_entr_cost, cross_entr_der, "CrossEntropy")

# %%
