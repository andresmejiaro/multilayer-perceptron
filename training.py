#%%
import pandas as pd
import layer as ly
import numpy as np
import argparse

def load_and_preprocess_data(name):
    t_data = pd.read_csv(f"{name}_TD.csv",header=None)
    t_target = pd.read_csv(f"{name}_TT.csv", header=None)
    v_data = pd.read_csv(f"{name}_VD.csv", header=None)
    v_target = pd.read_csv(f"{name}_VT.csv", header=None)
    t_data = np.array(t_data)
    t_target = np.array(t_target)
    t_target = np.append(t_target,1-t_target,axis=1)
    v_data = np.array(v_data)
    v_target = np.array(v_target)
    v_target = np.append(v_target, 1-v_target, axis=1)
    return t_data, t_target, v_data, v_target

def create_network(sizes, activation, cost, epochs = 100000):
    red = ly.Network(input_size=sizes[0],max_epoch=epochs)
    for i in range(len(sizes)):
        if i == 0:
            continue  
        if i == len(sizes) -1:
            red.layer_append(ly.Layer(sizes[i-1],sizes[i],activation, cost))    
        else:
            red.layer_append(ly.Layer(sizes[i-1],sizes[i],activation))
    return red

def select_activation(activation_text):
    activations = {"LeakyRelu":ly.leaky_relu,"Linear":ly.id_act,"Sigmoid":ly.sigmoid_act}
    if activation_text in activations.keys():
        return activations[activation_text]
    print(f"Activation not in {activations.keys()} returning sigmoid activation")
    return ly.sigmoid_act


def select_cost(cost_text):
    costs = {"MeanSq":ly.sq_cost,"CrossEntropy":ly.cross_entropy_cost}
    if cost_text in costs.keys():
        return costs[cost_text]
    print(f"Cost not in {costs.keys()} aborting")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="This script trains the data preprocesed with the preprocessing macro")
    parser.add_argument("-la","--layer",type=int, nargs='+', help = "size of hidden layers", required=True)
    parser.add_argument("-e","--epochs", type=int,nargs=1, help= "Number of epochs", default=100000)
    parser.add_argument("-lo","--loss", type=str,nargs=1, help= "Loss function name", default="CrossEntropy")
    parser.add_argument("-b","--batch_size", type=int,nargs=1, help= "Batch Size", default= -1)
    parser.add_argument("-lr","--learning_rate", type=float,nargs=1, help= "Learning Rate", default=0.001)
    parser.add_argument("-f","--file_name", type=str,nargs=1, help= "File Name prefix", required=True)
    parser.add_argument("-a","--activation", type=str,nargs=1, help= "Activation", default="Sigmoid")

    args = parser.parse_args()

    t_data, t_target, v_data, v_target = load_and_preprocess_data(args.file_name[0])

    sizes = [t_data.shape[1]]+ args.layer
    activation = select_activation(args.activation)
    loss = select_cost(args.loss)    
    red = create_network(sizes,activation,loss,args.epochs[0])
    
    red.train(t_data,t_target,val_input=v_data,val_observed=v_target,learning_rate=args.learning_rate[0])
    x=red.output(t_data)


if __name__ == "__main__":
    main()


#%%


# %%



# %%
#np.append(np.argmax(x,axis=1,keepdims=True),t_target[:,[1]],axis=1)
# %%
