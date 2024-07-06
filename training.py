# %%
import pandas as pd
import layer as ly
import numpy as np
import argparse
import joblib
import random


def load_and_preprocess_data(name, validation):
    t_data = pd.read_csv(f"{name}_TD.csv", header=None)
    t_target = pd.read_csv(f"{name}_TT.csv", header=None)
    t_data = np.array(t_data)
    t_target = np.array(t_target)
    t_target = np.append(t_target, 1-t_target, axis=1)
    if validation:
        v_data = pd.read_csv(f"{name}_VD.csv", header=None)
        v_target = pd.read_csv(f"{name}_VT.csv", header=None)
        v_data = np.array(v_data)
        v_target = np.array(v_target)
        v_target = np.append(v_target, 1-v_target, axis=1)
    else:
        v_data = v_target = None
    return t_data, t_target, v_data, v_target


def create_network(sizes, activation, cost, epochs=100000):
    red = ly.Network(input_size=sizes[0], max_epoch=epochs)
    for i in range(len(sizes)):
        if i == 0:
            continue
        if i == len(sizes) - 1:
            red.layer_append(
                ly.Layer(sizes[i-1], sizes[i], ly.softmax_act, cost))
        else:
            red.layer_append(ly.Layer(sizes[i-1], sizes[i], activation))
    return red


def select_activation(activation_text):
    activations = {"LeakyRelu": ly.leaky_relu, "Linear": ly.id_act,
                   "Sigmoid": ly.sigmoid_act, "Relu": ly.relu_act}
    if activation_text in activations.keys():
        return activations[activation_text]
    print(
        f"Activation {activation_text} not in {activations.keys()} returning sigmoid activation")
    return ly.sigmoid_act


def select_cost(cost_text):
    costs = {"MeanSq": ly.sq_cost, "CrossEntropy": ly.cross_entropy_cost}
    if cost_text in costs.keys():
        return costs[cost_text]
    print(f"Cost not in {costs.keys()} aborting")
    exit(1)


def select_training_method(training_method):
    methods = ["RMSProp", "GD"]
    if training_method in methods:
        return training_method
    print(f"Training Method not supported defaulting to Gradient Descent")
    return "GD"


def main():
    parser = argparse.ArgumentParser(
        description="This script trains the data preprocesed with the preprocessing macro")
    parser.add_argument("-la", "--layer", type=int, nargs='+',
                        help="size of hidden layers", required=True)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of epochs", default=100000)
    parser.add_argument("-lo", "--loss", type=str,
                        help="Loss function name", default="CrossEntropy")
    parser.add_argument("-b", "--batch_size", type=int,
                        help="Batch Size", default=-1)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="Learning Rate", default=0.01)
    parser.add_argument("-f", "--file_name", type=str,
                        help="File Name prefix", required=True)
    parser.add_argument("-a", "--activation", type=str,
                        help="Activation", default="Sigmoid")
    parser.add_argument("-m", "--method", type=str,
                        help="Training Method", default="GD")
    parser.add_argument("-s", "--seed", action='store_true',
                        help="Seed", default=False)
    parser.add_argument("-v", "--validation", action='store_false',
                        help="Include a Validation Set", default=True)

    args = parser.parse_args()

    if args.seed:
        random.seed(42)
        np.random.seed(42)
    t_data, t_target, v_data, v_target = load_and_preprocess_data(
        args.file_name, args.validation)

    sizes = [t_data.shape[1]] + args.layer
    activation = select_activation(args.activation)
    loss = select_cost(args.loss)
    training_method = select_training_method(args.method)
    red = create_network(sizes, activation, loss, args.epochs)

    j = red.train(t_data, t_target, val_input=v_data, val_observed=v_target,
                  learning_rate=args.learning_rate,
                  batch_size=args.batch_size, training_method=training_method)
    weights = [{"W": w.W, "b": w.b} for w in j]
    joblib.dump({"weights": weights, "loss": args.loss,
                "activation": args.activation}, f"{args.file_name}_model.joblib")


if __name__ == "__main__":
    main()


# %%


# %%


# %%
# np.append(np.argmax(x,axis=1,keepdims=True),t_target[:,[1]],axis=1)
# %%
