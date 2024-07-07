import layer as ly
import numpy as np
import pandas as pd
import training as tr
import joblib
import argparse
from sklearn.preprocessing import OneHotEncoder


def main():
    parser = argparse.ArgumentParser(
        description="This script predicts from new data")
    parser.add_argument("-m", "--model_name", type=str,
                         help="File Name prefix", required=True)
    parser.add_argument("-f", "--file_name", type=str,
                        help="data to predict", required=True)

    args = parser.parse_args()
    filen = args.model_name

    data = pd.read_csv(args.file_name, header=None)
    preprocess = joblib.load(f"{filen}_preprocess.joblib")
    model = joblib.load(f"{filen}_model.joblib")

    activation = tr.select_activation(model["activation"])
    loss = tr.select_cost(model["loss"])
    endsizes = [x["W"].shape[0] for x in model["weights"]]

    sizes = [model["weights"][0]["W"].shape[1]] + endsizes
    netw = tr.create_network(sizes, activation, loss)

    for a, b in zip(netw.layers, model["weights"]):
        a.W = b["W"]
        a.b = b["b"]

    y = data.iloc[:, [1]].copy()
    y = preprocess[1].transform(y)
    data = preprocess[0].transform(data.iloc[:, 2:])
    print(netw.cost_eval(data, y))


if __name__ == "__main__":
    main()


# %%
    