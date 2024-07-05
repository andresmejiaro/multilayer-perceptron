import layer as ly
import numpy as np
import pandas as pd
import training as tr
import joblib 
import argparse
from sklearn.preprocessing import OneHotEncoder



def main():
    parser = argparse.ArgumentParser(description="This script predicts from new data")
    parser.add_argument("-m","--model_name", type=str,nargs=1, help= "File Name prefix", required=True)
    parser.add_argument("-f","--file_name", type=str,nargs=1, help= "data to predict", required=True)

    args = parser.parse_args()
    filen=args.model_name[0]

    data = pd.read_csv("data/data.csv")
    preprocess = joblib.load(f"{filen}_preprocess.joblib")
    model = joblib.load(f"{filen}_model.joblib")

    activation= tr.select_activation(model["activation"])
    loss = tr.select_cost(model["loss"])
    endsizes = [x["W"].shape[0] for x in model["weights"]] 

    sizes = [model["weights"][0]["W"].shape[1]] + endsizes
    netw = tr.create_network(sizes,activation,loss)



    for a,b in zip(netw.layers,model["weights"]):
        a.W = b["W"]
        a.b = b["b"]
    
    y= data.iloc[:,[1]].copy()
    y = preprocess[1].transform(y)
    y = np.append(y, 1-y, axis= 1)
    data = preprocess[0].transform(data.iloc[:,2:])
    print(netw.cost_eval(data,y))
    


if __name__ == "__main__":
    main()





# %%
