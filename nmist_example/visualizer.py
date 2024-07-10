#%% 
import joblib
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib as plt
import time
import training as tr
import plotly.express as px
#%%

model = joblib.load(f"mnist_model.joblib")
#%%
activation = tr.select_activation(model["activation"])
loss = tr.select_cost(model["loss"])
endsizes = [x["W"].shape[0] for x in model["weights"]]
#%%

sizes = [model["weights"][0]["W"].shape[1]] + endsizes
netw = tr.create_network(sizes, activation, loss)

for a, b in zip(netw.layers, model["weights"]):
    a.W = b["W"]
    a.b = b["b"]

# %%

nor, ohc = joblib.load("Normalizer and encoder.joblib")
# %%

mnist = fetch_openml('mnist_784', version=1)

# %%
x, y = mnist['data'], mnist['target']

x_nor = nor.transform(x)
y = ohc.transform(np.array(y).reshape(-1,1))

# %%


nsample = np.random.randint(0,x.shape[0])
im =np.array(x.iloc[nsample,:]).reshape(28,28)
prediction = netw.output(x_nor[nsample,:].reshape(1,-1)) 
npred = np.argmax(prediction)
real = np.argmax(y[nsample])
px.imshow(im, color_continuous_scale="gray",title=f"Real: {real} Predicted: {npred}")




# %%
