#%%
import pandas as pd
import layer as ly
import numpy as np

#%%
t_data = pd.read_csv("Training_data.csv",header=None)
t_target = pd.read_csv("Training_target.csv", header=None)
v_data = pd.read_csv("Validation_data.csv", header=None)
v_target = pd.read_csv("Validation_target.csv", header=None)

# %%
t_data = np.array(t_data)
t_target = np.array(t_target)
t_target = np.append(t_target,1-t_target,axis=1)
v_data = np.array(v_data)
v_target = np.array(v_target)
v_target = np.append(v_target, 1-v_target)

#%%

red = ly.Network(input_size=t_data.shape[1],max_epoch=100000)
red.layer_append(ly.Layer(red.input_size,40,ly.sigmoid_act))
red.layer_append(ly.Layer(40,40,ly.sigmoid_act))
red.layer_append(ly.Layer(40,2,ly.sigmoid_act,ly.cross_entropy_cost))

#%%
red.train(t_data,t_target,val_input=v_data,val_observed=v_target)


# %%
x=red.output(t_data)



# %%
np.append(np.argmax(x,axis=1,keepdims=True),t_target[:,[1]],axis=1)
# %%
