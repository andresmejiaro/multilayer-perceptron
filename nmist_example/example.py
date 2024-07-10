#%%

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import layer as ly
import joblib
#%%

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


#%%
# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)

# Reshape the data
x, y = mnist['data'], mnist['target']

# %%
X_train, X_cv, y_train, y_cv = train_test_split(
        x, y, test_size=0.2)

# %%
ohc = OneHotEncoder(drop= None,sparse_output=False)

y_train2 = ohc.fit_transform(np.array(y_train).reshape(-1,1))
y_cv2 = ohc.transform(np.array(y_cv).reshape(-1,1))

# %%
X_train = np.array(X_train)
X_cv = np.array( X_cv)

# %%

nor =StandardScaler()
X_train = nor.fit_transform(X_train)
X_cv = nor.transform(X_cv)


#%%

red = create_network([784, 20, 20, 10],ly.leaky_relu,ly.cross_entropy_cost)



# %%

red.train(X_train,y_train2,val_input=X_cv, val_observed=y_cv2,learning_rate=0.1,batch_size=64,training_method="GD",gamma=0.99)
# %%
weights = [{"W": w.W, "b": w.b} for w in red.layers]
joblib.dump({"weights": weights, "loss": "CrossEntropy",
                "activation": "LeakyRelu"}, f"mnist_model.joblib")
# %%

joblib.dump([nor,ohc],"Normalizer and encoder.joblib")

# %%
