#%%

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


def normalize(x):
    mean = np.mean(x, axis=1)
    sd = np.std(x, axis = 1)
    return (x - mean)/sd


#%%



def main():
	if (len(sys.argv)  != 2):
		print("Wrong number or arguments")
		exit(1)
	else:
		try:
			df = pd.read_csv(sys.argv[1])
		except BaseException as e:
			print("Something wrong happened when opening the file")
			print(e)
			exit(1)
	y= df.iloc[:,[1]].copy()
	enc = OneHotEncoder(drop="first",sparse_output=False)
	std = StandardScaler()
	
	y = enc.fit_transform(y)
	df2 = df.drop(columns=df.columns[:2])
	#df2 = normalize(df2)


	X_train, X_cv, y_train, y_cv = train_test_split(df2, y, test_size= 0.2, random_state= 42)
	
	X_train = std.fit_transform(X_train)
	X_cv = std.transform(X_cv)

	try:
		pd.DataFrame(X_train).to_csv("Training_data.csv", index = False)
		pd.DataFrame(X_cv).to_csv("Validation_data.csv", index = False)
		pd.DataFrame(y_train).to_csv("Training_target.csv",index = False)
		pd.DataFrame(y_cv).to_csv("Validation_target.csv", index = False)
		joblib.dump(std,"standard_scaler.joblib")
	except BaseException as e:
		print("Something went wrong saving the files")
		exit(1)
	

if __name__ == "__main__":
	main()