python3 preprocesing.py data/data.csv 
python3 training.py -la 16 8 -e 10000 -lo "CrossEntropy"  -lr 0.005 -f data -a "LeakyRelu"  -m "MO"  
python3 predict.py -m data -f ./data/data.csv