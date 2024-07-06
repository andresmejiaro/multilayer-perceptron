python3 preprocesing.py data/data.csv 
python3 training.py -la 20 10 2 -e 100 -lo "CrossEntropy" -b 3 -lr 0.01 -f data -a "Sigmoid" -m "RMSProp" -s 
python3 predict.py -m data -f ./data/data.csv