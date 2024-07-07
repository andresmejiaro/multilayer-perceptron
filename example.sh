python3 preprocesing.py data/data.csv 
python3 training.py -la 30 30  2 -e 10000 -lo "CrossEntropy"  -lr 2 -f data -a "Sigmoid"  -s
python3 predict.py -m data -f ./data/data.csv