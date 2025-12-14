python3 preprocesing.py data/data.csv 
python3 training.py -la 30 30 -e 250 -lo "CrossEntropy"  -lr 0.001 -f data -a "Relu"  -m "MO"  -b 5 -l2 0.01 -s
python3 predict.py -m data -f ./data/data.csv