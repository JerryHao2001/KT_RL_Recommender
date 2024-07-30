# AKT Code implemented by Context-Aware Attentive Knowledge Tracing

Retrieved from: https://github.com/arghosh/AKT

## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 1.2.0 
- Scikit-learn 0.21.3
- Scipy 1.3.1
- Numpy 1.17.2



## Running AKT.
Here are some examples for using AKT-Rasch model (on ASSISTments2009 and ASSISTments2017 datasets):
```
python main.py --dataset assist2009_pid --model akt_pid 
python main.py --dataset assist2017_pid --model akt_pid 
```

Here is an example for using AKT-NonRasch model (on all datasets):
```
python3 main.py --dataset assist2015 --model akt_cid
```

