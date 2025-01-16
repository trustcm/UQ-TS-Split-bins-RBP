# UQ-TS-Split-bins-rbp
Perform UQ and TS on RBP binding site prediction and conduct bin sample screening.

## System requirement
```
python==3.8.16
torch==1.13.1+cu116
torch-geometric==2.5.3
torch-cluster==1.6.1+pt113cu116
torch-scatter==2.1.1+pt113cu116
torch-spline-conv==1.2.2+pt113cu116
fair-esm==2.0.0
```
You need to pay attention to the packages installed during the environment configuration:
```
pip install fair-esm
```
# Usage

## Train&Test  
1)Feature-based construction of protein graph datasets using ProteinGraphDataset;  
when predicting PDB files using ESMfold, please refer to https://github.com/biomed-AI/nucleic-acid-binding


2)Train model:
```
run .../endtrain.py
```

3)valid model for obtaining the temperature scaling parameter T value and the interval for sample screening.
It should be noted that the validation set should be consistent with that during training when validating.
```
run .../endvalid.py
```

4)Test model(uncal)
```
run .../prediction.py
```

5)Test model(TS)
```
run .../prediction_TS.py
```
