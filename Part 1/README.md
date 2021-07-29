## Directory Structure 

```
$root
├── data
├── data_prep.py
├── requirements.txt
├── sklearn_models.py

```

## Steps
1. Make sure you have all the required libraries installed from `requirements.txt`. 
2. Create a directory called `data` and put the `training.json` and `test.json` files in it.
3. Run `data_prep.py` to create the `train_data.csv`, `train_lables.csv` and `test_data.csv` files.
4. Run `sklearn_models.py` to create the `sklearn_models_pred.csv` file which contains the predictions of Random Forest and K Nearesr Neighbours models.
5. Run `main.py` to create the `final_pred.csv` file which contains the predictions of a coustom designed model.