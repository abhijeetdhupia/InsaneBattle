## Directory Structure 

```
$root
├── Part 1 
├── Part 2 
├── LICENSE 
├── README.md
├── requirements.txt

```

## Steps for Part 1 
1. Make sure you have all the required libraries installed from `requirements.txt`. 
2. Create a directory called `data` and put the `training.json` and `test.json` files in it.
3. Run `data_prep.py` to create the `train_data.csv`, `train_lables.csv` and `test_data.csv` files.
4. Run `sklearn_models.py` to create the `sklearn_models_pred.csv` file which contains the predictions of Random Forest and K Nearesr Neighbours models.
5. Run `main.py` to create the `final_pred.csv` file which contains the predictions of a coustom designed model.

## Steps for Part 2: Q1 and Q2
1. Make sure you have all the required libraries installed from `requirements.txt`.
2. Run `q12.py` to create the `users.csv` file which contains the Updated EOL ratings of the synthetic users. 

## Steps for Part 2: Q3 
1. Make sure you have all the required libraries installed from `requirements.txt`.
2. Run `app.py` flask file and go to `localhost:5000` to see the app.
3. Fill the form and submit it.
4. Updated User and AI ratings will be displayed on the page. 