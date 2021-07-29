# create a flask app which takes info from the form 
# and returns a json response

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify

np.random.seed(123)

def check_result(user_rep_count, ai_rep_count):
    if user_rep_count > ai_rep_count:
        return 'win'
    elif user_rep_count < ai_rep_count:
        return 'lose'
    else:
        return 'tie'

def new_ELO(rating_p1, rating_p2, actual_score_p1):
    K = 64
    p1 = 1.0/(1.0 + pow(10, ((rating_p1 - rating_p2) / 400)))
    p2 = 1-p1
    # p2 = 1.0/(1.0 + pow(10, ((rating_p2 - rating_p1) / 400)))
    
    # assuming player 1 is the user and player 2 is the AI
    # Player 1 won the battle 
    if actual_score_p1 == 'win':
        rating_p1 = rating_p1 + K * (1 - p1)
        rating_p2 = rating_p2 + K * (0 - p2)
    
    # Player 1 lost the battle  
    elif actual_score_p1 == 'lose':
        rating_p1 = rating_p1 + K * (0 - p1)
        rating_p2 = rating_p2 + K * (1 - p2)
    # Battle Tied
    else:
        rating_p1 = rating_p1 + K * (0.5 - p1)
        rating_p2 = rating_p2 + K * (0.5 - p2)
    return rating_p1, rating_p2

def new_main(user_rating, ai_rating, user_rep_count, level):
    while True:
        level = level.lower()
        if level == 'beginner':
            ai_rep_count = 25
        elif level == 'intermediate':
            ai_rep_count = 40
        elif level == 'advanced':
            ai_rep_count = 50
        elif level == 'elite':
            ai_rep_count = 80
        else:
            print("Please enter a valid level")
        break
    
    result = check_result(user_rep_count, ai_rep_count)
    user_rating, ai_rating = new_ELO(user_rating, ai_rating, result)
    return int(user_rating), int(ai_rating)


# define the app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# get the data from the form
@app.route('/', methods=['POST'])
def data():
    user_rating = 1000
    ai_rating = 1000
    # get the data from the form
    name = request.form['name']
    rep_count = int(request.form['rep_count'])
    level = request.form['level']
    
    new_user_rating, new_ai_rating = new_main(user_rating, ai_rating, rep_count, level)
    # make a response dict
    response = {}
    response['Name'] = name
    response['Rep Count'] = rep_count
    response['Level'] = level
    response['User Rating']= new_user_rating
    response['AI Rating']= new_ai_rating

    # when the form is submitted, return the jsonified response
    return jsonify(response)

    # return render_template('index.html')

# run the app   
if __name__ == '__main__':
    app.run(debug=True)