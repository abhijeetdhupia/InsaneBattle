"""
Fit Battle: User v/s AI Player 

Fit Battle - A fitness battle where the user is matched to an AI player.
Both the user and AI player do as many jumping jacks as possible in 5 minutes. 
Whoever has the higher rep count is the winner. Depending on the result, ELO Ratings of the user and AI player 
decreases, increases or stays the same.

ELO Update algorithm:

P1: Probability of winning of player with rating2
P2: Probability of winning of player with rating1.
P1 = (1.0 / (1.0 + pow(10, ((rating1 – rating2) / 400))));
P2 = (1.0 / (1.0 + pow(10, ((rating2 – rating1) / 400))));
Obviously, P1 + P2 = 1.
The rating of player is updated using the formula given below :-
rating1 = rating1 + K*(Actual Score – Expected score);
Take K = 64.  Actual score: Win - 1, Lose - 0, Tie - 0.5. 
"""

import numpy as np
import pandas as pd
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

# Q.1 Create a csv file with 100 synthetic users and their fitness levels between 20-90 

users = pd.DataFrame(np.random.randint(20,90,size=(100,1)), columns=['User_Rep_Count'])

# add empty columns for user_rating and ai_rating
users['User_Rating'] = 0
users['AI_Rating'] = 0


# Q.2 Simulate AI Battles and reach equilibrium ELO Ratings for each user in users
# for all the users in users calculate EOL Rating of user and update the ratings
# of the user and AI player

user_rating = 1000
ai_rating = 1000
for i in range(len(users)):
    ai_rating = 1000
    # if user_rep_count is less than 35 then he is a beginner 
    # if user_rep_count is greater than 35 and less than 60 then he is an intermediate
    # if user_rep_count is greater than 35 and less than 80 then he is an advanced
    # if user_rep_count is greater than 80 then he is an elite
    user_rep_count = users.iloc[i,0]
    if user_rating < 25:
        level = 'beginner'
    elif user_rating < 40:
        level = 'intermediate'
    elif user_rating < 80:
        level = 'advanced'
    else:
        level = 'elite'
    new_user_rating, new_ai_rating = new_main(user_rating=user_rating,ai_rating=ai_rating, user_rep_count=user_rep_count, level=level)
    # if any values is negative make them 0 
    if new_user_rating < 0:
        new_user_rating = 1000
    if new_ai_rating < 0:
        new_ai_rating = 1000

    user_rating, ai_rating = new_user_rating, new_ai_rating
    users.iloc[i,1] = user_rating
    users.iloc[i,2] = ai_rating

# save the users dataframe to csv file
users.to_csv('./data/users.csv', sep=',', encoding='utf-8')