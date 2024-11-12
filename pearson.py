import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import sqrt
import numpy as np
from django.contrib.auth.models import User
from .models import Movie, Rating

def generatepearsonRecommendation(request):
    movie = Movie.objects.all()
    rating = Rating.objects.all()
    x = []
    y = []
    A = []
    B = []
    C = []
    D = []

    # Movie Data Frames
    for item in movie:
        x = [item.id, item.title, item.genres]
        y += [x]
    movies_df = pd.DataFrame(y, columns=['movieId', 'title', 'genres'])

    # Rating Data Frames
    for item in rating:
        A = [item.user.id, item.movie.id, item.rating]
        B += [A]
    rating_df = pd.DataFrame(B, columns=['userId', 'movieId', 'rating'])
    rating_df['userId'] = rating_df['userId'].astype(np.int64)
    rating_df['movieId'] = rating_df['movieId'].astype(np.int64)
    rating_df['rating'] = rating_df['rating'].astype(np.float64)

    if request.user.is_authenticated:
        userid = request.user.id
        userInput = Rating.objects.select_related('movie').filter(user=userid)
        if userInput.count() == 0:
            recommenderQuery = None
            userInput = None
        else:
            for item in userInput:
                C = [item.movie.title, item.rating]
                D += [C]
            inputMovies = pd.DataFrame(D, columns=['title', 'rating'])
            inputMovies['rating'] = inputMovies['rating'].astype(np.float64)

            # Filtering out the movies by title
            inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
            inputMovies = pd.merge(inputId, inputMovies, on='title', how='inner')

            inputMovies = inputMovies[['movieId', 'title', 'genres', 'rating']]
            inputMovies['movieId'] = inputMovies['movieId'].astype(np.int64)

            # Filtering out users that have watched movies that the input has watched and storing it
            userSubset = rating_df[rating_df['movieId'].isin(inputMovies['movieId'].tolist())]
            print("pearson=")
            print(userSubset.head())

            # Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
            userSubsetGroup = userSubset.groupby(['userId'])

            # Sorting it so users with movie most in common with the input will have priority
            userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)
            print(userSubsetGroup[0:2])
            userSubsetGroup = userSubsetGroup[0:10]

            # Store the Pearson Correlation in a dictionary
            pearsonCorrelationDict = {}

            # For every user in subsetgroup
            for name, group in userSubsetGroup:
                group = group.sort_values(by='movieId')
                inputMovies = inputMovies.sort_values(by='movieId')
                nRatings = len(group)
                temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
                tempRatingList = temp_df['rating'].astype(float).tolist()
                tempGroupList = group['rating'].astype(float).tolist()

                Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
                Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
                Sxy = sum(i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(tempGroupList) / float(nRatings)

                if Sxx != 0 and Syy != 0:
                    pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
                else:
                    pearsonCorrelationDict[name] = 0

            pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
            pearsonDF.columns = ['similarityIndex']
            pearsonDF['userId'] = pearsonDF.index
            pearsonDF.index = range(len(pearsonDF))
            print("pearsondf=")
            print(pearsonDF.head())
            topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
            print(topUsers.head())
            topUsers['userId'] = topUsers['userId'].apply(lambda x: x[0] if isinstance(x, tuple) else x).astype(np.int64)

            topUsersRating = pd.merge(topUsers, rating_df, left_on='userId', right_on='userId', how='inner')
            topUsersRating['movieId'] = topUsersRating['movieId'].astype(np.int64)

            topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
            print("topUsersRating:")
            print(topUsersRating.head())

            tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
            tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
            print("temp=")
            print(tempTopUsersRating.head())
            recommendation_df = pd.DataFrame()
            recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating'] / tempTopUsersRating['sum_similarityIndex']
            recommendation_df['movieId'] = tempTopUsersRating.index
            print(recommendation_df.head(10))

            recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
            recommender = movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(35)['movieId'].tolist())]
            recommender1 = movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
            
            recommended_movies = recommender1.to_dict('records')
            print(recommender1)

            recommended_movies1 = recommender.to_dict('records')
            actual_movies = inputMovies['movieId'].tolist()
            recommended_movie_ids = [movie['movieId'] for movie in recommended_movies1]

            # Precision
            precision = len(set(recommended_movie_ids) & set(actual_movies)) / len(recommended_movie_ids)
            # Recall
            recall = len(set(recommended_movie_ids) & set(actual_movies)) / len(actual_movies)
            # Accuracy
            accuracy = len(set(recommended_movie_ids) & set(actual_movies)) / len(set(recommended_movie_ids) | set(actual_movies))
            print("pearson=")
            print("Precision:", precision)
            print("Recall:", recall)
            print("Accuracy:", accuracy)

            generate_confusion_matrix_graph(precision, recall, accuracy)
            

            return recommended_movies

def generate_confusion_matrix_graph(precision, recall, accuracy):
    confusion_matrix_data = [[precision, 1-precision], [1-recall, recall]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_data, annot=True, cmap='Reds', fmt='.2f', xticklabels=['Actual Negative', 'Actual Positive'], yticklabels=['Predicted Negative', 'Predicted Positive'])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix of pearson correlation (Accuracy: {:.2f})'.format(accuracy))
    plt.tight_layout()
    plt.savefig('pearson_confusion_matrix.png')
    plt.close()
