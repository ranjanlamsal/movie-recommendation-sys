from django.contrib.auth.models import User
from .models import Movie,Rating
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def generatecosineRecommendation(request):
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
    print("Movies DataFrame")
    print(movies_df)
    print(movies_df.dtypes)

    # Rating Data Frames
    print(rating)
    for item in rating:
        A = [item.user.id, item.movie, item.rating]
        B += [A]
    rating_df = pd.DataFrame(B, columns=['userId', 'movieId', 'rating'])
    print("Rating data Frame")
    rating_df['userId'] = rating_df['userId'].astype(str).astype(np.int64)
    rating_df['movieId'] = rating_df['movieId'].astype(str).astype(np.int64)
    rating_df['rating'] = rating_df['rating'].astype(str).astype(np.float64)
    print(rating_df)
    print(rating_df.dtypes)

    if request.user.is_authenticated:
        userid = request.user.id
        # select_related is a join statement in Django. It looks for foreign key and joins the table
        userInput = Rating.objects.select_related('movie').filter(user=userid)
        if userInput.count() == 0:
            recommenderQuery = None
            userInput = None
        else:
            for item in userInput:
                C = [item.movie.title, item.rating]
                D += [C]
            inputMovies = pd.DataFrame(D, columns=['title', 'rating'])
            print("Watched Movies by user dataframe")
            inputMovies['rating'] = inputMovies['rating'].astype(str).astype(np.float64)
            print(inputMovies.dtypes)

            # Filtering out the movies by title
            inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
            # Then merging it so we can get the movieId. It's implicitly merging it by title.
            # Assuming 'movieId' is the column on which you are merging
            inputMovies = pd.merge(inputId, inputMovies, on='title', how='inner')

            inputMovies = inputMovies[['movieId', 'title', 'genres', 'rating']]
            # Convert 'movieId' to int64
            inputMovies['movieId'] = inputMovies['movieId'].astype(np.int64)
            print(inputMovies)

            # Filtering out users that have watched movies that the input has watched and storing it
            userSubset = rating_df[rating_df['movieId'].isin(inputMovies['movieId'].tolist())]
            print("cosine=")
            print(userSubset.head())

            # Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
            userSubsetGroup = userSubset.groupby(['userId'])

            # Sorting it so users with movie most in common with the input will have priority
            userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)

            print(userSubsetGroup[0:2])

            userSubsetGroup = userSubsetGroup[0:10]

            # Store the cosine similarity in a dictionary
            cosine_similarity_dict = {}

            # For every user group in our subset
            for name, group in userSubsetGroup:
                # Let's start by sorting the input and current user group so the values aren't mixed up later on
                group = group.sort_values(by='movieId')
                inputMovies = inputMovies.sort_values(by='movieId')
                # Get the N for the formula
                n_ratings = len(group)
                # Get the review scores for the movies that they both have in common
                temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
                # And then store them in a temporary buffer variable in a list format to facilitate future calculations
                temp_rating_list = temp_df['rating'].tolist()
                # Let's also put the current user group reviews in a list format
                temp_group_list = group['rating'].tolist()
                # Now let's calculate the cosine similarity between two users
                dot_product = sum(i * j for i, j in zip(temp_rating_list, temp_group_list))
                magnitude1 = sqrt(sum(i ** 2 for i in temp_rating_list))
                magnitude2 = sqrt(sum(i ** 2 for i in temp_group_list))
                similarity = dot_product / (magnitude1 * magnitude2) if magnitude1 != 0 and magnitude2 != 0 else 0

                # Store the similarity in the dictionary
                cosine_similarity_dict[name] = similarity

            print(cosine_similarity_dict.items())

            cosine_df = pd.DataFrame.from_dict(cosine_similarity_dict, orient='index')
            cosine_df.columns = ['similarityIndex']
            cosine_df['userId'] = cosine_df.index
            cosine_df.index = range(len(cosine_df))
            print("cosinedf=")
            print(cosine_df.head())

            top_users = cosine_df.sort_values(by='similarityIndex', ascending=False)[0:50]
            print(top_users.head())

            # Convert 'userId' to int64 in topUsers DataFrame
            top_users['userId'] = top_users['userId'].apply(lambda x: x[0] if isinstance(x, tuple) else x).astype(np.int64)

            top_users_rating = pd.merge(top_users, rating_df, left_on='userId', right_on='userId', how='inner')
            # Ensure 'movieId' is of type int64 in topUsersRating
            top_users_rating['movieId'] = top_users_rating['movieId'].astype(np.int64)

            print(top_users_rating.head())

            # Multiplies the similarity by the user's ratings
            top_users_rating['weightedRating'] = top_users_rating['similarityIndex'] * top_users_rating['rating']
            # Debugging prints
            print("topUsersRating:")
            print(top_users_rating.head())
            

            # Applies a sum to the topUsers after grouping it up by userId
            temp_top_users_rating = top_users_rating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
            temp_top_users_rating.columns = ['sum_similarityIndex', 'sum_weightedRating']
            print("temp=")
            print(temp_top_users_rating.head())

            # Creates an empty dataframe
            recommendation_df = pd.DataFrame()
            # Now we take the weighted average
            recommendation_df['weighted average recommendation score'] = temp_top_users_rating['sum_weightedRating'] / temp_top_users_rating['sum_similarityIndex']
            recommendation_df['movieId'] = temp_top_users_rating.index
            print(recommendation_df.head(10))
            recommendation_df.head()

            recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
            recommender = movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(35)['movieId'].tolist())]
            recommender1 = movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
            print(recommender1)
            recommended_movies = recommender1.to_dict('records')

            # Calculating precision, recall, and accuracy
            recommended_movies1 = recommender.to_dict('records')
            actual_movies = inputMovies['movieId'].tolist()
            recommended_movie_ids = [movie['movieId'] for movie in recommended_movies1]

            # Precision
            precision = len(set(recommended_movie_ids) & set(actual_movies)) / len(recommended_movie_ids)
            # Recall
            recall = len(set(recommended_movie_ids) & set(actual_movies)) / len(actual_movies)
            # Accuracy
            accuracy = len(set(recommended_movie_ids) & set(actual_movies)) / len(set(recommended_movie_ids) | set(actual_movies))
            print("cosine=")
            print("Precision:", precision)
            print("Recall:", recall)
            print("Accuracy:", accuracy)

            generate_confusion_matrix_graph(precision, recall, accuracy)
           
            return recommended_movies


def generate_confusion_matrix_graph(precision, recall, accuracy):
    # Confusion matrix data
    confusion_matrix_data = [[precision, 1-precision], [1-recall, recall]]

    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_data, annot=True, cmap='Blues', fmt='.2f', xticklabels=['Actual Negative', 'Actual Positive'], yticklabels=['Predicted Negative', 'Predicted Positive'])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix of cosine similarity (Accuracy: {:.2f})'.format(accuracy))
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig('cosine_confusion_matrix.png')
    plt.close()
