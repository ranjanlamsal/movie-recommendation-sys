from django.shortcuts import render,redirect,HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from .forms import AddRatingForm
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from .models import Movie,Rating
from django.contrib import messages
import os,csv
from django.conf import settings
from .cosine import generatecosineRecommendation
from .pearson import generatepearsonRecommendation
from django.db.models import Q


# Create your views here.
def filterMovieByTitle():
    # Filtering by titles alphabetically
    allMovies = []
   
    titles_movie = Movie.objects.values('title', 'id')
    titles = {item["title"] for item in titles_movie}

    for title in titles:
        movies = Movie.objects.filter(title=title)
        print(movies)

        n = len(movies)
        nSlides = n // 4 + (n%4>0)
        allMovies.append([movies, range(1, nSlides), nSlides])

    params = {'allMovies': allMovies}
    return params

def filterMovieByGenre():
     #filtering by genres
    allMovies=[]
    genresMovie= Movie.objects.values('genres', 'id')
    genres= {item["genres"] for item in genresMovie}
    for genre in genres:
        movie=Movie.objects.filter(genres=genre)
        print(movie)
        n = len(movie)
        nSlides = n // 4 + (n % 4 > 0)
        allMovies.append([movie, range(1, nSlides), nSlides])
    params={'allMovies':allMovies }
    return params

 
def home(request):
    params=filterMovieByTitle()
    return render(request,'home.html',params)

@login_required(login_url='/login/')
def recommend(request):
    cosine_recommended_movies = generatecosineRecommendation(request)
    pearson_recommended_movies = generatepearsonRecommendation(request)
    if cosine_recommended_movies is None:
        cosine_recommended_movies = []

    if pearson_recommended_movies is None:
        pearson_recommended_movies = []
    combined_recommendations = [movie for movie in cosine_recommended_movies if movie in pearson_recommended_movies]

    params = {'cosine_recommendations': cosine_recommended_movies, 'pearson_recommendations': pearson_recommended_movies,'combined_recommendations': combined_recommendations}
   
    return render(request, 'recommend.html', params)


def register(request):
    if request.method =="POST":
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        password = request.POST.get('password')

        if not (first_name and last_name and username and password):
            messages.error(request, "All fields must be filled.")
            return redirect('/register/')

       
        if User.objects.filter(username=username).exists():
            messages.info(request, "Username already taken.")
            return redirect('/register/')

       
        user = User.objects.create(
            first_name=first_name,
            last_name=last_name,
            username=username,
        )
        user.set_password(password)
        user.save()

        file_path = 'C:/Users/amrit/Desktop/final8/rate.csv'
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                csv_user_id = row['userId']
                if csv_user_id == str(user.pk): 
                    movie_id = row['movieId']
                    rating_value = row['rating']
                    try:
                        movie = Movie.objects.get(pk=movie_id)
                    except Movie.DoesNotExist:
                        print(f"Movie with ID {movie_id} does not exist. Skipping rating import for this movie.")
                        continue

                    if Rating.objects.filter(user=user, movie=movie).exists():
                        print(f"Rating for user {user.pk} and movie {movie_id} already exists. Skipping duplicate.")
                        continue

                    # Create rating object
                    Rating.objects.create(user=user, movie=movie, rating=rating_value)

        messages.info(request, "Account created successfully with ratings imported if available.")
        return redirect('/register/')

    return render(request, 'register.html')
    


def login_page(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_superuser:
                messages.error(request, 'You are trying to login as an admin. Please use the admin login page.')
            else:
                login(request, user)
                return redirect('/home/')  
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'login.html')

def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_superuser:
                login(request, user)
                return redirect('dashboard')
            else:
                messages.error(request, 'You are trying to login as a user. Please use the user login page.')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'admin_login.html')


@login_required(login_url='/adminlogin/')
@staff_member_required
def addmovie(request):
    if request.method == 'POST':
        # Retrieve form data
        title = request.POST.get('title')
        genres = request.POST.get('genres')
        

        # Validate form data
        if not (title and genres):
            messages.error(request, "All fields must be filled.")
            return redirect('/addmovie/')  # Redirect to the same page if validation fails
        
        if Movie.objects.filter(title=title,genres=genres).exists():
            messages.info(request, "Movie already exists.")
            return redirect('/addmovie/')
        
        # Create a new movie instance
        Movie.objects.create(
            title=title,
            genres=genres,
        
            )

        messages.success(request, "Movie added successfully.")
        return redirect('/addmovie/')  # Redirect to the same page after successful submission

    genres = set()
    with open('C:/Users/amrit/Desktop/final8/movie.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            movie_genres = row[2]  # Assuming genres are in the third column (index 2)
            genres.add(movie_genres)

    return render(request, 'addmovie.html', {'genres': genres})



def import_movies_from_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            title = row.get('title', '')  # Use get() to avoid KeyError
            genres = row.get('genres', '')

            # Check if the movie already exists before adding
            if not Movie.objects.filter(title=title).exists():
                Movie.objects.create(
                    title=title,
                    genres=genres,
                )

# Call the import_movies_from_csv function only once during initialization
csv_file_path = os.path.join(settings.BASE_DIR, 'C:/Users/amrit/Desktop/final8/movie.csv')
import_movies_from_csv(csv_file_path)


def import_ratings_from_csv(file_path):
    # Open the CSV file
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Get user object if it exists
            user_id = row['userId']
            movie_id = row['movieId']
            rating_value = row['rating']

            try:
                user = User.objects.get(pk=user_id)
            except User.DoesNotExist:
                # Handle the case where user doesn't exist
                print(f"User with ID {user_id} does not exist. Skipping rating import for this user.")
                continue

            try:
                movie = Movie.objects.get(pk=movie_id)
            except Movie.DoesNotExist:
                # Handle the case where movie doesn't exist
                print(f"Movie with ID {movie_id} does not exist. Skipping rating import for this movie.")
                continue

            # Check if the rating already exists for this user-movie combination
            if Rating.objects.filter(user=user, movie=movie).exists():
                print(f"Rating for user {user_id} and movie {movie_id} already exists. Skipping duplicate.")
                continue

            # Create rating object
            Rating.objects.create(user=user, movie=movie, rating=rating_value)

# Call the function with the path to your CSV file
import_ratings_from_csv('C:/Users/amrit/Desktop/final8/rate.csv')





@login_required(login_url='/login/')
def dashboard(request):
    if request.user.is_authenticated:
        params = filterMovieByGenre()
        params['user'] = request.user
        search_query = request.GET.get('search', None)

        if search_query:
            # If there's a search query, filter movies by title containing the search query
            movies = Movie.objects.filter(Q(title__icontains=search_query))
        else:
            # If no search query, display all movies
            movies = Movie.objects.all()

        # Group movies by their genres
        grouped_movies = {}
        for movie in movies:
            genres = movie.genres.split(',')
            for genre in genres:
                if genre.strip() not in grouped_movies:
                    grouped_movies[genre.strip()] = []
                grouped_movies[genre.strip()].append(movie)

        params['grouped_movies'] = grouped_movies

        if request.method == 'POST':
            userid = request.POST.get('userid')
            movieid = request.POST.get('movieid')
            u = User.objects.get(pk=userid)
            m = Movie.objects.get(pk=movieid)
            rfm = AddRatingForm(request.POST)
            params['rform'] = rfm
            if rfm.is_valid():
                rat = rfm.cleaned_data['rating']
                count = Rating.objects.filter(user=u, movie=m).count()
                if count > 0:
                    messages.warning(request, 'Your review has already been submitted!!')
                else:
                    action = Rating(user=u, movie=m, rating=rat)
                    action.save()
                    messages.success(request, 'You have submitted' + ' ' + rat + ' ' + "star!")
                return redirect('/dashboard/')

        else:
            # Render the dashboard template
            rfm = AddRatingForm()
            params['rform'] = rfm

        return render(request, 'dashboard.html', params)
    else:
        return redirect('/login/')



def logout_page(request):
    logout(request)
    return redirect('/home/')




def confusion_matrix_cosine(request):
    # Open the image file
    with open('cosine_confusion_matrix.png', 'rb') as f:
        # Read the image content
        image_data = f.read()
    # Return the image data as HttpResponse with the appropriate content type
    return HttpResponse(image_data, content_type='image/png')


def confusion_matrix_pearson(request):
    with open('pearson_confusion_matrix.png', 'rb') as f:
        image_data = f.read()
    return HttpResponse(image_data, content_type='image/png')






