import pandas as pd
import numpy as np
from scipy import spatial
import operator

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('PATH_TO_DATASET/u.data', sep='', names=r_cols, usecols=range(3)) #here names sets the column names for the columsn loaded in the ratings variable ,USECOLS INSTRUCTS TO IMPORT ONLY THE FIRST 3 COLUMNS , THE RATINGS VARIABLE STORES THE DATASET IN PROPER COLUMNS , 
#ratings.head()  #this prints top few elements of ratings variable
movieProperties = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]}) #this will group the ratings of each moview as the number of movie views and the average rating of the movies 
#movieProperties.head()
movieNumRatings = pd.DataFrame(movieProperties['rating']['size']) #this command will normalize the movie data so that popularity is scaled on a scale of 0 to 1.
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))) # normalization done by - max rated #movie - min rated movie
#movieNormalizedNumRatings.head()
movieDict = {}
with open(r'PATH_TO_DATASET/u.item') as f:  #here u.item contains movie names and their genres.
    temp = ''
    for line in f:
        #line.decode("ISO-8859-1")
        fields = line.rstrip('\n').split('|')
        movieID = int(fields[0])  #extract movie ID
        name = fields[1]	##extract movie name	
        genres = fields[5:25]   ##extract movie genres
        genres = map(int, genres)
        movieDict[movieID] = (name, np.array(list(genres)), movieNormalizedNumRatings.loc[movieID].get('size'), movieProperties.loc[movieID].rating.get('mean'))  #make a movie dictionary

#now distance between 2 movies will be computed based on how similar their genres are.
def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB) #computes distances between genres of the 2 genre vectors
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB) # taking raw popularity difference between the 2 genres
    return genreDistance + popularityDistance 
    
ComputeDistance(movieDict[2], movieDict[4]) #calling the above function for movie no. 2 and movie. no. 4

#the distance will be between 0 and 1. More the distance, farther away the movies are.


def getNeighbors(movieID, K):   #this function defines neighbours for a specific value of K -  found out by using trial and error
    distances = []   #take a specific movie, compare its distances(both genre and popularity) with every other movie
    for movie in movieDict:
        if (movie != movieID):
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist))	            #distances added 
    distances.sort(key=operator.itemgetter(1))   		
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors

K = 11         #here I took K=11 owing to the dataset size  , k SHOULD BE ODD TO AVOID CONFLICT INCASE THE NEAREST NEIGHBOURS ARE IN SAME QUANTITY
avgRating = 0    #HERE THIS SHALL TAKE THE K TOP ONES FROM THE ABOVE FUNCTION AND USE THEM FOR EVALUATION
neighbors = getNeighbors(1, K)
for neighbor in neighbors:
    avgRating += movieDict[neighbor][3]   #set average rating to different movies. #COMPUTE AVERAGE RATING USING THE 10 NEIGHBOURS
    print (movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))   #
    
avgRating /= K



