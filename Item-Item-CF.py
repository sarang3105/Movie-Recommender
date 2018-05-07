

import numpy as np
import sys
import findspark
findspark.init()
from pyspark import SparkContext
from collections import defaultdict


# Making movie, item pairs 
#this function returns the movie pairs from the data set
#which are used to calculate similarity between movies 
#by the cosine similarity function

def moviePairs(userId, movies):
    for x,y in zip(*[iter(movies)]*2):
        if(len(movies)>1):
            return [(x[0],y[0]),(x[1],y[1])]

#This function returns the top 100 neighbors for each movie
def topKMoviesPerUser(movie, similarity_values, n=100):
    similarity_values= sorted(similarity_values, key= lambda x: x[1][0],reverse=True)
    return (movie, similarity_values[:n])

#this function creates key values pairs with one movie as key and other as 
#value along with its similarity and total occurences of the pair in the dataset
def movie_groups(movie_pair, similarity_size):
    sim_value= similarity_size[0]
    size=similarity_size[1]
    movie1= movie_pair[0]
    movie2= movie_pair[1]
    return movie1,(movie2,(sim_value,size))

#This function broadcasts the similarity of each movie with other 
#similar movies in the form of a dictionary
#with a movie as a key and its similar items as values

def retrieve_similarity(context, top_neighborsRDD ):
    movie_similarity= top_neighborsRDD.collect()
    movie_neighbor_dict= {}
    for (movie, similarity) in movie_similarity:
        movie_neighbor_dict[movie]= similarity
    movie_neighbors= context.broadcast(movie_neighbor_dict)
    return movie_neighbors
    

#This function calculates the Cosine-Similiarity by checking the movie-pairs recieved from the movie-pair funtion
def CosineSimilarity(movie_pairs, rating_pairs):
    xy_sum=0.0
    xx=0.0
    yy=0.0
    length=0
    for rating_value in rating_pairs:
        xy_sum+=np.dot(np.float(rating_value[0]),np.float(rating_value[1]))
        xx+= np.dot(np.float(rating_value[0]),np.float(rating_value[0]))
        yy+= np.dot(np.float(rating_value[1]),np.float(rating_value[1]))
        length+=1
    denominator=np.dot(np.sqrt(xx),np.sqrt(yy))
    
    if denominator!=0:
        cosineSimilarity= xy_sum/denominator
    else:
        cosineSimilarity=0.0
        
    return movie_pairs,(cosineSimilarity,length)
    

#This function calculates the Weighted Sum and predicts the mrating ffor other movies

def recommendation_weightedSum(userID, movie_ratings, similarity_size, n=30):
    weighted_sum_dict= defaultdict(float)
    sim_sum_dict = defaultdict(float)
    
    for (movies,rating) in movie_ratings:
        rate= float(rating)
        neighbors = similarity_size.get(movies, None)
        if neighbors:
            for (movie,(similarity, size)) in neighbors:
                item = int(movie)
                sim= float(similarity)
                
                if item != movies:
                    weighted_sum_dict[item] += sim * rate
                    sim_sum_dict[item] += sim
             
    movie_score= [(weight/ sim_sum_dict[movie], movie) for movie, weight in weighted_sum_dict.items()]
    movie_score.sort(reverse=True)
    return (userID, movie_score[:n])  

#this function returns the movie names dict for the corresponding movie ids present in the moviesRDD

def final_movies(sc, movieRDD, recommendedRDD):
    movie_list= movieRDD.collect()
    
    recommended= recommendedRDD.collect()
    movie_dict={}
    for movieId in recommended:
        for i in movie_list:
            if movieId==int(i[0]):
                movie_dict[movieId]=i[1]
    return movie_dict
        

#Spark Code

if __name__ == "__main__":
  if len(sys.argv) !=4:
    print >> sys.stderr, "Usage: <arg1:Rating.csv> <arg2:Movie.csv> <arg3:userId>"
    exit(-1)

sc = SparkContext(appName="Item-Item-CF")
#rating file from argument 1
rating_file= sc.textFile(sys.argv[1])
#movies file from argument 2
movies_file = sc.textFile(sys.argv[2])
#userId for whom recommendations are needed
userId= str(sys.argv[3])

#filter for removing index names from ratings.csv
header_rating=rating_file.first()
rating_data= rating_file.filter(lambda y: y!=header_rating )

#creating the RDD for ratings which has format (userID,(movieID, rating))
ratingRDD= rating_data.map(lambda line: line.split(','))\
                        .map(lambda r: (r[0],(r[1],r[2]))).groupByKey().cache()

#filter for removing index names from movies.csv
header_movies= movies_file.first()
movies_data=movies_file.filter(lambda m: m!=header_movies )

#creating RDD for movies which has format (movieID, movieName)
moviesRDD= movies_data.map(lambda line: line.split(','))\
                        .map(lambda m: (m[0],m[1])).cache()

#Splitting dataset into training=80% and test=20%
trainingRDD, testRDD= ratingRDD.randomSplit([8,2])

#call to moviePair function which returns movie pairs of format((movie-1, movie-2),(rating1,rating2))
movie_pairRDD=  trainingRDD.map(lambda x: moviePairs(x[0],x[1])).groupByKey().cache()


#map for calculating cosine similarity
similarityRDD=movie_pairRDD.map(lambda y:CosineSimilarity(y[0],y[1])).cache()

#forming movie groups by making one movie as key and other as value
movie_similarityRDD= similarityRDD.map(lambda x: movie_groups(x[0],x[1])).groupByKey().cache()


#getting top 100 neighbors 
top_neighbors= movie_similarityRDD.map(lambda x: topKMoviesPerUser(x[0],x[1]))


#call to function which returns top movies neihbors as dictionary
sim_values= retrieve_similarity(sc, top_neighbors)


#map to calculate the weighted sum
recommendations= trainingRDD.map(lambda x: recommendation_weightedSum(x[0],x[1],sim_values.value)).cache()
recommendations.collect()

#filtering recommendations for the user ID 
user_recommendations = sc.parallelize(recommendations.filter(lambda x:x[0]==userId).values().collect()[0])
movieId_recommended = user_recommendations.map(lambda x:x[1])


testData= testRDD.map(lambda c : (c[0],c[1])).cache()

recommendations= testData.map(lambda x: recommendation_weightedSum(x[0],x[1],sim_values.value,100)).cache()

#retrieving movie names
final = final_movies(sc, moviesRDD, movieId_recommended)

#printing final movie list
print 'The movies recommended for UserID',userId, 'are:'
for x in final:
    print x ,":", final[x]


