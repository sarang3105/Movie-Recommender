import numpy as np
import sys
from pyspark import SparkContext


#Function to calculate the mean of all the ratings of a user
def cal_mean(user_id,movie_info):
    ratings = movie_info
    number_of_ratings = len(movie_info)
    sum_of_ratings = 0.0
    if(number_of_ratings == 0):
        return (user_id, 0.0)
    else:
        for i in ratings:
            sum_of_ratings += i[1]
    return (user_id, sum_of_ratings/number_of_ratings)

#Function to find the pearson-corelation coefficient
def cal_pear(user_rating,user_mean_rating):
    movie_info_1 = user_rating[0][1]                 # movie info of the first user
    movie_info_2 = user_rating[1][1]                 # movie info of the second user
    movies_seen_by_neighbour = user_rating[1][1]     # movies that are seen by the neighbour and will be used to find the common movies between a user and its neigbour
    mean_rating_1 = user_mean_rating[0][1]           # will store the mean rating of the user
    mean_rating_2 = user_mean_rating[1][1]           # will store the mean rating of the neighbour of the user with whom the Karl-pearson ratio has to be calculated
    user_id = (user_rating[0][0],user_rating[1][0])  # will store the user id of the user and its neighbour
    prod = 0.0
    sum_difference_1 = 0.0
    sum_difference_2 = 0.0
    cnt=0                                            # will be used to keep the count of the common movies between a user and its neighbour
    for(i,j) in movie_info_1:                        # iterating over the movies_rated and ratings given by the user
        for (x,y) in movie_info_2:                   # iterating over the movies_rated and ratings given by the neighbour of the user
            if (i==x):                               # condition to check for the common movies between a user and its neighbour
                prod += ((j-mean_rating_1)*(y-mean_rating_2))
                sum_difference_1 += (j-mean_rating_1)**2
                sum_difference_2 += (y-mean_rating_2)**2
                cnt+=1
            else:
                continue
    Numerator = prod
    denom = (np.sqrt(sum_difference_1))*(np.sqrt(sum_difference_2))
    karl_coeff = np.divide(Numerator,denom)          # Calculating the co-efficient by dividing the numertor by the denominator
    return (user_id,(karl_coeff,cnt))                # The function is returning the user_id of the user and its neighbour alongwith their karl co-efficient and the number of movies they have rated in common
        
#Finding the top 100 neighbours of each user
def finding_top_neighbours(user_id, neighbour_info):
    sorted_neighbours = sorted(neighbour_info, key=lambda x:x[1], reverse=True)[:100]  #sorting the top 100 neighbours of the user on the basis of their karl-prason co-efficient value
    return (user_id, sorted_neighbours)     # returning the value in the form (x,[]) where x is the user_id and the list [] will contain its neighbour, their karl-coefficeint and the number of common movies

from collections import defaultdict         

'''This function will be used to calculate the recommendation that will be made to the user 
the input to this function is user , its neighbour , the movie list of the user and the number of movies that will be recommended
in this case it is 20'''
def recommendations(user, neighbors, user_movielist, n=20):
    sim= defaultdict(float)   # Initialising the sim dictionary which will be used to store the similarity of the user 
    weight= defaultdict(float) # This dictionary will be used to store the weighted sum
    user_movies= user_movielist.get(user,[])  #movies of the current user
    for (userId,similarity, size) in neighbors:  #iterating over the user_id of the neighbour, karl-coefficient of it and the number of common movies
        if (size>8):                             
            sim_user= user_movielist.get(userId,None) #storing the mmovies of the neighbour
            if sim_user:
                for (mov,rate) in sim_user:
                    if mov not in user_movies:
                        sim[mov]+=similarity
                        weight[mov]+= similarity*rate    
    user_val= [(movieId,weights/sim[movieId]) for (movieId,weights) in weight.items()]
    user_val.sort(key=lambda x:x[1], reverse=True)  #Sorting the movies in the descending order of their similarity values
    return (user, user_val[:n])  #Returning the top 20 movies

# This function will give the names of the recommended movies to the user
def movies_recommended(user,movie_list,all_movies):
    movie_name = []
    for (x,y) in movie_list:
        movie_name.append(all_movies[x])
    return (user, movie_name)

if __name__ == "__main__":
  if len(sys.argv) !=4:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)
  sc = SparkContext(appName="User-User")             

movies_data = sc.textFile(sys.argv[2])
movies_data_header = movies_data.take(1)[0] 
movies_rating_data = movies_data.filter(lambda x: x!=movies_data_header)\
.map(lambda x:x.split(",")).map(lambda x:(int(x[0]),x[1]))


user_data = sc.textFile(sys.argv[1])
usrnum=int(sys.argv[3])
user_data_header = user_data.first()
user_rating_data = user_data.filter(lambda x: x!=user_data_header)\
.map(lambda x:x.split(",")).map(lambda x: (int(x[0]),int(x[1]),float(x[2])))
    
#Splitting the ratings data set into a training and testing dataset
    
Train_Data, Test_Data = user_rating_data.randomSplit([8,2])
    
'''Formatting the data-set in the required format (x,[]) where x is the user_id
and the list [] will contain the movies_rated by the user and the rating given to the user'''
Test_Data_values = Test_Data.map(lambda x:(x[0],(x[1],x[2]))).groupByKey().map(lambda x:(x[0],list(x[1])))
Train_Data_values = Train_Data.map(lambda x:(x[0],(x[1],x[2]))).groupByKey().map(lambda x:(x[0],list(x[1])))

new_dict={}   # Taking the new dictionary to store the movies and rating of the user which will be broadcasted
for user,value in Train_Data_values.collect():  
    new_dict[user]= value    
    ib= sc.broadcast(new_dict)        #broadcasting the above used dictionary
        
# Below is the RDD which will contain the users and their mean-rating         
Train_Data_values_Mean = Train_Data_values.map(lambda x: cal_mean(x[0],x[1])).cache()
    
# Taking the cartesian product to form the pairs 
movies_list = Train_Data_values.cartesian(Train_Data_values)
    
'''Filtering out the above pairs by applying the condition x[0]<x[1] which will filter out the elements 
(for simplicity: it will give all the elements below the diagonal of the matrix)'''
movies_pair = movies_list.filter(lambda x: x[0]<x[1])
    
# Taking the cartesian product yet again to form the pairs of the mean_Rating this time
mean_rating_list = Train_Data_values_Mean.cartesian(Train_Data_values_Mean)

'''Filtering out the above pairs by applying the condition x[0]<x[1] which will filter out the elements 
(for simplicity: it will give all the elements below the diagonal of the matrix)''' 
mean_rating_pair = mean_rating_list.filter(lambda x:x[0]<x[1])
    
'''Zipping the elements so that we get the tuple of the form (((x,[]),(y,[])),((x,mean_x),(y,mean_y))) 
We'll send the above tuple in the cal_pear (function which will calulate the karl pearson coefficient) 
to calculate the co-efficient of the user and its neighbour'''
pair_RDD = movies_pair.zip(mean_rating_pair)

# The below RDD will store the karl pearson coefficient of the user and its neighbour
Similar_RDD = pair_RDD.map(lambda x: cal_pear(x[0],x[1])).cache()

'''Filtering out and formating the above RDD in the form (x,(y,co-efficeint,count)) where x is the user , y is its neighbour 
co-efficient is the karl pearson co-efficient and count is the number of movies they have in common'''
Similar_RDD_filter = Similar_RDD.filter(lambda x:(x[1][0]<1 and x[1][0]>0)).map(lambda x:(x[0][0],(x[0][1],x[1][0],x[1][1])))\
.groupByKey().sortByKey().mapValues(list)

#Below RDD contains the top 100 neighbour of the user
Top_Neighbours_RDD = Similar_RDD_filter.map(lambda x: finding_top_neighbours(x[0],x[1]))

#Below RDD contains the movies that are re-commended to the user , this RDD contains the id of the movie and the values of the similarity
final_recommendations = Top_Neighbours_RDD.map(lambda x:recommendations(x[0],x[1],ib.value)).cache()

#Taking a new dictionary to store the name of the movies based on their movie_id
new_dict1={}
for user,value in movies_rating_data.collect():
    new_dict1[user]= value.encode('ascii', 'ignore')   #using the .encode function to conver the unicode movies names to string
    ib1= sc.broadcast(new_dict1)  #broadcasting 
    

#Below RDD contains the name of the movies that are recommended
movies_recommended =  final_recommendations.map(lambda x:movies_recommended(x[0],x[1],ib1.value)).cache()

#Gettting the recommendation for the specific user in this case it is user 13
specific_user_recommendation = movies_recommended.filter(lambda x:x[0]==usrnum)
print (specific_user_recommendation.collect())
