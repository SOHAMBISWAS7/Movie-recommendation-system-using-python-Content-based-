import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#important_functions
def combine_features(row):#funtion for returning string of these features
    return row['keywords'] + " " +row['cast']+" "+row["genres"]+" "+row["director"]
def titlefinder(index):#function for title from index
    return datafile[datafile.index == index]["title"].values[0]
def indexfinder(title):#function for index from title
    return datafile[datafile.title == title]["index"].values[0]


datafile = pd.read_csv("moviedata.csv")#read file

features = ['keywords','cast','genres','director']#features based on which filtering is dine

for feature in features:#filing nil values with space
    datafile[feature] = datafile[feature].fillna('')

datafile["combined_features"] = datafile.apply(combine_features, axis=1)#combine all features in single column "combinde featues"

cv = CountVectorizer()#initialize vector function
count_matrix = cv.fit_transform(datafile["combined_features"])#using countvectorize and fit transform,we can find the distinct words
#in the lines of text and also the count of each word in the line.
cosine_sim = cosine_similarity(count_matrix)#used for finding the similarity btn the lines based on angle separation btn vectors

movie_user_likes = input("Enter movie of your chice:")
movie_index = indexfinder(movie_user_likes)
similar_movies =  list(enumerate(cosine_sim[movie_index]))

sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
i=0
print("Top 5 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
    print(titlefinder(element[0]))
    i=i+1
    if i>=5:
        break