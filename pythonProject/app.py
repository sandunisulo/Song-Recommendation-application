import numpy as np
import pandas as pd
import seaborn as sb
from PIL._imaging import display
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dataset1 = pd.read_csv("dataset.csv")
dataset2 = dataset1.head(1000)
print(dataset2['track_name'].head(20))


#Data set large thus we get 1000 amount of data rows for our project
#dataset = dataset2.sort_values(by=['popularity'] , ascending= False).head(500)
#print(dataset['track_name'].head(20))

#Create vector with genres data
song_vector = CountVectorizer()
song_vector.fit(dataset2['track_genre'])


def get_similar(song_name, data):

    character_features = song_vector.transform(data[data['track_name']==song_name]['track_genre']).toarray()
    
    #print(data['track_name'].head(20))
    #print(data[data['track_name']==song_name])
    #print(character_features)
    numeric_features = data[data['track_name']==song_name].select_dtypes(include=np.number).to_numpy()

    #Create variable for store similarities
    sim = []

    #store other songs details according to our searching song details
    #iterating trough table
    for idx, row in data.iterrows():
        names = row['track_name']

        #getting all details of other songs
        character_features_others = song_vector.transform(data[data['track_name']==names]['track_genre']).toarray()
        #print(character_features_others)
        numeric_features_others = data[data['track_name']==names].select_dtypes(include=np.number).to_numpy()

        #calculate similarities
        character_features_similarity =  cosine_similarity(character_features,character_features_others)[0][0]
        numeric_features_similarity = cosine_similarity(numeric_features,numeric_features_others)[0][0]

        sim.append(character_features_similarity+numeric_features_similarity)

    return sim

def get_songs(song_name,data):
    if data[data['track_name'] == song_name].shape[0] == 0:
        print("We can not find this kind of song please try another key word!")
        print("this songs may you know :")
        for song in data.sample(n=5)['track_name'].values:
            print(song)
        return


    simi = get_similar(song_name, data)
    data.loc[:, "similar_factor"] = simi

    data.sort_values(by=["similar_factor"],ascending=False,inplace=True)
    print(data["similar_factor"].tail(20))

    print("Song for you :")
    print(data['track_name'][2:15])

get_songs('Lonely This Christmas',dataset2)