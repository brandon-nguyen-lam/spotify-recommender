# Usage: python spotify.py playlist_url

import pandas as pd
import numpy as np
import json
import re
import sys
import itertools
import requests
import datetime
import base64

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

client_id = "client_id"
client_secret = "client_secret"
scope = 'user-library-read'

if len(sys.argv) > 1:
    playlist_url = sys.argv[1]
else:
    print("Usage: %s playlist link" % (sys.argv[0],))
    sys.exit()

print(playlist_url)


print("Parsing data...")
# Convert song dataset into panda dataframe
data = pd.read_csv("tracks.csv")
data.head()
genre_data = pd.read_csv("artists.csv")
# Genres list is actually a string. So I will have to parse it manually
genre_data["genres"].values[0]
# Split by , and then remove []s, and cut out surrounding 's. Also strip whitespace
genre_data["genres"] = genre_data["genres"].apply(lambda st: list(map((lambda s: s.replace("[","").replace("]","").strip()[1:-1]),st.split(","))))

# For each artist in original dataset, find their corresponding genres
# This way, we will know each songs genres and this will be used for our overall formula
# Artist lists are also string lists, so we parse those as well
data["id_artists"] = data["id_artists"].apply(lambda st: list(map((lambda s: s.replace("[","").replace("]","").strip()[1:-1]),st.split(","))))




# Here, we map each artist to their genres in the genre dataset. Then we append it to the overall dataset


# Un-listify multiple artists so that each artist in a song will have their own row
artists_exploded = data[['id_artists','id']].explode('id_artists')


# Tack genres onto each artist
artist_to_genre = artists_exploded.merge(genre_data[['name', 'genres', 'id']], how = 'left', left_on = 'id_artists', right_on = 'id')

# Remove data with only [nan]
remove_null = artist_to_genre[~artist_to_genre["genres"].isnull()]

# Combine artists of individual songs back together, and combine their genres into one list
combined = remove_null.groupby('id_x')['genres'].apply(list).reset_index()

# Combine the list of lists into one
combined['genres'] = combined['genres'].apply(lambda x: list(set(itertools.chain.from_iterable(x))))
combined["id"] = combined["id_x"] # Hack to get id_x to match to id for merging... I dont even know
data = data.merge(combined[['id','genres']], on="id",how = 'left')

print("Running TF-IDF...")
tfidfvec = TfidfVectorizer()

# Create tfidf-friendly format and run tfidf transform on it
# Also account for nans
def join(elem):
    try:
        return " ".join(elem)
    except:
        return "null"
tfidf =  tfidfvec.fit_transform(data['genres'].apply(join))

# Put tfidf output into dataframe
tfidf_data = pd.DataFrame(tfidf.toarray())

# Append genre and genre name so we can have genre values later in calculation
tfidf_data.columns = ['genre' + "|" + i for i in tfidfvec.get_feature_names_out()]


print("Normalizing Data...")
# Normalize loudness, instru, and tempo because they are outliers in the data. We want them to scale from 0-1 like everything else

loudness_max = data.loudness.max()
loudness_min = data.loudness.min()
instrumentalness_max = data.instrumentalness.max()
instrumentalness_min = data.instrumentalness.min()
tempo_max = data.tempo.max()
tempo_min = data.tempo.min()

data.loudness = data["loudness"].apply(lambda x: (x - loudness_min) / (loudness_max - loudness_min))
data.instrumentalness = data["instrumentalness"].apply(lambda x: (x - instrumentalness_min) / (instrumentalness_max - instrumentalness_min))
data.tempo = data["tempo"].apply(lambda x: (x - tempo_min) / (tempo_max - tempo_min))


print("Grabbing playlist data...")
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id,client_secret))
playlists = sp.playlist_items(playlist_url)
idSet = []
for item in playlists['items']:
    trackid = item["track"]["id"]
    match = data.loc[data["id"]==trackid]
    if len(match) > 0:
        idSet.append(match)

# Remove all unnecessary columns and combine TFIDF with final results
data=data.drop("duration_ms",axis=1)
data=data.drop("explicit",axis=1)
data=data.drop("id_artists",axis=1)
data=data.drop("release_date",axis=1)
data=data.drop("key",axis=1)
data=data.drop("mode",axis=1)
data=data.drop("time_signature",axis=1)
data=data.drop("genres",axis=1)

# Reset indexes so they match up for concat
tfidf_data.reset_index(drop = True, inplace=True)
data.reset_index(drop = True, inplace = True)
data = pd.concat([data, tfidf_data],axis=1)


indexList = []
for song in idSet:
    temp = song["id"].to_string()
    idx = temp.split()
    idx = idx[0]
    indexList.append(idx)
# indexList

userDataFrame = pd.DataFrame()
for song in indexList:
    userDataFrame = pd.concat([userDataFrame, data.loc[[int(song)]]])

if len(userDataFrame) == 0:
    print("Unfortunately, your playlist had 0 matches in the dataset. Try a different playlist")
    sys.exit()
else:
    print("Matches: " + str(len(userDataFrame)))


print("Generating recommendations...")

# Get everything not in the playlist, so we dont get recommended the same songs
data = data[~data['id'].isin(userDataFrame['id'].values)]

nums_only_playlist = userDataFrame.drop("id", axis=1).drop("name", axis=1).drop("artists", axis=1)
playlist = nums_only_playlist.sum(axis=0).divide(userDataFrame.count())

# Remove null rows (I dont know why they are there but they are)
playlist.dropna(axis=0, how='any',inplace=True)
data.dropna(axis=0, how='any',inplace=True)

# Use reshape since we only have a single sample
reshaped_playlist = playlist.values.reshape(1, -1)

# The .sum() thing with playlist sorts the columns alphabetically, so just do the same to data to line them up
data = data.reindex(sorted(data.columns), axis=1)

# Create actual numerical vector
nums_only = data.drop("id", axis=1).drop("name",axis=1).drop("artists", axis=1) #drop name again

# Cosine Formula
data['similarity'] = cosine_similarity(nums_only.values, reshaped_playlist)[:,0]

pd.set_option("display.max_rows", 300)
top_25_songs = data.sort_values('similarity',ascending = False).head(25)
print(top_25_songs[["name", "artists"]])
