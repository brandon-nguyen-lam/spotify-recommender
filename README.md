# Spotify Recommender
![Example Image](example.png)
[Playlist Used](https://open.spotify.com/playlist/2f64bv5oUNyUFQ2gUwfTA1?si=nif_aP2ZT2KAwoS7iptzKA)
## Getting Started

Download the [data](https://www.dropbox.com/scl/fo/4skf0zta047i01rl0e4fw/h?dl=0&rlkey=cfjcokgffymbayn3dnblewp7v) from DropBox and put inside the backend folder with spotify.py. After that, install these packages.

```
pip install spotipy
pip install numpy
pip install scikit-learn
pip install requests
```

## Running the program

```
python spotify.py [playlist_link_here]
```

## Important Information
- Dataset has 600k songs from 1920-2020 with data from Kaggle.
- The recommender does NOT put weight on date released for songs

