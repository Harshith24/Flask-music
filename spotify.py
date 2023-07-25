import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

# # Load environment variables from .env file
load_dotenv()

clientID = os.getenv("CLIENT_ID")
clientSecret = os.getenv("CLIENT_SECRET_KEY")

spCredential = SpotifyClientCredentials(clientID, clientSecret)
spConnect = spotipy.Spotify(client_credentials_manager = spCredential)

#search for playlist
playlists = spConnect.search(q='surprised', type='playlist', limit=5)

#print 
for playlist in playlists['playlists']['items']:
    playlist_name = playlist['name']
    playlist_owner = playlist['owner']['display_name']
    print(f"Playlist: {playlist_name} (by {playlist_owner})")

    # Get the playlist's tracks
    playlist_id = playlist['id']
    tracks = spConnect.playlist_tracks(playlist_id)

    for track in tracks['items']:
        track_name = track['track']['name']
        track_artist = track['track']['artists'][0]['name']
        print(f"Song: {track_name}\nArtist: {track_artist}\n")