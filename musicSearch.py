import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
import random

class SearchMusic():
    def __init__(self):
        # # Load environment variables from .env file
        load_dotenv()

        self.clientID = os.getenv("CLIENT_ID")
        self.clientSecret = os.getenv("CLIENT_SECRET_KEY")

        self.spCredential = SpotifyClientCredentials(self.clientID, self.clientSecret)
        self.spConnect = spotipy.Spotify(client_credentials_manager = self.spCredential)
    
    def search(self, emotion):
        #search for playlist
        playlists_response = self.spConnect.search(q=emotion, type='playlist', limit=40)
        playlists = []
        for playlist in playlists_response['playlists']['items']:
            playlist_name = playlist['name']
            playlist_id = playlist['id']
            playlist_url = playlist['external_urls']['spotify']
            playlists.append({'name': playlist_name, 'id': playlist_id, 'url': playlist_url})

        playlist = playlists[random.randint(0,35)]
        # playlist = playlist['playlists']['items']
        music_dict={}
        print(playlist)

        # Get the playlist's tracks
        playlist_id = playlist['id']
        playlist_name = playlist['name']
        spotify_url = playlist['url']
        # playlist_owner = playlist['owner']['display_name']
        # print(f"Playlist: {playlist_name} (by {playlist_owner})")
        tracks = self.spConnect.playlist_tracks(playlist_id)

        for track in tracks['items']:
            track_name = track['track']['name']
            track_artist = track['track']['artists'][0]['name']
            music_dict.update({track_artist:track_name})
            # print(f"Song: {track_name}\nArtist: {track_artist}\n")
        
        return playlist_name, music_dict, spotify_url