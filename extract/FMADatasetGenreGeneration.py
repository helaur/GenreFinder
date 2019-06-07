
import json

from utils import utils


class FMADatasetGenreGeneration:

    def read_song_id_and_find_genre(self, path_of_track_file, genres):

        tracks = utils.load(path_of_track_file)
        for gen in genres:
            all_ids = []
            for id, genre in tracks['track', "genre_top"].items():
                if(genre == gen):
                    all_ids.append('{:0>6}'.format(id))
            self.write_json_for_genres_all_song_ids(all_ids, gen)


    def write_json_for_genres_all_song_ids(self, ids, genre):

        data = [genre, ids]
        with open(r"..\data\%s.json" % genre, "w") as write_file:
            json.dump(data, write_file)

if __name__=="__main__":

    tester = FMADatasetGenreGeneration()
    # Need to download tracks.csv (250mb) from FMA git and place it in the extract folder or change path
    tester.read_song_id_and_find_genre(r"..\extract\tracks.csv", ["Hip-Hop"])



