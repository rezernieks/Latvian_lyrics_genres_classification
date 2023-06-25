#imports
import csv
import dask.dataframe as dd
import psutil
import pandas as pd
#import numpy as np

#constants
orig_path: str = "C:/Users/User/Desktop/Datasets/lyrics/archive/song_lyrics.csv"
lv_song_path: str = "C:/Users/User/PycharmProjects/Latvian_lyrics_genres_classification/lv_songs_lyrics.csv"
stopwords: list = ["aiz","ap","ar","apakš","ārpus","augšpus","bez","caur","dēļ","gar","iekš","iz","kopš","labad",
                   "lejpus","līdz","no","otrpus","pa","par","pār","pēc","pie","pirms","pret","priekš","starp","šaipus",
                   "uz","viņpus","virs","virspus","zem","apakšpus","un","bet","jo","ja","ka","lai","tomēr","tikko",
                   "turpretī","arī","kaut","gan","tādēļ","tā","ne","tikvien","vien","kā","ir","te","vai","kamēr",
                   "diezin","droši","diemžēl","nebūt","ik","it","taču","nu","pat","tiklab","iekšpus","nedz","tik",
                   "nevis","turpretim","jeb","iekam","iekām","iekāms","kolīdz","līdzko","tiklīdz","jebšu","tālab",
                   "tāpēc","nekā","itin","jā","jau","jel","nē","nezin","tad","tikai","vis","tak","iekams","būt","biju",
                   "biji","bija","bijām","bijāt","esmu","esi","esam","esat","būšu","būsi","būs","būsim","būsiet","tikt",
                   "tiku","tiki","tika","tikām","tikāt","tieku","tiec","tiek","tiekam","tiekat","tikšu","tiks","tiksim",
                   "tiksiet","tapt","tapi","tapāt","topat","tapšu","tapsi","taps","tapsim","tapsiet","kļūt","kļuvu",
                   "kļuvi","kļuva","kļuvām","kļuvāt","kļūstu","kļūsti","kļūst","kļūstam","kļūstat","kļūšu","kļūsi",
                   "kļūs","kļūsim","kļūsiet","varēt","varēju","varējām","varēšu","varēsim","var","varēji","varējāt",
                   "varēsi","varēsiet","varat","varēja","varēs"]


def filter_large_csv(input_file, output_file, language):
    chunk_size = 1000000  # Adjust the chunk size based on your available memory

    # Create an empty DataFrame to store the filtered data
    filtered_data = pd.DataFrame()

    # Read the CSV file in chunks and filter the data
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, quotechar='"', quoting=csv.QUOTE_ALL):
        filtered_chunk = chunk[chunk['language'] == language]
        filtered_data = pd.concat([filtered_data, filtered_chunk])

    # Write the filtered data to a new CSV file
    filtered_data.to_csv(output_file, index=False, quotechar='"', quoting=csv.QUOTE_ALL)


def calculate_chunk_size(total_memory, available_memory, percentage):
    memory_to_use = available_memory * percentage
    chunk_size = int(memory_to_use / psutil.virtual_memory().total * total_memory)
    return chunk_size


def create_lv_song_dataset():
    #atver orig_path
    ddf: dd = dd.read_csv(orig_path, usecols=["id", "tag", "lyrics", "language"])

    ddf = ddf[ddf.language == 'lv']
    print(ddf.columns)
    #df = df.dropna
    #df = df.drop_duplicates()
    print(ddf.npartitions)
    print(ddf.count(axis='columns'))
    #print(f"unikāli ieraksti: {df.value_counts}")
    print(ddf.dtypes.head(5))
    print(ddf.compute().head(5))
    #atver lv_songs_path
    #nolasa pa rindai
    #ja "language" ir 'lv' tad peivieno lv_songs_path

    #return


def inspect_dataset(path):

    return


if __name__ == '__main__':
    #filter_large_csv(orig_path, lv_song_path, 'lv')
    inspect_dataset(lv_song_path)
