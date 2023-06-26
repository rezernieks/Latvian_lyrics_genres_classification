#imports
import csv
#import dask.dataframe as dd
import psutil
import pandas as pd
from text_preprocessing import preprocess_text
from text_preprocessing import remove_punctuation, remove_number, remove_whitespace, remove_special_character
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
#import numpy as np

#constants
orig_path: str = "C:/Users/User/Desktop/Datasets/lyrics/archive/song_lyrics.csv"
lv_song_path: str = "C:/Users/User/PycharmProjects/Latvian_lyrics_genres_classification/lv_songs_lyrics.csv"
lv_song_preproc_path: str = "C:/Users/User/PycharmProjects/Latvian_lyrics_genres_classification/lv_songs_lyrics_preprocessed.csv"
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
feature_names: list = []


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


def inspect_dataset(path):
    f = pd.read_csv(path)
    print("file read")
    print(f.head())
    return


def preprocess_dataset(path, save_path):
    drop_list = ['language_cld3', 'language_ft', 'id', 'language', 'year', 'features', 'views', 'year', 'title',
                 'artist']
    f = pd.read_csv(path)
    f = f.drop(drop_list, axis=1)
    f['lyrics'] = f['lyrics'].apply(lambda x: f"\"{preprocess(x)}\"")
    print(f.head())
    print(f['lyrics'][0])
    f.to_csv(save_path, index=False, encoding="utf-8")


def preprocess(document: str) -> str:
    #stemmer = LatvianStemmer
    document = document.lower()
    document = preprocess_text(document, [remove_special_character, remove_number, remove_whitespace, remove_punctuation])
    #document = word_tokenize(document)
    #document = [word for word in document if word not in lv_stopwords]
    #document = [stemmer.stem(word) for word in document]
    #document = ' '.join(document)
    return document


def to_test_and_train_sets(path, split_size=0.20):
    f = pd.read_csv(path)
    X_train, X_test, y_train, y_test = train_test_split(f['lyrics'], f['tag'], test_size=split_size,
                                                        random_state=42)
    return [X_train, X_test, y_train, y_test]


def create_bow_vector(sets: list = None):
    if sets is None:
        sets = to_test_and_train_sets(lv_song_preproc_path)
    vectorizer = CountVectorizer(analyzer='word')
    X_train_bow = vectorizer.fit_transform(sets[0])
    global feature_names
    feature_names = vectorizer.get_feature_names_out()
    X_test_bow = vectorizer.transform(sets[1].tolist())
    return [X_train_bow, X_test_bow, sets[2], sets[3]]


def create_tfidf_vector(sets: list = None):
    if sets is None:
        sets = to_test_and_train_sets(lv_song_preproc_path)
    tfidfvectorizer = TfidfVectorizer(analyzer='word')
    X_train_tfidf = tfidfvectorizer.fit_transform(sets[0])
    global feature_names
    feature_names = tfidfvectorizer.get_feature_names_out()
    X_test_tfidf = tfidfvectorizer.transform(sets[1])
    return [X_train_tfidf, X_test_tfidf, sets[2], sets[3]]


def model_report(cl_rep):
    return [round(100*cl_rep['macro avg']['precision']),
            round(100*cl_rep['weighted avg']['precision']),
            round(100*cl_rep['macro avg']['recall']),
            round(100*cl_rep['weighted avg']['recall']),
            round(100*cl_rep['accuracy']),
            round(100*cl_rep['macro avg']['f1-score']),
            round(100*cl_rep['weighted avg']['f1-score'])]


def model_table(res, name, attr):
    mydata = res
    head = attr + ["Pmavg", "Pwavg", "Rmavg", "Rwavg", "A", "f1mavg", "f1wavg"]
    print(tabulate(mydata, headers=head, tablefmt="grid"))
    pd.DataFrame(mydata, columns=head).to_csv(f'{name}.csv', index=False)
    return


def select_vect_type(vect):
    sets = []
    if vect == "BoW":
        sets = create_bow_vector()
    if vect == "TF-IDF":
        sets = create_tfidf_vector()
    return sets


def all_mnb():
    def mnb_and_evaluate(alpha, sets: list = None, vect: str = None):
        sets = select_vect_type(vect)
        model = MultinomialNB(alpha=alpha)
        model.fit(sets[0], sets[2])
        y_pred = model.predict(sets[1])
        report = classification_report(sets[3], y_pred, output_dict=True)
        return model_report(report)

    res = []
    for v in ["BoW", "TF-IDF"]:
        for a in range(1,21):
            res.append([v, a/100]+mnb_and_evaluate(a/100, vect=v))
    return res

def all_lr():
    def lr_and_evaluate(m_it, c, sets: list = None, vect: str = None):
        sets = select_vect_type(vect)
        model = LogisticRegression(multi_class="multinomial", max_iter=m_it, penalty='l2', C=c)
        model.fit(sets[0], sets[2])
        y_pred = model.predict(sets[1])
        report = classification_report(sets[3], y_pred, output_dict=True)
        return model_report(report)

    res = []
    for v in ["BoW", "TF-IDF"]:
        for i in [500,1000,2000]:
            for c in [0.01, 0.1, 1, 10, 100]:
                print(f'vect: {v}, i: {i}, c: {c}')
                res.append([v, i, c]+lr_and_evaluate(m_it=i, c=c, vect=v))
    return res



if __name__ == '__main__':
    #filter_large_csv(orig_path, lv_song_path, 'lv')
    #inspect_dataset(lv_song_path)
    #preprocess_dataset(lv_song_path, lv_song_preproc_path)
    #model_table(all_mnb(), "mnb", ["v", "a"])
    model_table(all_lr(), "lr", ["v", "max_iter", "C"])
