import pandas as pd
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

# Funkcja do usuwania jednej literki
def remove_letter(word):
    if len(word) < 2:
        return word
    i = random.randint(0, len(word)-1)
    return word[:i] + word[i+1:]    # usuwa jedna litere

# Funkcja do tworzenia literówek
def make_mistake(word):
    if len(word) < 4:
        return word
    i = random.randint(0, len(word) - 2)
    return word[:i] + word[i+1] + word[i] + word[i+2:]  # zamienia 2 litery miejscami

# Funkcja do dodawania spacji w slowie
def add_spacebar(word):
    i=random.randint(0, len(word) - 2)
    return word[:i]+ " " + word[i:]

# Funkcja do losowania blednej nazwy leku
def randomize_wrong_drug(drug, wrong_drug):
    new_wrong_drug = remove_letter(remove_letter(drug))
    new_wrong_drug = add_spacebar(new_wrong_drug)
    new_wrong_drug = make_mistake(new_wrong_drug)

    if wrong_drug.__contains__(new_wrong_drug):
        return randomize_wrong_drug(drug, wrong_drug)
    else:
        return new_wrong_drug

# Wczytywanie danych
random.seed(12)
df = pd.read_csv("drugs410.csv")
drugs = df["drug_name"].tolist()  # wczytywanie listy lekow

# Powielenie poprawnych lekow i tworzenie blednych lekow
correct_drugs = []
wrong_drugs = []
for drug in drugs:
    for i in range (0, 5):
        correct_drugs.append(drug)
        wrong_drugs.append(randomize_wrong_drug(drug, wrong_drugs))

# Wszystko po * (train_count, block_size) musi być nazwane.
# Dzieli sekwencję na dwa zbiory co block_size elementow
#  - indeksy <od 0 do train_count) → train
#  - indeksy <od train_count do block_size) → test
#
# Np. train_count=3; block_size=5
# Co 5 elementow:
#  - indeksy 0, 1, 2 → train
#  - indeksy 3, 4 → test
def split_blocks(seq, *, train_count, block_size):
    if train_count > block_size or block_size<0 or train_count<0:
        raise ValueError("Require 0 ≤ train_count ≤ block_size")
    train_data, test_data = [], []
    # step through the sequence block by block
    for i in range(0, len(seq), block_size):
        train_data.extend(seq[i : i + train_count])              # <0 to train_count) → train
        test_data.extend(seq[i + train_count : i + block_size])  # <train_count to block_size) → test
    return train_data, test_data

# Machine learning

# 1. Podział na dane treningowe i testowe
correct_drugs_train, correct_drugs_test = split_blocks(correct_drugs, train_count=3, block_size=5)  # 3 of 5 train; 2 of 5 test
wrong_drugs_train, wrong_drugs_test = split_blocks(wrong_drugs, train_count=3, block_size=5) # 3 of 5 train; 2 of 5 test

# 2. Zamiana tekstów na liczby (wektoryzacja)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 1))
wrong_drugs_train_vectorized = vectorizer.fit_transform(wrong_drugs_train) #1230 blednych lekow
wrong_drugs_test_vectorized = vectorizer.fit_transform(wrong_drugs_test) #802 blednych lekow

# 3. Trenowanie modelu przez rozne klasyfikatory
results=pd.DataFrame(columns=('klasyfikator','ocena','czas_uczenia','czas_testu')) #tworzenie tabelki

def run(x_train, y_train, x_test, y_test, clf, clf_name):
    global results  #results jest zmienną globalną

    s=time.time() #czas rozpoczęcia procesu trenowania
    x=clf.fit(x_train, y_train) #trening na danych treningowych
    time_train=time.time()-s #czas trenowania

    s=time.time() #czas rozpoczęcia procesu testowania
    score_tst=x.score(x_test,y_test) #ocena modelu na danych testowych
    time_test=time.time()-s #czas testowania

    new_row = pd.DataFrame({'klasyfikator':[clf_name], 'ocena':[score_tst], 'czas_uczenia':[time_train], 'czas_testu':[time_test]})
    results = pd.concat([results, new_row], ignore_index=True) 

clfs=[
    KNeighborsClassifier(n_neighbors=3, metric="cosine", weights="distance"),
    KNeighborsClassifier(n_neighbors=4, metric="cosine", weights="distance"),
    KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance"),
    KNeighborsClassifier(n_neighbors=6, metric="cosine", weights="distance"),
    MultinomialNB(),
    NearestCentroid(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression()]

for clf in clfs:
   run(wrong_drugs_train_vectorized, correct_drugs_train, wrong_drugs_test_vectorized, correct_drugs_test, clf, str(clf))
print(tabulate(results, headers="keys", tablefmt="fancy_grid"))