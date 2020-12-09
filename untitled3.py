import pandas as pd 
# biblioteka do analizy danych
import numpy as np 
# biblioteka do obliczeń matematycznych
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
# sxikit-learn biblioteka uczenia maszynowego. 
# Zawiera m.in. algorytmy klasyfikacji, regresji, klastrowania.
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer 
# NLTK – zestaw bibliotek i programów do symbolicznego i statystycznego przetwarzania języka naturalnego.
from surprise import Reader
from surprise import SVD 
# Rozkład według wartości osobliwych
from surprise import Dataset
from surprise.model_selection import cross_validate
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import warnings; warnings.simplefilter('ignore') 
# blokuje komunikaty ostrzegawcze o np. niektualnej wersji

filmy_dane = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane_edit/filmy_dane.csv')  
# wczytanie pliku .csv z danymi takimi jak tytuł, język, data produkcji. Rozmiar (45466, 24).

filmy_dane['genres'] = filmy_dane['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])  
# formuła oceny wazonej. 
# !!! muszę ją w wolnej chwili dokładnie zinterpretować

liczba_glosow = filmy_dane[filmy_dane['vote_count'].notnull()]['vote_count'].astype('int') 
# do zmiennej liczba głosów, przypisujemy zbiór danych o filmie, nastepnie wyciągamy kolumne 'liczba głosów' 
# oraz tylko te wartosci które nie są nulami i zmieniami typ danych na 'int'.
# Dlatego nasza kolumna 'liczba_glosow' ma (45460, ).
# O 6 mniej niż nasz obiekt DataFrame 'filmy_dane'. 

srednia_glosow = filmy_dane[filmy_dane['vote_average'].notnull()]['vote_average'].astype('int')
# dane o filmach przypisujemy do zmiennej 'srednia_glosow'(45460, ).
# Nastepnie wyciągamy kolumne 'srednia_glosow' oraz tylko te wartosci które nie są nulami i zmieniami typ danych na 'int'.

srednia_ze_sredniej_liczby_glosow = srednia_glosow.mean()
# liczymy srednią, ze sredniej liczby głosów. Wynosi 5.24.

liczba_glosow_kwantyl = liczba_glosow.quantile(0.867) 
# ustawienie liczby głosów tak, (oddane głosy tmdb) aby film mogł sie znaleźć na liscie proponowanych pozycji.
# Obecne ustawienie = min. 100 oddanych glosow.   

filmy_dane['year'] = pd.to_datetime(filmy_dane['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
# Dodaje kolumne 'year' do obiektu dataframe film_dane.

zakwalifikowany = filmy_dane[(filmy_dane['vote_count'] >= liczba_glosow_kwantyl) & (filmy_dane['vote_count'].notnull()) & (filmy_dane['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]  
# do zmiennej 'zakwalifikowany' przypsiujemy dane o filmach. 
# Złożenia: liczba glosow jest >= 100 (tyle wynosi zmienna liczba_glosow_kwantyl) i liczba głosów nie jest nulem i srednia glosów nie jest nullem.
# Dodatkowo pracuje tylko na kolumnach: 'title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres'.
# Czyli mamy 6055 tytułow spełniających nasz warunek. 
# Te filmy są przypsane do obiektu DataFrame o nazwie 'zakwalifikowany'. 

zakwalifikowany['vote_count'] = zakwalifikowany['vote_count'].astype('int')  
# zmiana typu danych w kolumnie 'vote_count' z float64 na 'int'.

zakwalifikowany['vote_average'] = zakwalifikowany['vote_average'].astype('int')  
# zmiana typu danych w kolumnie 'vote_count' z float64 na 'int'.

# Aby zakwalifikować się do listy filmow proponowanych, film musi mieć co najmniej 100 głosów na tmdb.
# Widzimy również, że średnia ocena filmu na tmdb to 5,244 w skali od 0 do 10. 
# 6055 filmów kwalifikują się do umieszczenia na naszym wykresie.

def ocena_wazona(x):
    l_liczba_glosow = x['vote_count']
    s_srednia_glosow = x['vote_average']
    return (l_liczba_glosow / (l_liczba_glosow + liczba_glosow_kwantyl) * s_srednia_glosow) + (liczba_glosow_kwantyl / (liczba_glosow_kwantyl + l_liczba_glosow) * srednia_ze_sredniej_liczby_glosow)
# funkcja ocena_wazona 
# !!! dzięki tej funkcji okreslamy wagę ocen dla poszczególnego filmu?   
# muszę ją w wolnej chwili dokładnie zinterpretować

zakwalifikowany['ocena_wazona'] = zakwalifikowany.apply(ocena_wazona,axis=1)  
# dodanie kolumny 'ocena_wazona' do obiektu 'zakwalifikowany'. Obecny rozmiar to (6055, 7). Czyli o jedną kolumnę więcej niż poprzednio.

zakwalifikowany = zakwalifikowany.sort_values('ocena_wazona', ascending=False).head(500)  
# Posortowanie danych wg. oceny_wazonej. Ograniczenie tytułów do 500.
# 500 tytułów będzie branych pod uwage jako filmy proponowane. 
# !!! Z tmdb są pobrani użytkownicy a z imdb są pobrane filmy? Muszę to sprawdzić.

# Budowa funkcji która buduje wykres dla poszczególnych gatunkow.

gatunek_filmu = filmy_dane.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1,drop=True)  
# tworzymy nowy obiekt typu Series zawierający wyłącznie 'genres' zawierające gatunek filmu i jego id.
# !!! muszę ją w wolnej chwili dokładnie zinterpretować
# po co wyciagamy wszytskie agtunki filmów? Rozmiar (91106, ). Aby później je z joinować z obiektem 'filmy_dane'?

gatunek_filmu.name = 'kategoria'
# zmiany nazwy kolumny tabeli 'gatunek_filmu' z '0' na 'kategoria'

filmy_dane_kategoria = filmy_dane.drop('genres', axis=1).join(gatunek_filmu) 
# Obiekt DataFrame 'filmy_dane_bez_genres' już nie ma skomplikowanej kolumny 'genres'. Nie wiem co miałem na mysli.
# Utworzono obiekt DataFrame filmy_dane_kategoria, który ma na końcu dodaną kolumne 'kategoria'.
# Obiekt nie różni się niczym od obiektu 'filmy_dane' więc po co on jest?

def buduj_wykres(kategoria, percentile=0.867):
    df = filmy_dane_kategoria[filmy_dane_kategoria['belongs_to_collection'] == kategoria]
    liczba_glosow = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    srednia_glosow = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    srednia_ze_sredniej_liczby_glosow = srednia_glosow.mean()
    liczba_glosow_kwantyl = liczba_glosow.quantile(percentile)

    zakwalifikowany = df[(df['vote_count'] >= liczba_glosow_kwantyl) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    zakwalifikowany['vote_count'] = zakwalifikowany['vote_count'].astype('int')
    zakwalifikowany['vote_average'] = zakwalifikowany['vote_average'].astype('int')

    zakwalifikowany['ocena_wazona'] = zakwalifikowany.apply(lambda x: (x['vote_count'] / (x['vote_count'] + liczba_glosow_kwantyl) * x['vote_average']) + (liczba_glosow_kwantyl / (liczba_glosow_kwantyl + x['vote_count']) * srednia_ze_sredniej_liczby_glosow), axis=1)
    zakwalifikowany = zakwalifikowany.sort_values('ocena_wazona', ascending=False).head(500)

    return zakwalifikowany
# !!! muszę ją w wolnej chwili dokładnie zinterpretować

# buduj_wykres('Romance').head(15)

dane_id = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane_edit/dane_id_male.csv')  
# wczytanie danych o: id_film, id_imdb, id_tmdb (9125 3).

dane_id = dane_id[dane_id['tmdbId'].notnull()]['tmdbId'].astype('int')  
# pozostawienie tylko kolumny 'tmdbId' w tabeli 'dane_id'.

filmy_dane = filmy_dane.dropna(subset=['release_date'])
# uwzględnienie kolumny 'release_date'. Usuniecie  wartoci nan. Rozmiar (45379, 24) a było (45466, 24).
filmy_dane = filmy_dane[filmy_dane['release_date'].str.contains("^[0-9]{4}-[0-9]{2}-[0-9]{2}")]
# usunięcie trzech błędnych rekordów z przesuniętą datą

filmy_dane['id'] = filmy_dane['id'].astype('int')  
# zmiana typu danych z 'object' na 'int'.

filmy_dane_join_dane_id = filmy_dane[filmy_dane['id'].isin(dane_id)]  
# nowy obiekt DataFrame zawierający te filmy/ dane z kolumną dane_id.

# Rekomendacja oparta na opisie filmu
# Najpierw spróbujmy zbudować rekomendację, korzystając z opisów filmów i sloganów.
# Nie mamy miernika ilościowego, aby ocenić wydajność naszej maszyny, więc będzie to musiało być wykonane jakościowo.

filmy_dane_join_dane_id['tagline'] = filmy_dane_join_dane_id['tagline'].fillna('')
# Dzięki funkcji 'fillna('')' zastępujemy wartosci nan w kolumnie 'tagline', pustym polem.

filmy_dane_join_dane_id['description'] = filmy_dane_join_dane_id['overview'] + filmy_dane_join_dane_id['tagline']  
# dodanie do obiektu dataframe kolummny 'description' 
# po co 'overievw' i 'tagline'?

filmy_dane_join_dane_id['description'] = filmy_dane_join_dane_id['description'].fillna('')  
# Dzięki funkcji 'fillna('')' zastępujemy wartosci nan w kolumnie 'description', pustym polem.

wektoryzator_tfId = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0,stop_words='english')  
# TfidfVectorizer przypisuje wagę każdemu tokenowi która zależy nie tylko od jego częstotliwości w dokumencie, 
# ale także od tego, jak powtarzający się termin występuje 

tfidf_macierz = wektoryzator_tfId.fit_transform(filmy_dane_join_dane_id['description'])  
# Dopasuj do danych, a następnie przekształć je.

tfidf_macierz.shape
# !!! skąd ten rozmiar się wziął (9099, 268124)? Co tam jest przechowywane?
# Przechowywane są tam nasze wartopci numeryczne/ tokeny?

# Podobieństwo cosinusowe
# Będę używał podobieństwa cosinusów, aby obliczyć wielkość liczbową, która oznacza podobieństwo między dwoma filmami.
# Ponieważ użyliśmy wektoryzatora TF-IDF, obliczenie iloczynu skalarnego bezpośrednio da nam wynik podobieństwa cosinusów.
# Dlatego użyjemy linear_kernel sklearn zamiast cosine_similarities, ponieważ jest znacznie szybsze.

podobienstwo_cosinusowe = linear_kernel(tfidf_macierz,tfidf_macierz)  
# "linear_kernel używamy gdy dane można rozdzielić."
# x = linear_kernel(próbki, funkcje). Dlaczego podajemy dwukrotnie te same dane?

podobienstwo_cosinusowe[0]
# !!! jak interpretować tą liste? 
# Czy jest ta macierz terminów, w której każdy wiersz reprezentuje dokument,
# a każda kolumna jest adresowana do tokenu?

# Mamy teraz macierz podobieństwa cosinusów parami dla wszystkich filmów w naszym zbiorze danych.
# Następnym krokiem jest napisanie funkcji zwracającej 30 najbardziej podobnych filmów na podstawie
# wyniku podobieństwa cosinusowego.

filmy_dane_join_dane_id = filmy_dane_join_dane_id.reset_index()
# !!! Dlaczego dodalimy nowa kolumnę indeks?

tytuly = filmy_dane_join_dane_id['title']
# przypisanie tytułów z obiektu DataFrame 'filmy_dane_join_dane_id' do zmiennej tytuly

indeksy = pd.Series(filmy_dane_join_dane_id.index, index=filmy_dane_join_dane_id['title'])  
# przypisanie do kolumny 'indeksy' tytułów i indeksów.
# !!! jak rozumiem, filmy_dane_join_dane_id.index nadał indeks tytłom wyciągniętym z kolumny 
# 'filmy_dane_join_dane_id'? Po co?

#indeksy.name = 'indeks'

def uzyskane_rekomendacje(tytul):
    indeks = indeksy[tytul]
    wynik_symulacji = list(enumerate(podobienstwo_cosinusowe[indeks]))
    wynik_symulacji = sorted(wynik_symulacji, key=lambda x: x[1], reverse=True)
    wynik_symulacji = wynik_symulacji[1:31]
    indeksy_filmowe = [i[0] for i in wynik_symulacji]
    return tytuly.iloc[indeksy_filmowe]
# !!! musze to w wolnej chwili zinterpretować dokładnie.

uzyskane_rekomendacje('Batman Returns').head(5)

uzyskane_rekomendacje('Harry Potter and the Half-Blood Prince').head(5)
# rekomendacja nie uwzględniająca użytkownika. Szuka poprostu filmów podobnych do siebie.

# Moje pytania:
# - czy ten sposób rekomendacji rzeczywicie ma zaimplementowaną "sztuczną intelgigencję" ?
#   Jesli dobrze rozumiem, to zostały porównanie opisy filmów a poźniej zjoinowane tabele, czy tak?

obsada = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane_edit/credits.csv') 
# wczytanie danych dotyczące obsady wraz z osobami realizującymi film.

slowa_kluczowe = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane_edit/keywords.csv') 
# wczytanie danych zawierających id z tmdb oraz id słów kluczowych powiązanych z filmem.

filmy_dane = filmy_dane.merge(obsada, on='id') 
# filmy_dane miały (45376, 25). A po łączeniu (45451, 27).
# Dodano dwie kolumny obsada (cast) i osoby realizujące film (crew).

filmy_dane = filmy_dane.merge(slowa_kluczowe, on='id') 
# Dodano do obiektu DataFrame 'filmy_dane kolumne 'keywords'.
# filmy_dane ma łącznie 28 kolumn (46540, 28), miało (45451, 27)

filmy_dane_join_dane_id = filmy_dane[filmy_dane['id'].isin(dane_id)] 
# dodanie kolumny id do obiektu DataFrame 'film_dane_join_dane_id'. Roz. (9219, 28), a było (9099, 27).

# Mamy teraz obsadę, ekipę, gatunki i napisy końcowe w jednej tabeli 'filmy_dane_join_dane_id'. 
# Będziemy teraz brali pod uwagę reżysera, oraz głóWnych bohaterów i idpowiadających im aktorów. 
# Wybierzemy 3 najlepszych aktorów, którzy pojawią się na liście.

filmy_dane_join_dane_id['cast'] = filmy_dane_join_dane_id['cast'].apply(literal_eval) 
# apply umożliwia użytkownikom przekazywanie funkcji. !!! Co to robi?
filmy_dane_join_dane_id['crew'] = filmy_dane_join_dane_id['crew'].apply(literal_eval) 
# !!! Co to robi?
filmy_dane_join_dane_id['keywords'] = filmy_dane_join_dane_id['keywords'].apply(literal_eval)
# !!! Co to robi?

filmy_dane_join_dane_id['cast_size'] = filmy_dane_join_dane_id['cast'].apply(lambda x: len(x)) 
# dodanie kolumny 'cast_size'
filmy_dane_join_dane_id['crew_size'] = filmy_dane_join_dane_id['crew'].apply(lambda x: len(x))
# dodanie kolumny 'crew_size'.

def otrzymanie_rezysera(x): 
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
# funkcja na otrzymanie reżysera

filmy_dane_join_dane_id['Director'] = filmy_dane_join_dane_id['crew'].apply(otrzymanie_rezysera)
# użycie funkcji otrzymanie rezysera która wyciąga reżysera imie i nazwisko i dodaje jako nową kolumnę w 'filmy_dane_join_dane_id'



##### TEGO NIE WCZYTYWAŁEM #####



filmy_dane_join_dane_id['cast'] = filmy_dane_join_dane_id['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []) 
# !!! musze to dokładnie zinterpretować

filmy_dane_join_dane_id['cast'] = filmy_dane_join_dane_id['cast'].apply(lambda x: x[:3] if len(x) >=3 else x) 
# !!! musze to dokładnie zinterpretować

filmy_dane_join_dane_id['keywords'] = filmy_dane_join_dane_id['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []) 
# !!! musze to dokładnie zinterpretować

filmy_dane_join_dane_id['cast'] = filmy_dane_join_dane_id['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x]) 
# !!! musze to dokładnie zinterpretować

filmy_dane_join_dane_id['Director'] = filmy_dane_join_dane_id['Director'].astype('str').apply(lambda x: str.lower(x.replace(" ", ""))) 
# !!! musze to dokładnie zinterpretować

filmy_dane_join_dane_id['Director'] = filmy_dane_join_dane_id['Director'].apply(lambda x: [x,x, x]) 
# !!! musze to dokładnie zinterpretować

# Słowa kluczowe
# Wykonamy niewielką ilość wstępnego przetwarzania naszych słów kluczowych przed ich użyciem. 
# Pierwszym krokiem jest obliczenie częstości występowania każdego słowa kluczowego, które pojawia się w zbiorze danych.

gatunek_filmu = filmy_dane_join_dane_id.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
# !!! musze to dokładnie zinterpretować. Liczba rekordów została zmniejszona z 91106 na 64407.



##### DO TEGO MIEJSCA #####


gatunek_filmu.name = 'keyword'
# zmiana nazy kolumny z '0' na 'keywords'.

gatunek_filmu = gatunek_filmu.value_counts()
# zmiana wielkos‡ci obiektu na 12940 rekordów. Są teraz dwie kolumny Index (czyli słowo kluczowe) i keyords jako liczba.
# Słowa kluczowe występują w częstotliwościach od 1 do 610. 

# Nie stosujemy słów kluczowych, które występują tylko raz. 
# Dlatego można je bezpiecznie usunąć. Na koniec przekonwertujemy 
# każde słowo na jego rdzeń, aby słowa takie jak Psy i Pies były traktowane tak samo.

gatunek_filmu = gatunek_filmu[gatunek_filmu > 1]
# usunięcie słów które występują tylko 1 raz.

stemmer = SnowballStemmer('english')
# wybranie języka dla stemmera

stemmer.stem('asked')
# sprawdzenie działania stemmera

def filtr_slow_kluczowych(x): # 
    words = []
    for i in x:
        if i in gatunek_filmu:
            words.append(i)
    return words
# !!! musze to dokładnie zinterpretować.




filmy_dane_join_dane_id['keywords'] = filmy_dane_join_dane_id['keywords'].apply(filtr_slow_kluczowych)
# !!!

filmy_dane_join_dane_id['keywords'] = filmy_dane_join_dane_id['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
# !!!

filmy_dane_join_dane_id['keywords'] = filmy_dane_join_dane_id['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x]) # usuwane są spacje
# !!!

filmy_dane_join_dane_id['soup'] = filmy_dane_join_dane_id['keywords'] + filmy_dane_join_dane_id['cast'] + filmy_dane_join_dane_id['Director'] + filmy_dane_join_dane_id['genres'] # 
# !!!

filmy_dane_join_dane_id['soup'] = filmy_dane_join_dane_id['soup'].apply(lambda x: ' '.join(x)) #
# !!!






zlicz = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english') 
# !!!!

zlicz_macierz = zlicz.fit_transform(filmy_dane_join_dane_id['soup'])
# !!!

podobienstwo_cosinusowe = cosine_similarity(zlicz_macierz, zlicz_macierz) 
# zmiany wartosci w tabeli 'podobienstwo_cosinusowe' tj. (9219, 9219). Czy słowo występuje czy nie występuje.
# !!!

filmy_dane_join_dane_id = filmy_dane_join_dane_id.reset_index()
# dodanie kolumny 'index'.

tytuly = filmy_dane_join_dane_id['title']
# !!!

indeksy = pd.Series(filmy_dane_join_dane_id.index, index=filmy_dane_join_dane_id['title'])
# !!! 

uzyskane_rekomendacje('Batman Returns').head(5)
# pokazuje tylko te filmy, które są Tima Burtona

czytelnik = Reader()
# !!! po co to? 

oceny = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane_edit/ratings_small.csv') 
# wczytanie użytkownika, filmu i oceny.

dane = Dataset.load_from_df(oceny[['userId', 'movieId', 'rating']], czytelnik)
# !!! co to jest?

svd = SVD()
# !!! o tym poczytać

cross_validate (svd, dane, measures=['RMSE', 'MAE'], cv=5, verbose=True) 
# !!! o tym poczytać. To jest chyba nasze wyników po trenowaniu. RMSE o tym wiadczy.

# Trenujmy teraz na naszym zbiorze danych i dojdźmy do prognoz.

trainset = dane.build_full_trainset()
# !!! trenowanie? Troche za szybkie.

# svd.train(trainset) # błąd 'SVD' object has no attribute 'train'
# Wybierzmy użytkownika 5000 i sprawdźmy, jakie oceny wystawił. Raczej użytkownika o id = 1.

oceny[oceny['userId'] == 1]
# wyswietlenie wszytskich filmow i ocen którzy uzytkownik 1 udzielił.

svd.predict(1, 302, 3) 
# W przypadku filmu o identyfikatorze 302 otrzymujemy szacunkową prognozę na 2,686. 
# Jedną z zaskakujących cech tego systemu rekomendacji jest to, że nie obchodzi go, 
# czym jest film (lub co zawiera). Działa wyłącznie na podstawie przypisanego identyfikatora 
# filmu i próbuje przewidzieć oceny na podstawie tego, jak inni użytkownicy przewidzieli film.
# W tej sekcji spróbuję zbudować prostą hybrydową rekomendację, która łączy techniki, które wdrożyliśmy 
# w silnikach opartych na treści i opartych na filtrach współpracy. Oto jak to będzie działać:
# Dane wejściowe: identyfikator użytkownika i tytuł filmu
# Wynik: podobne filmy posortowane na podstawie oczekiwanych ocen danego użytkownika.

##def konwertuj_int(x): #
  ##  try:
    ##    return int(x)
    ##except:
      ##  return np.nan
    
mapa_id = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane_edit/dane_id_male.csv')[['movieId', 'tmdbId']] 
# wczytanie idków dotyczacyh filmóW. Mamy nowy obiekt DataFrame 'mapa_id' z dwiema kolumnami tj. 'movieID' i 'tmdbId'.
# utworzenie nowego obiektu DataFrame mapa_id

#mapa_id['tmdbId'] = mapa_id['tmdbId'].apply(konwertuj_int) #

mapa_id.columns = ['movieId', 'id']
# !!! nie wiem po co

mapa_id = mapa_id.merge(filmy_dane_join_dane_id[['title', 'id']], on='id').set_index('title') 
# zmiana w obiekcie DataFrame 'mapa_id'. 'Index' zmieniono na 'title', a 'tmdbId' na 'id'.

#id_map = id_map.set_index('tmdbId') # tak było zakomentowane na stronie

mapa_indeksow = mapa_id.set_index('id') 
# utworzenie nowej tabeli o naziwe 'indices_map' zawierającej 'id' oraz 'movieId'.

def hybryda(userId, title): 
    # wskazniki tytułow póxniej bierze id filmów 
    idx = indeksy[title] 
    # movie id = id filmow, druga tab. to id rekomendacji.
    tmdbId = mapa_id.loc[title]['id']
    print(idx)
    movie_id = mapa_id.loc[title]['movieId']
    
    wynik_cosunisowy = list(enumerate(podobienstwo_cosinusowe[int(idx)])) # nie ma znaczenia większego
    wynik_cosunisowy = sorted(wynik_cosunisowy, key=lambda x: x[1], reverse=True) # jak wszytskie filmy są podobne do tego jednego. 
    wynik_cosunisowy = wynik_cosunisowy[1:10] # tylko 25 obiecujących folmów pokazano. Wyprowadzić gdzie indziej ten parametrów.
    indeksy_filmowe = [i[0] for i in wynik_cosunisowy] # później znajdujemy indkesy tych filmów.
    
    filmy = filmy_dane_join_dane_id.iloc[indeksy_filmowe][['title', 'vote_count', 'vote_average', 'year', 'id']] # później patrzymy z którego roku, etc.
    filmy['est'] = filmy['id'].apply(lambda x: svd.predict(userId, mapa_indeksow.loc[x]['movieId']).est) # tu robimy predyckje. Wiesz kim jest nasz user, i powiedz jak spodobało się te 25 filmów.
    filmy = filmy.sort_values('est', ascending=False) # przerobienie funkjci aby wykorzystywała get_recommendaition
    return filmy.head(5)

hybryda(1, 'Avatar') # wygenerowanie danych w kodzie
hybryda(500, 'Avatar') # wygenerowanie danych w kodzie



# obejrzały 10 filmów i uznały czy rzeczywicie te filmy są dobre.
# przerobić raitngs_small. To jest w kontekcie oceny.

# stworzyc zbior testowy
# rating_small podzielimy na dwa zbiory
# uzytkownik ktory dał przynajmniej 10 ocen. 
# TEST




















