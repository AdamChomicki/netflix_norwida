import pandas as pd # biblioteka do analizy danych
import numpy as np # biblioteka do obliczeń matematycznych
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # sxikit-learn biblioteka uczenia maszynowego. Zawiera m.in. algorytmy klasyfikacji, regresji, klastrowania.
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer # NLTK – zestaw bibliotek i programów do symbolicznego i statystycznego przetwarzania języka naturalnego.
from surprise import Reader
from surprise import SVD # Rozkład według wartości osobliwych
from surprise import Dataset
from surprise.model_selection import cross_validate

import warnings; warnings.simplefilter('ignore') # czego dotyczy to ostrzeżenie?

filmy_dane = pd.read_csv(r'C:\Users\Adam\Desktop\netflix_dane\filmy_dane.csv')  # wczytanie pliku .csv z danymi takimi jak tytuł, język, data produkcji. Rozmiar (45466, 24).
filmy_dane.head()  # wyswietlenie 5 pierwszych wierszy

filmy_dane['genres'] = filmy_dane['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])  # formuła oceny wazonej

# Następnym krokiem jest ustalenie odpowiedniej wartości m, czyli minimalnej liczby głosów wymaganych do umieszczenia
# na wykresie. Jako wartość odcięcia użyjemy 95. percentyla. Innymi słowy, aby film znalazł się na listach przebojów,
# musi mieć więcej głosów niż co najmniej 95% filmów na liście.
# Zbuduję naszą ogólną listę Top 250 i zdefiniuję funkcję do tworzenia wykresów dla określonego gatunku.

liczba_glosow = filmy_dane[filmy_dane['vote_count'].notnull()]['vote_count'].astype('int')  # wyciągnięcie kolumny 'vote_count' i przypisanie jej typu 'int' (45460, ).
srednia_glosow = filmy_dane[filmy_dane['vote_average'].notnull()]['vote_average'].astype('int')  # wyciągnięcie kolumny 'vote_average' i przypisanie jej typu 'int' (45460, ).
srednia_glosow_mean = srednia_glosow.mean()  #
srednia_glosow_mean  # wyswietlenie sredniej oceny głosów w konsoli?

liczba_glosow_kwantyl = liczba_glosow.quantile(0.85)  # wartosc odciecia 0.95 percentyla # ZMIANA TUTAJ NP. ## zmiana na z 0.95 na 0.85.
liczba_glosow_kwantyl  # wygenerowanie liczby 434.0 w konsoli. Co to jest m?

filmy_dane['year'] = pd.to_datetime(filmy_dane['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)  # ?

zakwalifikowany = filmy_dane[(filmy_dane['vote_count'] >= liczba_glosow_kwantyl) & (filmy_dane['vote_count'].notnull()) & (filmy_dane['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]  # Wycięcie kolumn z nawioasu do nowej tabeli o nazwie 'qualified'.
zakwalifikowany['vote_count'] = zakwalifikowany['vote_count'].astype('int')  # zmiana typu kolumny 'vote_count' na 'int'.
zakwalifikowany['vote_average'] = zakwalifikowany['vote_average'].astype('int')  # zmiana typu kolumny 'vote_count' na 'int'.
zakwalifikowany.shape  # pokazuje rozmiar tabeli 'qualified' w konsoli tj. (2274, 6)

# Dlatego, aby zakwalifikować się do listy przebojów, film musi mieć co najmniej 434 głosów na TMDB.
# Widzimy również, że średnia ocena filmu na TMDB to 5,244 w skali 10. 2274 Filmy kwalifikują się do umieszczenia
# na naszym wykresie.

def ocena_wazona(x):  # ocena wazona
    l_liczba_glosow = x['vote_count']
    s_srednia_glosow = x['vote_average']
    return (l_liczba_glosow / (l_liczba_glosow + liczba_glosow_kwantyl) * s_srednia_glosow) + (liczba_glosow_kwantyl / (liczba_glosow_kwantyl + l_liczba_glosow) * srednia_glosow_mean)

zakwalifikowany['wr'] = zakwalifikowany.apply(ocena_wazona,axis=1)  # dodanie oceny ważonej do kolumny 'qualified'. Teraz ma (2274, 7)

zakwalifikowany = zakwalifikowany.sort_values('wr', ascending=False).head(500)  # ograniczenie w danych tabeli 'qualified' do 250. Teraz jest (250, 7).

zakwalifikowany.head(15)  # wyswietlenie 15 wierszy w konsoli z tytułem i oceną ważoną.

# Widzimy, że trzy filmy Christophera Nolana, Incepcja, Mroczny rycerz i Międzygwiezdny znajdują się na samym szczycie
# naszej listy. Wykres wskazuje również na silne nastawienie użytkowników TMDB do poszczególnych gatunków i reżyserów.
# Skonstruujmy teraz naszą funkcję, która buduje wykresy dla poszczególnych gatunków.
# W tym celu użyjemy rozluźnienia naszych warunków domyślnych do 85. percentyla zamiast 95.

gatunek_filmu = filmy_dane.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1,drop=True)  # pojawienie się tabeli 's' o rozmiarze (91106, ).
gatunek_filmu.name = 'genre'  #
gen_filmy_dane = filmy_dane.drop('genres', axis=1).join(gatunek_filmu)  # utworzenie tabeli 'gen_md' o roziarze (93548, 25).

def buduj_wykres(genre, percentile=0.80):  ## zmiana z 0.85 na 0.80
    df = gen_filmy_dane[gen_filmy_dane['genre'] == genre]
    liczba_glosow = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    srednia_glosow = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    srednia_glosow_mean = srednia_glosow.mean()
    liczba_glosow_kwantyl = liczba_glosow.quantile(percentile)

    zakwalifikowany = df[(df['vote_count'] >= liczba_glosow_kwantyl) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    zakwalifikowany['vote_count'] = zakwalifikowany['vote_count'].astype('int')
    zakwalifikowany['vote_average'] = zakwalifikowany['vote_average'].astype('int')

    zakwalifikowany['wr'] = zakwalifikowany.apply(lambda x: (x['vote_count'] / (x['vote_count'] + liczba_glosow_kwantyl) * x['vote_average']) + (liczba_glosow_kwantyl / (liczba_glosow_kwantyl + x['vote_count']) * srednia_glosow_mean), axis=1)
    zakwalifikowany = zakwalifikowany.sort_values('wr', ascending=False).head(250)

    return

# Zobaczmy, jak działa nasza metoda, wyświetlając 15 najlepszych filmów romantycznych
# (filmy romantyczne prawie w ogóle nie pojawiały się na naszej ogólnej liście najpopularniejszych filmów,
# mimo że są jednym z najpopularniejszych gatunków filmowych).

### buduj_wykres('Romance').head(5) # problem z wyswietleniem 5 wierszy z uwagi na obiekt.

dane_id_male = pd.read_csv(r'C:\Users\Adam\Desktop\netflix_dane\dane_id_male.csv')  # wczytanie danych i utworzenie tabeli 'links_small'.
dane_id_male = dane_id_male[dane_id_male['tmdbId'].notnull()]['tmdbId'].astype('int')  # pozostawienie tylko kolumny 'tmdbl' z tabeli 'links_small'.

filmy_dane = filmy_dane.drop([19730, 29503, 35587])  # usunięcie trzech idków i jednej kolumny tj. rozmiar (45463, 25).

filmy_dane['id'] = filmy_dane['id'].astype('int')  # zmiana kolumny 'id' na 'int'.

smd = filmy_dane[filmy_dane['id'].isin(dane_id_male)]  # utworzenie tablicy 'smd' o roz. (9099, 25). smd - co to za tablica???
smd.shape  # wyswietlenie kształtu w konsoli tj. (9099, 25)

# Rekomendacja oparta na opisie filmu
# Najpierw spróbujmy zbudować rekomendację, korzystając z opisów filmów i sloganów.
# Nie mamy miernika ilościowego, aby ocenić wydajność naszej maszyny, więc będzie to musiało być wykonane jakościowo.

smd['tagline'] = smd['tagline'].fillna('')  # nie wiem co robi
smd['description'] = smd['overview'] + smd['tagline']  # nie wiem co robi
smd['description'] = smd['description'].fillna('')  # nie wiem co robi

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0,stop_words='english')  # pojawienie się nowej tabel 'tf' # co to za tablica 'tf' ???
tfidf_matrix = tf.fit_transform(smd['description'])  # pojawienie się nowej tablicy 'tfidf_matrix' # co ta za tablica 'tfidf_matrix' ???

tfidf_matrix.shape  # wyswietlenie kształtu tabeli w konsoli tj. rozm. (9099, 26124)

# Podobieństwo cosinusowe
# Będę używał podobieństwa cosinusów, aby obliczyć wielkość liczbową, która oznacza podobieństwo między dwoma filmami.
# Ponieważ użyliśmy wektoryzatora TF-IDF, obliczenie iloczynu skalarnego bezpośrednio da nam wynik podobieństwa cosinusów.
# Dlatego użyjemy linear_kernel sklearn zamiast cosine_similarities, ponieważ jest znacznie szybsze.

podobienstwo_cosinusowe = linear_kernel(tfidf_matrix,tfidf_matrix)  # pojawienie się nowej tablicy 'cosine_sim' o rozm. (9099, 9099). # OD TĄD DO ok 340 do wyrzucenia

podobienstwo_cosinusowe[0]  # wyswietlenie tablicy w konsoli

# Mamy teraz macierz podobieństwa cosinusów parami dla wszystkich filmów w naszym zbiorze danych.
# Następnym krokiem jest napisanie funkcji zwracającej 30 najbardziej podobnych filmów na podstawie
# wyniku podobieństwa cosinusowego.

smd = smd.reset_index()
tytuly = smd['title']
indeksy = pd.Series(smd.index, index=smd['title'])  # pojawienie się nowej tablicy 'indicnes' o rozm. (9099, ).

def uzyskane_rekomendacje(title):  # O TYM MOWILISMY
    idx = indeksy[title]
    wynik_symulacji = list(enumerate(podobienstwo_cosinusowe[idx]))
    wynik_symulacji = sorted(wynik_symulacji, key=lambda x: x[1], reverse=True)
    wynik_symulacji = wynik_symulacji[1:31]
    indeksy_filmowe = [i[0] for i in wynik_symulacji]
    return tytuly.iloc[indeksy_filmowe]

# Wszystko gotowe. Spróbujmy teraz uzyskać najlepsze rekomendacje dla kilku filmów i zobaczmy, jak dobre są one.

uzyskane_rekomendacje('Batman Returns').head(10)  # wyswietla filmy w konsoli

uzyskane_rekomendacje('Jurassic Park').head(10)  # wyswietla 10 filmów w konsoli

# Widzimy, że w przypadku Mrocznego Rycerza nasz system jest w stanie zidentyfikować go jako film o Batmanie, 
# a następnie polecić inne filmy o Batmanie jako jego najlepsze rekomendacje. Ale niestety to wszystko, 
# co ten system może obecnie zrobić. Nie jest to zbyt przydatne dla większości ludzi, ponieważ nie bierze 
# pod uwagę bardzo ważnych cech, takich jak obsada, ekipa, reżyser i gatunek, które określają ocenę i popularność filmu. 
# Ktoś, kto lubił Mrocznego Rycerza, prawdopodobnie polubiłby go bardziej z powodu Nolana i nienawidziłby 
# Batmana Forever i każdego innego filmu o niespełniającym standardzie z serii Batman.
# Dlatego będziemy używać znacznie bardziej sugestywnych metadanych niż omówienie i slogan. 
# W następnej podsekcji stworzymy bardziej wyrafinowaną rekomendację, która uwzględni gatunek, słowa kluczowe,
# obsadę i ekipę.
# Rekomendujący oparty na metadanych
# Aby zbudować naszą standardową rekomendację treści opartą na metadanych, będziemy musieli połączyć nasz 
# aktualny zbiór danych z załogą i zestawami danych słów kluczowych. Przygotujmy te dane jako nasz pierwszy krok.

obsada = pd.read_csv(r'C:\Users\Adam\Desktop\netflix_dane\obsada.csv') # pojawienie się nowej tabeli 'credits' o rozm. (45476, 3).
slowa_kluczowe = pd.read_csv(r'C:\Users\Adam\Desktop\netflix_dane\slowa_kluczowe.csv') # pojawienie się nowej tablicy 'keywords' o rozm. (46419, 2).
slowa_kluczowe['id'] = slowa_kluczowe['id'].astype('int') # zmiana kolumny 'id' na typ 'int'.
obsada['id'] = obsada['id'].astype('int') # zmiana kolumny 'id' na typ 'int'.

filmy_dane['id'] = filmy_dane['id'].astype('int') 
filmy_dane.shape # wyswietlenie kształtu w konsoli (45466, 25)
filmy_dane = filmy_dane.merge(obsada, on='id') 
filmy_dane = filmy_dane.merge(slowa_kluczowe, on='id') 

smd = filmy_dane[filmy_dane['id'].isin(dane_id_male)] 
smd.shape # wyswietlenie kształtu w konsoli (9219, 28).

# Mamy teraz obsadę, ekipę, gatunki i napisy końcowe w jednej ramce danych. Porozmawiajmy trochę więcej, 
# korzystając z następujących intuicji:
# Załoga: Spośród ekipy wybierzemy tylko reżysera jako nasz film, ponieważ pozostali nie wnoszą tak dużego 
# wkładu w klimat filmu.
# Obsada: Wybór Cast jest nieco trudniejszy. Mniej znani aktorzy i drugorzędne role nie wpływają tak naprawdę
# na opinię ludzi o filmie. Dlatego musimy wybrać tylko głównych bohaterów i odpowiadających im aktorów. 
# Arbitralnie wybierzemy 3 najlepszych aktorów, którzy pojawią się na liście kredytów.

smd['cast'] = smd['cast'].apply(literal_eval) # nie wiem co robi
smd['crew'] = smd['crew'].apply(literal_eval) # nie wiem co robi
smd['keywords'] = smd['keywords'].apply(literal_eval) # nie wiem co robi
smd['cast_size'] = smd['cast'].apply(lambda x: len(x)) # nie wiem co robi
smd['crew_size'] = smd['crew'].apply(lambda x: len(x)) # nie wiem co robi

def otrzymanie_scenarzysty(x): #
    for i in x:
        if i['job'] == 'Screenplay':
            return i['name']
    return np.nan

smd['screenplay'] = smd['crew'].apply(otrzymanie_scenarzysty) # 
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []) #
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x) #
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []) #
smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x]) #

# Moje podejście do budowania rekomendującego będzie wyjątkowo hakerskie. Planuję stworzyć zrzut metadanych dla każdego filmu, który zawiera gatunki, reżysera, głównych aktorów i słowa kluczowe. Następnie używam wektoryzatora zliczania, aby utworzyć naszą macierz liczenia, tak jak to zrobiliśmy w zalecaniu opisu. Pozostałe kroki są podobne do tego, co zrobiliśmy wcześniej: obliczamy podobieństwa cosinusowe i zwracamy filmy, które są najbardziej podobne.
# Oto kroki, które wykonuję, przygotowując dane o moich gatunkach i napisach:
# Usuń spacje i konwertuj na małe litery ze wszystkich naszych funkcji. W ten sposób nasz silnik nie pomyli Johnny'ego Deppa i Johnny'ego Galeckiego.
# Wspomnij reżysera 3 razy, aby nadać mu większą wagę w stosunku do całej obsady.

smd['screenplay'] = smd['screenplay'].astype('str').apply(lambda x: str.lower(x.replace(" ", ""))) #
smd['screenplay'] = smd['screenplay'].apply(lambda x: [x,x, x]) #

# Słowa kluczowe
# Wykonamy niewielką ilość wstępnego przetwarzania naszych słów kluczowych przed ich użyciem. Pierwszym krokiem jest obliczenie częstości występowania każdego słowa kluczowego, które pojawia się w zbiorze danych.

gatunek_filmu = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True) #
gatunek_filmu.name = 'keyword' #
gatunek_filmu = gatunek_filmu.value_counts() #
gatunek_filmu[:5] # wyswietlenie w konsoli jakis informacji 

# Słowa kluczowe występują w częstotliwościach od 1 do 610. 
# Nie stosujemy słów kluczowych, które występują tylko raz. 
# Dlatego można je bezpiecznie usunąć. Na koniec przekonwertujemy 
# każde słowo na jego rdzeń, aby słowa takie jak Psy i Pies były traktowane tak samo.

gatunek_filmu = gatunek_filmu[gatunek_filmu > 1]
stemmer = SnowballStemmer('english') # utworzenie tablicy 'stemmer'.
stemmer.stem('dogs') # zwrócenie w konsoli nazwy 'dog'.

def filtr_slow_kluczowych(x): # 
    words = []
    for i in x:
        if i in gatunek_filmu:
            words.append(i)
    return words

smd['keywords'] = smd['keywords'].apply(filtr_slow_kluczowych) #
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x]) #
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x]) # usuwane są spacje
smd['soup'] = smd['keywords'] + smd['cast'] + smd['screenplay'] + smd['genres'] # 
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x)) #

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english') # utworzenie tabeli 'count'. Można dodac 1,2,3.
count_matrix = count.fit_transform(smd['soup']) # metoda fit

cosine_sim = cosine_similarity(count_matrix, count_matrix) # zmiany wartosci w tabeli 'cosine_sim' tj. (9219, 9219). Czy słowo występuje czy nie występuje.

smd = smd.reset_index() # 

tytuly = smd['title'] # 
indeksy = pd.Series(smd.index, index=smd['title']) # 

# Ponownie użyjemy funkcji get_recommendations, którą napisaliśmy wcześniej. 
# Ponieważ zmieniły się nasze wyniki podobieństwa cosinusów, spodziewamy się,
# że da nam to inne (i prawdopodobnie lepsze) wyniki. Poszukajmy ponownie The Dark Knight 
# i zobaczmy, jakie rekomendacje otrzymam tym razem.

uzyskane_rekomendacje('The Godfather').head(10) # generuje dane w konsoli. # O FUNKCJI GET RECCOMENDATION 

# Jestem znacznie bardziej zadowolony z rezultatów, które uzyskuję tym razem. 
# Wydaje się, że zalecenia uznały inne filmy Christophera Nolana 
# (ze względu na dużą wagę przypisaną reżyserowi) i umieściły je jako najlepsze rekomendacje. 
# Podobało mi się oglądanie The Dark Knight, a także niektórych innych na liście, 
# w tym Batman Begins, The Prestige i The Dark Knight Rises.
# Możemy oczywiście eksperymentować na tym silniku, wypróbowując różne wagi
# dla naszych funkcji (reżyserzy, aktorzy, gatunki), ograniczając liczbę słów kluczowych, 
# których można użyć w zupie, ważąc gatunki na podstawie ich częstotliwości, 
# pokazując tylko filmy o tym samym języki itp.
# Pozwólcie, że otrzymam również rekomendacje dotyczące innego filmu, Wredne dziewczyny, 
# który jest ulubionym filmem mojej dziewczyny.

uzyskane_rekomendacje('Mean Girls').head(10) # generuje dane w konsoli.

# Popularność i oceny.
# Jedną rzeczą, którą zauważamy w naszym systemie rekomendacji, jest to, 
# że poleca filmy niezależnie od ocen i popularności. Prawdą jest, że 
# Batman i Robin mają wiele podobnych postaci w porównaniu z Mrocznym Rycerzem, 
# ale był to okropny film, którego nie należy nikomu polecać.
# Dlatego dodamy mechanizm usuwania złych filmów i zwracanych filmów, 
# które są popularne i miały dobrą reakcję krytyczną.
# Wezmę 25 najlepszych filmów na podstawie wyników podobieństwa i obliczę głos 
# dla filmu 60-centylowego. Następnie, używając tego jako wartości m, 
# obliczymy ważoną ocenę każdego filmu, korzystając ze wzoru IMDB, tak jak to zrobiliśmy 
# w sekcji Prosty polecający.

#def ulepszone_rekomendacje(title): # O TYM MOWILISMY # może z tego zrezygnować. # możemy zostać consine similitary
#    idx = indeksy[title]
#    wynik_cosunisowy = list(enumerate(cosine_sim[idx]))
#    wynik_cosunisowy = sorted(wynik_cosunisowy, key=lambda x: x[1], reverse=True)
#    wynik_cosunisowy = wynik_cosunisowy[1:26]
#    indeksy_filmowe = [i[0] for i in wynik_cosunisowy]
#    
#    filmy = smd.iloc[indeksy_filmowe][['title', 'vote_count', 'vote_average', 'year']]
#    liczba_glosow = filmy[filmy['vote_count'].notnull()]['vote_count'].astype('int')
#    srednia_glosow = filmy[filmy['vote_average'].notnull()]['vote_average'].astype('int')
#    srednia_glosow_mean = srednia_glosow.mean()
#    liczba_glosow_kwantyl = liczba_glosow.quantile(0.60)
#    zakwalifikowany = filmy[(filmy['vote_count'] >= liczba_glosow_kwantyl) & (filmy['vote_count'].notnull()) & (filmy['vote_average'].notnull())]
#    zakwalifikowany['vote_count'] = zakwalifikowany['vote_count'].astype('int')
#    zakwalifikowany['vote_average'] = zakwalifikowany['vote_average'].astype('int')
#    zakwalifikowany['wr'] = zakwalifikowany.apply(ocena_wazona, axis=1)
#    zakwalifikowany = zakwalifikowany.sort_values('wr', ascending=False).head(10)
#    return zakwalifikowany

#ulepszone_rekomendacje('The Dark Knight') # wyswietlenie w konsoli danych o tytułach filmów.

# Pozwólcie, że poznam także rekomendacje dotyczące Wrednych dziewczyn, ulubionego filmu mojej dziewczyny.

#ulepszone_rekomendacje('Mean Girls') # wyswietlenie w konsoli danych o tytułach filmów.

# Niestety Batman i Robin nie znikają z naszej listy rekomendacji. Wynika to prawdopodobnie z faktu, 
# że ma ocenę 4, która jest tylko nieco poniżej średniej w TMDB. Z pewnością nie zasługuje na 4, 
# kiedy niesamowite filmy, takie jak Mroczny rycerz Powstaje, mają tylko 7. 
# Jednak niewiele możemy z tym zrobić. W związku z tym zakończymy tutaj naszą sekcję dotyczącą 
# rekomendacji opartych na treści i wrócimy do niej, gdy będziemy budować silnik hybrydowy.
# Filtrowanie oparte na współpracy
# Nasz silnik oparty na treści ma poważne ograniczenia. Może tylko sugerować filmy, 
# które są zbliżone do określonego filmu. Oznacza to, że nie jest w stanie wychwycić 
# gustów i przedstawiać rekomendacji dla różnych gatunków.
# Ponadto silnik, który zbudowaliśmy, nie jest tak naprawdę osobisty, ponieważ nie odzwierciedla 
# osobistych upodobań i uprzedzeń użytkownika. Każdy, kto zapyta nasz silnik o rekomendacje 
# oparte na filmie, otrzyma te same rekomendacje dla tego filmu, niezależnie od tego, kim jest.
# Dlatego w tej sekcji użyjemy techniki zwanej Collaborative Filtering, 
# aby przedstawić zalecenia obserwatorom filmów. Filtrowanie zespołowe opiera się na założeniu, 
# że użytkownicy podobni do mnie mogą być wykorzystywani do przewidywania, jak bardzo
# spodoba mi się dany produkt lub usługa, z których korzystali / doświadczyli, a ja nie.
# Nie będę wdrażać filtrowania opartego na współpracy od podstaw. 
# Zamiast tego skorzystam z biblioteki Surprise, która wykorzystywała niezwykle wydajne algorytmy, 
# takie jak dekompozycja wartości osobliwych (SVD), aby zminimalizować
# RMSE (Root Mean Square Error) i dać świetne zalecenia.

czytelnik = Reader() # utworzenie tabeli 'Reader'
oceny = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane/oceny_small.csv') # zaciągnięcie danych. Utworzenie tabeli 'ratings' o roz. (100004, 4).
oceny.head() # wyswietlenie danych w konsoli.
dane = Dataset.load_from_df(oceny[['userId', 'movieId', 'rating']], czytelnik) # O TYM MOWILISMY

# data.split(n_folds=5) # błąd AttributeError: 'DatasetAutoFolds' object has no attribute 'split'
svd = SVD() # pojawienie się nowej tablicy 'svd'

cross_validate (svd, dane, measures=['RMSE', 'MAE'], cv=5, verbose=True) # zmieniłem z 'evaluate(svd, data, measures=['RMSE', 'MAE'])'. Otrzymanie wyniku w konsoli.

# Otrzymujemy średni Root Mean Sqaure Error 0,8963, który jest więcej niż wystarczająco dobry w naszym przypadku. Trenujmy teraz na naszym zbiorze danych i dojdźmy do prognoz.

trainset = dane.build_full_trainset()

# svd.train(trainset) # błąd 'SVD' object has no attribute 'train'
# Wybierzmy użytkownika 5000 i sprawdźmy, jakie oceny wystawił. Raczej użytkownika o id = 1.

oceny[oceny['userId'] == 1] # wyswietlenie danych w konsoli.
svd.predict(1, 302, 3) # wyswietlenie danych w konsoli.

# W przypadku filmu o identyfikatorze 302 otrzymujemy szacunkową prognozę na 2,686. 
# Jedną z zaskakujących cech tego systemu rekomendacji jest to, że nie obchodzi go, 
# czym jest film (lub co zawiera). Działa wyłącznie na podstawie przypisanego identyfikatora 
# filmu i próbuje przewidzieć oceny na podstawie tego, jak inni użytkownicy przewidzieli film.
# W tej sekcji spróbuję zbudować prostą hybrydową rekomendację, która łączy techniki, które wdrożyliśmy 
# w silnikach opartych na treści i opartych na filtrach współpracy. Oto jak to będzie działać:
# Dane wejściowe: identyfikator użytkownika i tytuł filmu
# Wynik: podobne filmy posortowane na podstawie oczekiwanych ocen danego użytkownika.

def konwertuj_int(x): #
    try:
        return int(x)
    except:
        return np.nan
    
    
mapa_id = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane/dane_id_male.csv')[['movieId', 'tmdbId']] # zaciagniecie danych. Utworzenie nowej tabeli 'id_map' o rozm. (9125, 2).
mapa_id['tmdbId'] = mapa_id['tmdbId'].apply(konwertuj_int) #
mapa_id.columns = ['movieId', 'id'] #
mapa_id = mapa_id.merge(smd[['title', 'id']], on='id').set_index('title') #
#id_map = id_map.set_index('tmdbId') # tak było zakomentowane na stronie
mapa_indeksow = mapa_id.set_index('id') # utworzenie nowej tabeli o naziwe 'indices_map' o rozm. (9219, 1)

def hybryda(userId, title): # wskazniki tytułow póxniej bierze id filmów 
    idx = indeksy[title] # movie id = id filmow, druga tab. to id rekomendacji.
    tmdbId = mapa_id.loc[title]['id']
    print(idx)
    movie_id = mapa_id.loc[title]['movieId']
    
    wynik_cosunisowy = list(enumerate(cosine_sim[int(idx)])) # nie ma znaczenia większego
    wynik_cosunisowy = sorted(wynik_cosunisowy, key=lambda x: x[1], reverse=True) # jak wszytskie filmy są podobne do tego jednego. 
    wynik_cosunisowy = wynik_cosunisowy[1:26] # tylko 25 obiecujących folmów pokazano. Wyprowadzić gdzie indziej ten parametrów.
    indeksy_filmowe = [i[0] for i in wynik_cosunisowy] # później znajdujemy indkesy tych filmów.
    
    filmy = smd.iloc[indeksy_filmowe][['title', 'vote_count', 'vote_average', 'year', 'id']] # później patrzymy z którego roku, etc.
    filmy['est'] = filmy['id'].apply(lambda x: svd.predict(userId, mapa_indeksow.loc[x]['movieId']).est) # tu robimy predyckje. Wiesz kim jest nasz user, i powiedz jak spodobało się te 25 filmów.
    filmy = filmy.sort_values('est', ascending=False) # przerobienie funkjci aby wykorzystywała get_recommendaition
    return filmy.head(10)

hybryda(1, 'Avatar') # wygenerowanie danych w kodzie
hybryda(500, 'Avatar') # wygenerowanie danych w kodzie


# jak porównać wyniki naszych rekomendacji
# obejrzały 10 filmów i uznały czy rzeczywicie te filmy są dobre.
# przerobić na raitngs_small być może. To jest w kontekcie oceny.
# jak porównywać modele, jak je oceniają. Do klasteryzacji, do rekomendacji <--.
# Dowiedzieć się tego.
# Częsć userow moze być testerami naszej aplikacji.
