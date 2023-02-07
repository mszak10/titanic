# analiza, obróbka i wizualizacja danych
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# uczenie maszynowe
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
import keras
from keras.layers import Dense

# inne
import time

sns.set_style('darkgrid')

# Importowanie zbiorów danych treningowych i testowych
train_df = pd.read_csv(r"data\train.csv")  # dane treningowe
test_df = pd.read_csv(r"data\test.csv")  # dane testowe

# dzielenie danych na numeryczne (stosunkowe) i kategoryczne (skala porządkowa)
df_num = train_df[['Age', 'SibSp', 'Parch', 'Fare']]  # dane numeryczne
df_kat = train_df[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

# Usunięcie kolumn "Ticket" i "Cabin"
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

# Tworzenie listy zbiorów danych do przetwarzania
titanic_data = [train_df, test_df]

# Wyciąganie tytułu z każdej nazwy
for data in titanic_data:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Zastępowanie rzadkich tytułów pojedynczą wartością "Rare"
for data in titanic_data:
    data['Title'] = data['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

# Grupowanie "Age" w zakresy
train_df['AgeBand'] = pd.cut(train_df['Age'], 8)

# Grupowanie "Fare" w zakresy
train_df['FareBand'] = pd.qcut(train_df['Fare'], 5)


def model_f(all, tr, te):
    # Celem tej funkcji jest wstępne przetworzenie i oczyszczenie zbioru danych Titanic.
    # Mapuje zmienne kategoryczne takie jak 'Title' i 'Sex' na wartości liczbowe,
    # Wypełnia brakujące wartości, tworzy nowe cechy i usuwa niepotrzebne kolumny.
    # Końcowym rezultatem jest oczyszczony i wstępnie przetworzony zbiór danych gotowy do modelowania.
    # Wszystko to jest wewnątrz funkcji, aby zostało wywołane tylko wewnątrz funkcji predict()
    # Gdyby te działania były wykonane poza funkcją, przed główną częścią programu..
    # .. użytkownik wydziałby dane wyłącznie w postaci arbitralnych numerków (np Title=1 zamiast Title=Mr)
    # all = titanic_data ; tr = train_df ; te = test_df

    # mapowanie kolumny "Title" obu ramek danych na wartości numeryczne.
    # Mapowanie jest zdefiniowane w słowniku title_mapping.
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for data in all:
        data['Title'] = data['Title'].map(title_mapping)
        data['Title'] = data['Title'].fillna(0)

    # kolumna 'Sex' jest wypełniana brakującymi wartościami i również jest mapowana na wartości numeryczne.
    for data in all:
        data['Sex'].fillna('unknown', inplace=True)
        data['Sex'] = data['Sex'].map({'female': 1, 'male': 0, 'unknown': -1}).astype(int)

    tr['Sex'] = all[0]['Sex']
    te['Sex'] = all[1]['Sex']

    # Kolumna "AgeBand" jest tworzona poprzez pogrupowanie kolumny "Age" w przedziały
    # i zakodowanie tych przedziałów jako wartości numerycznych.
    tr['AgeBand'] = tr['AgeBand'].cat.codes
    tr['AgeBand'] = tr['AgeBand'].astype(int)

    te['AgeBand'] = pd.cut(te['Age'], 8)
    te['AgeBand'] = te['AgeBand'].cat.codes
    te['AgeBand'] = te['AgeBand'].astype(int)

    # Kolumna "AgeClass" jest tworzona przez pomnożenie kolumn "AgeBand" i "Pclass".
    for data in titanic_data:
        data['AgeClass'] = data['AgeBand'] * data['Pclass']

    # Kolumny "Name" i "PassengerId" usuwa się z ramki danych "tr", a kolumnę "Name" usuwa się z ramki danych "te".
    tr = tr.drop(['Name', 'PassengerId'], axis=1)
    te = te.drop(['Name'], axis=1)
    all = [train_df, test_df]

    # Mediana wieku jest obliczana dla każdej kombinacji "Pclass" i "Sex" i używana do wypełnienia wszelkich
    # brakujących wartości w kolumnie "Age" w obu ramkach danych.
    medians = tr.groupby(['Pclass', 'Sex'])['Age'].median()

    def fill_age(row):
        if pd.isnull(row['Age']):
            return medians[row['Pclass'], row['Sex']]
        return row['Age']

    tr['Age'] = tr.apply(fill_age, axis=1)

    # Tworzy się nową kolumnę "IsAlone", aby wskazać, czy pasażer podróżował sam czy nie.
    for data in all:
        data['IsAlone'] = 0
        data.loc[data['SibSp'] + data['Parch'] + 1 == 1, 'IsAlone'] = 1

    # Z obu ramek danych usuwa się kolumny "Parch", "SibSp" i "Age".
    tr = tr.drop(['Parch', 'SibSp'], axis=1)
    te = te.drop(['Parch', 'SibSp'], axis=1)
    all = [tr, te]

    tr = tr.drop(columns=['Age'])
    te = te.drop(columns=['Age'])

    # Wypełnia się brakujące wartości w kolumnie "Embarked" i odwzorowuje kolumnę na wartości liczbowe.
    for data in all:
        data['Embarked'].fillna('S', inplace=True)

    for data in all:
        data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    tr['Embarked'] = all[0]['Embarked']
    te['Embarked'] = all[1]['Embarked']

    # Kolumna FareBand jako wartości liczbowe
    tr['FareBand'] = tr['FareBand'].cat.codes
    tr['FareBand'] = tr['FareBand'].astype(int)

    te['FareBand'] = pd.cut(te['Fare'], 8)
    te['FareBand'] = te['FareBand'].cat.codes
    te['FareBand'] = te['FareBand'].astype(int)

    # Na koniec kolumna "Fare" jest usuwana z obu ramek danych
    tr = tr.drop(columns=['Fare'])
    te = te.drop(columns=['Fare'])

    # Zwracane są wstępnie przetworzone ramki danych
    return all, tr, te


def importance(X, im):
    importance_index = np.argsort(im)

    names = [X.columns[i] for i in importance_index]

    plt.barh(range(X.shape[1]), im[importance_index])
    plt.yticks(range(X.shape[1]), names)
    plt.title("Wykres ukazujący znaczenie poszczególnych cech")
    plt.xlabel("Waga cechy")
    plt.ylabel("Nazwy cech")
    plt.show()


# Słowniki z opcjami do menu
main_menu = {
    1: 'Wyświetl dane',
    2: 'Analizuj dane',
    3: 'Przewidywanie (po wykonaniu program zakończy działanie)',  # ze względu na zmodelowane dane
    4: 'Wyjdź',
}

view_menu = {
    1: 'Wyświetl pierwsze 5 rzędów',
    2: 'Wyświetl ostatnie 5 rzędów',
    3: 'Wyświetl informacje nt. zbioru danych',
    4: 'Wyświetl statystyki danych treningowych liczbowych',
    5: 'Wyświetl statystyki danych treningowych kategorycznych',
    6: 'Menu główne',
}

analyze_menu = {
    1: 'Wykres słupkowy danych numerycznych',
    2: 'Mapa cieplna korelacji',
    3: 'Wykres cena biletu, a przeżycie',
    4: 'Wykres płeć pasasżera, a przeżycie',
    5: 'Wykres liczba rodzeństwa, a przeżycie',
    6: 'Wykres liczba rodziców/dzieci, a przeżycie',
    7: 'Wykres wiek, a przeżycie',
    8: 'Wykres tytuł (Mr/Mrs/Miss..), a przeżycie',
    9: 'Menu główne',
}

predict_menu = {
    1: 'Sieć neuronowa (z konfiguracją parametrów)',
    2: 'Drzewo decyzyjne',
    3: 'Regresja logistyczna',
    4: 'Random Forest',
    5: 'Support Vector Machines',
    6: 'Naive Bayes classifier',
    7: 'k-Nearest Neighbors (KNN)',
    8: 'Perceptron',
    9: 'Relevance Vector Machine (RVM)',
    10: 'Menu główne',
}


# Drukowanie powyższych menu (typ menu jako argument)
def print_menu(menu):
    for key in menu.keys():
        print(key, '--', menu[key])


def view():
    print_menu(view_menu)  # Pobieranie od użytkownika opcji, którą chce wybrać
    try:
        option = int(input("Wybierz opcję: "))
    except:
        print("Wystąpił błąd, spróbuj ponownie")
        option = 6
    """
    Opis opcji:
    1: 'Wyświetl pierwsze 5 rzędów',
    2: 'Wyświetl ostatnie 5 rzędów',
    3: 'Wyświetl informacje nt. zbioru danych',
    4: 'Wyświetl statystyki danych treningowych liczbowych',
    5: 'Wyświetl statystyki danych treningowych kategorycznych',
    6: 'Menu główne',
    """
    # Wybieranie wybranej opcji
    if option == 1:
        print(train_df.head())
        time.sleep(wait_time)
    elif option == 2:
        print(train_df.tail())
        time.sleep(wait_time)
    elif option == 3:
        print('Dane treningowe')
        print(train_df.info())
        print('=' * 40)
        print('Dane testowe')
        print(test_df.info())
        time.sleep(wait_time)
    elif option == 4:
        print(train_df.describe())
        time.sleep(wait_time)
    elif option == 5:
        print(train_df.describe(include=['O']))
        time.sleep(wait_time)
    elif option == 6:
        pass
    else:
        print('Nieprawidłowa opcja. Proszę wprowadzić liczbę pomiędzy 1 a 6.')


def analyze():
    print_menu(analyze_menu)
    try:
        option = int(input("Wybierz opcję: "))
    except:
        print("Wystąpił błąd, spróbuj ponownie")
        option = 9

    """
    Opis opcji:
    1: 'Wykres słupkowy danych numerycznych',
    2: 'Mapa cieplna korelacji',
    3: 'Wykres cena biletu, a przeżycie',
    4: 'Wykres płeć pasasżera, a przeżycie',
    5: 'Wykres liczba rodzeństwa, a przeżycie',
    6: 'Wykres liczba rodziców/dzieci, a przeżycie',
    7: 'Wykres wiek, a przeżycie',
    8: 'Wykres tytuł (Mr/Mrs/Miss..), a przeżycie',
    9: 'Menu główne',
    """
    if option == 1:
        for i in df_num.columns:
            plt.hist(df_num[i], bins=20)
            plt.title(i)
            plt.show()
    elif option == 2:
        sns.heatmap(df_num.corr())
        plt.show()
    elif option == 3:
        sns.barplot(x='FareBand', y='Survived', data=train_df)
        plt.title("Cena biletu, a przeżycie")
        plt.show()
        time.sleep(wait_time)  # Oczekiwanie przez określony czas przed powrotem do głównego menu
    elif option == 4:
        sns.barplot(x='Sex', y='Survived', data=train_df)
        plt.title("Płeć, a przeżycie")
        plt.show()
        time.sleep(wait_time)
    elif option == 5:
        sns.barplot(x='Parch', y='Survived', data=train_df)
        plt.title("Liczba rodzeństwa, a przeżycie")
        plt.show()
        time.sleep(wait_time)
    elif option == 6:
        sns.barplot(x='SibSp', y='Survived', data=train_df)
        plt.title("Liczba dzieci/rodziców, a przeżycie")
        plt.show()
        time.sleep(wait_time)
    elif option == 7:
        sns.barplot(x='AgeBand', y='Survived', data=train_df)
        plt.title("Wiek, a przeżycie")
        plt.show()
        time.sleep(wait_time)
    elif option == 8:
        title_survived = train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
        sns.barplot(x='Title', y='Survived', data=title_survived)
        plt.ylabel("Survival")
        plt.xlabel("Title")
        plt.title("Tytuł, a przeżycie")
        plt.show()
        time.sleep(wait_time)
    elif option == 9:
        pass
    else:
        print('Nieprawidłowa opcja. Proszę wprowadzić liczbę pomiędzy 1 a 6.')


def predict():
    all_data, train, test = model_f(titanic_data, train_df, test_df)
    print("Zmodelowane dane:")
    print(train.head())
    print_menu(predict_menu)

    try:
        option = int(input("Wybierz opcję: "))
    except:
        print("Wystąpił błąd, spróbuj ponownie")
        option = 10

    X_train = train.drop("Survived", axis=1)
    Y_train = train["Survived"]
    # X_test = test.drop("PassengerId", axis=1).copy()
    # submission = pd.read_csv(r"data\gender_submission.csv")
    # Y_test = submission["Survived"]

    """
    Opis opcji:
    1: 'Sieć neuronowa (z konfiguracją parametrów)',
    2: 'Drzewo decyzyjne',
    3: 'Regresja logistyczna',
    4: 'Random Forest',
    5: 'Support Vector Machines',
    6: 'Naive Bayes classifier',
    7: 'k-Nearest Neighbors (KNN)',
    8: 'Perceptron',
    9: 'Relevance Vector Machine (RVM)',
    10: 'Menu główne',
    """
    if option == 1:
        model = keras.Sequential()
        print("Konfiguracja:")
        liczba_warstw = int(input("Podaj liczbę ukrytych warstw: "))
        i = 0
        while i < liczba_warstw:
            layer = int(input(f"Podaj liczbę węzłów w wartwie nr {i + 1}: "))
            model.add(Dense(units=layer, activation='relu', input_dim=X_train.shape[1]))  # pierwsza ukryta wartwa
            # if i == 0:
            #     model.add(Dense(units=layer, activation='relu', input_dim=X_train.shape[1]))  # pierwsza ukryta wartwa
            # else:
            #     model.add(Dense(units=layer, activation='relu', input_dim=X_train.shape[1]))  # pozostałe warstwy
            i += 1
        print("Wybierz funkcje aktywacjyną na węźle wyjściowym: ")
        print("1 -- relu")
        print("2 -- sigmoid")
        print("3 -- softmax")
        activation = input("Podaj numer: ")
        if activation == 1:
            model.add(Dense(units=1, activation='relu'))
        elif activation == 2:
            model.add(Dense(units=1, activation='sigmoid'))
        elif activation == 3:
            model.add(Dense(units=1, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=100, batch_size=10)
        score = model.evaluate(X_train, Y_train, verbose=0)  # ewaluacja modelu
        print("Training Accuracy: ", round(score[1] * 100, 4))
    elif option == 2:
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, Y_train)
        # Y_pred = decision_tree.predict(X_test)
        acc_decision_tree = round(decision_tree.score(X_train, Y_train * 100, 4))
        print("Training Accuracy: ", acc_decision_tree)

        # Nanoszenie wagi cech na wykres
        importances = decision_tree.feature_importances_
        importance(X_train, importances)
    elif option == 3:
        logreg = LogisticRegression()
        logreg.fit(X_train, Y_train)
        # Y_pred = logreg.predict(X_test)
        acc_logreg = round(logreg.score(X_train, Y_train), 4)
        print("Training Accuracy: ", acc_logreg * 100)

        importances = logreg.coef_[0]
        importance(X_train, importances)
    elif option == 4:
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(X_train, Y_train)
        # Y_pred = random_forest.predict(X_test)
        random_forest.score(X_train, Y_train)
        acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
        print("Training Accuracy: ", acc_random_forest)

        importances = random_forest.feature_importances_
        importance(X_train, importances)
    elif option == 5:
        svc = SVC()
        svc.fit(X_train, Y_train)
        # Y_pred = svc.predict(X_test)
        acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
        print("Training Accuracy: ", acc_svc)

        importances = svc.coef_[0]
        importance(X_train, importances)
    elif option == 6:
        gaussian = GaussianNB()
        gaussian.fit(X_train, Y_train)
        # Y_pred = gaussian.predict(X_test)
        acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
        print("Training Accuracy: ", acc_gaussian)

        importances = gaussian.theta_[-1]
        importance(X_train, importances)
    elif option == 7:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, Y_train)
        # Y_pred = knn.predict(X_test)
        acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
        print("Training Accuracy: ", acc_knn)

        result = permutation_importance(knn, X_train, Y_train, n_repeats=10, random_state=0)
        importances = result.importances_mean
        importance(X_train, importances)
    elif option == 8:
        perceptron = Perceptron()
        perceptron.fit(X_train, Y_train)
        # Y_pred = perceptron.predict(X_test)
        acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
        print("Training Accuracy: ", acc_perceptron)

        importances = perceptron.coef_[0]
        importance(X_train, importances)
    elif option == 9:
        pass
    else:
        print('Nieprawidłowa opcja. Proszę wprowadzić liczbę pomiędzy 1 a 9.')
    exit()


#  clear = lambda: os.system('cls')
print("Witaj w analizie danych dot. pasażerów Titanica")
print("Konfiguracja:")
try:
    wait_time = int(input("Jak długo chciałbyś widzieć wybrane dane (w sekundach) "))
except:
    print("Wystąpił błąd, ustawiam domyślne 2 sekundy")
    wait_time = 2

while True:
    print_menu(main_menu)

    try:
        menu_option = int(input("Wybierz opcję: "))
    except:
        print("Wystąpił błąd, spróbuj ponownie")
        continue

    if menu_option == 1:
        view()
    elif menu_option == 2:
        analyze()
    elif menu_option == 3:
        predict()
    elif menu_option == 4:
        print('Dziękuję za korzystanie z programu!')
        exit()
    else:
        print('Nieprawidłowa opcja. Proszę wprowadzić liczbę pomiędzy 1 a 4.')
