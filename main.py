# analiza, obróbka i wizualizacja danych
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')
import random as rnd

# uczenie maszynowe
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import os
import keras
from keras.models import Sequential
from keras.layers import Dense

train_df = pd.read_csv(r"data\train.csv")  # dane treningowe
test_df = pd.read_csv(r"data\test.csv")  # dane testowe
titanic_data = [train_df, test_df]  # wszystkie dane

# dzielenie danych na numeryczne (stosunkowe) i kategoryczne (skala porządkowa)
df_num = train_df[['Age', 'SibSp', 'Parch', 'Fare']]  # dane numeryczne
df_kat = train_df[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
titanic_data = [train_df, test_df]

for data in titanic_data:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for data in titanic_data:
    data['Title'] = data['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')


def model_f(all, tr, te):
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for data in all:
        data['Title'] = data['Title'].map(title_mapping)
        data['Title'] = data['Title'].fillna(0)

    for data in all:
        data['Sex'].fillna('unknown', inplace=True)
        data['Sex'] = data['Sex'].map({'female': 1, 'male': 0, 'unknown': -1}).astype(int)

    tr['Sex'] = all[0]['Sex']
    te['Sex'] = all[1]['Sex']

    tr['AgeBand'] = pd.cut(tr['Age'], 8)

    tr['AgeBand'] = tr['AgeBand'].cat.codes
    tr['AgeBand'] = tr['AgeBand'].astype(int)

    te['AgeBand'] = pd.cut(te['Age'], 8)
    te['AgeBand'] = te['AgeBand'].cat.codes
    te['AgeBand'] = te['AgeBand'].astype(int)

    medians = tr.groupby(['Pclass', 'Sex'])['Age'].median()

    def fill_age(row):
        if pd.isnull(row['Age']):
            return medians[row['Pclass'], row['Sex']]
        return row['Age']

    for data in all:
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    tr['Age'] = tr.apply(fill_age, axis=1)

    for data in all:
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

    tr = tr.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    te = te.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    all = [tr, te]

    for data in all:
        data['AgeClass'] = data['AgeBand'] * data['Pclass']

    tr = tr.drop(columns=['Age'])
    te = te.drop(columns=['Age'])

    for data in all:
        data['Embarked'].fillna('S', inplace=True)

    for data in all:
        data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    tr['Embarked'] = all[0]['Embarked']
    te['Embarked'] = all[1]['Embarked']

    tr['FareBand'] = pd.qcut(tr['Fare'], 5)

    tr['FareBand'] = tr['FareBand'].cat.codes
    tr['FareBand'] = tr['FareBand'].astype(int)

    te['FareBand'] = pd.cut(te['Fare'], 8)
    te['FareBand'] = te['FareBand'].cat.codes
    te['FareBand'] = te['FareBand'].astype(int)

    tr = tr.drop(columns=['Fare'])
    te = te.drop(columns=['Fare'])

    return all, tr, te


main_menu = {
    1: 'Wyświetl dane',
    2: 'Analizuj dane',
    # 3: 'Przewidywanie (po wykonaniu program zakończy działanie)',  # ze względu na zmodelowane dane
    3: 'Wyjdź',
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

# predict_menu = {
#     1: 'Sieć neuronowa (z konfiguracją parametrów)',
#     2: 'Drzewo decyzyjne',
#     3: 'Regresja logistyczna',
#     4: 'Random Forest',
#     5: 'Support Vector Machines',
#     6: 'Naive Bayes classifier',
#     7: 'k-Nearest Neighbors (KNN)',
#     8: 'Perceptron',
#     9: 'Relevance Vector Machine (RVM)',
#     10: 'Menu główne',
# }


def print_menu(menu):
    for key in menu.keys():
        print(key, '--', menu[key])


def view():
    print_menu(view_menu)
    option = int(input("Wybierz opcję: "))
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
    option = int(input("Wybierz opcję: "))
    if option == 1:
        for i in df_num.columns:
            plt.hist(df_num[i], bins=20)
            plt.title(i)
            plt.show()
    elif option == 2:
        sns.heatmap(df_num.corr())
        plt.show()
    elif option == 3:
        plt.hist(df_num['Fare'], bins=20)
        plt.title("Fare")
        plt.show()
        time.sleep(wait_time)
    elif option == 4:
        plt.hist(df_num['Sex'], bins=20)
        plt.title("Sex")
        plt.show()
        time.sleep(wait_time)
    elif option == 5:
        plt.hist(df_num['Parch'], bins=20)
        plt.title("Parch")
        plt.show()
        time.sleep(wait_time)
    elif option == 6:
        plt.hist(df_num['SibSp'], bins=20)
        plt.title("SibSp")
        plt.show()
        time.sleep(wait_time)
    elif option == 7:
        plt.hist(df_num['Age'], bins=20)
        plt.title("Age")
        plt.show()
        time.sleep(wait_time)
    elif option == 8:
        title_survived = train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
        sns.barplot(x='Title', y='Survived', data=title_survived)
        plt.ylabel("Survival")
        plt.xlabel("Title")
        plt.title("Survival Rates based on Title")
        plt.show()
        time.sleep(wait_time)
    elif option == 9:
        pass
    else:
        print('Nieprawidłowa opcja. Proszę wprowadzić liczbę pomiędzy 1 a 6.')


# def predict():
#     all_data, train, test = model_f(titanic_data, train_df, test_df)
#     print_menu(predict_menu)
#     option = int(input("Wybierz metodę przewidywania: "))
#     X_train = train.drop("Survived", axis=1)
#     Y_train = train["Survived"]
#     X_test = test.drop("PassengerId", axis=1).copy()
#
#     model = keras.Sequential()
#
#     if option == 1:
#         print("Konfiguracja:")
#         liczba_warstw = int(input("Podaj liczbę ukrytych warstw"))
#         i = 0
#         layer = []
#         while i < liczba_warstw:
#             layer = int(input(f"Podaj liczbę węzłów w wartwie nr {i + 1}"))
#             model.add(Dense(units=layer, activation='relu', input_dim=X_train.shape[1]))  # pierwsza ukryta wartwa
#             i += 1
#         print("Wybierz funkcje aktywacjyną: ")
#         print("1 -- relu")
#         print("2 -- sigmoid")
#         print("3 -- softmax")
#         activation = input("Podaj numer: ")
#         if activation == 1:
#             model.add(Dense(units=1, activation='relu'))
#             model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         elif activation == 2:
#             model.add(Dense(units=1, activation='sigmoid'))
#             model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         elif activation == 3:
#             model.add(Dense(units=1, activation='softmax'))
#             model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         score = model.evaluate(X_train, Y_train, verbose=0)  # ewaluacja modelu
#         print("Training Accuracy: ", round(score[1] * 100, 4))
#     elif option == 2:
#         decision_tree = DecisionTreeClassifier()
#         decision_tree.fit(X_train, Y_train)
#         Y_pred = decision_tree.predict(X_test)
#         acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 4)
#         print(acc_decision_tree)
#     elif option == 3:
#         logreg = LogisticRegression()
#         logreg.fit(X_train, Y_train)
#         Y_pred = logreg.predict(X_test)
#         print(round(accuracy_score(Y_train, Y_pred) * 100, 4))
#     elif option == 4:
#         random_forest = RandomForestClassifier(n_estimators=100)
#         random_forest.fit(X_train, Y_train)
#         Y_pred = random_forest.predict(X_test)
#         random_forest.score(X_train, Y_train)
#         acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
#         print(acc_random_forest)
#     elif option == 5:
#         svc = SVC()
#         svc.fit(X_train, Y_train)
#         Y_pred = svc.predict(X_test)
#         acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
#         print(acc_svc)
#     elif option == 6:
#         gaussian = GaussianNB()
#         gaussian.fit(X_train, Y_train)
#         Y_pred = gaussian.predict(X_test)
#         acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
#         print(acc_gaussian)
#     elif option == 7:
#         knn = KNeighborsClassifier(n_neighbors=3)
#         knn.fit(X_train, Y_train)
#         Y_pred = knn.predict(X_test)
#         acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
#         print(acc_knn)
#     elif option == 8:
#         perceptron = Perceptron()
#         perceptron.fit(X_train, Y_train)
#         Y_pred = perceptron.predict(X_test)
#         acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
#         print(acc_perceptron)
#     elif option == 9:
#         pass
#     else:
#         print('Nieprawidłowa opcja. Proszę wprowadzić liczbę pomiędzy 1 a 6.')
#     exit()


clear = lambda: os.system('cls')
print("Witaj w analizie danych dot. pasażerów Titanica")
print("Konfiguracja:")
try:
    wait_time = int(input("Jak długo chciałbyś widzieć wybrane dane (w sekundach) "))
except:
    wait_time = 5

while True:
    print_menu(main_menu)
    menu_option = int(input("Wybierz opcję: "))
    if menu_option == 1:
        view()
    elif menu_option == 2:
        analyze()
    # elif menu_option == 3:
    #     predict()
    elif menu_option == 3:
        print('Dziękujemy za korzystanie z programu!')
        exit()
    else:
        print('Nieprawidłowa opcja. Proszę wprowadzić liczbę pomiędzy 1 a 4.')
