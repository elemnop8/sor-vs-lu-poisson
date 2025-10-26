""" Dieses Modul enthält Hilfsfunktionen zur Verarbeitung von Eingaben,
    zur Zeiterfassung und zur Erstellung von Plots.
    Author: M. Nguyen, E. Tarielashvili.
    pylint Version 3.1.0
    pylint score: 10/10
"""

import math
import time
import matplotlib.pyplot as plt
import numpy as np

def read_number(question: str,
                lower_limit: float = -math.inf,
                upper_limit: float = math.inf,
                data_type: type = float
                ):
    """
    Überprüfung der Nutzeingabe auf ihre Gültigkeit.

    Parameter
    ----------
    question : str 
        Frage, die der Nutzer beantworten soll
    lower_limit : float, standard: -math.inf
        Der Nutzer soll keinen kleineren Wert als diesen angeben können
    upper_limit : float, standard: math.inf
        Der Nutzer soll keinen größeren Wert als diesen angeben können
    data_type : type (<class 'float'> oder <class 'int'>)
        Eingabe des Nutzers soll als dieser Datentyp eingelesen werden können
    
    Returns
    -------
    nutzereingabe : data_type
        Eingegebene Zahl wird im Datentyp data_type zurückgegeben
    """
    wert = True
    while wert:
        try:
            # Nutzereingabe abfragen
            x = input(question)
            # Programmabbruch bei einer leeren Eingabe
            if x == "":
                raise ValueError("Leere Eingabe. Das Programm wurde abgebrochen.")
            nutzereingabe = data_type(x)
            # Überprüfen, ob die Eingabe im gültigen Bereich liegt
            if nutzereingabe > upper_limit:
                print("Ungültige Eingabe. Bitte geben Sie eine Zahl kleinergleich "
                      f"{upper_limit} ein. Zum Abbrechen Eingabe leer lassen.")
            elif nutzereingabe < lower_limit:
                print("Ungültige Eingabe. Bitte geben Sie eine Zahl größergleich"
                      f" {lower_limit} ein. Zum Abbrechen Eingabe leer lassen.")
            else:
                wert = False
                return nutzereingabe
        except ValueError as e:
            if str(e) == "Leere Eingabe. Das Programm wurde abgebrochen.":
                raise
            print("Ungültige Eingabe. Bitte geben Sie eine Zahl mit" +
                      f" dem Datentyp {data_type} ein. Zum Abbrechen Eingabe leer lassen.")
    return None

def zahlenliste():
    """
    Erstellen einer Zahlenliste

    Parameters
    ----------
    None

    Returns
    ---------
    list
    """
    liste = []
    while True:
        eingabe = input("Bitte geben Sie eine positive Zahl ein: \n"
                        "Um die Liste der Zahlen zu beenden, lassen Sie die Eingabe frei \n")
        # Programm wird beendet, wenn der Benutzer 'q' eingibt
        if eingabe == '':
            break
        try:
            zahl = float(eingabe)
            if zahl > 0:
                liste.append(zahl)
                print(f"Die Zahl {zahl} wurde hinzugefügt.")
            else:
                print("Nur positive Zahlen sind erlaubt. Versuche es erneut.")
        # erzeugt einen ValueError, wenn keine Zahl oder nicht 'q' eingeben wurde
        except ValueError:
            print("Ungültige Eingabe. Bitte geben Sie eine positive Zahl ein.")
    return liste

def save_plot(filename=str):
    """
    Speichert das aktuelle Diagramm unter einem angegebenen Dateinamen.

    Fragt den Benutzer, ob er das Diagramm speichern möchte und speichert es 
    anschließend im aktuellen Verzeichnis als PNG-Datei.

    Parameter
    ----------
    filename : str
        Der Dateiname, unter dem der Plot gespeichert wird.
    """
    save = read_number(f"Wollen Sie den Plot unter dem Namen {filename} speichern?\n"
                                "Es sollte keine Datei mit dem gleichen Dateiname im Ordner "+
                                "des Programms vorliegen"+", da diese sonst überschrieben wird."
                                "\n(1 für ja, 2 für nein): ", 1, 2, int)
    if save ==1:
        plt.savefig(filename, dpi=600)
        print(f"Plot gespeichert als {filename}.")

def measure_time(func, *args, num_repeats=5, **kwargs):
    """
    Misst die durchschnittliche Ausführungszeit einer Funktion.

    Diese Funktion führt die angegebene Funktion `func` mehrmals aus und berechnet
    den Durchschnitt der Ausführungszeiten. Dies ist nützlich, um die Performance 
    einer Funktion zu analysieren.

    Parameter
    ----------
    func : function
        Die Funktion, deren Ausführungszeit gemessen werden soll.
    *args : tuple
        Die Argumente, die an die Funktion übergeben werden.
    num_repeats : int, optional
        Die Anzahl der Wiederholungen der Funktion, standardmäßig 5.
    **kwargs : dict
        Zusätzliche benannte Argumente für die Funktion.

    Returns
    -------
    float
        Die durchschnittliche Ausführungszeit in Millisekunden.
    """
    total_time = 0
    for _ in range(num_repeats):
        start_time = time.perf_counter_ns()
        func(*args, **kwargs)
        end_time = time.perf_counter_ns()
        # time in milliseconds
        total_time += (end_time - start_time) / 1000000
    return total_time / num_repeats

def main():
    """
    Hauptfunktion des Programms.

    Diese Funktion dient als Einstiegspunkt und steuert die Interaktionen mit dem
    Benutzer. Sie fragt nach einer Zahl, speichert Plots und misst die Ausführungszeit
    von Funktionen.
    """
    print("Willkommen zum Programm!")

    # Frage den Benutzer nach einer Zahl (z.B., der Anzahl der Iterationen oder einer anderen Zahl)
    number = read_number("Bitte geben Sie eine Zahl ein "
                         "(z.B. Anzahl der Iterationen): ", 1, 100, int)
    print(f"Sie haben {number} gewählt.")

    # Bereite einige Beispielwerte für py_logspace vor
    start_exp = read_number("Geben Sie den Startexponenten für die "
                            "logarithmische Verteilung an: ", -5, 5, int)
    stop_exp = read_number("Geben Sie den Endexponenten für die "
                           "logarithmische Verteilung an: ", -5, 5, int)
    num_values = read_number("Geben Sie die Anzahl der Werte in "
                             "der logarithmischen Verteilung an: ", 2, 100, int)

    # Logarithmisch verteilte Zahlen erzeugen
    values = py_logspace(start_exp, stop_exp, num_values)
    print(f"Logarithmisch verteilte Zahlen: {values}")

    # Beispiel-Funktion definieren, die die Zeitmessung verwendet
    def sample_function(a, b):
        """ Eine einfache Beispiel-Funktion, die die Zeitmessung nutzt. """
        return a ** b

    # Messen der Zeit für die Ausführung der Beispiel-Funktion
    time_taken = measure_time(sample_function, 2, 10, num_repeats=3)
    print(f"Durchschnittliche Ausführungszeit der Funktion: {time_taken:.4f} ms")

    # Diagramm erstellen (Beispiel für die Visualisierung von Werten)
    plt.plot(values, np.log(values), label="log(x)")
    plt.xlabel("Werte")
    plt.ylabel("log(Werte)")
    plt.title("Beispielplot")
    plt.legend()
    # Frage den Benutzer, ob er den Plot speichern möchte
    save_plot("log_values_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
