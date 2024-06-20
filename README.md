# GIFTS
### Projekt nr. 2 na przedmiot inteligencja obliczeniowa
##### 4 semestr Informatyki Praktycznej UG

---

## Opis projektu

Projekt ten obejmuje system rekomendacji prezentów oparty na analizie tekstu przy użyciu modelu językowego BERT. System analizuje preferencje użytkowników oraz opisy produktów, oblicza podobieństwo kosinusowe między nimi i generuje rekomendacje na podstawie tych wyników.

## Funkcje

1. **Przetwarzanie tekstu**: Wykorzystanie NLTK do tokenizacji, usuwania stopwords oraz przygotowania tekstu do analizy.

2. **Modele językowe**: Użycie BERT do generowania reprezentacji tekstowych i obliczania podobieństwa kosinusowego między preferencjami użytkownika a opisami produktów.

3. **Obliczanie podobieństwa**: Wykorzystanie podobieństwa kosinusowego do określenia dopasowania produktów do preferencji użytkownika.

4. **Generowanie rekomendacji**: Na podstawie wyników analizy generowanie listy rekomendowanych produktów.

5. **Wizualizacja wyników**: Tworzenie wykresów prezentujących wyniki analizy danych, takich jak rozkład podobieństwa czy wizualizacje osadzeń produktów.

## Instalacja i uruchomienie

1. **Instalacja zależności**:
```bash
pip install -r requirements.tx
```

2. **Uruchomienie projektu:**
Uruchom skrypt `embeddings-enh.py`, który generuje osadzenia produktów na podstawie preferencji użytkowników oraz oblicza podobieństwa i generuje rekomendacje.


3. **Wizualizacja wyników:**
Sprawdź wygenerowane wykresy i pliki wynikowe w katalogu `product_categories_data`, które zawierają wyniki analizy oraz rekomendacje dla poszczególnych produktów.

4. **Uruchomienie aplikacji:**

## Serwer (Python)
Część serwerowa (oraz przykłady procesu trenowania) bazują na interpreterze Python. W katalogu `gifts_backend` uruchamiamy plik `app.py` za pomoca polecenia:
```bash
python3 app.py
```

## Klient (JavaScript)
Część klienta bazuje na JavaScriptowym frameworku Next.js. Zanim uruchomimy aplikację, tak samo jak w przypadku Pythona, musimy pobrać zależności. Wykonujemy to w katalogu `gifts_frontend` za pomocą polecenia:
```bash
npm install
```

Następnie w tym samym katalogu `gifts_frontend` uruchamiamy aplikację za pomocą polecenia:
```bash
npm run dev
```