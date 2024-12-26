from typing import List, Tuple, Dict
import re
import spacy
from spacy.cli import download
import itertools
import logging
import yaml


logger = logging.getLogger(__name__)

def load_stopwords() -> set:
    """Ładuje stop words z pliku YAML."""
    logger.debug("Rozpoczęcie ładowania stopwords z pliku YAML")
    try:
        with open('stopwords.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            stopwords = set(config.get('stopwords', []))
            logger.debug(f"Pomyślnie załadowano stopwords. Przykładowe słowa: {list(stopwords)[:5]}")
            return stopwords
    except Exception as e:
        logger.error(f"Błąd podczas ładowania stopwords: {e}")
        return set()


try:
    logger.debug("Próba załadowania modelu językowego spaCy")
    nlp = spacy.load('pl_core_news_lg')
    logger.debug("Model językowy spaCy załadowany pomyślnie")
except OSError:
    logger.info("Instalacja modelu języka polskiego...")
    from spacy.cli import download
    download('pl_core_news_lg')
    nlp = spacy.load('pl_core_news_lg')
    logger.debug("Model językowy spaCy zainstalowany i załadowany pomyślnie")


class SearchQueryOptimizer:
    def __init__(self):
        logger.debug("Inicjalizacja SearchQueryOptimizer")
        self.nlp = nlp
        self.stop_words = load_stopwords()
        logger.info(f"Załadowano {len(self.stop_words)} stop words")
        logger.debug(f"Przykładowe stop words: {list(self.stop_words)[:5]}")

    def get_synonyms(self, word: str) -> List[str]:
        """Znajduje synonimy dla danego słowa."""
        logger.debug(f"Rozpoczęcie wyszukiwania synonimów dla słowa: '{word}'")
        synonyms = set()
        doc = self.nlp(word)
        
        for token in doc:
            ms = token.vector_norm
            logger.debug(f"Analizowanie tokenu '{token.text}', vector_norm: {ms}")
            if ms != 0:
                similar_words = []
                processed_words = 0
                for w in token.vocab:
                    processed_words += 1
                    if processed_words % 10000 == 0:
                        logger.debug(f"Przetworzono {processed_words} słów z vocabulary")
                    if w.is_lower and w.has_vector and w.vector_norm != 0:
                        similarity = token.similarity(w)
                        if similarity > 0.7 and w.text != token.text:
                            similar_words.append((w.text, similarity))
                            logger.debug(f"Znaleziono podobne słowo: '{w.text}' z podobieństwem: {similarity:.3f}")
                
                similar_words.sort(key=lambda x: x[1], reverse=True)
                for w, sim in similar_words[:3]:
                    synonyms.add(w)
                    logger.debug(f"Dodano synonim: '{w}' z podobieństwem: {sim:.3f}")
        
        logger.debug(f"Zakończono wyszukiwanie synonimów dla '{word}'. Znaleziono: {list(synonyms)}")
        return list(synonyms)

    def extract_individual_words(self, phrase: str) -> List[str]:
        """Ekstrahuje pojedyncze znaczące słowa z frazy."""
        logger.debug(f"Rozpoczęcie ekstrakcji słów z frazy: '{phrase}'")
        doc = self.nlp(phrase.lower())
        words = []
        
        for token in doc:
            logger.debug(f"Analiza tokenu: '{token.text}', POS: {token.pos_}")
            if (token.text not in self.stop_words and 
                len(token.text) > 2 and 
                token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN']):
                words.append(token.text)
                logger.debug(f"Dodano słowo: '{token.text}'")
        
        logger.debug(f"Zakończono ekstrakcję słów z frazy '{phrase}'. Wynik: {words}")
        return words

    def extract_key_phrases(self, text: str) -> List[str]:
        """Wyodrębnia kluczowe frazy z tekstu."""
        logger.debug(f"Rozpoczęcie ekstrakcji fraz kluczowych z tekstu: '{text}'")
        doc = self.nlp(text.lower())
        key_phrases = []
        current_phrase = []
        
        for token in doc:
            logger.debug(f"Analiza tokenu: '{token.text}', POS: {token.pos_}")
            if token.text not in self.stop_words and len(token.text) > 2:
                if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN']:
                    current_phrase.append(token.text)
                    logger.debug(f"Dodano token '{token.text}' do aktualnej frazy")
                else:
                    if current_phrase:
                        phrase = ' '.join(current_phrase)
                        key_phrases.append(phrase)
                        logger.debug(f"Dodano frazę: '{phrase}'")
                        current_phrase = []
        
        if current_phrase:
            phrase = ' '.join(current_phrase)
            key_phrases.append(phrase)
            logger.debug(f"Dodano ostatnią frazę: '{phrase}'")
        
        logger.debug(f"Zakończono ekstrakcję fraz. Wynik: {key_phrases}")
        return key_phrases

    def create_extended_queries(self, phrases: List[str]) -> List[str]:
        """Tworzy rozszerzone zapytania, rozbijając frazy na pojedyncze słowa."""
        logger.debug(f"Rozpoczęcie tworzenia rozszerzonych zapytań dla fraz: {phrases}")
        extended_queries = []
        
        # Najpierw dodajemy oryginalne frazy
        for phrase in phrases:
            if phrase:
                extended_queries.append(f'"{phrase}"')
                logger.debug(f"Dodano oryginalną frazę: '{phrase}'")
        
        # Następnie rozbijamy każdą frazę na pojedyncze słowa
        individual_words = []
        for phrase in phrases:
            words = self.extract_individual_words(phrase)
            individual_words.extend(words)
            logger.debug(f"Dodano słowa z frazy '{phrase}': {words}")
        
        # Dodajemy synonimy dla każdego słowa
        word_variations = {}
        for word in individual_words:
            logger.debug(f"Wyszukiwanie synonimów dla słowa: '{word}'")
            synonyms = self.get_synonyms(word)
            word_variations[word] = [word] + synonyms
            logger.debug(f"Znalezione warianty dla '{word}': {word_variations[word]}")

        # Generujemy kombinacje słów i ich synonimów
        for i in range(1, min(len(individual_words) + 1, 3)):
            logger.debug(f"Generowanie kombinacji {i} słów")
            for words_combo in itertools.combinations(individual_words, i):
                variations = [word_variations[word] for word in words_combo]
                for variation in itertools.product(*variations):
                    query = ' '.join(variation)
                    if query not in extended_queries:
                        extended_queries.append(query)
                        logger.debug(f"Dodano nowe zapytanie: '{query}'")

        logger.debug(f"Zakończono generowanie zapytań. Liczba zapytań: {len(extended_queries)}")
        return extended_queries[:10]

    def optimize_query(self, question: str) -> Dict[str, List[str]]:
        """
        Główna funkcja optymalizująca zapytanie.
        Zwraca słownik z zapytaniami podstawowymi i rozszerzonymi.
        """
        logger.debug(f"Rozpoczęcie optymalizacji zapytania: '{question}'")
        
        # Czyszczenie i normalizacja pytania
        cleaned_question = re.sub(r'[^\w\s]', ' ', question)
        normalized_question = ' '.join(cleaned_question.split())
        logger.debug(f"Znormalizowane pytanie: '{normalized_question}'")
        
        # Wyodrębnienie kluczowych fraz
        key_phrases = self.extract_key_phrases(normalized_question)
        logger.debug(f"Wyodrębnione frazy kluczowe: {key_phrases}")
        
        # Tworzenie zapytań podstawowych
        basic_queries = []
        for phrase in key_phrases:
            if phrase:
                basic_queries.append(f'"{phrase}"')
                logger.debug(f"Dodano podstawowe zapytanie: '{phrase}'")
        
        # Tworzenie zapytań rozszerzonych
        extended_queries = self.create_extended_queries(key_phrases)
        logger.debug(f"Utworzono {len(extended_queries)} zapytań rozszerzonych")
        
        result = {
            "basic_queries": basic_queries,
            "extended_queries": extended_queries
        }
        logger.debug(f"Wynik optymalizacji: {result}")
        return result

def get_optimized_queries(question: str) -> Dict[str, List[str]]:
    """
    Funkcja pomocnicza zwracająca zoptymalizowane zapytania.
    """
    logger.debug(f"Wywołanie get_optimized_queries z pytaniem: '{question}'")
    optimizer = SearchQueryOptimizer()
    queries = optimizer.optimize_query(question)
    
    result = {
        "keyword_query": ' '.join(queries["basic_queries"]),
        "context_queries": queries["extended_queries"]
    }
    logger.debug(f"Wynik get_optimized_queries: {result}")
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    question = "Jakie produkty są oferowane?"
    logger.debug(f"Rozpoczęcie testu z pytaniem: '{question}'")
    results = get_optimized_queries(question)
    print("Zapytanie podstawowe:", results["keyword_query"])
    print("\nZapytania rozszerzone:")
    for i, query in enumerate(results["context_queries"], 1):
        print(f"{i}. {query}")
    logger.debug("Zakończenie testu")