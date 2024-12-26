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
    try:
        with open('stopwords.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return set(config.get('stopwords', []))
    except Exception as e:
        logger.error(f"Błąd podczas ładowania stopwords: {e}")
        return set()


try:
    nlp = spacy.load('pl_core_news_lg')
except OSError:
    logger.info("Instalacja modelu języka polskiego...")
    from spacy.cli import download
    download('pl_core_news_lg')
    nlp = spacy.load('pl_core_news_lg')


class SearchQueryOptimizer:
    def __init__(self):
        self.nlp = nlp
        self.stop_words = load_stopwords()
        logger.info(f"Załadowano {len(self.stop_words)} stop words")

    def get_synonyms(self, word: str) -> List[str]:
        """Znajduje synonimy dla danego słowa."""
        synonyms = set()
        doc = self.nlp(word)
        
        for token in doc:
            ms = token.vector_norm
            if ms != 0:
                similar_words = []
                for w in token.vocab:
                    if w.is_lower and w.has_vector and w.vector_norm != 0:
                        similarity = token.similarity(w)
                        if similarity > 0.7 and w.text != token.text:
                            similar_words.append((w.text, similarity))
                
                similar_words.sort(key=lambda x: x[1], reverse=True)
                for w, _ in similar_words[:3]:
                    synonyms.add(w)
        
        logger.debug(f"Znalezione synonimy dla '{word}': {list(synonyms)}")
        return list(synonyms)

    def extract_individual_words(self, phrase: str) -> List[str]:
        """Ekstrahuje pojedyncze znaczące słowa z frazy."""
        doc = self.nlp(phrase.lower())
        words = []
        
        for token in doc:
            if (token.text not in self.stop_words and 
                len(token.text) > 2 and 
                token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN']):
                words.append(token.text)
        
        logger.debug(f"Wyekstrahowane słowa z frazy '{phrase}': {words}")
        return words

    def extract_key_phrases(self, text: str) -> List[str]:
        """Wyodrębnia kluczowe frazy z tekstu."""
        doc = self.nlp(text.lower())
        key_phrases = []
        current_phrase = []
        
        for token in doc:
            if token.text not in self.stop_words and len(token.text) > 2:
                if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN']:
                    current_phrase.append(token.text)
                else:
                    if current_phrase:
                        key_phrases.append(' '.join(current_phrase))
                        current_phrase = []
        
        if current_phrase:
            key_phrases.append(' '.join(current_phrase))
        
        logger.debug(f"Wyekstrahowane frazy z tekstu '{text}': {key_phrases}")
        return key_phrases

    def create_extended_queries(self, phrases: List[str]) -> List[str]:
        """Tworzy rozszerzone zapytania, rozbijając frazy na pojedyncze słowa."""
        extended_queries = []
        
        # Najpierw dodajemy oryginalne frazy
        for phrase in phrases:
            if phrase:
                extended_queries.append(f'"{phrase}"')
        
        # Następnie rozbijamy każdą frazę na pojedyncze słowa
        individual_words = []
        for phrase in phrases:
            words = self.extract_individual_words(phrase)
            individual_words.extend(words)
        
        # Dodajemy synonimy dla każdego słowa
        word_variations = {}
        for word in individual_words:
            synonyms = self.get_synonyms(word)
            word_variations[word] = [word] + synonyms

        # Generujemy kombinacje słów i ich synonimów
        for i in range(1, min(len(individual_words) + 1, 3)):  # maksymalnie 2 słowa w kombinacji
            for words_combo in itertools.combinations(individual_words, i):
                variations = [word_variations[word] for word in words_combo]
                for variation in itertools.product(*variations):
                    query = ' '.join(variation)
                    if query not in extended_queries:
                        extended_queries.append(query)

        logger.debug(f"Wygenerowane rozszerzone zapytania: {extended_queries}")
        return extended_queries[:10]  # Ograniczamy liczbę zapytań

    def optimize_query(self, question: str) -> Dict[str, List[str]]:
        """
        Główna funkcja optymalizująca zapytanie.
        Zwraca słownik z zapytaniami podstawowymi i rozszerzonymi.
        """
        # Czyszczenie i normalizacja pytania
        question = re.sub(r'[^\w\s]', ' ', question)
        question = ' '.join(question.split())
        
        # Wyodrębnienie kluczowych fraz
        key_phrases = self.extract_key_phrases(question)
        
        # Tworzenie zapytań podstawowych
        basic_queries = []
        for phrase in key_phrases:
            if phrase:
                basic_queries.append(f'"{phrase}"')
        
        # Tworzenie zapytań rozszerzonych
        extended_queries = self.create_extended_queries(key_phrases)
        
        return {
            "basic_queries": basic_queries,
            "extended_queries": extended_queries
        }

def get_optimized_queries(question: str) -> Dict[str, List[str]]:
    """
    Funkcja pomocnicza zwracająca zoptymalizowane zapytania.
    """
    optimizer = SearchQueryOptimizer()
    queries = optimizer.optimize_query(question)
    
    return {
        "keyword_query": ' '.join(queries["basic_queries"]),
        "context_queries": queries["extended_queries"]
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    question = "Jakie produkty są oferowane?"
    results = get_optimized_queries(question)
    print("Zapytanie podstawowe:", results["keyword_query"])
    print("\nZapytania rozszerzone:")
    for i, query in enumerate(results["context_queries"], 1):
        print(f"{i}. {query}")