from typing import List, Dict
import spacy
import yaml
import logging
from spacy.cli import download
import re

logger = logging.getLogger(__name__)

def load_config() -> dict:
    with open('config.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_stopwords() -> set:
    with open('stopwords.yaml', 'r', encoding='utf-8') as file:
        return set(yaml.safe_load(file).get('stopwords', []))

try:
    nlp = spacy.load('pl_core_news_lg')
except OSError:
    logger.info("Installing Polish language model...")
    download('pl_core_news_lg')
    nlp = spacy.load('pl_core_news_lg')

class SearchQueryOptimizer:
    def __init__(self):
        self.nlp = nlp
        self.stop_words = load_stopwords()
        self.config = load_config()
        self.min_results = self.config.get('search', {}).get('min_results', 10)

    def remove_stopwords(self, text: str) -> str:
        logger.debug("Removing stopwords from: %s", text)
        tokens = self.nlp(text.lower())
        filtered_words = [token.text for token in tokens if token.text not in self.stop_words]
        return ' '.join(filtered_words)

    def get_synonyms(self, text: str) -> List[str]:
        logger.debug("Generating synonyms for: %s", text)
        doc = self.nlp(text.lower())
        synonyms = set()

        for token in doc:
            if token.vector_norm:
                similar_words = []
                for word in token.vocab:
                    if (word.is_lower and word.has_vector and
                        word.vector_norm and word.text != token.text):
                        similarity = token.similarity(word)
                        if similarity > 0.7:
                            similar_words.append((word.text, similarity))

                similar_words.sort(key=lambda x: x[1], reverse=True)
                synonyms.update(word for word, _ in similar_words[:3])

        logger.debug("Found synonyms: %s", list(synonyms))
        return list(synonyms)

    def split_into_words(self, text: str) -> List[str]:
        logger.debug("Splitting into words: %s", text)
        doc = self.nlp(text.lower())
        words = [token.text for token in doc
                if token.text not in self.stop_words
                and len(token.text) > 2]
        return words

    def process_input_text(self, text: str) -> str:
        # Usuwanie znaków specjalnych oprócz myślnika
        processed = re.sub(r'[^\w\s-]', ' ', text)

        # Znajdowanie słów z myślnikiem i otaczanie ich cudzysłowami
        words = processed.split()
        processed_words = []

        for word in words:
            if '-' in word:
                processed_words.append(f'"{word}"')
            else:
                processed_words.append(word)

        return ' '.join(processed_words)

    def optimize_query(self, question: str) -> Dict[str, List[str]]:
        queries = {'keyword_query': '', 'context_queries': []}
        clean_query = self.process_input_text(question)
        clean_query = self.remove_stopwords(clean_query)
        queries['keyword_query'] = clean_query

        synonyms = [syn for word in clean_query.split() for syn in self.get_synonyms(word)]
        individual_words = self.split_into_words(clean_query)

        queries['context_queries'] = list(set(synonyms + individual_words))

        return queries

def get_optimized_queries(question: str) -> Dict[str, List[str]]:
    optimizer = SearchQueryOptimizer()
    return optimizer.optimize_query(question)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    sample_question = "Jakie są najlepsze praktyki programowania?"
    results = get_optimized_queries(sample_question)
    print("\nWyniki optymalizacji zapytania:")
    print("1. Podstawowe:", results['basic'])
    print("2. Rozszerzone:", results['extended'])
    print("3. Pojedyncze słowa:", results['split'])