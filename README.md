
### Przykładowe zapytanie

```
curl -X 'POST' \
  'http://localhost:8000/search' \
  -H 'accept: application/json' \
  -H 'X-Auth-Token: 1234567890' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Jakich klientów obsługuje Alfavox ?"
  }'
```


```
curl -X 'POST' \
  'http://localhost:8000/search' \
  -H 'accept: application/json' \
  -H 'X-Auth-Token: 1234567890' \
  -H 'Content-Type: application/json' \
  -d '{
        "provider": "anthropic",
        "model": "claude-3-opus-20240229",
        "question": "Jakich klientów obsługuje Alfavox ?"
      }'

```

### Endpoint do sprawdzania dostępnych modeli:



Teraz możesz:

1. Wybrać model w żądaniu:
```json
{
    "question": "What is quantum computing?",
    "provider": "anthropic",
    "model": "claude-3-opus-20240229"
}
```

2. Sprawdzić dostępne providery i modele przez endpoint /providers:
```bash
curl -H "X-Auth-Token: your-token" http://localhost:8000/providers
```

Dostaniesz odpowiedź w stylu:
```json
{
    "openai": {
        "available": true,
        "models": ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"],
        "default_model": "gpt-4-turbo-preview"
    },
    "anthropic": {
        "available": true,
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
        "default_model": "claude-3-opus-20240229"
    },
    "ollama": {
        "available": true,
        "models": ["llama2", "mistral", "mixtral"],
        "default_model": "llama2"
    }
}
```

Jeśli model nie zostanie wybrany w żądaniu, używany jest model domyślny zdefiniowany w config.yaml dla danego providera.



### `.env`

```
# API Authentication
API_KEY_NAME=X-Auth-Token
API_KEY=

# OpenAI Configuration
OPENAI_API_KEY=
OPENAI_ORG_ID=

# Anthropic Configuration
ANTHROPIC_API_KEY=

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_API_KEY=

# Google Search Configuration
GOOGLE_CSE_KEY=
GOOGLE_CSE_ID=
```