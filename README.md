

```
curl -X 'POST' \
  'http://localhost:8000/search' \
  -H 'accept: application/json' \
  -H 'X-Auth-Token: 1234567890' \
  -H 'Content-Type: application/json' \
  -d '{\n  "question": "Jakich klientów obsługuje Alfavox ?"
  }'
```