# Backend - BTC Scenario Logger

## Rodar local
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## API
- POST `/api/start` inicia o loop (análise contínua)
- POST `/api/stop` para o loop
- POST `/api/reset` limpa logs/contadores
- GET `/api/status` cards + snapshot + último ciclo
- GET `/api/logs?limit=300` logs

> É um logger/estudo (não executa ordens reais).


### Binance endpoint
Por padrão usa `https://data-api.binance.vision` (market data). Você pode trocar definindo a env `BINANCE_SPOT_BASE`.


- POST `/api/start` usa `entry_timeout_seconds` (padrão 600s).
