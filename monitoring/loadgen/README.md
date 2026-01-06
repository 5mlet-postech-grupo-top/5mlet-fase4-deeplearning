Load generator for monitoring tests

Files
- `generate_load.py` : Python script that issues a mix of GET/POST requests to the app.

Quick start
1. (Optional) Create virtualenv and install deps:

```bash
python -m venv .venv
. .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows PowerShell
python -m pip install -r requirements.txt
```

2. Run a 60s test:

```bash
python monitoring/loadgen/generate_load.py --host http://localhost:8000 --concurrency 30 --duration 60 --rate 200
```

Notes
- The script hits `/health`, `/metrics`, `/stocks/AAPL/status`, `/predict` (POST), and an invalid endpoint to produce errors.
- If you run the app inside `docker-compose` the host is still `http://localhost:8000` because `docker-compose` maps ports to host.
- Adjust `--rate` and `--concurrency` to control the load.
