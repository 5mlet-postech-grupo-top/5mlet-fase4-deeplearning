#!/usr/bin/env python3
"""
Simple load generator for the LSTM API to produce traffic for Prometheus/Grafana.
Usage:
  python generate_load.py --host http://localhost:8000 --concurrency 50 --duration 60

This script issues a mix of GET and POST requests against common endpoints to produce
successful and error responses for metrics testing.
"""
import argparse
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import requests

DEFAULT_ENDPOINTS = [
    ("GET", "/health"),
    ("GET", "/metrics"),
    # a POST to /predict may produce 4xx/5xx depending on model/data state — useful for errors
    ("POST", "/predict"),
    # invalid endpoint to generate errors
    ("GET", "/not-found-endpoint")
]

# default symbols to include in download/train tests
DEFAULT_SYMBOLS = [
    "AAPL",
    "KLBN4.SA",
    "PETR4.SA",
    "VALE3.SA",
    "BBAS3.SA",
    "BBSE3.SA",
    "CXSE3.SA",
    "CYRE3.SA",
    "CSMG3.SA",
]

stats_lock = threading.Lock()
stats = {"requests": 0, "errors": 0}


def worker(host: str, duration: int, sleep_interval: float):
    end = time.time() + duration
    session = requests.Session()

    while time.time() < end:
        method, path = random.choice(DEFAULT_ENDPOINTS)
        url = host.rstrip("/") + path
        try:
            if method == "GET":
                r = session.get(url, timeout=5)
            else:
                # simple body for predict; many responses might be 4xx if no data/model
                r = session.post(url, json={"symbol": "AAPL", "last_n_days": 60, "n_days": 1}, timeout=10)

            with stats_lock:
                stats["requests"] += 1
                if r.status_code >= 500 or r.status_code < 200:
                    stats["errors"] += 1
        except Exception:
            with stats_lock:
                stats["requests"] += 1
                stats["errors"] += 1
        time.sleep(sleep_interval)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:8000", help="Base URL of the app")
    p.add_argument("--concurrency", type=int, default=20, help="Number of concurrent worker threads")
    p.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    p.add_argument("--rate", type=float, default=10.0, help="Approx requests per second total")
    p.add_argument("--per-endpoint", type=int, default=0, help="Number of requests to send for each endpoint (overrides duration/rate)")
    p.add_argument("--symbols", type=str, default="", help="Comma-separated list of symbols to include (overrides default list)")
    args = p.parse_args()

    # parse symbols (comma separated) or use default
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = DEFAULT_SYMBOLS

    # compute per-thread sleep to achieve approx desired rate
    if args.concurrency <= 0:
        args.concurrency = 1

    per_thread_rps = max(args.rate / args.concurrency, 0.1)
    sleep_interval = 1.0 / per_thread_rps

    print(f"Starting load: host={args.host} concurrency={args.concurrency} duration={args.duration}s target_rps={args.rate} per_endpoint={args.per_endpoint}")

    # If per-endpoint count is requested, build a job queue and run workers that
    # consume items until the queue is empty. This sends exactly N requests per endpoint.
    if args.per_endpoint and args.per_endpoint > 0:
        q = queue.Queue()
        # base endpoints (exclude any /stocks/* entries to avoid static AAPL duplication)
        base_endpoints = []
        for method, path in DEFAULT_ENDPOINTS:
            if path.startswith("/stocks/"):
                continue
            base_endpoints.append((method, path))

        # enqueue non-symbol endpoints
        for method, path in base_endpoints:
            for _ in range(args.per_endpoint):
                q.put((method, path, None))

        # add per-symbol endpoints: predict (body), download, train, status
        for sym in symbols:
            for _ in range(args.per_endpoint):
                q.put(("POST", "/predict", sym))
                q.put(("POST", f"/stocks/{sym}/download", sym))
                q.put(("POST", f"/stocks/{sym}/train", sym))
                q.put(("GET", f"/stocks/{sym}/status", sym))

        print(f"Queue jobs={q.qsize()} (requests total={q.qsize()})")

        def worker_queue(host: str, q: queue.Queue):
            session = requests.Session()
            while True:
                try:
                    method, path, sym = q.get_nowait()
                except queue.Empty:
                    break
                url = host.rstrip("/") + path
                try:
                    # handle GET
                    if method == "GET":
                        r = session.get(url, timeout=10)
                    else:
                        # method is POST; use symbol-aware payloads
                        if path == "/predict":
                            payload = {"symbol": (sym or "AAPL"), "last_n_days": 60, "n_days": 1}
                            r = session.post(url, json=payload, timeout=20)
                        elif path.endswith("/download"):
                            payload = {"start_date": "2018-01-01"}
                            r = session.post(url, json=payload, timeout=60)
                        else:
                            # train or other POSTs
                            r = session.post(url, timeout=20)

                    with stats_lock:
                        stats["requests"] += 1
                        if r.status_code >= 500 or r.status_code < 200:
                            stats["errors"] += 1
                except Exception:
                    with stats_lock:
                        stats["requests"] += 1
                        stats["errors"] += 1
                finally:
                    q.task_done()

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for _ in range(args.concurrency):
                ex.submit(worker_queue, args.host, q)

            # wait until queue is processed
            try:
                q.join()
            except KeyboardInterrupt:
                print("Interrupted by user")

    else:
        print(f"Each thread rps~{per_thread_rps:.2f} sleep_interval={sleep_interval:.3f}s")

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = []
            for _ in range(args.concurrency):
                futures.append(ex.submit(worker, args.host, args.duration, sleep_interval))

            # progress reporting
            start = time.time()
            try:
                while time.time() - start < args.duration:
                    time.sleep(5)
                    with stats_lock:
                        r = stats["requests"]
                        e = stats["errors"]
                    elapsed = int(time.time() - start)
                    print(f"elapsed={elapsed}s requests={r} errors={e}")
            except KeyboardInterrupt:
                print("Interrupted by user")

    print("Load test finished")
    with stats_lock:
        print(f"Total requests={stats['requests']} errors={stats['errors']}")


if __name__ == '__main__':
    main()
