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
import requests

DEFAULT_ENDPOINTS = [
    ("GET", "/health"),
    ("GET", "/metrics"),
    ("GET", "/stocks/AAPL/status"),
    # a POST to /predict may produce 4xx/5xx depending on model/data state — useful for errors
    ("POST", "/predict"),
    # invalid endpoint to generate errors
    ("GET", "/not-found-endpoint")
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
    args = p.parse_args()

    # compute per-thread sleep to achieve approx desired rate
    if args.concurrency <= 0:
        args.concurrency = 1

    per_thread_rps = max(args.rate / args.concurrency, 0.1)
    sleep_interval = 1.0 / per_thread_rps

    print(f"Starting load: host={args.host} concurrency={args.concurrency} duration={args.duration}s target_rps={args.rate}")
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
