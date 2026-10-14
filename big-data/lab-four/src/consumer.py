import argparse
import json
import math
import time
import uuid

from kafka import KafkaConsumer

from common import BOOTSTRAP, GROUP, TOPIC


class Average:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.low = math.inf
        self.high = -math.inf

    def add(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.low = min(self.low, value)
        self.high = max(self.high, value)

    @property
    def std(self):
        return math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idle", type=float, default=10.0)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--from-start", action="store_true")
    args = parser.parse_args()

    group = f"{GROUP}-{uuid.uuid4().hex[:6]}" if args.from_start else GROUP
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=group,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=int(args.idle * 1000) if args.idle else float("inf"),
    )
    print(f"subscribed to {TOPIC} as group {group}")
    total = Average()
    channels = {}
    started = time.perf_counter()
    try:
        for message in consumer:
            record = json.loads(message.value.decode("utf-8"))
            seconds = record["service_time_s"]
            total.add(seconds)
            channels.setdefault(record["channel"], Average()).add(seconds)
            print(f"p{message.partition} offset {message.offset:4}: {message.key.decode()} "
                  f"{record['channel']:12} {seconds:6.1f} s | average over "
                  f"{total.count} clients {total.mean:6.1f} s")
            if total.count % args.every == 0:
                print(f"  -- {total.count} clients: average {total.mean:.1f} s "
                      f"({total.mean / 60:.2f} min), std {total.std:.1f} s, "
                      f"min {total.low:.1f} s, max {total.high:.1f} s")
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

    print(f"\n== Summary: {total.count} clients in {time.perf_counter() - started:.1f} s ==")
    if total.count:
        print(f"average service time {total.mean:.1f} s ({total.mean / 60:.2f} min), "
              f"std {total.std:.1f} s, min {total.low:.1f} s, max {total.high:.1f} s")
        for name, stats in sorted(channels.items(), key=lambda kv: -kv[1].mean):
            print(f"  {name:12} {stats.count:4} clients, average {stats.mean:6.1f} s")


if __name__ == "__main__":
    main()
