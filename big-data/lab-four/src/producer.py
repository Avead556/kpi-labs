import argparse
import json
import random
import time
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

from common import BOOTSTRAP, CHANNELS, PARTITIONS, REPLICATION, REQUESTS, TOPIC


def ensure_topic():
    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP, client_id="lab4-admin")
    try:
        admin.create_topics([NewTopic(TOPIC, num_partitions=PARTITIONS,
                                      replication_factor=REPLICATION)])
        print(f"topic {TOPIC} created: {PARTITIONS} partition, replication {REPLICATION}")
    except TopicAlreadyExistsError:
        print(f"topic {TOPIC} already exists")
    finally:
        admin.close()


def client_record(number, rng):
    channel = rng.choice(list(CHANNELS))
    mean, spread = CHANNELS[channel]
    seconds = max(5.0, rng.gauss(mean, spread))
    return {
        "client_id": f"C-{number:06d}",
        "channel": channel,
        "request": rng.choice(REQUESTS),
        "service_time_s": round(seconds, 1),
        "served_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--count", type=int, default=100)
    parser.add_argument("-d", "--delay", type=float, default=0.05)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    ensure_topic()
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        acks="all",
    )
    rng = random.Random(args.seed)
    started = time.perf_counter()
    for number in range(args.start, args.start + args.count):
        record = client_record(number, rng)
        meta = producer.send(TOPIC, key=record["client_id"].encode(),
                             value=json.dumps(record).encode("utf-8")).get(timeout=10)
        print(f"sent {record['client_id']} {record['channel']:12} "
              f"{record['service_time_s']:6.1f} s -> partition {meta.partition}, "
              f"offset {meta.offset}")
        if args.delay:
            time.sleep(args.delay)
    producer.flush()
    producer.close()
    print(f"{args.count} records in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
