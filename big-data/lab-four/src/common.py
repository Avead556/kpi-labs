BOOTSTRAP = "localhost:9092"
TOPIC = "telecom-client-service-time"
PARTITIONS = 1
REPLICATION = 1
GROUP = "service-time-metrics"

CHANNELS = {
    "call-center": (240, 90),
    "chat": (420, 150),
    "store": (600, 200),
    "self-service": (60, 20),
}
REQUESTS = ["tariff change", "billing question", "no signal", "SIM replacement",
            "roaming", "internet setup", "contract termination"]
