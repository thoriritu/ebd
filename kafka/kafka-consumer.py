import json
from kafka import KafkaConsumer

# Kafka settings
BROKER_URL = 'localhost:9092'
IOT_TOPIC = 'iot-stream'
SCADA_TOPIC = 'scada-stream'
MES_TOPIC = 'mes-stream'


def create_consumer(broker_url: str, topics: list) -> KafkaConsumer:
    """Create and configure a Kafka consumer subscribing to multiple topics."""
    consumer = KafkaConsumer(
        *topics,
        bootstrap_servers=[broker_url],
        auto_offset_reset='earliest',  # start from beginning
        enable_auto_commit=True,
        group_id='my-consumer-group',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    return consumer


def consume_and_print(consumer: KafkaConsumer):
    """Poll messages endlessly and print them."""
    print(f"Subscribed to: {consumer.subscription()}\nListening for messages...\n")
    for message in consumer:
        topic = message.topic
        partition = message.partition
        offset = message.offset
        value = message.value
        print(f"[{topic}][partition:{partition}][offset:{offset}] :: {value}")


def main():
    topics = [IOT_TOPIC, SCADA_TOPIC, MES_TOPIC]
    consumer = create_consumer(BROKER_URL, topics)
    consume_and_print(consumer)


if __name__ == '__main__':
    main()