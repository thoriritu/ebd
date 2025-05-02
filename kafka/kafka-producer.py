import json
import time
import threading
import pandas as pd
from kafka import KafkaProducer

# File paths (update as needed)
IOT_CSV = 'data/generated_data/future-iot.csv'
SCADA_CSV = 'data/generated_data/future-scada.csv'
MES_CSV = 'data/generated_data/future-mes.csv'

# Kafka settings
BROKER_URL = 'localhost:9092'
IOT_TOPIC = 'iot-stream'
SCADA_TOPIC = 'scada-stream'
MES_TOPIC = 'mes-stream'


def create_producer(broker_url: str) -> KafkaProducer:
    """Create and configure a Kafka producer."""
    return KafkaProducer(bootstrap_servers=[broker_url])


def load_and_prepare(csv_path: str, date_col: str) -> pd.DataFrame:
    """Load CSV into DataFrame, parse dates, and convert timestamp to ISO string."""
    df = pd.read_csv(csv_path, parse_dates=[date_col], keep_default_na=False, na_values=[])
    # Convert Timestamp to standard string for JSON serialization
    df[date_col] = df[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df


def stream_data(producer: KafkaProducer, df: pd.DataFrame, topic: str, interval: float):
    """
    Send grouped messages at a fixed interval.
    :param df: DataFrame with a Timestamp column as string
    :param topic: Kafka topic to send to
    :param interval: sleep time between groups (in seconds)
    """
    # Group by timestamp string
    for timestamp, group in df.groupby('Timestamp'):
        records = group.to_dict(orient='records')
        for record in records:
            producer.send(topic, json.dumps(record).encode('utf-8'))
        producer.flush()
        print(f"Sent {len(records)} records to '{topic}' for {timestamp}")
        time.sleep(interval)


def main():
    # Initialize producer
    producer = create_producer(BROKER_URL)

    # Load and prepare each dataset
    iot_df = load_and_prepare(IOT_CSV, 'Timestamp')
    scada_df = load_and_prepare(SCADA_CSV, 'Timestamp')
    scada_df['Alarm_Code'] = scada_df['Alarm_Code'].replace('None', '')

    mes_df = load_and_prepare(MES_CSV, 'Timestamp')

    # Create and start threads for each stream
    threads = [
        threading.Thread(target=stream_data, args=(producer, iot_df[:144], IOT_TOPIC, 10)),            # run till 144000 for 24th EOD
        threading.Thread(target=stream_data, args=(producer, scada_df[:96], SCADA_TOPIC, 30)),    # run till 9600 for 24th EOD
        threading.Thread(target=stream_data, args=(producer, mes_df[:24], MES_TOPIC, 60)),        # run till 2400 for 24th EOD
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

        # Cleanup
    print('All threads finished. Closing producer.')
    producer.close()


if __name__ == '__main__':
    main()