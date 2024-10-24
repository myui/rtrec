import os
import asyncio
import boto3
from botocore.exceptions import ClientError
import signal
from typing import List, Dict, Set
from rtrec.models import Fast_SLIM_MSE as SlimMSE

kinesis_client = boto3.client('kinesis')
training_lock = asyncio.Lock()

recommender = SlimMSE(alpha=0.1, beta=1.0, lambda1=0.0002, lambda2=0.0001, min_value=-5, max_value=10)

async def fetch_shard_data(stream_name: str, stop_event: asyncio.Event, active_tasks: Dict[str, asyncio.Task]):
    """Continuously discover shards and manage tasks for them."""
    while not stop_event.is_set():
        try:
            response = kinesis_client.describe_stream(StreamName=stream_name)
            shards = response['StreamDescription']['Shards']

            # Add tasks for new shards
            for shard in shards:
                shard_id = shard['ShardId']
                if shard_id not in active_tasks:
                    print(f"Starting task for new shard: {shard_id}")
                    task = asyncio.create_task(fetch_and_process_records(stream_name, shard_id, stop_event))
                    active_tasks[shard_id] = task

            # Cancel tasks for removed shards
            active_shard_ids = {shard['ShardId'] for shard in shards}
            for shard_id in list(active_tasks.keys()):
                if shard_id not in active_shard_ids:
                    print(f"Stopping task for removed shard: {shard_id}")
                    active_tasks[shard_id].cancel()
                    await active_tasks[shard_id]  # Wait for task to finish
                    del active_tasks[shard_id]

            await asyncio.sleep(5.0)  # Throttle shard discovery

        except ClientError as e:
            print(f"Error discovering shards: {e}")
            await asyncio.sleep(5.0)  # Retry after a delay

async def fetch_and_process_records(stream_name: str, shard_id: str, stop_event: asyncio.Event):
    """Fetch and process records from a specific shard."""
    try:
        shard_iterator = await get_latest_iterator(stream_name, shard_id)

        while not stop_event.is_set():
            try:
                response = kinesis_client.get_records(ShardIterator=shard_iterator, Limit=10000)
                records = response['Records']
                shard_iterator = response.get('NextShardIterator')

                if records:
                    print(f"Processing {len(records)} records from shard {shard_id}")
                    await process_records(records)

                await asyncio.sleep(1.0)  # Respect API limits

            except kinesis_client.exceptions.ExpiredIteratorException:
                print(f"Shard iterator expired for {shard_id}. Resetting...")
                shard_iterator = await get_latest_iterator(stream_name, shard_id)

            except kinesis_client.exceptions.ProvisionedThroughputExceededException:
                print(f"Throughput limit exceeded on shard {shard_id}. Backing off...")
                await asyncio.sleep(2.0)

    except ClientError as e:
        print(f"Error fetching records from shard {shard_id}: {e}")
    except asyncio.CancelledError:
        print(f"Task for shard {shard_id} cancelled.")
    finally:
        print(f"Task for shard {shard_id} shutting down.")

async def get_latest_iterator(stream_name: str, shard_id: str) -> str:
    """Get the latest shard iterator for a shard."""
    response = kinesis_client.get_shard_iterator(
        StreamName=stream_name,
        ShardId=shard_id,
        ShardIteratorType='LATEST'
    )
    return response['ShardIterator']

async def process_records(records: List[Dict]):
    """Synchronously process records with a lock."""
    async with training_lock:
        await asyncio.to_thread(run_task, records)

def run_task(records: List[Dict]):
    """Blocking task to simulate training."""
    for record in records:
        data = record['Data']
        # Process the data
        print(f"Processing record: {data.decode()}")
        # Update the recommender model
        recommender.fit(data.decode())

async def shutdown(stop_event: asyncio.Event, active_tasks: Dict[str, asyncio.Task]):
    """Signal all tasks to stop and wait for completion."""
    print("Shutting down gracefully...")
    stop_event.set()  # Signal tasks to stop

    # Cancel all running tasks
    for task in active_tasks.values():
        task.cancel()

    # Wait for tasks to complete, handling cancellation exceptions
    await asyncio.gather(*active_tasks.values(), return_exceptions=True)
    print("Shutdown complete.")

def signal_handler(stop_event: asyncio.Event, active_tasks: Dict[str, asyncio.Task]):
    """Handle SIGINT/SIGTERM signals."""
    print("Received shutdown signal.")
    asyncio.create_task(shutdown(stop_event, active_tasks))

async def main(stream_name: str):
    """Main entry point for the consumer."""
    stop_event = asyncio.Event()
    active_tasks: Dict[str, asyncio.Task] = {}

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler, stop_event, active_tasks)
    loop.add_signal_handler(signal.SIGTERM, signal_handler, stop_event, active_tasks)

    await fetch_shard_data(stream_name, stop_event, active_tasks)

if __name__ == "__main__":
    stream_name = os.getenv("KINESIS_STREAM_NAME")
    if not stream_name:
        raise ValueError("KINESIS_STREAM_NAME environment variable not set")
    asyncio.run(main(stream_name))
