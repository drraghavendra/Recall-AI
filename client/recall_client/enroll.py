from recall_client.api import ApiError, RecallApi
from recall_client.audio import record


def enroll_voice(server_url: str, seconds: float, sample_rate: int) -> None:
    print(f"Recording {seconds:.0f}s — talk normally, as you would in a conversation...")
    pcm = record(seconds, sample_rate)
    print("Uploading...")
    try:
        RecallApi(server_url).enroll_self_voice(pcm)
    except ApiError as e:
        raise SystemExit(f"Enrollment failed: {e}") from e
    print("Voice enrolled: your speech will be tagged as 'self' and kept out of other people's personas.")
