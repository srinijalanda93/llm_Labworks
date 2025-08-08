from google.cloud import storage, speech
import os

# Set credentials path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/srinija/Downloads/calendar-chatbot-project-4952ba51de3c.json"

def transcribe_audio(bucket_name, audio_file_name, output_file_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Download audio file from bucket
        local_audio_path = f"/tmp/{audio_file_name}"
        print("⬇️ Downloading audio from GCS...")
        bucket.blob(audio_file_name).download_to_filename(local_audio_path)

        # Transcribe
        print("🎤 Transcribing audio...")
        speech_client = speech.SpeechClient()
        with open(local_audio_path, 'rb') as f:
            audio_content = f.read()

        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        response = speech_client.recognize(config=config, audio=audio)

        # Save transcription
        transcript = "\n".join([result.alternatives[0].transcript for result in response.results])
        local_transcript_path = f"/tmp/{output_file_name}"
        with open(local_transcript_path, 'w') as f:
            f.write(transcript)

        # Upload to GCS
        print("⬆️ Uploading transcription to GCS...")
        bucket.blob(output_file_name).upload_from_filename(local_transcript_path)
        print(f"✅ Transcription saved to gs://{bucket_name}/{output_file_name}")

    except Exception as e:
        print("❌ ERROR:", str(e))

# Call the function
transcribe_audio("jk1997", "audio.wav", "transcription.txt")
