from google.cloud import texttospeech
import os

# Set the path to your valid credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/srinija/Downloads/calendar-chatbot-project-4952ba51de3c.json"

def synthesize_text(text_input, output_audio_file):
    # Create a Text-to-Speech client
    client = texttospeech.TextToSpeechClient()

    # Set the text input
    synthesis_input = texttospeech.SynthesisInput(text=text_input)

    # Configure the voice
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-D",  # You can change the voice
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Configure the audio output
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3  # Or LINEAR16 for WAV
    )

    # Perform the request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Save the output audio file
    with open(output_audio_file, "wb") as out:
        out.write(response.audio_content)
        print(f"---Successfully executed--- Audio content written to file: {output_audio_file}")

# Example usage
synthesize_text("Hello Srinija! This is your Google Cloud Text to Speech test.", "output.mp3")
