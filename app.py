from flask import Flask, request, jsonify
import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline
import os

app = Flask(__name__)
recognizer = sr.Recognizer()

# Initialize the translation pipelines
translator = pipeline('translation', model='facebook/nllb-200-3.3B')

def recognize_speech_from_audio_file(file_path):
    audio = AudioSegment.from_file(file_path)
    audio.export("converted.wav", format="wav")

    with sr.AudioFile("converted.wav") as source:
        audio_data = recognizer.record(source)
        try:
            text_query = recognizer.recognize_google(audio_data)
            print("You said:", text_query)
            return text_query
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            return None

@app.route('/recognize', methods=['POST'])
def recognize_speech():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filename = file.filename
    file.save(filename)
    
    print(f'Processing file: {filename}')
    text_query = recognize_speech_from_audio_file(filename)

    if text_query is None:
        return jsonify({"error": "Could not recognize speech"}), 500

    os.remove(filename)
    os.remove("converted.wav")

    # Translate recognized text
    text_translated = translator(text_query, src_lang="eng_Latn", tgt_lang="asm_Beng")[0]['translation_text']
    text_translated2 = translator(text_translated, src_lang="asm_Beng", tgt_lang="hin_Deva")[0]['translation_text']

    response = {
        "recognized_text": text_query,
        "translated_text_asm": text_translated,
        "translated_text_hin": text_translated2
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
