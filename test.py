from stts import (
    text_to_speech_kokoro
)

text = '「もしおれがただ偶然、そしてこうしようというつもりでなくここに立っているのなら、ちょっとばかり絶望するところだな」と、そんなことが彼の頭に思い浮かんだ。'

text_to_speech_kokoro(text, "ja", "temp/kokoro_ja.wav")
