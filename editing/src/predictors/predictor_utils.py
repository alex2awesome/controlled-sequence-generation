import unidecode

def clean_text(text, special_chars=["\n", "\t"]):
    text = unidecode.unidecode(text)
    for char in special_chars:
        text = text.replace(char, " ")
    return text

