from html.parser import HTMLParser
from io import StringIO
import unicodedata
import re

def add_space_after_tag(s: str) -> str:
    return s.replace('>', '> ')

class HTML_tag_stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_url_parts(s: str) -> str:
    for part in ['.com','.nl','https:','http:','www.','.de','.co.uk', '.eu']:
        if part in s:
            s = s.replace(part,' ')
    return s

def strip_tags(html: str) -> str:
    s = HTML_tag_stripper()
    s.feed(html)
    return s.get_data()

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def strip_punctuation(s: str) -> str:
    for ch in ['\\','`','*','_','{','}','[',']','(',')','>','#','+','.','!','$','\'']:
        if ch in s:
            s = s.replace(ch,' ')
    return s

def strip_non_alphabetical(s: str) -> str:
    return re.sub(r'[^a-zA-Z -]+', ' ', s)

def strip_extra_whitespace(s: str) -> str:
    return " ".join(s.split())

def clean_text(text: str) -> str:
    text = str(text) # cast to str
    text = add_space_after_tag(text) # before stripping tags, lets add a space after each html tag
    text = strip_tags(text) # strip html
    text = strip_url_parts(text) # strip domain extensions
    text = strip_accents(text) # remove diacritics
    text = text.lower() #lowercase
    text = strip_punctuation(text) # punctuation
    text = strip_non_alphabetical(text) # alphanummeric
    text = strip_extra_whitespace(text) # extra whitespace
    # some kind of data anonimization to remove Phone numbers and street addresses
    # TODO: now dealt with by removing all numbers
    return text