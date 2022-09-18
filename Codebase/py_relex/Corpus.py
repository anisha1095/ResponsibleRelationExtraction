import spacy
import en_core_web_sm

spacy_nlp = en_core_web_sm.load()


class Corpus:
    def __init__(self, raw_text):
        self.raw_text = raw_text

    def sentence_segmentation(self):
        doc = spacy_nlp(self.raw_text)
        return [sent.text for sent in doc.sents]