import pickle
import fasttext
import smart_open
import numpy as np
from Preprocessing.data_preprocessor import load_txt_from_s3, DataPreprocessor

S3_URL = 'facebook_fasttext/wiki_word_vectors.pkl'

with smart_open.open(S3_URL, 'rb') as f:
    words_dict = pickle.load(f)

ZERO_WORD = np.zeros(300)
for v in words_dict.values():
    ZERO_WORD = np.zeros_like(v)
    break

def run():
    loader = DataPreprocessor()
    gersti = loader.load_from_s3('GerSti/cleaned.pkl')
    gersti['fasttext'] = gersti['attr_page_title'].apply(handler)


def handler(input):
    text = input.strip().lower()
    vectors = [
        words_dict[token] / np.linalg.norm(words_dict[token]) if token in words_dict else ZERO_WORD for token in fasttext.tokenize(text)
    ]
    if len(vectors) > 0:
        mean_vector = np.array(vectors).mean(axis=0)
    else:
        mean_vector = ZERO_WORD
    return mean_vector.tolist()
