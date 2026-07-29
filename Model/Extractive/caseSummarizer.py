import nltk
import re
import math
import numpy as np
import operator
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'maxent_ne_chunker'])
wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                continue
    return continuous_chunk

def generate_summary(text):
    with open('dictionary.txt', 'r', encoding='utf-8') as f:
        legal_terms = [line.strip() for line in f]
    
    text = re.sub(r'\n+', ' ', text)          
    text = re.sub(r'\s+', ' ', text).strip()   
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    
    processed_sentences = []
    df_vec = {}
    
    for sentence in sentences:
        processed = []
        words = word_tokenize(sentence.lower())
        for w in words:
            if w not in stop_words:
                lemma = wordnet_lemmatizer.lemmatize(w)
                processed.append(lemma)
                df_vec[lemma] = df_vec.get(lemma, 0) + 1
        processed_sentences.append(processed)
    
    total_sentences = len(sentences)
    
    sentence_scores = []
    
    for i, (original_sent, processed) in enumerate(zip(sentences, processed_sentences)):
        tf = {}
        length = len(processed)
        for word in processed:
            tf[word] = tf.get(word, 0) + 1/length
        
        tfidf_sum = 0
        for word in processed:
            idf = math.log(total_sentences / df_vec[word]) if df_vec.get(word) else 0
            tfidf_sum += tf[word] * idf
        
        ne_list = get_continuous_chunks(original_sent)
        e = len(ne_list)/len(processed) if processed else 0
        d = 1 if any(char.isdigit() for char in original_sent) else 0
        
        legal_matches = sum(1 for term in legal_terms if term.lower() in original_sent.lower())
        g = legal_matches/len(processed) if processed else 0
        
        score = tfidf_sum + (0.2 * d + 0.3 * e + 1.5 * g)
        sentence_scores.append((original_sent, score, len(word_tokenize(original_sent))))
    
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    
    summary = []
    word_count = 0
    target_length = 500
    
    for sent, score, length in sorted_sentences:
        if word_count + length <= target_length or not summary:
            summary.append(sent)
            word_count += length
        else:
            break
    
    return ' '.join(summary)