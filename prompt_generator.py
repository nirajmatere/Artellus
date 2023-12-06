# !pip install nltk
# !pip install spacy
# !python -m spacy download en_core_web_sm
# !pip install scikit-learn
# !pip install textblob



import re
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# coded by Niraj

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    
    return ' '.join(tokens)


def extract_technical_keywords(text, num_keywords=5):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    
    # Extract nouns and technical terms
    technical_keywords = [word for word, pos in tagged_words if pos.startswith('N') or pos.startswith('JJ')]
    print(technical_keywords)
    # Limit to num_keywords
    return technical_keywords[:num_keywords]


def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    technical_keywords = extract_technical_keywords(preprocess_text(input_text))
    # entities.append(technical_keywords)    
    for tech_key in technical_keywords:
        entities.append(tech_key)
    return entities


def extract_keywords(text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    keywords = [feature_names[i] for i in tfidf_matrix.sum(axis=0).argsort()[0, ::-1][:5]]  # Extract top 5 keywords
    return keywords


def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = "positive" if blob.sentiment.polarity > 0 else "negative" if blob.sentiment.polarity < 0 else "neutral"
    return sentiment



def construct_prompt(entities, keywords, sentiment):
    # Convert entities and keywords to strings
    entities_str = ', '.join(map(str, entities))
    keywords_str = ', '.join(map(str, keywords[0][0]))

    # Construct the prompt
    prompt = f"Generate an image featuring {entities_str} with a {sentiment} mood, emphasizing {keywords_str}."
    return prompt


def generate_prompt(input_text):
    preprocessed_text = preprocess_text(input_text)
    # print(preprocessed_text)
    
    # Example usage:
    entities = extract_entities(preprocessed_text)
    print(entities)

    # Example usage:
    keywords = extract_keywords(preprocessed_text)
    # print(keywords)

    # Example usage:
    sentiment = analyze_sentiment(input_text)
    # print(sentiment)

    prompt = construct_prompt(entities, keywords, sentiment)
    return prompt
    # print(prompt)


# input_text = "Deep learning is a branch of artificial intelligence that involves training neural networks on large datasets. These networks can learn complex patterns and representations, enabling them to perform tasks such as image recognition, natural language processing, and speech synthesis. The advancements in deep learning have led to breakthroughs in various fields, including healthcare, finance, and autonomous vehicles."


input_text = "Mount Everest, the highest peak in the world, is located in the Himalayas. The stunning landscape offers breathtaking views of snow-capped mountains and serene valleys. Visitors often describe the experience as awe-inspiring. The local flora and fauna, including rare species, add to the unique charm of the region. Exploring the Everest Base Camp is a popular adventure for trekking enthusiasts."



prompt = generate_prompt(input_text)

print(prompt)