#! /usr/bin/env python2

from collections import defaultdict
import json
import re

import nltk
from nltk.corpus import stopwords
from transformers import pipeline

from pattern.en import parse, Sentence, mood
from pattern.db import csv
from pattern.vector import Document, NB

# Download resources automatically if not installed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# https://github.com/clips/pattern/issues/295#issuecomment-841625057
try:
    parse('dummy sentence')
except RuntimeError:
    pass


def readText():
    """
    Reads the text from a text file.
    """
    with open("tst.txt", "rb") as f:
        text = f.read().decode('utf-8-sig')
    return text


def chunkSentences(text):
    """
    Parses text into parts of speech tagged with parts of speech labels.

    Used for reference: https://gist.github.com/onyxfish/322906
    """
    sentences = nltk.sent_tokenize(text)
    tokenizedSentences = [nltk.word_tokenize(sentence)
                          for sentence in sentences]
    taggedSentences = [nltk.pos_tag(sentence)
                       for sentence in tokenizedSentences]
    if nltk.__version__[0:2] == "2.":
        chunkedSentences = nltk.batch_ne_chunk(taggedSentences, binary=True)
    else:
        chunkedSentences = nltk.ne_chunk_sents(taggedSentences, binary=True)
    return chunkedSentences


def extractEntityNames(tree, _entityNames=None):
    """
    Creates a local list to hold nodes of tree passed through, extracting named
    entities from the chunked sentences.

    Used for reference: https://gist.github.com/onyxfish/322906
    """
    if _entityNames is None:
        _entityNames = []
    try:
        if nltk.__version__[0:2] == "2.":
            label = tree.node
        else:
            label = tree.label()
    except AttributeError:
        pass
    else:
        if label == 'NE':
            _entityNames.append(' '.join([child[0] for child in tree]))
        else:
            for child in tree:
                extractEntityNames(child, _entityNames=_entityNames)
    return _entityNames


def buildDict(chunkedSentences, _entityNames=None):
    """
    Uses the global entity list, creating a new dictionary with the properties
    extended by the local list, without overwriting.

    Used for reference: https://gist.github.com/onyxfish/322906
    """
    if _entityNames is None:
        _entityNames = []

    for tree in chunkedSentences:
        extractEntityNames(tree, _entityNames=_entityNames)

    return _entityNames


def removeStopwords(entityNames, customStopWords=None):
    """
    Brings in stopwords and custom stopwords to filter mismatches out.
    """
    # Memoize custom stop words
    if customStopWords is None:
        with open("customStopWords.txt", "rb") as f:
            customStopwords = f.read().decode('utf-8-sig').split(', ')

    for name in entityNames:
        if name in stopwords.words('english') or name in customStopwords:
            entityNames.remove(name)


def getMajorCharacters(entityNames):
    """
    Adds names to the major character list if they appear frequently.
    """
    return {name for name in entityNames if entityNames.count(name) > 10}


def splitIntoSentences(text):
    """
    Split sentences on .?! "" and not on abbreviations of titles.
    Used for reference: http://stackoverflow.com/a/8466725
    """
    sentenceEnders = re.compile(r"""
    # Split sentences on whitespace between them.
    (?:               # Group for two positive lookbehinds.
      (?<=[.!?])      # Either an end of sentence punct,
    | (?<=[.!?]['"])  # or end of sentence punct and quote.
    )                 # End group of two positive lookbehinds.
    (?<!  Mr\.   )    # Don't end sentence on "Mr."
    (?<!  Mrs\.  )    # Don't end sentence on "Mrs."
    (?<!  Ms\.   )    # Don't end sentence on "Ms."
    (?<!  Jr\.   )    # Don't end sentence on "Jr."
    (?<!  Dr\.   )    # Don't end sentence on "Dr."
    (?<!  Prof\. )    # Don't end sentence on "Prof."
    (?<!  Sr\.   )    # Don't end sentence on "Sr."
    \s+               # Split on whitespace between sentences.
    """, re.IGNORECASE | re.VERBOSE)
    return sentenceEnders.split(text)


def compareLists(sentenceList, majorCharacters):
    """
    Compares the list of sentences with the character names and returns
    sentences that include names.
    """
    characterSentences = defaultdict(list)
    for sentence in sentenceList:
        for name in majorCharacters:
            if re.search(r"\b(?=\w)%s\b(?!\w)" % re.escape(name),
                         sentence,
                         re.IGNORECASE):
                characterSentences[name].append(sentence)
    return characterSentences


def extractMood(characterSentences):
    """
    Analyzes the sentence using grammatical mood module from pattern.
    """
    characterMoods = defaultdict(list)
    for key, value in characterSentences.items():
        for x in value:
            try:
                characterMoods[key].append(mood(Sentence(parse(str(x), lemmata=True))))
            except:
                characterMoods[key].append(-1)

    return characterMoods


def extractSentiment(characterSentences):
    """
    Trains a Naive Bayes classifier object with the reviews.csv file, analyzes
    the sentence, and returns the tone.
    """
    nb = NB()
    characterTones = defaultdict(list)
    for review, rating in csv("reviews.csv"):
        nb.train(Document(review, type=int(rating), stopwords=True))
    for key, value in characterSentences.items():
        for x in value:
            try:
                characterTones[key].append(nb.classify(str(x)))
            except:
                characterTones[key].append(-1)
    return characterTones


def extractVisualAppearanceWords(characterSentences):
    """
    Extracts words describing the visual appearance of a character.
    """
    text = " ".join(characterSentences)
    tagged_words = nltk.pos_tag(nltk.word_tokenize(text))
    visual_appearance_words = [word for word, pos in tagged_words if pos.startswith('JJ') or pos.startswith('NNP')]
    return visual_appearance_words


def abstractiveSummarization(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, no_repeat_ngram_size=2)
    print(summary)
    return summary[0]['summary_text']


def generateCharacterProfilesWithSummarization(characterSentences, characterTones, characterMoods):
    """
    Generates descriptive profiles for each character based on sentences, tones, and moods
    using abstractive summarization.
    """
    characterProfiles = {}

    for character, sentences in characterSentences.items():
        tones = characterTones.get(character, [])
        moods = characterMoods.get(character, [])

        # Perform abstractive summarization on major traits
        major_traits = ". ".join(sentences)
        summarization_result = abstractiveSummarization(major_traits)

        # Calculate overall sentiment and prevalent mood
        overall_sentiment = sum(tones) / len(tones) if tones else None
        prevalent_mood = max(set(moods), key=moods.count) if moods else None

        # Build character profile
        profile = {
            'Character': character,
            'MajorTraits': summarization_result,
            'OverallSentiment': overall_sentiment,
            'PrevalentMood': prevalent_mood
        }

        characterProfiles[character] = profile

    return characterProfiles

# ... (previous code)


# Print or use character profiles as needed
def writeAnalysis(sentenceAnalysis):
    """
    Writes the sentence analysis to a text file in the same directory.
    """
    with open("sentenceAnalysis.txt", "w") as f:
        for item in sentenceAnalysis.items():
            f.write("%s:%s\n" % item)


def writeToJSON(sentenceAnalysis):
    """
    Writes the sentence analysis to a JSON file in the same directory.
    """
    with open("sentenceAnalysis.json", "w") as f:
        json.dump(sentenceAnalysis, f)


if __name__ == "__main__":
    text = readText()

    chunkedSentences = chunkSentences(text)
    entityNames = buildDict(chunkedSentences)
    removeStopwords(entityNames)
    majorCharacters = getMajorCharacters(entityNames)

    sentenceList = splitIntoSentences(text)
    characterSentences = compareLists(sentenceList, majorCharacters)
    characterMoods = extractMood(characterSentences)
    characterTones = extractSentiment(characterSentences)

    # Merges sentences, moods and tones together into one dictionary on each
    # character.
    sentenceAnalysis = defaultdict(list,
                                   [(k, [characterSentences[k],
                                         characterTones[k],
                                         characterMoods[k]])
                                    for k in characterSentences])

    
# Analyze the text with abstractive summarization
    characterProfilesWithSummarization = generateCharacterProfilesWithSummarization(
        characterSentences, characterTones, characterMoods
    )
    # print(sentenceAnalysis)
    # characterDescription = extractVisualAppearanceWords()
    for character, profile in characterProfilesWithSummarization.items():
        print(f"Character: {character}")
        print(f"Major Traits (Summarized): {profile['MajorTraits']}")
        print(f"Overall Sentiment: {profile['OverallSentiment']}")
        print(f"Prevalent Mood: {profile['PrevalentMood']}")
        print("\n")



    writeAnalysis(sentenceAnalysis)
    writeToJSON(sentenceAnalysis)
