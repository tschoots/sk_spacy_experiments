# this sample came from
# https://nicschrading.com/project/Intro-to-NLP-with-spaCy/

import inspect
from spacy.en import English
parser = English()

from numpy import dot
from numpy.linalg import norm

def PrintFrame():
    callerframerecord = inspect.stack()[1]      # 0 represents this line
                                                # 1 represents line at caller

    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    print(info.filename)                         # __FILE__    -> SpacyTest.py
    print(info.function)                         # __FUNCTION__ -> Main
    print(info.lineno)                           # __LINE__     -> 13

def PrintLineNr():
    callerframerecord = inspect.stack()[1]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    print("Line nr : ", info.lineno)

def read_file(file_name):
    with open(file_name, 'r') as file:
        return file.read()

questions = read_file("wikidata_testsentences.txt")

print(questions)

multiSentence = "There  is an art, it says, or rather, a knack to flying." \
                 "The knack lies in learning how to throw yourself at the ground and miss." \
                 "In the beginning the Universe was created. This has made a lot of people "\
                 "very angry and been widely regarded as a bad move."

parsedData = parser(multiSentence)

for i, token in enumerate(parsedData):
    print(i)
    print("original:", token.orth, token.orth_)
    print("lowercased:", token.lower, token.lower_)
    print("lemma:", token.lemma, token.lemma_)
    print("shape:", token.shape, token.shape_)
    print("prefix:", token.prefix, token.prefix_)
    print("suffix:", token.suffix, token.suffix_)
    print("log probability:", token.prob)
    print("Brown cluster id:", token.cluster)
    print("----------------------------------------")
    #if i > 1:
    #    break


# Let's look at the sentences
sents = []
# the "sents" property returns spans
# spans have indices into the original string
# where each index value represents a token
for span in parsedData.sents:
    # go from the start to the en of each span, returning each token in the sentence
    # combine each token using join()
    print('span start {} , span end {}'.format(span.start,  span.end) )
    sent = ''.join(parsedData[i].string for i in range(span.start, span.end)).strip()
    sents.append(sent)

for sentence in sents:
    print(sentence)


# Let's look at the part of speech tags of the first sentence
for span in parsedData.sents:
    sent = [parsedData[i] for i in range(span.start, span.end)]
    break
for token in sent:
    print(token.orth_, token.pos_)


# Let's look at the dependencies of this example
example = "The boy with the spotted dog quickly ran after the firetruck."
parsedEx = parser(example)
# shown as: original token, dependency tag, head word, left dependents, right dependents
print()
PrintFrame()
PrintLineNr()
print("\nDependencies : ", example)
for token in parsedEx:
    print(token.orth_, token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])


#let's look at the named entities of this example
example = "Apple's stock dropped dramatically after the death of Steve Jobs in October." \
          "When he was living in Paris near the Eiffel tower."
parsedEx = parser(example)
print()
PrintLineNr()
print(example)
print("----------------------  entities  -----------------------")
for token in parsedEx:
    print(token.orth_, token.ent_type_ if token.ent_type_ != "" else "(not an entity)")

print("---------------------- named entities only -----------------------")
# if you just want the entities and nothing else , you can do access the parsed examples "ents" property like this:
ents = list(parsedEx.ents)
for entity in ents:
    print(entity.label, entity.label_, ' '.join(t.orth_ for t in entity))


messyData = "lol that is rly funny :) This is gr8 i rate it 8/8!!!"
print()
PrintLineNr()
print("--------------- messy data -------------------")
print(messyData)
parsedData = parser(messyData)
for token in parsedData:
    print(token.orth_, token.pos_, token.lemma_)
# it does pretty well! Note that it does fail on the token "gr8),
# taking it as a verb rather than an adjective meaning "great"
# and "lol" probably isn't a noun...it's more like an interjection


# spaCy has word vector representation built in!
# you can access known words from the parser's vocabulary
nasa = parser.vocab['NASA']

# cosine similarity
cosine = lambda v1, v2: dot(v1, v2)  / (norm(v1) * norm(v2))

#gather all known words, take only the lowercased versions
allWords = list({w for w in parser.vocab if w.has_vector and w.orth_.islower() and w.lower_ != "nasa"})

# sort by similarity to NASA
allWords.sort(key=lambda w: cosine(w.vector, nasa.vector))
allWords.reverse()
print()
PrintLineNr()
print("top 10 most similar words to NASA: ")
for word in allWords[:10]:
    print(word.orth_)


# Let's see if it can figure out this analogy
# Man is to King as Woman is to ??
king = parser.vocab['king']
man = parser.vocab['man']
woman = parser.vocab['woman']

result = king.vector - man.vector + woman.vector

# gather all known words, take only the lower cases versions
allWords = list({w for w in parser.vocab if w.has_vector and w.orth_.islower() and w.lower_ != "king" and w.lower_ != "man" and w.lower_ != "woman"})
# sort by similarity to the result
allWords.sort(key=lambda w: cosine(w.vector, result))
allWords.reverse()
print("\n-----------------------------------\nMan is to King as Woman is to ??\nTop 3 closest for king - man - woman:")
for word in allWords[:3]:
    print(word.orth_)