import spacy
nlp = spacy.load('en')

doc = nlp("The 22-year-old recently won ATP Challenger tournament.")

for tok in doc:
  print(tok.text, "...", tok.dep_)

print("              ")
print("#####################")
print("              ")

doc = nlp("Nagal won the first set.")

for tok in doc:
  print(tok.text, "...", tok.dep_)

print("              ")
print("#####################")
print("              ")

import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher
from spacy.tokens import Span

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)
'exec(%matplotlib inline)'

# import wikipedia sentences
candidate_sentences = pd.read_csv("/Users/Fabi/PycharmProjects/wiki_sentences_v2.csv")
candidate_sentences.shape

print("Output:")
print(candidate_sentences.shape)
print("       ")
candidate_sentences['sentence'].sample(5)
print("Output:")
print(candidate_sentences['sentence'].sample(5))
print("       ")
print("*********")

doc = nlp("the drawdown process is governed by astm standard d823")

print("Output:")
print("         ")
for tok in doc:
  print(tok.text, "...", tok.dep_)

print("          ")
# Perfect! There is only one subject (‘process’) and only one object (‘standard’). You can check for other sentences in a similar manner.

def get_entities(sent):
  ## chunk 1
  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""  # dependency tag of previous token in the sentence
  prv_tok_text = ""  # previous token in the sentence

  prefix = ""
  modifier = ""

  #############################################################

  for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " " + tok.text

      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " " + tok.text

      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier + " " + prefix + " " + tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""

        ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier + " " + prefix + " " + tok.text

      ## chunk 5
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text
  #############################################################

  return [ent1.strip(), ent2.strip()]

film = nlp("the film had 200 patents")
for tok in film:
  print(tok.text, "...", tok.dep_)
get_entities("the film had 200 patents")
print("Output:")
print(get_entities("the film had 200 patents"))

print("          ")
entity_pairs = []

for i in tqdm(candidate_sentences["sentence"]):
  entity_pairs.append(get_entities(i))

entity_pairs[10:20]
print(entity_pairs[10:20])

# As you can see, there are a few pronouns in these entity pairs such as ‘we’, ‘it’, ‘she’, etc. We’d like to have proper nouns or nouns instead. Perhaps we can further improve the get_entities( ) function to filter out pronouns. For the time being, let’s leave it as it is and move on to the relation extraction part.

# Relation / Predicate Extraction
#
# This is going to be a very interesting aspect of this article. Our hypothesis is that the predicate is actually the main verb in a sentence.
#
# For example, in the sentence — “Sixty Hollywood musicals were released in 1929”, the verb is “released in” and this is what we are going to use as the predicate for the triple generated from this sentence.
#
# The function below is capable of capturing such predicates from the sentences. Here, I have used spaCy’s rule-based matching:

def get_relation(sent):

  doc = nlp(sent)

  # Matcher class object
  matcher = Matcher(nlp.vocab)

  #define the pattern
  pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}]

  matcher.add("matching_1", None, pattern)

  matches = matcher(doc)
  k = len(matches) - 1

  span = doc[matches[k][1]:matches[k][2]]

  return(span.text)

# The pattern defined in the function tries to find the ROOT word or the main verb in the sentence. Once the ROOT is identified, then the pattern checks whether it is followed by a preposition (‘prep’) or an agent word. If yes, then it is added to the ROOT word.

get_relation("John completed the task")
print(get_relation("John completed the task"))

# Similarly, let’s get the relations from all the Wikipedia sentences:

relations = [get_relation(i) for i in
             tqdm(candidate_sentences['sentence'])]

# Let’s take a look at the most frequent relations or predicates that we have just extracted:

pd.Series(relations).value_counts()[:50]
print(pd.Series(relations).value_counts()[:50])
print("          ")

# It turns out that relations like “A is B” and “A was B” are the most common relations. However, there are quite a few relations that are more associated with the overall theme — “the ecosystem around movies”. Some of the examples are “composed by”, “released in”, “produced”, “written by” and a few more.


# Build a Knowledge Graph
#
# We will finally create a knowledge graph from the extracted entities (subject-object pairs) and the predicates (relation between entities).

# Let’s create a dataframe of entities and predicates:
# extract subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

# Next, we will use the networkx library to create a network from this dataframe. The nodes will represent the entities and the edges or connections between the nodes will represent the relations between the nodes.
#
# It is going to be a directed graph. In other words, the relation between any connected node pair is not two-way, it is only from one node to another. For example, “John eats pasta”:
# create a directed-graph from a dataframe
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="composed by"], "source", "target",
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(6,6))
pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()

# That’s a much cleaner graph. Here the arrows point towards the composers. For instance, A.R. Rahman, who is a renowned music composer, has entities like “soundtrack score”, “film score”, and “music” connected to him in the graph above.
#
# Let’s check out a few more relations.
#
# Since writing is an important role in any movie, I would like to visualize the graph for the “written by” relation:

G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="written by"], "source", "target",
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(6,6))
pos = nx.spring_layout(G, k = 0.5)
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()

# Awesome! This knowledge graph is giving us some extraordinary information. Guys like Javed Akhtar, Krishna Chaitanya, and Jaideep Sahni are all famous lyricists and this graph beautifully captures this relationship.
#
# Let’s see the knowledge graph of another important predicate, i.e., the “released in”:
#
# G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="released in"], "source", "target",
#                           edge_attr=True, create_using=nx.MultiDiGraph())
#
# plt.figure(figsize=(12,12))
# pos = nx.spring_layout(G, k = 0.5)
# nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
# plt.show()

# I can see quite a few interesting information in this graph. For example, look at this relationship — “several action horror movies released in the 1980s” and “pk released on 4844 screens”. These are facts and it shows us that we can mine such facts from just text. That’s quite amazing!
# ######################################################
# End Notes
#
# In this article, we learned how to extract information from a given text in the form of triples and build a knowledge graph from it.
#
# However, we restricted ourselves to use sentences with exactly 2 entities. Even then we were able to build quite informative knowledge graphs. Imagine the potential we have here!
#
# I encourage you to explore this field of information extraction more to learn extraction of more complex relationships. In case you have any doubt or you want to share your thoughts, please feel free to use the comments section below.



