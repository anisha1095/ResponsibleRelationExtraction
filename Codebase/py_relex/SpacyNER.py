import os
import sys
import csv
import spacy
import pandas as pd
from tqdm import tqdm
import networkx as nx
import en_core_web_sm
from shutil import copyfile
import itertools
from itertools import combinations
from collections import defaultdict

ENT_TYPES = ["PERSON", "GPE", "LOC", "ORG", "NORP"]
FLAIR_ENT_TYPES = ["PERSON", "GPE", "LOC", "ORG", "NORP"]
# ontonotes use person and have 18 entity types, while conll03 only have 4 entity types and use per
tagged_tokens_header = ["text", "pos", "dep", "head", "begin_char", "end_char", "word_index", "sent_index", "spacy_entity_id", "spacy_ent_type"]
rel_headers = ["sent_idx", "sent_text", "rel_type", "ent1_text", "ent2_text", "ent1_type", "ent2_type", "ent1_id", "ent2_id", "in_between_text", "sdp_dist", "sdp_path", "sdp_dep_pos", "const_pos"]

nlp = spacy.load("en_core_web_sm")

from .Corpus import Corpus
# spacy.load('en_core_web_sm')

class SpacyNER(Corpus):
    def __init__(self, raw_text):
        super().__init__(raw_text)
    
    def add_entity_annotation(self, ents, token_info): # ents in doc.ents format/ this is only for spacy
        for ent in ents:
            if token_info["char_start"] >= ent[1].start_char and token_info["char_end"] <= ent[1].end_char:
                return [ent[1].label_, ent[0]]
        return ["", ""]   

    def generate_tokens_n_rels(self, sents):
        tagged_tokens = []
        spacy_rels = []

        for sent_idx, sent_text in enumerate(tqdm(sents[:])):
            doc = nlp(sent_text)
            ents_info = [[str(sent_idx) + "_" + str(idx), ent] for idx, ent in enumerate(doc.ents) if ent.label_ in ENT_TYPES]
#             print(ents_info)
            
            edges = []
            sent_tokens = []
            
            for word_idx, token in enumerate(doc):

                for child in token.children:
                    edges.append((token.i, child.i))
                
                token_info = {}
                token_info["sent_idx"] = sent_idx
                token_info["word_idx"] = word_idx
                token_info["text"] = token.text
                token_info["pos_tags"] = token.pos_
                token_info["dep"] = token.dep_
                token_info["head"] = token.head.i
                token_info["char_start"] = token.idx
                token_info["char_end"] = token.idx + len(token.text)          
                token_info["entity_type"], token_info["entity_id"] = self.add_entity_annotation(ents_info, token_info)
                
                tagged_token = [token.text, token.pos_, token.dep_, token.head.i, token.idx, token.idx + len(token.text), word_idx, sent_idx, token_info["entity_id"], token_info["entity_type"] ]
                
                sent_tokens.append(tagged_token)
                
            tagged_tokens.extend(sent_tokens)
            
            # spacy
#             print(len(ents_info))
            if len(ents_info) < 2:
                if len(ents_info) == 1:
                    spacy_rels.append([sent_idx, sent_text, "no_relation", ents_info[0][1].text, "", ents_info[0][1].label_, "", ents_info[0][0], "", ""])
                else:
                    spacy_rels.append([sent_idx, sent_text, "no_relation", "", "", "", "", "", "", ""])
            else:
                ent_pairs = [comb for comb in combinations(ents_info, 2)]

                for ent_pair in ent_pairs:
                    ent_pair = list(ent_pair)

                    ent1_id = ent_pair[0][0]
                    ent1_text, ent1_type, ent1_range = ent_pair[0][1].text, ent_pair[0][1].label_, [ent_pair[0][1].start_char, ent_pair[0][1].end_char]

                    ent2_id = ent_pair[1][0]
                    ent2_text, ent2_type, ent2_range = ent_pair[1][1].text, ent_pair[1][1].label_, [ent_pair[1][1].start_char, ent_pair[1][1].end_char]

                    ent_range = ent1_range + ent2_range
                    ent_range.sort()
                    in_between_text = sent_text[ent_range[1]:ent_range[2]]

                    dep_list = [token[1] for token in sent_tokens]

                    ent1_word_idx_range = [token[6] for token in sent_tokens if token[8]==ent1_id]
                    ent1_word_idx = ent1_word_idx_range[-1]

                    ent2_word_idx_range = [token[6] for token in sent_tokens if token[8]==ent2_id]
                    ent2_word_idx = ent2_word_idx_range[-1]  

    #                 ent1_word_idx = [token[6] for token in tagged_tokens if token[8]==ent1_id][-1]
    #                 ent2_word_idx = [token[6] for token in tagged_tokens if token[8]==ent2_id][-1]
                    if ent1_word_idx < ent2_word_idx:
                        in_between_const = dep_list[ent1_word_idx_range[-1]+1:ent2_word_idx_range[0]]
                    else:
                        in_between_const = dep_list[ent2_word_idx_range[-1]+1:ent1_word_idx_range[0]]

                    graph = nx.Graph(edges)
                    try:
    #                     dep_dist = nx.shortest_path_length(graph, source=ent1_word_idx, target=ent2_word_idx)
                        dep_path = nx.shortest_path(graph, source=ent1_word_idx, target=ent2_word_idx)

                        for idx in ent1_word_idx_range:
                            if idx in dep_path:
                                dep_path.remove(idx)

                        for idx in ent2_word_idx_range:
                            if idx in dep_path:
                                dep_path.remove(idx)
                        dep_dist = len(dep_path)                        
                        in_between_dep = [dep_list[idx] for idx in dep_path]

                    except:
                        dep_dist = "no_path_found"
                        dep_path = "no_path_found"
                        in_between_dep = "no_path_found"


                    spacy_rels.append([sent_idx, sent_text, "", ent1_text, ent2_text, ent1_type, ent2_type, ent1_id, ent2_id, in_between_text, dep_dist, dep_path, in_between_dep, in_between_const])
        return [tagged_tokens, spacy_rels]
            