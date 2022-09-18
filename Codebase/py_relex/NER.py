import networkx as nx
import pandas as pd
from itertools import combinations


def spacy_flair_intersection(spacy_tts, flair_tts):
    # need to make a new ent_id and Type
    intersection_tts = []
    for idx, spacy_tt in enumerate(spacy_tts):
        if spacy_tt[9]!='' and flair_tts[idx][9]!='':
            ent_id = spacy_tt[8]
            if spacy_tt[9] == flair_tts[idx][9]:
                ent_type = spacy_tt[9]
            else:
                ent_type = spacy_tt[9] + "_" + flair_tts[idx][9]
            tt = spacy_tt[0:8] + [ent_id, ent_type]
        else:
            tt = spacy_tt[0:8] + ["", ""]
        intersection_tts.append(tt)
    return intersection_tts


def spacy_flair_union(spacy_tts, flair_tts):
    # need to make a new ent_id and Type
    union_tts = []
    for idx, spacy_tt in enumerate(spacy_tts):
        if spacy_tt[9]!='' and flair_tts[idx][9]!='':
            ent_id = spacy_tt[8]
            if spacy_tt[9] == flair_tts[idx][9]:
                ent_type = spacy_tt[9]
            else:
                ent_type = spacy_tt[9] + "_" + flair_tts[idx][9]
            tt = spacy_tt[0:8] + [ent_id, ent_type]
        else:
            ent_id = spacy_tt[8] + flair_tts[idx][8]
            ent_type = spacy_tt[9] + flair_tts[idx][9]
            tt = spacy_tt[0:8] + [ent_id, ent_type]
        union_tts.append(tt)
    return union_tts


def tagged_tokens_2_rels(s_f_i):
    tagged_tokens_header = ["text", "pos", "dep", "head", "begin_char", "end_char", "word_index", "sent_index", "entity_id", "ent_type"]
    sfi_tagged_tokens = pd.DataFrame(s_f_i, columns = tagged_tokens_header)
    rels = []
    for sent_idx, g in sfi_tagged_tokens.groupby("sent_index"):
        edges = []
        tagged_token_list = g.values.tolist()
        for idx, token in g.iterrows():
            edges.append((token["word_index"], token["head"]))
        
#         print(g)
        ent_ids = list(set(g["entity_id"].tolist()))
        if "" in ent_ids:
            ent_ids.remove("")

        sent_text_list = g["text"].tolist()
        sent_pos_list = g["pos"].tolist()
        sent_text = " ".join(sent_text_list)

        if len(ent_ids) < 2:
            if len(ent_ids) == 1:
                ent_id = ent_ids[0]
                ent_text = " ".join([token[0] for token in tagged_token_list if token[8]==ent_id])
                ent_type = [token[9] for token in tagged_token_list if token[8]==ent_id][0]
                rels.append([sent_idx, sent_text, "no_relation", ent_text, '', ent_type, '', ent_id, "", "", "no_path_found", "no_path_found", "no_path_found", ""])
            else:
                rels.append([sent_idx, sent_text, "no_relation", "", "", "", "", "", "", "", "no_path_found", "no_path_found", "no_path_found", ""])
        else:
            ent_pairs = [comb for comb in combinations(ent_ids, 2)]

            for ent_pair in ent_pairs:
                ent_pair = list(ent_pair)
                ent1_id = ent_pair[0]
                ent2_id = ent_pair[1]
                ent1_type = [token[9] for token in tagged_token_list if token[8]==ent1_id][0]
                ent2_type = [token[9] for token in tagged_token_list if token[8]==ent2_id][0]
                ent1_text = " ".join([token[0] for token in tagged_token_list if token[8]==ent1_id])
                ent2_text = " ".join([token[0] for token in tagged_token_list if token[8]==ent2_id])
                ent1_idx_range = [token[6] for token in tagged_token_list if token[8]==ent1_id]
                ent2_idx_range = [token[6] for token in tagged_token_list if token[8]==ent2_id]
                ent1_word_idx = ent1_idx_range[-1]
                ent2_word_idx = ent2_idx_range[-1]

                if ent1_idx_range[0]>ent2_idx_range[0]:
                    in_between_text = sent_text_list[ent2_idx_range[-1]+1:ent1_idx_range[0]]
                    in_between_const = sent_pos_list[ent2_idx_range[-1]+1:ent1_idx_range[0]]
                else:
                    in_between_text = sent_text_list[ent1_idx_range[-1]+1:ent2_idx_range[0]]
                    in_between_const = sent_pos_list[ent1_idx_range[-1]+1:ent2_idx_range[0]]
                
                in_between_text = " ".join(in_between_text)
                graph = nx.Graph(edges)

                try:
                    dep_path = nx.shortest_path(graph, source=ent1_word_idx, target=ent2_word_idx)

                    for idx in ent1_idx_range:
                        if idx in dep_path:
                            dep_path.remove(idx)

                    for idx in ent2_idx_range:
                        if idx in dep_path:
                            dep_path.remove(idx)
                    dep_dist = len(dep_path)                        
                    in_between_dep = [sent_pos_list[idx] for idx in dep_path]

                except:
                    dep_dist = "no_path_found"
                    dep_path = "no_path_found"
                    in_between_dep = "no_path_found"            

                rels.append([sent_idx, sent_text, "", ent1_text, ent2_text, ent1_type, ent2_type, ent1_id, ent2_id, in_between_text, dep_dist, dep_path, in_between_dep, in_between_const])

    return rels