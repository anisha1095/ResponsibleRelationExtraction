import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import ast
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from .RelationExtraction import RelationExtraction

class ConstNetwork(RelationExtraction):
    def __init__(self, tagged_tokens, gt_relations, negative_flag):
        super().__init__(tagged_tokens, gt_relations, negative_flag)

    def extract_const_triple(self, token_list, e1_id, e2_id): # e1 need to be before e2
        e1, const, e2 = [], [], []
        for token in token_list:
            if token[8] == e1_id:
                e1.append(token)
            if token[8] == e2_id:
                e2.append(token)
            if e1!=[] and e2==[]:
                const.append(token)
            if e1==[] and e2!=[]:
                const.append(token)
        return[e1, e2, const]
    
    def const_analysis_raw_analysis(self):
        
        const_patterns = [[rel[2], rel[13]] for rel in tqdm(self.gt_relations)]
        return pd.DataFrame(const_patterns, columns=["rel_type", "const"])
         

        # gt_tokens_df = pd.DataFrame(self.tagged_tokens, columns = ["token.text", "token.pos_", "token.dep_", "token.head", "token.idx", "end", "word_idx", "sent_idx", "entity_mention_id", "entity_type"])
        # const_triples = []

        # for rel in tqdm(self.gt_relations):
        #     sent_idx = rel[0]
        #     ent1_id, ent2_id = rel[7], rel[8]
        #     token_list = gt_tokens_df.loc[gt_tokens_df["sent_idx"]==rel[0]].values.tolist()

        #     rel_triple = self.extract_const_triple(token_list, ent1_id, ent2_id)

        #     e1_pos = [token[1] for token in rel_triple[0]]
        #     e2_pos = [token[1] for token in rel_triple[1]]
        #     predicate_pos = [ token[1] for token in rel_triple[2]]

        #     const_triples.append([e1_pos, e2_pos, predicate_pos])

        # const_triple_df = pd.DataFrame(const_triples, columns=["e1_pos", "e2_pos", "predicate"])

        # return const_triple_df

    # find all pattern match. pattern-->[subject, object, predicate]
    # match pattern with an or logic. e.g., if any thing in e1 in pattern_e1, then it matches.
    def const_patterns(self, pattern):
        pattern_match = []
        for rel in self.gt_relations:
            const_list = rel[13]
            if const_list==pattern:
                flag = 1
            else:
                flag = 0
            
            pattern_match.append([rel[2], flag])

        const_df = pd.DataFrame(pattern_match)
        const_df.columns = ["gt", "const"]
        return const_df



        # truth_table = []
        # gt_tf_table = [0 if item[2]== self.negative_flag else 1 for item in self.gt_relations]   

        # const_triple_df = self.const_analysis_raw_analysis()
        # for idx, triple in const_triple_df.iterrows():
        #     e1_pos, e2_pos, predicate = triple["e1_pos"], triple["e2_pos"], triple["predicate"]
        #     e1_pos_flag, e2_pos_flag, predicate_pos_flag = False, False, False
        #     for pos in e1_pos:
        #         if pos in pattern[0]:
        #             e1_pos_flag = True                

        #     for pos in e2_pos:
        #         if pos in pattern[1]:
        #             e2_pos_flag = True
            
        #     for pos in predicate:
        #         if pos in pattern[2]:
        #             predicate_pos_flag = True

        #     if  e1_pos_flag==True and e2_pos_flag==True and predicate_pos_flag==True:
        #         flag = 1
        #     else:
        #         flag = 0

        #     truth_table.append(flag)
        # const_df = pd.DataFrame([gt_tf_table, truth_table]).T
        # const_df.columns = ["gt", "const"]
        # return const_df


# class ConstNetwork(RelationExtraction):
#     def __init__(self, tagged_tokens, gt_relations, negative_flag):
#         super().__init__(tagged_tokens, gt_relations, negative_flag)

#     def extract_const_triple(self, token_list, e1_id, e2_id): # e1 need to be before e2
#         e1, const, e2 = [], [], []
#         for token in token_list:
#             if token[8] == e1_id:
#                 e1.append(token)
#             if token[8] == e2_id:
#                 e2.append(token)
#             if e1!=[] and e2==[]:
#                 const.append(token)
#             if e1==[] and e2!=[]:
#                 const.append(token)
#         return[e1, e2, const]
    
#     def const_patterns(self):

#         gt_tokens_df = pd.DataFrame(self.tagged_tokens, columns = ["token.text", "token.pos_", "token.dep_", "token.head", "token.idx", "end", "word_idx", "sent_idx", "entity_mention_id", "entity_type"])
#         const_triples = []

#         for rel in tqdm(self.gt_relations):
#             sent_idx = rel[0]
#             ent1_id, ent2_id = rel[7], rel[8]
#             token_list = gt_tokens_df.loc[gt_tokens_df["sent_idx"]==rel[0]].values.tolist()

#             rel_triple = self.extract_const_triple(token_list, ent1_id, ent2_id)

#             e1_pos = [token[1] for token in rel_triple[0]]
#             e2_pos = [token[1] for token in rel_triple[1]]
#             predicate_pos = [ token[1] for token in rel_triple[2]]

#             const_triples.append([e1_pos, e2_pos, predicate_pos])

#         const_triple_df = pd.DataFrame(const_triples, columns=["e1_pos", "e2_pos", "predicate"])

#         return const_triple_df

#     # find all pattern match. pattern-->[subject, object, predicate]
#     # match pattern with an or logic. e.g., if any thing in e1 in pattern_e1, then it matches.
#     def const_analysis_raw_analysis(self, pattern):
#         truth_table = []
#         gt_tf_table = [0 if item[2]== self.negative_flag else 1 for item in self.gt_relations]   

#         const_triple_df = self.const_patterns()
#         for idx, triple in const_triple_df.iterrows():
#             e1_pos, e2_pos, predicate = triple["e1_pos"], triple["e2_pos"], triple["predicate"]
#             e1_pos_flag, e2_pos_flag, predicate_pos_flag = False, False, False
#             for pos in e1_pos:
#                 if pos in pattern[0]:
#                     e1_pos_flag = True                

#             for pos in e2_pos:
#                 if pos in pattern[1]:
#                     e2_pos_flag = True
            
#             for pos in predicate:
#                 if pos in pattern[2]:
#                     predicate_pos_flag = True

#             if  e1_pos_flag==True and e2_pos_flag==True and predicate_pos_flag==True:
#                 flag = 1
#             else:
#                 flag = 0

#             truth_table.append(flag)
#         const_df = pd.DataFrame([gt_tf_table, truth_table]).T
#         const_df.columns = ["gt", "const"]
#         return const_df

#     # # analysis
#     # def const_analysis(gt_relations, tagged_tokens, negative_flag, pattern):
#     #     const_df = self.const_analysis_raw_analysis(gt_relations, tagged_tokens, negative_flag, pattern)
#     #     gt = const_df["gt"].tolist()
#     #     pred = const_df["const"].tolist()
#     #     confusion_info = self.calculate_confusion_matrix(gt, pred).ravel()
#     #     const_df = pd.DataFrame(confusion_info).T
#     #     const_df.columns = ["tn", "fp", "fn", "tp"]
#     #     const_df.index = [pattern]
#     #     return const_df
#     #     # return {"tn":confusion_info[0], "fp":confusion_info[1], "fn":confusion_info[2], "tp":confusion_info[3]}
