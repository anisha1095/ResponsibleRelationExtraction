import pandas as pd
# import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import ast
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from .RelationExtraction import RelationExtraction

class DepNetwork(RelationExtraction):
    def __init__(self, tagged_tokens, gt_relations, negative_flag):
        super().__init__(tagged_tokens, gt_relations, negative_flag)

    # two characteristics: 1. distribution of distance. 2. distribution of specific pattern occurrence 
    def dependency_classifier(self, dep_length, tree_length):
        if dep_length == '' or dep_length == 'no_path_found':
            return ''
        if dep_length <= tree_length:
            return 1
        else:
            return 0

    # return raw features
    def dependency_analysis_raw_analysis(self):
        dep_rels = [[item[2], item[10]] for item in self.gt_relations]
        dep_test = []
        gt_tf_table = [0 if item[0]== self.negative_flag else 1 for item in dep_rels ]   

        for tree_length in range(0, 10):
            dep_test.append([self.dependency_classifier(rel[1], tree_length) for rel in dep_rels])
        raw_analysis = [gt_tf_table] + dep_test

        dep_headers = ["dep_"+str(i) for i in range(10)]
        column_headers = ["gt"] + dep_headers    

        raw_analysis_df = pd.DataFrame(raw_analysis).T
        raw_analysis_df.columns = column_headers
        return raw_analysis_df

    # return confusion matrix
    def dependency_analysis(self):
        raw_analysis_df = self.dependency_analysis_raw_analysis()
        # print(raw_analysis_df)
        raw_analysis_df.replace('', np.nan, inplace=True)

        raw_analysis_df.dropna(inplace=True)
        # print(raw_analysis_df)
        pd_index = list(range(0, 10))

        raw_analysis = raw_analysis_df.T.values.tolist()
        # print(raw_analysis)
        confusion_info = [confusion_matrix( raw_analysis[0], raw_analysis[idx]).ravel() for idx in range(1, 11)]
        return pd.DataFrame(confusion_info, columns = ["tn", "fp", "fn", "tp"])
        # return pd.DataFrame(confusion_info)

    def dep_patterns(self, dep_ptn=[]):
        if dep_ptn == []:
            const_patterns = [[rel[2], rel[12]] for rel in tqdm(self.gt_relations)]
            return pd.DataFrame(const_patterns, columns=["rel_type", "dep"])
        else:
            pattern_match = []
            for rel in self.gt_relations:
                dep_list = rel[12]
                if dep_list == dep_ptn:
                    flag = 1
                else:
                    flag = 0
                pattern_match.append([rel[2], flag])
            dep_df = pd.DataFrame(pattern_match)
            dep_df.columns = ["gt", "dep"]
            return dep_df

        # for rel in self.gt_relations:
        #     if rel[11] == "" or rel[11] == "no_path_found":
        #         flag = "no_path_found"


        # gt_tokens_df = pd.DataFrame(self.tagged_tokens, columns = ["token.text", "token.pos_", "token.dep_", "token.head", "token.idx", "end", "word_idx", "sent_idx", "entity_mention_id", "entity_type"])
        # dep_list = []
        # # text_list = []
        # index_list = []
        # dep_list_df = pd.DataFrame()

        # for rel in tqdm(self.gt_relations):
        #     # print(rel)
        #     sent_idx = rel[0]
        #     print(rel[10], rel[11], rel[12])
        #     if rel[10]!="" and rel[10]!="no_path_found":
        #         dep_idx = ast.literal_eval(rel[11]) # get the index of words in the dependency tree
        #         pos = ast.literal_eval(rel[12]) # get the index of pos
        #     else:
        #         dep_idx = "na"
        #         pos = "na"

        #     index_list.append(dep_idx)
        #     dep_list.append(pos)

        # dep_list_df["dep_pos"] = pd.Series(dep_list)
        # # dep_list_df["dep_text"] = pd.Series(text_list)
        # dep_list_df["dep_length"] = pd.Series(index_list)
        # if dep_ptn==[]:
        #     return dep_list_df
        #     # pass
        # else:
        #     preds = []
            
        #     for dep in dep_list:
        #         flag = 0
        #         if dep == dep_ptn:
        #             flag = 1                
        #         # for ptn_token in dep_ptn:
        #         #     if ptn_token in dep:
        #         #         flag = 1
        #         preds.append(flag)
        #     dep_list_df["pred"] = pd.Series(preds)
        #     return dep_list_df  