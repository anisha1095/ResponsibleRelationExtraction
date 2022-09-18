import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import ast
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from .RelationExtraction import RelationExtraction


class CoNetwork(RelationExtraction):
    def __init__(self, tagged_tokens, gt_relations, negative_flag):
        super().__init__(tagged_tokens, gt_relations, negative_flag)

    # based on window size to decide whether one rel is positive or negative. 
    def co_occurrence_rel_classifier(self, rel, window_size):
        if window_size == "sentence":
            return 1
        return 1 if rel <= window_size else 0

    # create dataframe which stores all the conditional results --> should be merged and output for feature engineering
    def co_occurrence_analysis_raw_analysis(self):
        rels = [[item[2], len(item[9].split())] for item in self.gt_relations]
        co_occurrence_test = []
        gt_tf_table = [0 if item[0]== self.negative_flag else 1 for item in rels ]
        # co-occurrence with different window size
        for window_size in range(0, 10):
            co_occurrence_test.append([self.co_occurrence_rel_classifier(rel[1], window_size) for rel in rels])
        co_occurrence_test.append([self.co_occurrence_rel_classifier(rel[1], "sentence") for rel in rels]) # sentence level
        # have t-f table for each item, indexed by gt_relation index (basically with the same sieze)
        raw_analysis = [gt_tf_table] + co_occurrence_test
        co_headers = ["cooc_"+str(i) for i in range(10)]
        column_headers = ["gt"] + co_headers + ["co_sent"]    
        raw_analysis_df = pd.DataFrame(raw_analysis).T
        raw_analysis_df.columns = column_headers
        return raw_analysis_df
        
    # create confusion_matrix 
    def co_occurrence_analysis(self):
        raw_analysis_df = self.co_occurrence_analysis_raw_analysis()
        pd_index = list(range(0, 10)) + ["sentence"]
        raw_analysis = raw_analysis_df.T.values.tolist()

        #tn, fp, fn, tp
        confusion_info = [confusion_matrix(raw_analysis[0], raw_analysis[idx]).ravel() for idx in range(1, 12)]
        return pd.DataFrame(confusion_info, columns = ["tn", "fp", "fn", "tp"], index=pd_index)