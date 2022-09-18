class RelationExtraction:
    def __init__(self, tagged_tokens, gt_relations, negative_flag):
        self.tagged_tokens = tagged_tokens
        self.gt_relations = gt_relations
        self.negative_flag = negative_flag