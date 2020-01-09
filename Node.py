import numpy as n

class NodeClass:
    def __init__(self, it_deep, arr_Label):
        self.subNode = {}               # 本节点的子节点
        self.it_selfFeatureRow = None   # 本节点使用的特征(样本的行)
        self.it_selfFeatureVal = None   # 本节点使用的特征(样本的行)的取值

        self.it_deep = it_deep + 1      # 本节点的深度
        self.arr_Label = arr_Label      # 本节点使用的训练集
        self.probability()              # 本节点的预测概率

        self.bl_isPredict = False       # 预测时是否使用了本节点,用于指导预测可视化

        self.it_featureRow = None       # 子节点使用的特征(样本的行)

    def addSubNode(self, it_featureRow, it_featureValue, inst_subNode):
        if self.it_featureRow is not None:
            if it_featureRow != self.it_featureRow:
                raise Exception('addSubNode() 输入的参数\'特征行 featureRow\'错误!')
        self.it_featureRow = it_featureRow              # 子节点所使用的特征
        self.subNode[it_featureValue] = inst_subNode    # 特征的多个取值的子节点
        inst_subNode.it_selfFeatureRow = it_featureRow
        inst_subNode.it_selfFeatureVal = it_featureValue

    def probability(self):
        arr_L = self.arr_Label
        arr_nL = n.sum(arr_L, axis=1)
        arr_sumL = arr_nL.sum()
        arr_proL = (arr_nL / arr_sumL)
        self.arr_proLabel = arr_proL  # 每个标签的概率

    def isLeaf(self):
        if len(self.subNode) == 0:
            return True
        else:
            return False










