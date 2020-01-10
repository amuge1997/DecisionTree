import numpy as n

class NodeClass:
    def __init__(self,tp_selfInfo, arr_Label,it_deep):
        self.subNode = {}               # 本节点的子节点
        self.tp_subInfo = None          # 记录子节点特征与取值

        self.tp_selfInfo = tp_selfInfo      # tp_info[0] 本节点使用的特征(样本的行)  tp_info[1] 本节点使用的特征(样本的行)的取值
        self.it_deep = it_deep + 1      # 本节点的深度
        self.ar_Label = arr_Label      # 本节点使用的训练集
        self.bl_isPredict = False       # 预测时是否使用了本节点,用于指导预测可视化
        self.probability()  # 本节点的预测概率


    def addSubNode(self,ins_subNode,tp_subInfo):
        self.tp_subInfo = (tp_subInfo[0],tp_subInfo[1])              # 子节点所使用的特征
        self.subNode[tp_subInfo] = ins_subNode    # 特征的多个取值的子节点

    def probability(self):
        arr_L = self.ar_Label
        arr_nL = n.sum(arr_L, axis=1)
        arr_sumL = arr_L.shape[1]
        arr_proL = (arr_nL / arr_sumL)
        self.arr_proLabel = arr_proL  # 每个标签的概率

    def isLeaf(self):
        if len(self.subNode) == 0:
            return True
        else:
            return False










