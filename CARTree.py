import numpy as n
from DecisionTree.Node import NodeClass
from DecisionTree.DTreeDraw import DTreeDrawClass
from DecisionTree.TreeSave import TreeSaveClass

class CARTree:
    # 离散决策树
    '''
    输入数据 arr_X:
            每行为一个样本,
            每个样本的同一行都表示一类特征
    输入标签 arr_L:
            每行为一个样本,每个标签向量只有一行为 1 ,其他为 0

    注意 DTree().predict() 只能预测一个样本 输入形式为 array( [ [1,2,3,1,5] ] )
    '''

    def __init__(self,arr_X,arr_L):
        arr_X = arr_X.T
        arr_L = arr_L.T
        self.it_featureNum = arr_X.shape[0]
        self.root = NodeClass(tp_selfInfo=(None,None),arr_Label=arr_L,it_deep=0)
        self.node({'X':arr_X,'L':arr_L},self.root)

    # 终止判断
    def isStop(self,arr_gain=None,it_deep=None,it_minSample=None):
        fl_maxGin = n.max(arr_gain)
        if arr_gain is not None and fl_maxGin < 0.01:    # 信息增益过小, 停止
            return True
        if it_deep is not None and it_deep >= 15:        # 树过深, 停止
            return True
        if it_minSample is not None and it_minSample <= 1:        # 树过深, 停止
            return True
        return False

    # 节点处理
    def node(self,dc_sample,ins_node):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']

        arr_LNum = arr_L.shape[1]

        if self.isStop(it_deep=ins_node.it_deep):
            print('deep stop')
            return
        if self.isStop(it_minSample=arr_LNum):      # 纯度计算,如果只有一个样本 calPurity 会出问题,所以加样本数量停止条件
            print('min stop')
            return

        fl_nowPurity = self.calPurity({'X': None, 'L': arr_L})         # 计算当前节点的纯度, 用于计算信息增益
        arr_everyFSplitPoint,arr_everyFPurityExpect = self.all_feature_purity_expect({'X': arr_X, 'L': arr_L})   # 计算每一个特征分割后的纯度
        arr_gain = fl_nowPurity - arr_everyFPurityExpect       # 计算增益

        if self.isStop(arr_gain):
            print('gain stop')
            return

        it_selectFeatureIndex = arr_gain.argmax()                     # 获取信息增益最大的特征(行)索引数组
        fl_splitPoint = arr_everyFSplitPoint[it_selectFeatureIndex]
        arr_selectArrXbyIdx = arr_X[it_selectFeatureIndex]               # 获取信息增益最大的特征(行)样本

        arr_left = n.where(arr_selectArrXbyIdx <= fl_splitPoint)[0]
        arr_leftX = arr_X[:, arr_left]
        arr_leftL = arr_L[:, arr_left]
        tp_subInfo = (it_selectFeatureIndex, fl_splitPoint, False)
        ins_subNode = NodeClass(tp_subInfo, arr_leftL, ins_node.it_deep)
        ins_node.addSubNode(ins_subNode=ins_subNode, tp_subInfo=tp_subInfo)
        self.node({'X': arr_leftX, 'L': arr_leftL}, ins_subNode)  # 递归创建节点

        arr_righ = n.where(arr_selectArrXbyIdx > fl_splitPoint)[0]
        arr_righX = arr_X[:, arr_righ]
        arr_righL = arr_L[:, arr_righ]
        tp_subInfo = (it_selectFeatureIndex, fl_splitPoint, True)
        ins_subNode = NodeClass(tp_subInfo, arr_righL, ins_node.it_deep)
        ins_node.addSubNode(ins_subNode=ins_subNode, tp_subInfo=tp_subInfo)
        self.node({'X': arr_righX, 'L': arr_righL}, ins_subNode)  # 递归创建节点

    # 计算所有特征纯度的期望
    def all_feature_purity_expect(self,dc_sample):
        arr_X = dc_sample['X']
        it_featureNum = arr_X.shape[0]                   # 特征数量

        arr_everyFSplitPoint = n.zeros(it_featureNum)
        arr_everyFPurityExpect = n.zeros(it_featureNum)    # 记录所有特征的分割纯度

        # 计算使用某个特征进行分割后的纯度的期望
        for it_featureIndex in range(it_featureNum):   # it_featureIndex特征索引,即第几个特征  arr_aFX每个样本的该索引特征取值
            fl_splitPoint,fl_purityExpect = self.calPurityExpection(dc_sample,it_featureIndex)
            arr_everyFSplitPoint[it_featureIndex] = fl_splitPoint
            arr_everyFPurityExpect[it_featureIndex] = fl_purityExpect # 所有特征的纯度的期望

        return arr_everyFSplitPoint,arr_everyFPurityExpect

    # 计算单个特征的纯度期望,CART树是二叉树
    def calPurityExpection(self,dc_sample,it_featureIndex):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']
        arr_oneRowX = arr_X[it_featureIndex, :]
        it_LNum = arr_L.shape[1]

        # 计算出分割点
        # 计算每个分割点对应的纯度
        # 选择纯度最大的分割点
        arr_oneRowXSort = n.sort(arr_oneRowX)

        arr_zero = n.array([0])
        arr_temp = n.append(arr_zero,arr_oneRowXSort[:-1])
        arr_split = 0.5*(arr_temp + arr_oneRowXSort)[1:]

        arr_uniqueSplit = n.unique(arr_split)

        arr_purityExpect = n.zeros(arr_uniqueSplit.shape[0])
        for it_spliti,fl_split in enumerate(arr_uniqueSplit):
            fl_purityExpect = 0
            arr_index1 = n.where(arr_oneRowX <= fl_split)[0]
            arr_newL1 = arr_L[:,arr_index1]
            it_newLNum1 = arr_newL1.shape[1]
            fl_purity1 = self.calPurity(dc_sample={'X':None,'L':arr_newL1})
            fl_purityExpect += it_newLNum1/it_LNum * fl_purity1

            arr_index2 = n.where(arr_oneRowX > fl_split)[0]
            arr_newL2 = arr_L[:, arr_index2]
            it_newLNum2 = arr_newL2.shape[1]
            fl_purity2 = self.calPurity(dc_sample={'X':None,'L':arr_newL2})
            fl_purityExpect += it_newLNum2 / it_LNum * fl_purity2

            arr_purityExpect[it_spliti] = fl_purityExpect
        fl_minPurityExpect = n.min(arr_purityExpect)                    # 越小越好
        fl_sqlitPoint = arr_uniqueSplit[n.argmin(arr_purityExpect)]

        return fl_sqlitPoint,fl_minPurityExpect

    # 计算单个特征的单个取值的纯度
    def calPurity(self,dc_sample):
        fl_zero = 1e-6          # 定义一个极小值代替 0

        arr_L = dc_sample['L']
        arr_nL = n.sum(arr_L, axis=1)  # 样本每个类别数量
        arr_sumL = arr_L.shape[1]  # 样本数量

        if arr_sumL == 0:
            return 0.0

        arr_proL = arr_nL / arr_sumL    # 样本概率 = 样本每个类别数量 / 样本数量
        arr_proL = n.where(arr_proL < fl_zero, fl_zero, arr_proL)   # 将样本概率中接近 0 的值进行替换, 否则 0*log(0) 将出现问题
        fl_purity = - n.sum(arr_proL * n.log(arr_proL))             # 计算纯度
        return fl_purity

    # 预测
    def predict(self, arr_X, bl_isShowPredict=False):
        # 递归预测
        def pred(node, arr_X, bl_isShowPredict=False):
            if bl_isShowPredict:
                node.bl_isPredict = True
            if node.isLeaf():  # 如果到达了叶子节点,则返回预测概率
                return node.arr_proLabel
            it_featureIndex = node.tp_subInfo[0]
            fl_splitPoint = node.tp_subInfo[1]
            fl_featureValue = arr_X[it_featureIndex, 0]
            bl = False if fl_featureValue <= fl_splitPoint else True
            tp_subKey = (it_featureIndex, fl_splitPoint, bl)
            if tp_subKey not in node.subNode:  # 由于训练时样本不断分割,可能出现子节点样本中的特征没有该取值,此时直接返回奔节点的预测概率
                return node.arr_proLabel
            arr_pre = pred(node=node.subNode[tp_subKey], arr_X=arr_X, bl_isShowPredict=bl_isShowPredict)
            return arr_pre
        arr_X = arr_X.T
        if arr_X.shape[0] != self.it_featureNum:
            raise ValueError('DTree.pred() 输入特征的维度与训练样本不一致!')
        arr_pre = pred(self.root, arr_X, bl_isShowPredict=bl_isShowPredict)
        return arr_pre.copy()


    # 决策树可视化
    def show(self):
        Dtd = DTreeDrawClass(self.root)
        Dtd.drawTree()
        Dtd.show()
    # 决策树预测可视化
    def predictShow(self, arr_X):
        arr_pre = self.predict(arr_X, bl_isShowPredict=True)
        Dtd = DTreeDrawClass(self.root, bl_isShowPredict=True)
        Dtd.drawTree()
        Dtd.show()
        return arr_pre

    # 决策树可视化保存
    def showSave(self,sr_savePath):
        TreeSaveClass(self.root,sr_savePath)


if __name__ == '__main__':
    arr_X = n.array([
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1],
        [2, 1, 1],
    ])
    arr_L = n.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])
    # arr_aX = n.array([
    #     [1, 2, 1]
    # ])
    #
    # arr_pre = T.predict(arr_aX)
    # print('预测概率: {}'.format(arr_pre))

    from SBLIB.SB_Data import arr_XData2,arr_LData2
    T = CARTree(arr_XData2, arr_LData2)
    dc = {
        'face': 1,  # 0
        'rf': 2,    # 1
        'tui': 1,   # 2
        'high': 1,  # 3
        'fm': 1,    # 4
        'age': 1,   # 5
    }

    ls = [v for k, v in dc.items()]
    X = n.array([ls])
    arr_pre = n.round(T.predict(X), 2)

    print('zhengru 概率: {}\nhouru   概率: {}'.format(arr_pre[0], arr_pre[1]))

    T.showSave('T.png')






























