import numpy as n
from DecisionTree.Node import NodeClass
from DecisionTree.TreeDraw import DTreeDrawClass
from DecisionTree.TreeSave import TreeSaveClass

class DTree:
    # 离散决策树
    '''
    输入数据 arr_X:
            每行为一个样本,
            每个样本的同一行都表示一类特征
    输入标签 arr_L:
            每行为一个样本,每个标签向量只有一行为 1 ,其他为 0

    注意 DTree().predict() 只能预测一个样本 输入形式为 array( [ [1,2,3,1,5] ] )
    '''

    def __init__(self,arr_X,arr_L,fl_minGain=0.01,it_maxDeep=15,it_minSample=1):
        self.fl_minGain = fl_minGain        # 最小增益
        self.it_maxDeep = it_maxDeep        # 最大深度
        self.it_minSample = it_minSample    # 最小样本数

        arr_X = arr_X.T
        arr_L = arr_L.T
        self.it_featureNum = arr_X.shape[0]
        self.root = NodeClass(tp_selfInfo=(None,None),arr_Label=arr_L,it_deep=0)
        self.node({'X':arr_X,'L':arr_L},self.root)

    # 终止判断
    def isStop(self,arr_gain=None,it_deep=None,it_minSample=None):
        fl_maxGin = n.max(arr_gain)
        if arr_gain is not None and fl_maxGin < self.fl_minGain:            # 信息增益过小, 停止
            return True
        if it_deep is not None and it_deep >= self.it_maxDeep:              # 树过深, 停止
            return True
        if it_minSample is not None and it_minSample <= self.it_minSample:  # 样本过少, 停止
            return True
        return False

    # 节点处理
    def node(self,dc_sample,ins_node):
        arr_L = dc_sample['L']

        arr_LNum = arr_L.shape[1]                       # 样本数量
        if self.isStop(it_minSample=arr_LNum):
            return

        fl_nowPurity = self.calPurity({'X': None, 'L': arr_L})      # 计算当前节点的纯度, 用于计算信息增益
        arr_everyFeaturePurityExpect = self.all_feature_purity_expect({'X': arr_X, 'L': arr_L})   # 计算每一个特征分割后的纯度
        arr_gain = fl_nowPurity - arr_everyFeaturePurityExpect      # 计算信息增益

        if self.isStop(arr_gain=arr_gain,it_deep=ins_node.it_deep):
            return

        it_selectFeatureIndex = arr_gain.argmax()                   # 获取信息增益最大的特征(行)索引数组
        arr_selectArrXbyIdx = arr_X[it_selectFeatureIndex]          # 获取信息增益最大的特征(行)样本
        set_featureValue = set(arr_selectArrXbyIdx)
        for it_featureVal in set_featureValue:
            arr_colIndex = n.where(arr_selectArrXbyIdx == it_featureVal)[0]

            # 根据特征的每一个取值将样本分离
            arr_newX = arr_X[:, arr_colIndex]
            arr_newL = arr_L[:, arr_colIndex]

            tp_subInfo = (it_selectFeatureIndex,it_featureVal)
            ins_subNode = NodeClass(tp_subInfo,arr_newL,ins_node.it_deep)   # 在确定了选取的特征行后, 该特征的每个取值作为一个子节点
            ins_node.addSubNode(ins_subNode=ins_subNode,tp_subInfo=tp_subInfo)
            self.node({'X': arr_newX, 'L': arr_newL}, ins_subNode)          # 递归创建节点

    # 计算所有特征纯度的期望
    def all_feature_purity_expect(self,dc_sample):
        it_featureNum = self.it_featureNum                  # 特征数量
        arr_allFPurityExpect = n.zeros(it_featureNum)       # 记录所有特征后的分割纯度

        # 计算使用某个特征进行分割后的纯度的期望
        for it_featureIndex in range(it_featureNum):        # it_featureIndex特征索引,即第几个特征  arr_aFX每个样本的该索引特征取值
            fl_purityExpect = self.calPurityExpection(dc_sample,it_featureIndex)
            arr_allFPurityExpect[it_featureIndex] = fl_purityExpect # 所有特征的纯度的期望
        return arr_allFPurityExpect

    # 计算单个特征的纯度期望
    def calPurityExpection(self,dc_sample,it_featureIndex):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']
        arr_oneRowX = arr_X[it_featureIndex,:]
        it_LNum = arr_X.shape[1]
        set_enableValue = set(arr_oneRowX)  # 一个特征的所有可能的取值
        fl_purityExpect = 0  # 一个特征的纯度期望
        for i in set_enableValue:  # 遍历一个特征的每个取值
            arr_colIndex = n.where(arr_oneRowX == i)[0]  # 拥有该特征取值的样本的索引
            arr_newL = arr_L[:, arr_colIndex]  # 一个特征的一个取值的所有样本的标签集合
            fl_purity = self.calPurity({'X': None, 'L': arr_newL})  # 一个特征的一个取值的纯度
            it_newLNum = arr_newL.shape[1]  # 一个特征的一个取值的样本数量
            fl_pro = it_newLNum / it_LNum  # 一个特征的一个取值的样本数量 在 所有样本 中的比例
            fl_purityExpect += fl_pro * fl_purity  # 一个特征的纯度的期望
        return fl_purityExpect

    # 计算单个特征的单个取值的纯度
    def calPurity(self,dc_sample):
        fl_zero = 1e-6          # 定义一个极小值代替 0

        arr_L = dc_sample['L']

        arr_nL = n.sum(arr_L, axis=1)   # 样本每个类别数量
        arr_sumL = arr_L.shape[1]         # 样本数量
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
            it_featureValue = arr_X[it_featureIndex, 0]
            tp_subKey = (it_featureIndex, it_featureValue)
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
    def show(self,ls_othName=None):
        DTreeDrawClass(self.root,ls_othName=ls_othName)

    # 决策树预测可视化
    def predictShow(self,arr_aX,ls_othName=None,sr_title=None):
        arr_pre = self.predict(arr_aX,bl_isShowPredict=True)
        DTreeDrawClass(self.root,bl_isShowPredict=True,ls_othName=ls_othName,sr_title=sr_title)
        return arr_pre

    # 决策树可视化保存
    def showSave(self, sr_savePath,ls_othName=None):
        TreeSaveClass(ins_root=self.root, sr_savePath=sr_savePath,ls_othName=ls_othName)

if __name__ == '__main__':

    # 测试

    arr_X = n.array([
        [1,1,1],
        [1,2,1],
        [1,1,1],
        [2,1,1],
    ])

    arr_L = n.array([
        [1,0,0],
        [0,1,0],
        [1,0,0],
        [0,0,1],
    ])

    T = DTree(arr_X,arr_L)

    arr_aX = n.array([
        [1,2,1]
    ])

    arr_pre = T.predict(arr_aX)
    # print('预测概率: {}'.format(arr_pre))





















