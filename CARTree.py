import numpy as n
from DecisionTree.Node import NodeClass
from DecisionTree.DTreeDraw import DTreeDrawClass
from DecisionTree.TreeSave import TreeSaveClass


# 分类回归决策树 CART
class CARTree:
    def __init__(self,arr_X,arr_L,fl_minGain=0.01,it_maxDeep=15,it_minSample=1):
        self.fl_minGain = fl_minGain        # 最小增益
        self.it_maxDeep = it_maxDeep        # 最大深度
        self.it_minSample = it_minSample    # 最小样本数

        arr_X = arr_X.T                     # 样本数据
        arr_L = arr_L.T                     # 样本标签
        self.it_featureNum = arr_X.shape[0] # 特征数量
        self.root = NodeClass(tp_selfInfo=(None,None),arr_Label=arr_L,it_deep=0)    # 根节点
        self.node({'X':arr_X,'L':arr_L},self.root)          # 进行节点处理

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
        arr_X = dc_sample['X']                      # 进入该节点的样本数据
        arr_L = dc_sample['L']                      # 进入该节点的样本标签

        arr_LNum = arr_L.shape[1]                   # 样本数量

        if self.isStop(it_deep=ins_node.it_deep):   # 深度条件判断
            return
        if self.isStop(it_minSample=arr_LNum):      # 纯度条件判断,如果只有一个样本 calPurity 会出问题,所以加样本数量停止条件
            return

        fl_nowPurity = self.calPurity({'X': None, 'L': arr_L})      # 计算当前节点的纯度, 用于计算信息增益
        # 计算每一个特征的分割点和分割后的纯度
        arr_everyFSplitPoint,arr_everyFPurityExpect = self.all_feature_purity_expect({'X': arr_X, 'L': arr_L})
        arr_gain = fl_nowPurity - arr_everyFPurityExpect            # 计算增益

        if self.isStop(arr_gain):                   # 增益条件判断
            return

        it_selectFeatureIndex = arr_gain.argmax()                   # 获取信息增益最大的特征(行)索引
        fl_splitPoint = arr_everyFSplitPoint[it_selectFeatureIndex] # 分割点
        arr_selectArrXbyIdx = arr_X[it_selectFeatureIndex]          # 获取信息增益最大的特征(行)样本数组

        arr_left = n.where(arr_selectArrXbyIdx <= fl_splitPoint)[0] # 左子节点,<=分割点
        arr_leftX = arr_X[:, arr_left]                              # 左子节点样本数据
        arr_leftL = arr_L[:, arr_left]                              # 左子节点样本标签
        tp_subInfo = (it_selectFeatureIndex, fl_splitPoint, False)  # 左子节点信息
        ins_subNode = NodeClass(tp_subInfo, arr_leftL, ins_node.it_deep)    # 左子节点
        ins_node.addSubNode(ins_subNode=ins_subNode, tp_subInfo=tp_subInfo) # 父节点连接左子节点
        self.node({'X': arr_leftX, 'L': arr_leftL}, ins_subNode)    # 递归创建子节点

        arr_righ = n.where(arr_selectArrXbyIdx > fl_splitPoint)[0]  # 同上
        arr_righX = arr_X[:, arr_righ]
        arr_righL = arr_L[:, arr_righ]
        tp_subInfo = (it_selectFeatureIndex, fl_splitPoint, True)
        ins_subNode = NodeClass(tp_subInfo, arr_righL, ins_node.it_deep)
        ins_node.addSubNode(ins_subNode=ins_subNode, tp_subInfo=tp_subInfo)
        self.node({'X': arr_righX, 'L': arr_righL}, ins_subNode)

    # 计算所有特征纯度的期望
    def all_feature_purity_expect(self,dc_sample):
        it_featureNum = self.it_featureNum                  # 特征数量

        arr_everyFSplitPoint = n.zeros(it_featureNum)       # 用于记录每一个特征的分割点
        arr_everyFPurityExpect = n.zeros(it_featureNum)     # 用于记录每一个特征的纯度的期望

        # 计算使用每个特征进行分割后的纯度的期望
        for it_featureIndex in range(it_featureNum):        # 计算第 it_featureIndex 个特征的分割点和纯度期望
            fl_splitPoint,fl_purityExpect = self.calPurityExpection(dc_sample,it_featureIndex)
            arr_everyFSplitPoint[it_featureIndex] = fl_splitPoint       # 记录每一个特征的分割点
            arr_everyFPurityExpect[it_featureIndex] = fl_purityExpect   # 记录每一个特征的纯度的期望
        return arr_everyFSplitPoint,arr_everyFPurityExpect

    # 计算单个特征的纯度期望,CART树是二叉树
    def calPurityExpection(self,dc_sample,it_featureIndex):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']
        arr_oneRowX = arr_X[it_featureIndex, :]             # 取出样本的第 it_featureIndex 个特征的值
        it_LNum = arr_L.shape[1]                            # 该节点的样本的总数

        # 计算出分割点,计算每个分割点对应的纯度,选择纯度最大的分割点
        arr_oneRowXSort = n.sort(arr_oneRowX)               # 对特征的值进行排序

        arr_zero = n.array([0])
        arr_temp = n.append(arr_zero,arr_oneRowXSort[:-1])
        arr_split = 0.5*(arr_temp + arr_oneRowXSort)[1:]    # 求出每个值的中点作为候选分割点

        arr_uniqueSplit = n.unique(arr_split)               # 取出相同的分割点

        arr_purityExpect = n.zeros(arr_uniqueSplit.shape[0])    # 用于记录每个分割点带来的纯度的期望
        for it_spliti,fl_split in enumerate(arr_uniqueSplit):
            fl_purityExpect = 0                             # 一个分割点纯度期望
            arr_index1 = n.where(arr_oneRowX <= fl_split)[0]# 所有特征的取值<=该分割点的样本索引
            if arr_index1.shape[0] > 0:
                arr_newL1 = arr_L[:,arr_index1]                 # 所有特征的取值<=该分割点的样本集合,用于计算纯度
                it_newLNum1 = arr_newL1.shape[1]                # 这个集合的数量
                fl_purity1 = self.calPurity(dc_sample={'X':None,'L':arr_newL1}) # 计算这个集合的纯度
                fl_purityExpect += it_newLNum1/it_LNum * fl_purity1             # 左期望
            else:
                fl_purityExpect += 0.0

            arr_index2 = n.where(arr_oneRowX > fl_split)[0]
            if arr_index2.shape[0] > 0:
                arr_newL2 = arr_L[:, arr_index2]
                it_newLNum2 = arr_newL2.shape[1]
                fl_purity2 = self.calPurity(dc_sample={'X':None,'L':arr_newL2})
                fl_purityExpect += it_newLNum2 / it_LNum * fl_purity2  # 左期望+右期望,得到纯度总期望
            else:
                fl_purityExpect += 0.0

            arr_purityExpect[it_spliti] = fl_purityExpect
        fl_minPurityExpect = n.min(arr_purityExpect)                        # 值越小,纯度越高
        fl_sqlitPoint = arr_uniqueSplit[n.argmin(arr_purityExpect)]         # 纯度最高的分割点

        return fl_sqlitPoint,fl_minPurityExpect                             # 返回该特征的 分割点,纯度期望

    # 计算单个特征的单个取值的纯度
    def calPurity(self,dc_sample):
        fl_zero = 1e-6                      # 定义一个极小值代替 0

        arr_L = dc_sample['L']
        arr_nL = n.sum(arr_L, axis=1)       # 样本集合中每个类别数量
        arr_sumL = arr_L.shape[1]           # 样本集合中的样本数量

        arr_proL = arr_nL / arr_sumL        # 样本概率 = 样本每个类别数量 / 样本数量
        arr_proL = n.where(arr_proL < fl_zero, fl_zero, arr_proL)   # 将样本概率中接近 0 的值进行替换, 否则 0*log(0) 将出现问题
        fl_purity = - n.sum(arr_proL * n.log(arr_proL))             # 计算该样本集合的纯度
        return fl_purity

    # 预测
    def predict(self, arr_X, bl_isShowPredict=False):
        # 递归预测
        def pred(node, arr_X, bl_isShowPredict=False):
            if bl_isShowPredict:                        # 可视化时是否绘制路径
                node.bl_isPredict = True                # 当节点的这个属性置位时,决策树可视化将把该节点更换颜色
            if node.isLeaf():                           # 如果到达了叶子节点,则返回预测概率
                return node.arr_proLabel
            it_featureIndex = node.tp_subInfo[0]        # 子节点所选用的特征
            fl_splitPoint = node.tp_subInfo[1]          # 子节点选用的特征的分割点
            fl_featureValue = arr_X[it_featureIndex, 0] # 该样本的该特征的取值
            bl = False if fl_featureValue <= fl_splitPoint else True    # 该样本取值与分割点取值的比较
            tp_subKey = (it_featureIndex, fl_splitPoint, bl)            # 子节点的键
            arr_pre = pred(node=node.subNode[tp_subKey], arr_X=arr_X, bl_isShowPredict=bl_isShowPredict)    # 递归预测
            return arr_pre                              # 返回概率

        arr_X = arr_X.T
        if arr_X.shape[0] != self.it_featureNum:        # 预测样本特征维度与训练样本特征维度要求一致
            raise ValueError('DTree.pred() 输入特征的维度与训练样本不一致!')
        arr_pre = pred(self.root, arr_X, bl_isShowPredict=bl_isShowPredict) # 递归预测
        return arr_pre.copy()                           # 返回预测概率

    # 决策树可视化
    def show(self,ls_othName=None):
        DTreeDrawClass(self.root,ls_othName=ls_othName)
    # 决策树预测可视化
    def predictShow(self,arr_X,ls_othName=None,sr_title=None):
        arr_pre = self.predict(arr_X, bl_isShowPredict=True)
        DTreeDrawClass(self.root, bl_isShowPredict=True,ls_othName=ls_othName,sr_title=sr_title)
        return arr_pre

    # 决策树可视化保存
    def showSave(self,sr_savePath,ls_othName=None):
        TreeSaveClass(ins_root=self.root,sr_savePath=sr_savePath,ls_othName=ls_othName)
































