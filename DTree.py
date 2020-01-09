import numpy as n
from DecisionTree.Node import NodeClass
from DecisionTree.DTreeDraw import DTreeDrawClass

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

    def __init__(self,arr_X,arr_L):
        arr_X = arr_X.T
        arr_L = arr_L.T
        self.it_featureNum = arr_X.shape[0]
        self.root = NodeClass(0,arr_L)
        self.node({'X':arr_X,'L':arr_L},self.root)

    # 终止判断
    def isStop(self,fl_gain=None,it_deep=None):
        fl_maxGin = fl_gain.max()
        if fl_gain is not None and fl_maxGin < 0.01:    # 信息增益过小, 停止
            return True
        if it_deep is not None and it_deep >= 5:        # 树过深, 停止
            return True
        return False

    # 节点处理
    def node(self,dc_sample,ins_node):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']

        fl_nowPurity = self.purity({'X': None, 'L': arr_L})         # 计算当前节点的纯度, 用于计算信息增益
        arr_everyFeaturePurityExpect = self.all_feature_purity_expect({'X': arr_X, 'L': arr_L})   # 计算每一个特征分割后的纯度
        fl_gain = fl_nowPurity - arr_everyFeaturePurityExpect       # 计算信息增益

        if self.isStop(fl_gain,ins_node.it_deep):
            return

        it_maxGainFeatureRow = fl_gain.argmax()                     # 获取信息增益最大的特征(行)索引数组
        arr_maxGainArrX = arr_X[it_maxGainFeatureRow]               # 获取信息增益最大的特征(行)样本
        set_ = set(arr_maxGainArrX)
        for i in set_:
            arr_col = n.where(arr_maxGainArrX == i)[0]

            # 根据特征的每一个取值将样本分离
            arr_aNewX = arr_X[:, arr_col]
            arr_aNewL = arr_L[:, arr_col]

            inst_subNode = NodeClass(ins_node.it_deep,arr_aNewL)        # 在确定了选取的特征行后, 该特征的每个取值作为一个子节点
            ins_node.addSubNode(it_featureRow= it_maxGainFeatureRow, it_featureValue= i, inst_subNode= inst_subNode)  # 将子节点连接到父节点上
            self.node({'X': arr_aNewX, 'L': arr_aNewL}, inst_subNode)   # 递归创建节点

    # 计算所有特征纯度的期望
    def all_feature_purity_expect(self,dc_sample):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']

        it_featureNumSum = arr_X.shape[0]                   # 特征数量
        it_sampleSum = arr_X.shape[1]                       # 样本数量

        arr_allFPurityExpect = n.zeros(it_featureNumSum)    # 记录所有特征后的分割纯度

        # 计算使用某个特征进行分割后的纯度的期望
        for it_featureIndex, arr_aFX in enumerate(arr_X):   # it_featureIndex特征索引,即第几个特征  arr_aFX每个样本的该索引特征取值
            set_enableValue = set(arr_aFX)                  # 一个特征的所有可能的取值
            fl_oneFPurityExpect = 0                         # 一个特征的纯度期望

            for i in set_enableValue:                       # 遍历一个特征的每个取值
                arr_colIndex = n.where(arr_aFX == i)[0]     # 拥有该特征取值的样本的索引
                arr_aFaVL = arr_L[:, arr_colIndex]          # 一个特征的一个取值的所有样本的标签集合
                fl_purity = self.purity({'X': None, 'L': arr_aFaVL})    # 一个特征的一个取值的纯度
                it_aFaVNumSum = arr_aFaVL.shape[1]          # 一个特征的一个取值的样本数量
                fl_pro = it_aFaVNumSum / it_sampleSum       # 一个特征的一个取值的样本数量 在 所有样本 中的比例
                fl_oneFPurityExpect += fl_pro * fl_purity   # 一个特征的纯度的期望

            arr_allFPurityExpect[it_featureIndex] = fl_oneFPurityExpect # 所有特征的纯度的期望
        return arr_allFPurityExpect

    # 计算纯度
    def purity(self,dc_sample):
        fl_zero = 1e-6          # 定义一个极小值代替 0

        arr_L = dc_sample['L']

        arr_nL = n.sum(arr_L, axis=1)   # 样本每个类别数量
        arr_sumL = arr_nL.sum()         # 样本数量
        arr_proL = arr_nL / arr_sumL    # 样本概率 = 样本每个类别数量 / 样本数量
        arr_proL = n.where(arr_proL < fl_zero, fl_zero, arr_proL)   # 将样本概率中接近 0 的值进行替换, 否则 0*log(0) 将出现问题
        fl_purity = - n.sum(arr_proL * n.log(arr_proL))             # 计算纯度
        return fl_purity

    # 预测
    def predict(self,arr_aX,bl_isShowPredict=False):
        arr_aX = arr_aX.T
        if arr_aX.shape[0] != self.it_featureNum:
            raise ValueError('DTree.pred() 输入特征的维度与训练样本不一致!')
        arr_pre = self.pred(self.root,arr_aX,bl_isShowPredict=bl_isShowPredict)
        return arr_pre.copy()
    # 递归预测
    def pred(self,node,arr_aX,bl_isShowPredict=False):
        if bl_isShowPredict:
            node.bl_isPredict = True
        if node.isLeaf():                       # 如果到达了叶子节点,则返回预测概率
            return node.arr_proLabel
        it_featureRow = node.it_featureRow
        it_featureValue = arr_aX[it_featureRow,0]
        if it_featureValue not in node.subNode: # 由于训练时样本不断分割,可能出现子节点样本中的特征没有该取值,此时直接返回奔节点的预测概率
            return node.arr_proLabel
        arr_pre = self.pred(node= node.subNode[it_featureValue],arr_aX=arr_aX,bl_isShowPredict=bl_isShowPredict)
        return arr_pre

    # 决策树可视化
    def show(self):
        Dtd = DTreeDrawClass(self.root)
        Dtd.drawTree()
        Dtd.show()

    # 决策树预测可视化
    def predictShow(self,arr_aX):
        arr_pre = self.predict(arr_aX,bl_isShowPredict=True)
        Dtd = DTreeDrawClass(self.root,bl_isShowPredict=True)
        Dtd.drawTree()
        Dtd.show()
        return arr_pre

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

    from DecisionTree.DTreeDraw import DTreeDraw

    Dtd = DTreeDraw(T)
    Dtd.drawTree()
    Dtd.show()





















