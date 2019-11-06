import numpy as n



class DDTree:
    # 离散决策树
    '''
    输入数据 arr_X:
            每列为一个样本,
            每个样本的同一行都表示一类特征
    输入标签 arr_L:
            每列为一个样本,每个标签向量只有一行为 1 ,其他为 0
    '''

    class NodeClass:
        def __init__(self,it_deep,arr_Label):
            self.subNode = {}
            # 特征(样本的行)
            self.it_featureRow = None
            self.it_deep = it_deep + 1
            # 这个节点使用的训练集
            self.arr_Label = arr_Label
            # 这个节点的预测
            self.probability()

        def add_subNode(self,it_featureRow,it_featureValue,inst_subNode):
            self.it_featureRow = it_featureRow
            self.subNode[it_featureValue] = inst_subNode

        def probability(self):
            arr_L = self.arr_Label
            arr_nL = n.sum(arr_L, axis=1)
            arr_sumL = arr_nL.sum()
            arr_proL = (arr_nL / arr_sumL)
            # 每个标签的概率
            self.arr_proLabel = arr_proL

        def isLeaf(self):
            if len(self.subNode) == 0:
                return True
            else:
                return False

    def __init__(self,arr_X,arr_L):
        self.root = self.NodeClass(0,arr_L)
        self.node({'X':arr_X,'L':arr_L},self.root)

    # 终止判断
    def isStop(self,fl_gain):
        fl_maxGin = fl_gain.max()
        # 信息增益过小时停止
        if fl_maxGin < 0.1:
            return True
        return False

    # 节点处理
    def node(self,dc_sample,ins_node):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']

        # 计算当前节点的纯度
        fl_nowPurity = self.purity({'X': None, 'L': arr_L})
        # 计算所有特征分割后的纯度
        arr_allFeaturePurityExpect = self.all_feature_purity_expect({'X': arr_X, 'L': arr_L})
        # 计算信息增益
        fl_gain = fl_nowPurity - arr_allFeaturePurityExpect

        if self.isStop(fl_gain):
            return

        # 获取信息增益最大的特征(行)
        it_maxGainRow = fl_gain.argmax()
        arr_maxGainArrX = arr_X[it_maxGainRow]
        set_ = set(arr_maxGainArrX)
        for i in set_:
            arr_col = n.where(arr_maxGainArrX == i)[0]
            # 根据特征的每一个取值将样本分开
            arr_aNewX = arr_X[:, arr_col]
            arr_aNewL = arr_L[:, arr_col]

            inst_subNode = self.NodeClass(ins_node.it_deep,arr_aNewL)
            ins_node.add_subNode(it_featureRow=it_maxGainRow, it_featureValue=i, inst_subNode=inst_subNode)
            self.node({'X': arr_aNewX, 'L': arr_aNewL}, inst_subNode)

    # 计算所有特征纯度的期望
    def all_feature_purity_expect(self,dc_sample):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']

        it_featureNumSum = arr_X.shape[0]
        it_sampleSum = arr_X.shape[1]
        # 记录所有特征后的分割纯度
        arr_allFPurityExpect = n.zeros(it_featureNumSum)
        for it_featureNum, arr_aFX in enumerate(arr_X):
            # 计算使用某个特征进行分割后的纯度的期望
            set_ = set(arr_aFX)
            fl_aFPurityExpect = 0
            for i in set_:
                arr_col = n.where(arr_aFX == i)[0]
                # 一个特征的一个取值的所有样本的标签集合
                arr_aFaVL = arr_L[:, arr_col]
                fl_purity = self.purity({'X': None, 'L': arr_aFaVL})
                it_aFaVNumSum = arr_aFaVL.shape[1]
                # 一个特征的一个取值的所有样本的标签集合 占 所有样本 的比例
                fl_pro = it_aFaVNumSum / it_sampleSum
                # 一个特征的纯度的期望
                fl_aFPurityExpect += fl_pro * fl_purity
            # 所有特征的纯度的期望
            arr_allFPurityExpect[it_featureNum] = fl_aFPurityExpect
        return arr_allFPurityExpect

    # 计算纯度
    def purity(self,dc_sample):
        fl_zero = 1e-6

        arr_X = dc_sample['X']
        arr_L = dc_sample['L']
        # 统计标签占比
        arr_nL = n.sum(arr_L, axis=1)
        arr_sumL = arr_nL.sum()
        arr_proL = arr_nL / arr_sumL
        # 由于浮点数 0*log0 并非为 0 ,因此定义一个极小值代替 0
        arr_proL = n.where(arr_proL < fl_zero, fl_zero, arr_proL)
        # 计算纯度
        fl_purity = - n.sum(arr_proL * n.log(arr_proL))
        return fl_purity

    # 预测
    def predict(self,node,arr_aX):
        if node.isLeaf():
            print('概率: {}'.format(node.arr_proLabel))
            return
        it_featureRow = node.it_featureRow
        it_featureValue = arr_aX[it_featureRow,0]
        self.predict(node= node.subNode[it_featureValue],arr_aX=arr_aX)

if __name__ == '__main__':
    # arr_X = n.array([
    #     [1, 2, 1, 1],
    #     [1, 2, 1, 3],
    #     [1, 2, 1, 1],
    #     [0, 0, 0, 0],
    # ], dtype=n.int)
    arr_X = n.array([
        [1, 1, 1, 2],
        [1, 2, 1, 1],
        [1, 1, 1, 1],
    ])
    arr_L = n.array([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])

    T = DDTree(arr_X,arr_L)
    arr_aX = n.array([
        [1],
        [2],
        [1]
    ])
    T.predict(T.root,arr_aX)








