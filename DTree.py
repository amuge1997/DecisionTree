import numpy as n



class DDTree:
    # 离散决策树

    class TreeClass:
        def __init__(self):
            pass

    
    def __init__(self):
        self.Tree = self.TreeClass()

    def node(self,dc_sample):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']

        # 计算当前节点的纯度
        fl_nowPurity = self.purity({'X': None, 'L': arr_L})
        # 计算所有特征分割后的纯度
        arr_allFeaturePurityExpect = self.all_feature_purity_expect({'X': arr_X, 'L': arr_L})
        # 计算信息增益
        fl_gain = fl_nowPurity - arr_allFeaturePurityExpect
        it_maxGain = fl_gain.argmax()

        # 删去信息增益最大的特征
        arr_newX = n.delete(arr_X, it_maxGain, axis=0)

        arr_maxGainArrX = arr_X[it_maxGain]
        set_ = set(arr_maxGainArrX)
        for i in set_:
            arr_col = n.where(arr_maxGainArrX == i)[0]
            arr_aNewX = arr_newX[:, arr_col]
            arr_aNewL = arr_L[:, arr_col]
            print(arr_aNewX)


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
            # print(fl_aFPurityExpect)
            arr_allFPurityExpect[it_featureNum] = fl_aFPurityExpect
        # print(arr_allFPurityExpect)
        return arr_allFPurityExpect

    def purity(self,dc_sample):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']

        # 统计标签占比
        arr_nL = n.sum(arr_L, axis=1)
        arr_sumL = arr_nL.sum()
        arr_proL = arr_nL / arr_sumL

        # 计算纯度
        fl_purity = - n.sum(arr_proL * n.log(arr_proL))
        return fl_purity













