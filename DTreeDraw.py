import numpy as np
from tkinter import *

class DTreeDrawClass:
    def __init__(self,ins_root,bl_isShowPredict=False):     # bl_isShowPredict 是否预测可视化
        self.Troot =ins_root
        self.it_maxX = 0
        self.it_maxY = 0

        self.it_wid = 170
        self.it_hei = 72
        self.it_xoffset = 100
        self.it_yoffset = 40

        self.calMaxXY(self.Troot, {'x': 0, 'y': 0}, {'x': 0, 'y': 0})

        self.bl_isShowPredict = bl_isShowPredict
        self.tk = Tk()                                      # 主控件
        self.tk.geometry('1000x950')
        self.tk.resizable(width=False,height=False)         # 不可缩放
        tp_scrollregion = (0,0,self.it_wid*(self.it_maxX+2),self.it_hei*(self.it_maxY+3))   # 限制滚动区域
        self.cv = Canvas(self.tk,background='#afafaf', width=1000, height=950,scrollregion=tp_scrollregion)  # 画布
        self.cv.pack()
        vbar = Scrollbar(self.tk, orient=VERTICAL)          # 竖直滚动条
        vbar.place(x=0, width=20, height=400)
        vbar.configure(command=self.cv.yview)
        hbar = Scrollbar(self.tk, orient=HORIZONTAL)        # 竖直滚动条
        hbar.place(x=20, width=400, height=20)
        hbar.configure(command=self.cv.xview)

    def calMaxXY(self, node, node_xy, node_lxy):
        x, y = node_xy['x'], node_xy['y']
        if x>self.it_maxX:
            self.it_maxX = x
        if y>self.it_maxY:
            self.it_maxY = y
        if node.isLeaf():
            return node_xy['y']  # 如果是叶子节点则返回当前高度
        x += 1
        for k, subNode in node.subNode.items():
            y = self.calMaxXY(subNode, {'x': x, 'y': y}, node_lxy)
            y += 1
        return y - 1  # 返回当前高度

    # 决策树可视化
    def drawTree(self):
        self.drawWalk(self.Troot, {'x': 0, 'y': 0},{'x':0,'y':0})
    # 决策树可视化递归
    def drawWalk(self, node, node_xy, node_lxy):
        x, y = node_xy['x'], node_xy['y']

        it_wid = self.it_wid
        it_hei = self.it_hei
        it_xoffset = self.it_xoffset
        it_yoffset = self.it_yoffset
        it_drawSrcX = x*it_wid + it_xoffset         # 节点参考点坐标X
        it_drawSrcY = y*it_hei + it_yoffset         # 节点参考点坐标Y
        it_rectangleX = it_drawSrcX+it_wid-40       # 矩形右下角坐标X
        it_rectangleY = it_drawSrcY+it_hei          # 矩形右下角坐标Y

        self.cv.create_rectangle(it_drawSrcX, it_drawSrcY, it_rectangleX, it_rectangleY-4,width=1,fill='#9fafbf')  # 节点矩形
        if node.bl_isPredict:                       # 如果需要绘制预测路径
            if self.bl_isShowPredict:               # 如果该节点在预测时被使用到, 则使用黄色框
                self.cv.create_rectangle(it_drawSrcX, it_drawSrcY, it_rectangleX, it_rectangleY-4,width=2,fill='#dfefff')  # 节点矩形
                node.bl_isPredict = False           # 清空预测置位

        it_featureIndex = node.tp_selfInfo[0]
        fl_featureValue = node.tp_selfInfo[1]
        it_selfInfoLen = len(node.tp_selfInfo)

        sr_pro = '概率: {}'.format(np.round(node.arr_proLabel,2)) # 概率
        sr_fea = '特征: {}'.format(it_featureIndex)               # 特征(行)
        if it_selfInfoLen == 2:
            sr_val = '取值: {}'.format(fl_featureValue)  # 特征(行)的取值
        elif it_selfInfoLen == 3:
            bl = node.tp_selfInfo[2]
            if bl:
                sr_val = '取值: ＞{}'.format(fl_featureValue)               # 特征(行)的取值
            else:
                sr_val = '取值: ≤{}'.format(fl_featureValue)  # 特征(行)的取值
        sr_num = '数量: {}'.format(node.ar_Label.shape[1])        # 该节点的样本数量
        self.cv.create_text(it_drawSrcX + 10, it_drawSrcY + 2, text=sr_fea, anchor='nw')
        self.cv.create_text(it_drawSrcX + 10, it_drawSrcY + 17, text= sr_val, anchor='nw')
        self.cv.create_text(it_drawSrcX + 10, it_drawSrcY + 32, text=sr_num, anchor='nw')
        self.cv.create_text(it_drawSrcX + 10, it_drawSrcY + 47, text= sr_pro, anchor='nw')

        tp_thisxy = (it_drawSrcX,it_drawSrcY + it_hei/2)        # 当前节点坐标
        tp_lastxy = (node_lxy['x'],node_lxy['y'] + it_hei/2)    # 父节点坐标
        it_r = 4
        self.cv.create_oval(tp_lastxy[0]-it_r,tp_lastxy[1]-it_r,tp_lastxy[0]+it_r,tp_lastxy[1]+it_r,fill='black')
        self.cv.create_line([tp_lastxy,tp_thisxy],arrow=LAST)   # 节点指向箭头

        node_lxy = {'x':it_rectangleX,'y':it_drawSrcY}
        if node.isLeaf():
            return node_xy['y']             # 如果是叶子节点则返回当前高度
        x += 1
        for k, subNode in node.subNode.items():
            y = self.drawWalk(subNode, {'x': x, 'y': y},node_lxy)
            y += 1
        return y - 1                        # 返回当前高度

    # 打开主循环进行显示
    def show(self):
        self.tk.mainloop()























