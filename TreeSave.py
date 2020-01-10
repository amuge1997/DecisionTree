
import numpy as np
from PIL import Image
from PIL import ImageDraw,ImageFont

class TreeSaveClass:
    def __init__(self,ins_root,sr_savePath):
        self.Troot = ins_root
        self.sr_savePath = sr_savePath

        self.it_wid = 250
        self.it_hei = 105
        self.it_xoffset = 100
        self.it_yoffset = 40

        self.it_maxX = 0
        self.it_maxY = 0
        self.calMaxXY(self.Troot, {'x': 0, 'y': 0}, {'x': 0, 'y': 0})

        array = np.ndarray((self.it_hei*(self.it_maxY+3),self.it_wid*(self.it_maxX+2), 3), np.uint8)
        array[:, :, 0] = 255
        array[:, :, 1] = 255
        array[:, :, 2] = 255

        self.image = Image.fromarray(array)
        self.draw = ImageDraw.Draw(self.image)

        self.font = ImageFont.truetype('C:\Windows\Fonts\consola.ttf', 20)
        self.drawWalk(self.Troot,{'x': 0, 'y': 0},{'x':0,'y':0})
        self.image.save(self.sr_savePath)


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
    # 递归
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

        self.draw.rectangle((it_drawSrcX,it_drawSrcY,it_rectangleX,it_rectangleY-4),fill='#9fafbf',outline=10)

        it_featureIndex = node.tp_selfInfo[0]
        fl_featureValue = node.tp_selfInfo[1]
        it_selfInfoLen = len(node.tp_selfInfo)

        sr_pro = 'pro: {}'.format(np.round(node.arr_proLabel,2)) # 概率
        sr_fea = 'fea: {}'.format(it_featureIndex)               # 特征(行)
        if it_selfInfoLen == 2:
            sr_val = 'val: {}'.format(fl_featureValue)  # 特征(行)的取值
        elif it_selfInfoLen == 3:
            bl = node.tp_selfInfo[2]
            if bl:
                sr_val = 'val: >{}'.format(fl_featureValue)               # 特征(行)的取值
            else:
                sr_val = 'val: <={}'.format(fl_featureValue)  # 特征(行)的取值
        sr_num = 'num: {}'.format(node.ar_Label.shape[1])        # 该节点的样本数量

        self.draw.text((it_drawSrcX + 10,it_drawSrcY + 10),text=sr_fea,fill='black',font=self.font)
        self.draw.text((it_drawSrcX + 10, it_drawSrcY + 30), text=sr_val,fill='black',font=self.font)
        self.draw.text((it_drawSrcX + 10, it_drawSrcY + 50), text=sr_num,fill='black',font=self.font)
        self.draw.text((it_drawSrcX + 10, it_drawSrcY + 70), text=sr_pro,fill='black',font=self.font)

        tp_thisxy = (it_drawSrcX,it_drawSrcY + it_hei/2)        # 当前节点坐标
        tp_lastxy = (node_lxy['x'],node_lxy['y'] + it_hei/2)    # 父节点坐标
        self.draw.line((tp_thisxy,tp_lastxy),fill='black',width=3)

        node_lxy = {'x':it_rectangleX,'y':it_drawSrcY}
        if x>self.it_maxX:
            self.it_maxX = x
        if y>self.it_maxY:
            self.it_maxY = y
        if node.isLeaf():
            return node_xy['y']             # 如果是叶子节点则返回当前高度
        x += 1
        for k, subNode in node.subNode.items():
            y = self.drawWalk(subNode, {'x': x, 'y': y},node_lxy)
            y += 1
        return y - 1                        # 返回当前高度








