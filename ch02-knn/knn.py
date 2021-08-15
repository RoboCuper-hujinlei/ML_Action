import numpy as np
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


def createDataSet():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])  # (4, 2)
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


data, label = createDataSet()
print('data:\n', data)
print('label:\n', label)
print()

'''
函数说明:kNN算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
'''


# inx = np.array([101, 23])
def classify0(inX, dataSet, labels, k):
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]

    # 计算与每个样本的差值
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # np.tile(A, reps)  
    sqDiffMat = diffMat ** 2

    sqDistances = sqDiffMat.sum(axis=1)  # sum(1)行相加
    print('sqDistances:', sqDistances)
    distances = sqDistances ** 0.5  # 平方根

    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndicies = distances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}

    for i in range(k):
        votelable = labels[sortedDistIndicies[i]]  # 取出前k个元素的类别
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[votelable] = classCount.get(votelable, 0) + 1
        # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


print(np.tile([0, 0], (3, 1)))
print(np.tile([0, 0], (3, 1)).shape)

print()
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 提到轴 先看数组的维度。这是数组中有三个一维数组，所以是二维数组。shape(3, 3)   索引为[0,1]
print('a[:2, 1:]:', a[:2, 1:])

b = np.arange(20).reshape(4, 5)
print(b.shape)
# 数组转置
print(b.transpose().shape)


# 项目2 约会网站配对效果判定

# 处理数据
def file2matrix(filenames):
    fr = open(filenames)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))

    classLabelVector = []
    # 行索引值
    index = 0
    print('returnMat.shape:', returnMat.shape)
    for line in arrayOfLines:
        # s.strip(rm) 当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        listFromLine = line.split('\t')

        returnMat[index, :] = listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


'''
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
'''


def showdatas(datingDataMat, datingLabels):
    '''
    数据可视化：
    datingDataMat - 特征矩阵
    datingLabels - 分类Labe
    :return:
    '''
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列, 不共享x轴和y轴, fig画布的大小为(13,8)

    # 当 nrow=2, nclos=2 时, 代表fig画布被分为四个区域, axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')

    # 画出散点图, 以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据, 散点大小为15, 透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占', FontProperties=font)

    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()


if __name__ == '__main__':
    '''
        1、电影分类
    '''
    inx = np.array([101, 23])
    test_class = classify0(inx, data, label, 3)
    print('test_class:', test_class)

    '''
        2、约会网站配对效果判定
    '''
    # 数据清洗部分
    datingTestSet = '../data/datingTestSet.txt'
    data, label = file2matrix(datingTestSet)
    print('data:\n', data)
    print('label:\n', label)
    showdatas(data, label)

    '''
    3、
    '''
