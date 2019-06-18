import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from sklearn.cluster import KMeans
import warnings

# 忽略一些版本不兼容等警告
warnings.filterwarnings("ignore")

#读取数据
def loadData(file_path):
    file_object = open(file_path, 'rU')
    try:
        lines = []
        for line in file_object:
            line = line.rstrip('\n').split()
            line = [float(x) for x in line]
            lines.append(line)
    finally:
        file_object.close()
    return np.array(lines)

#求欧氏距离
def distance(data,centers):
    dis = np.zeros((data.shape[0],centers.shape[0]))
    for i in range(len(data)):
        for j in range(len(centers)):
            dis[i,j] = np.linalg.norm(data[i] - centers[j])
    return dis

#找到k个中心点中最近的一个
def closest_centroid(data,centers):
    dis = distance(data,centers)
    closest = np.argmin(dis,1)
    return closest

#k-means
def K_Means(data,k):

    centers = np.random.choice(np.arange(-25,25,0.1),(k,3))#随机生成k个样本点
    print("初始centers:\n",centers)
    flag = True
    m = 1
    while flag:
        flag = False
        for i in range(data.shape[0]):# 把每一个数据点划分到离它最近的中心点
            closest = closest_centroid(data, centers)
            if m!=1:
                if (closest - closest_old).any(): flag = True#只要closest还在变化flag就为True
            for j in range(k):
                if len(data[closest == j])!=0:  #不为空再继续
                    centers[j] = np.mean(data[closest == j], axis=0)  # 均值向量

            closest_old = closest
            m = 0
    return centers, closest


if __name__ == "__main__":

    data = loadData('data.txt')
    print(data)



    kmeans = KMeans(n_clusters=3).fit(data)
    centers,closest = K_Means(data,3)

    print("centers:\n",centers)
    print("自带函数计算的centers:\n",kmeans.cluster_centers_)

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(centers[:, 0], centers[:, 1:2], centers[:, 2:], marker='^', c='r', label='Centers')

    color = ['g','b','y']
    for i in range(3):
        print("第"+str(i+1)+"个簇中的点为:\n", data[closest == i])
        x = data[closest == i][:, 0]
        y = data[closest == i][:, 1:2]
        z = data[closest == i][:, 2:]
        ax.scatter(x, y, z, c=color[i], label='cluster'+str(i))

    # 绘制图例
    ax.legend(loc='best')

    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'r'})
    ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'g'})
    ax.set_xlabel('X', fontdict={'size': 10, 'color': 'b'})

    plt.show()