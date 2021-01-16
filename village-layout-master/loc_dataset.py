import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from data.rendered import RenderedComposite
import torch
import random


class LocDataset(Dataset):
    def __init__(self, source, block_category):
        if not os.path.exists(source):
            print('数据文件路径未找到')

        self.block_category = block_category
        self.data_dir = [source + '/cnts/{}_blocks'.format(self.block_category),
                         source + '/info/{}_blocks'.format(self.block_category)]
        self.cnts_files = os.listdir(self.data_dir[0])
        self.info_files = os.listdir(self.data_dir[1])
        self.data_len = len(self.cnts_files)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # idx的范围是从0到 len(self)的索引
        txt_name = [self.cnts_files[idx], self.info_files[idx]]
        with open(self.data_dir[0] + '/' + txt_name[0], 'r') as cnts_f:
            cnts = eval(cnts_f.read())
        with open(self.data_dir[1] + '/' + txt_name[1], 'r') as info_f:
            info = eval(info_f.read())

        num_houses = random.randint(0, len(cnts[0]))        # 随机数，记录当前场景中房屋数
        existing_cnts = cnts[1:num_houses]                  # 当前场景中已存在的房屋
        existing_cnts.append(cnts[-1])                      # 添加block数据
        existing_info = info[0:num_houses - 1]              # 存储对应房屋详细信息

        move_cnts = cnts[num_houses-1:-1]                   # 存储移除的房屋
        move_cnts.append(cnts[-1])                          # 添加block数据
        move_info = info[num_houses - 1:]                   # 存储移除的房屋详细信息

        # 对当前场景进行渲染，作为位置网络模型的输入  移除了部分对象的场景渲染图
        for i in range(num_houses):
            input = RenderedComposite(existing_cnts, existing_info).get_composite()

        # 测试移除的房子渲染图
        for i in range(len(cnts[0]) - num_houses + 1):
            moved_house = RenderedComposite(move_cnts, move_info).get_composite()

        # 记录移除的房屋的 质心centroids，作为输出的target
        centroids = []
        temp = RenderedComposite(move_cnts, move_info)
        for i in range(len(cnts[0]) - num_houses + 1):
            x, y = temp.get_movedhouse_center(i)                                # 记录移除的房屋质心坐标
            centroids.append((x // 4, y // 4, move_info[i]['label']))           # 质心和对应房屋类别标签

        output = torch.zeros((64, 64)).long()

        for (x, y, label) in centroids:
            output[y, x] = label

        return input, output, moved_house


# 测试tesor画图
def test_drawpic(t):
    image = plt.imshow(t, cmap='gray')
    plt.show()


def test_output(t):
    image = plt.imshow(t.squeeze(0), cmap='gray')
    plt.show()


if __name__ == "__main__":
    data_dir = './txt_data_divide'
    block_category = 3
    dataset = LocDataset(data_dir, block_category)
    data_loader = DataLoader(dataset)
    data_iter = iter(data_loader)
    input, output, moved_house = next(data_iter)
    test_drawpic(input[0][1])
    test_drawpic(moved_house[0][1])
    test_output(output)
