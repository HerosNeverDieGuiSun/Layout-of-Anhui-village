import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from data.rendered import RenderedComposite
import torch
import random


class CategoryDataset(Dataset):
    def __init__(self, source, block_category, counts_house_categories, p_removing):
        if not os.path.exists(source):
            print('数据文件路径未找到')

        self.block_category = block_category                        # 当前选择的block类别
        self.counts_house_categories = counts_house_categories      # 房屋类别计数,共有10个类别
        self.p_removing = p_removing                                # 移除概率
        self.data_dir = [source + '/cnts/{}_blocks'.format(self.block_category), source + '/info/{}_blocks'.format(self.block_category)]
        self.cnts_files = os.listdir(self.data_dir[0])              # 获取当前block类别下的cnts的txt文件目录
        self.info_files = os.listdir(self.data_dir[1])              # 获取当前block类别下的info的txt文件目录
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

        existing_categories = torch.zeros(self.counts_house_categories)     # 先设10个空房屋类别[0,0,...,0]
        existing_cnts = cnts[1:-1]                                          # 现有的cnts中房屋轮廓数据
        existing_info = info                                                # 现有的info房屋信息数据
        future_cnts = []                                                    # 未来存储cnts
        future_info = []                                                    # 未来存储info

        temp_cnts = existing_cnts[:]
        temp_info = existing_info[:]

        # 移除房屋
        for i in range(len(temp_info)):
            if random.uniform(0, 1) > self.p_removing:
                future_cnts.append(temp_cnts[i])
                future_info.append(temp_info[i])
                existing_cnts.remove(temp_cnts[i])
                existing_info.remove(temp_info[i])
            else:
                existing_categories[temp_info[i]['label']] += 1         # 给当前存在的房屋类别标签数+1

        existing_cnts.append(cnts[-1])                                  # 重新添加road数据
        heads = [i['label'] for i in existing_info]
        existing_cnts.insert(0, heads)                                  # 重新插入现有block中房屋类别标签列
        x = RenderedComposite(existing_cnts, existing_info).get_composite()

        # 添加<stop>类别
        y = torch.zeros(1).long()
        if len(future_info) == 0:
            y[-1] = 10
        else:
            y_label = random.choice(future_info)['label']
            y[0] = y_label

        return x, y, existing_categories


# 测试tesor画图
def test_drawpic(t):
    print(t.shape)
    image = plt.imshow(t.squeeze(), cmap='gray')
    plt.show()


if __name__ == "__main__":
    data_dir = './txt_data_divide'
    block_category = 3  # 选择当前的block类别为第3种
    num_house_categories = 10  # 房屋类别有10种
    p_removing = 0.5
    dataset = CategoryDataset(data_dir, block_category, num_house_categories, p_removing)
    data_loader = DataLoader(dataset)
    data_iter = iter(data_loader)
    x, y, existing_categories = next(data_iter)

    # 测试
    test_drawpic(x[0][1])
    print(y)
    print(existing_categories)
