import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from data.rendered import RenderedComposite
import torch
import random
import math
import pdb

class OrientDataset(Dataset):
    def __init__(self, source, block_category, counts_house_categories):
        if not os.path.exists(source):
            print('数据文件路径未找到')

        self.block_category = block_category                        # 当前选择的block类别
        self.counts_house_categories = counts_house_categories      # 房屋类别计数,共有10个类别
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

        num_houses = random.randint(1, len(cnts[0]) - 1)    # 随机数，记录当前场景中房屋数
        existing_cnts = cnts[1:num_houses + 1]              # 当前场景中已存在的房屋
        existing_cnts.append(cnts[-1])                      # 添加block数据
        existing_info = info[0:num_houses]                  # 存储对应房屋详细信息
        move_cnts = cnts[num_houses + 1:]                   # 存储移除的房屋和block数据
        move_info = info[num_houses:]                       # 存储移除的房屋详细信息


        input_img = RenderedComposite(existing_cnts, existing_info).get_composite()     # input_img

        existing_categories = torch.zeros(self.counts_house_categories)                 # 记录已存在房屋类别
        for i in existing_info:
            existing_categories[i['label']] += 1

        # 先将移除的房屋的 angle 全部转换为 (cos,sin)
        for i in range(len(move_info)):
            angle = move_info[i]['angle']
            if (angle >= 0 and angle < 180):
                trans_angle = [math.cos(angle), 1]
            if (angle >= 180 and angle < 360):
                trans_angle = [math.cos(angle), 0]
            move_info[i]['angle'] = trans_angle

        # Normalize the coordinates to [-1, 1], with (0,0) being the image center
        temp = RenderedComposite(move_cnts, move_info)
        for i in range(len(move_info)):
            x, y = temp.get_movedhouse_center(i)
            x_ = ((x / 256) - 0.5) * 2
            y_ = ((y / 256) - 0.5) * 2
            move_info[i]['center'] = x_, y_

        select_house = random.choice(move_info)             # 再从移除的房屋 里面随机选择一个房子

        cat = torch.LongTensor([select_house['label']])     # 记录cat

        loc = torch.Tensor([select_house['center']])        # 记录loc

        cos = select_house['angle'][0]
        sin = select_house['angle'][1]
        orient = torch.Tensor([cos, sin])                    # 记录orient

        return input_img, existing_categories, cat, loc, orient


# 测试tesor画图
def test_drawpic(t):
    print(t.shape)
    image = plt.imshow(t.squeeze(), cmap='gray')
    plt.show()


if __name__ == "__main__":
    data_dir = './txt_data_divide'
    block_category = 3  # 选择当前的block类别为第3种
    num_house_categories = 10  # 房屋类别有10种
    dataset = OrientDataset(data_dir, block_category, num_house_categories)
    data_loader = DataLoader(dataset)
    data_iter = iter(data_loader)
    td_img, td_counts, td_cat, td_loc, td_orient = next(data_iter)

    # 测试
    test_drawpic(td_img[0][1])
    print(td_counts)
    print(td_cat)
    print(td_loc)
    print(td_orient)



# import math
# # # 测试
# if __name__ == "__main__":
#     data_dir = './txt_data_divide/info'
#     with open(data_dir + '/' + '3_blocks/44_info_3.txt', 'r') as info_f:
#         info = eval(info_f.read())
#
#     for i in range(len(info)):
#         temp_angle = info[i]['angle']
#         if(temp_angle >=0 and temp_angle <= 180):
#             trans_angle = [math.cos(temp_angle), 1]
#         if(temp_angle >180 and temp_angle <= 360):
#             trans_angle = [math.cos(temp_angle), 0]
#         info[i]['angle'] = trans_angle
#
#     a = info
#     print('a')




