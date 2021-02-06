import torch
from data.rendered import RenderedComposite
from category import NextCategory
from loc import Location
from orient import Orientation
from dims import Dims
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

args = {'data_folder': './txt_data_divide',
        'save_dir': './trainedModels',
        'epoch': 4,
        'save_every_n_epochs': 1,
        'block_category': 3,
        'counts_house_categories': 10,
        'latent_dim': 200,
        'lr': 0.0005,
        'batch_size': 3,
        'train_percentage': 0.8,
        'validation_percentage': 0.2,
        'latent_size': 10,
        'hidden_size': 40,
        'use_cuda': True
        }


class LayoutSynth:
    def __init__(self, testcnts, categories):
        testcnts.insert(0, [])
        self.testcnts = testcnts
        self.info = []
        self.categories = categories

        # Loads trained models and build up NNs
        self.model_cat = self._load_category_model()
        self.model_location, self.fc_location = self._load_location_model()
        self.model_orient = self._load_orient_model()
        self.model_dims = self._load_dims_model()

        self.softmax = nn.Softmax(dim=1)
        self.softmax.cuda()

        self.current_rendered = RenderedComposite(self.testcnts, self.info)
        self.current_composites = self.generate_current_composites()
        self.counts = []
        self.house_num = 0

    def synth_block(self):
        while True:
            print("New Room")
            next_cat = self.sample_next_cat()
            if next_cat != 10 and self.house_num < 5:
                x, y, orient, dims = self.sample_everything_else(next_cat)
                door_len, wall_len = dims[0], dims[1]
                self.updata_house_info(next_cat, x, y, orient, door_len, wall_len)
                self.house_num += 1
            else:
                break
        return self.current_composites

    def _load_category_model(self):
        model_cat = NextCategory(args['counts_house_categories'] + 4, args['counts_house_categories'],
                                 args['latent_dim'])
        model_cat.load_state_dict(torch.load('./trainedModels/nextcat_3.pt'))
        model_cat.eval()
        model_cat.cuda()

        return model_cat

    def _load_location_model(self):
        model_location = Location(args['counts_house_categories'] + 1, args['counts_house_categories'] + 4)
        model_location.load_state_dict(torch.load('./trainedModels/nextloc_3.pt'))
        model_location.eval()
        model_location.cuda()

        return model_location, None

    def _load_orient_model(self):
        model_orient = Orientation(args['latent_size'], args['hidden_size'], args['counts_house_categories'] + 4)
        model_orient.load('./trainedModels/model_3.pt')
        model_orient.eval()
        model_orient.cuda()

        return model_orient

    def _load_dims_model(self):
        model_dims = Dims(args['latent_size'], args['hidden_size'], args['counts_house_categories'] + 4)
        model_dims.load('./trainedModels/dims_model_3.pt')
        model_dims.eval()
        model_dims.cuda()

        return model_dims

    def generate_current_composites(self):
        # print(self.cnts)
        return self.current_rendered.get_composite()

    def make_counts(self):
        counts = torch.zeros(self.categories)
        for i in self.testcnts[0]:
            counts[i] += 1
        return counts

    def sample_next_cat(self):
        with torch.no_grad():
            input_scene = self.generate_current_composites().unsqueeze(0).cuda()
            counts = self.make_counts().unsqueeze(0).cuda()
            next_cat = self.model_cat(input_scene, counts)
        return next_cat[0][0]

    def sample_everything_else(self, target_cat):
        self.location_map = None
        while True:
            x, y = self._sample_location(target_cat)
            w = 256
            x_ = ((x / w) - 0.5) * 2
            y_ = ((y / w) - 0.5) * 2
            loc = torch.Tensor([x_, y_]).unsqueeze(0).cuda()
            print('loc is:{}, {}'.format(x, y))
            target_cat = 0
            orient = self._sample_orient(loc, target_cat)
            sin, cos = float(orient[0][1]), float(orient[0][0])
            print('orient is:{}, {}'.format(sin, cos))

            dims = self._sample_dims(loc, orient, target_cat)
            print('dims are: ', dims)

        return x, y, orient, dims

    def _sample_location(self, target_cat):
        if self.location_map is None:
            self.location_map = self._create_location_map(target_cat)

        loc = int(torch.distributions.Categorical(probs=self.location_map.view(-1)).sample())
        x, y = loc // 256, loc % 256
        # Clears sampled location so it does not get resampled again
        self.location_map[x][y] = 0
        return x, y

    def _create_location_map(self, target_cat):
        inputs = self.generate_current_composites().unsqueeze(0)
        with torch.no_grad():
            inputs = inputs.cuda()
            outputs = self.model_location(inputs)
            outputs = self.softmax(outputs)
            # pdb.set_trace()
            target_cat = 0
            outputs = F.upsample(outputs, mode='bilinear', scale_factor=4).squeeze()[target_cat + 1]
            outputs[self.current_composites[0] == 0] = 0  # block_mask
            outputs[self.current_composites[1] == 1] = 0  # house_mask
            # location_map = outputs.cpu()
            location_map = outputs
            location_map = location_map / location_map.sum()

            return location_map

    def _sample_orient(self, loc, category):
        orient = torch.Tensor([math.cos(0), math.sin(0)]).unsqueeze(0).cuda()
        input_img = self.generate_current_composites().unsqueeze(0)
        input_img_orient = self.inverse_xform_img(input_img, loc, orient, 64)
        noise = torch.randn(1, 10)
        # pdb.set_trace()
        orient = self.model_orient.generate(noise, input_img_orient, category)

        return orient

    def inverse_xform_img(self, img, loc, orient, output_size):
        batch_size = img.shape[0]
        matrices = torch.zeros(batch_size, 2, 3)
        cos = orient[:, 0]
        sin = orient[:, 1]
        matrices[:, 0, 0] = cos
        matrices[:, 1, 1] = cos
        matrices[:, 0, 1] = -sin
        matrices[:, 1, 0] = sin
        matrices[:, 0, 2] = loc[:, 1]
        matrices[:, 1, 2] = loc[:, 0]
        out_size = torch.Size((batch_size, img.shape[1], output_size, output_size))
        grid = F.affine_grid(matrices, out_size)
        return F.grid_sample(img, grid)

    def _sample_dims(self, loc, orient, category):
        input_img = self.generate_current_composites().unsqueeze(0)
        input_img_dims = self.inverse_xform_img(input_img, loc, orient, 64)
        noise = torch.randn(1, 10)
        dims = self.model_dims.generate(noise, input_img_dims, category)

        return dims

    def updata_house_info(self, label, x, y, angle, door_len, wall_len):
        new_vercoordinate = self.caculate_vercoordinate(x, y, angle, door_len, wall_len)
        print(new_vercoordinate)
        self.current_rendered.add_categories_map(label, new_vercoordinate)
        self.current_composites = self.generate_current_composites()

    def caculate_vercoordinate(self, x, y, angle, door_len, wall_len):
        left = x - wall_len / 2
        top = y - door_len / 2
        right = x + wall_len / 2
        bottom = y + door_len / 2
        origin = [[left, bottom], [left, top], [right, top], [right, bottom]]
        new_vercoordinate = []
        for i in range(len(origin)):
            temp = []
            new_x = (origin[i][0] - x) * math.cos(angle) - (origin[i][0] - y) * math.sin(angle) + x
            new_y = (origin[i][0] - x) * math.sin(angle) + (origin[i][0] - y) * math.cos(angle) + y
            temp.append(new_x)
            temp.append(new_y)
            new_vercoordinate.append(temp)
        return new_vercoordinate


def test_drawpic(t):
    image = plt.imshow(t.squeeze(), cmap='gray')
    plt.show()


if __name__ == '__main__':
    with open('./test_cnts.txt') as f:
        cnts = [eval(f.read())]
    a = LayoutSynth(cnts, 10)
    b = a.synth_block()
    print('房子个数:', a.house_num)
    test_drawpic(b[0])
    test_drawpic(b[1])
    # # print(a.synth_block())
