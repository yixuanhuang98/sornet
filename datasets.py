'''
MIT License

Copyright (c) 2022 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as fn
import h5py
import json
import numpy as np
import torch
import os
import pickle


class NormalizeInverse(transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


normalize_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

denormalize_rgb = transforms.Compose([
    NormalizeInverse(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
    transforms.ToPILImage(),
])

normalize_gripper = transforms.Normalize(mean=[0.03228], std=[0.01])

denormalize_gripper = NormalizeInverse(mean=[0.03228], std=[0.01])


def get_ground_truth_relations(obj_pose_1, obj_extent_1, obj_pose_2, obj_extent_2): # to start, assume no orientation
    action = []
    anchor_pose = obj_pose_1
    anchor_pose_max = []
    anchor_pose_min = []
    #print(len(obj_extent_1))
    for i in range(len(obj_extent_1)):
        anchor_pose_max.append(obj_pose_1[i] + obj_extent_1[i])
        anchor_pose_min.append(obj_pose_1[i] - obj_extent_1[i])

    other_pose = obj_pose_2
    other_pose_max = []
    other_pose_min = []
    for i in range(len(obj_extent_2)):
        other_pose_max.append(obj_pose_2[i] + obj_extent_2[i])
        other_pose_min.append(obj_pose_2[i] - obj_extent_2[i])

    if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]:
        if(anchor_pose[0] < other_pose[0]):
            action.append(1)
            action.append(0)
        else:
            action.append(0)
            action.append(1)
    else:
        action.append(0)
        action.append(0)
    if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
        if(anchor_pose[1] < other_pose[1]):
            action.append(1)
            action.append(0)
        else:
            action.append(0)
            action.append(1)                  
    else:
        action.append(0)
        action.append(0)
    if((other_pose[2] - anchor_pose[2]) > 0):
        current_extents = np.array(anchor_pose_max) - np.array(anchor_pose_min)
    else:
        current_extents = np.array(other_pose_max) - np.array(other_pose_min)
    if np.abs(other_pose[2] - anchor_pose[2]) > 0.04 and np.abs(other_pose[2] - anchor_pose[2]) < 0.12:
        if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
            if((other_pose[2] - anchor_pose[2]) > 0): # above
                action.append(1)
                action.append(0)
            else:  # below
                action.append(0)
                action.append(1)
        else:
            action.append(0)
            action.append(0)
    else:
        action.append(0)
        action.append(0)
    return action
    

def build_predicates(objects, unary, binary):
    pred_names = [pred % obj for pred in unary for obj in objects]
    obj1 = [obj for _ in range(len(objects) - 1) for obj in objects]
    obj2 = [obj for i in range(1, len(objects)) for obj in np.roll(objects, -i)]
    pred_names += [pred % (o1, o2) for pred in binary for o1, o2 in zip(obj1, obj2)]
    return pred_names


class CLEVRDataset(Dataset):
    def __init__(self, scene_file, obj_file, max_nobj, rand_patch):
        self.obj_file = obj_file
        self.obj_h5 = None
        self.scene_file = scene_file
        self.scene_h5 = None
        with h5py.File(scene_file) as scene_h5:
            self.scenes = list(scene_h5.keys())
        self.max_nobj = max_nobj
        self.rand_patch = rand_patch

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        if self.obj_h5 is None:
            self.obj_h5 = h5py.File(self.obj_file)
        if self.scene_h5 is None:
            self.scene_h5 = h5py.File(self.scene_file)

        scene = self.scene_h5[self.scenes[idx]]
        #print(BytesIO(scene['image'][()]))
        #intput_image = 
        #print(intput_image)
        img = normalize_rgb(Image.open(BytesIO(scene['image'][()])).convert('RGB'))
        #print(img.shape)

        objects = scene['objects'][()].decode().split(',')
        obj_patches = []
        for obj in objects:
            patch_idx = 0
            if self.rand_patch:
                patch_idx = torch.randint(len(self.obj_h5[obj]), ()).item()
            patch = normalize_rgb(Image.open(BytesIO(self.obj_h5[obj][patch_idx])))
            obj_patches.append(patch)
        for _ in range(len(obj_patches), self.max_nobj):
            obj_patches.append(torch.zeros_like(obj_patches[0]))
        #print(torch.zeros_like(obj_patches[0]).shape)
        obj_patches = torch.stack(obj_patches)

        relations, mask = [], []
        ids = np.arange(self.max_nobj)
        for relation in scene['relations']:
            for k in range(1, self.max_nobj):
                for i, j in zip(ids, np.roll(ids, -k)):
                    if i >= len(objects) or j >= len(objects):
                        relations.append(0)
                        mask.append(0)
                    else:
                        relations.append(relation[i][j])
                        mask.append(relation[i][j] != -1)
        relations = torch.tensor(relations).float()
        mask = torch.tensor(mask).float()
        # print('max objects', self.max_nobj)
        # print('relations shape', relations.shape)
        # print('mask shape', mask.shape)

        return img, obj_patches, relations, mask


class IssacDataset(Dataset):
    def __init__(self, train_dir_list, max_nobj = 10, rand_patch = False):
        # self.obj_file = obj_file
        # self.obj_h5 = None
        # self.scene_file = scene_file
        # self.scene_h5 = None
        max_size = 20

        self.train_dir_list = train_dir_list#[:max_size]
        # print(self.train_dir_list)
        # print(self.train_dir_list)
        files = sorted(os.listdir(self.train_dir_list))[:max_size]
        self.train_pcd_path = [
            os.path.join(self.train_dir_list, p) for p in files if 'demo' in p]
        print(len(self.train_pcd_path))
        self.data_list = []
        for train_dir in self.train_pcd_path[:max_size]:
            print(train_dir)            
            with open(train_dir, 'rb') as f:
                data, attrs = pickle.load(f)
            self.data_list.append([data, attrs])
            # print('data', data)
            # print('attrs', attrs)
        
        self.max_nobj = max_nobj
        self.rand_patch = rand_patch

    def __len__(self):
        print('len', len(self.train_pcd_path))
        return len(self.train_pcd_path)

    def __getitem__(self, idx):
        print('enter')
        step = 0
        #scene = self.scene_h5[self.scenes[idx]]
        # current_rgb = self.data_list[idx][0]['rgb'][step]
        # current_rgb = torch.tensor(current_rgb)
        # resize_rgb = fn.resize(current_rgb, size=[224])

        # resize_rgb = np.zeros((3,320,480))
        # print(resize_rgb.shape)
        input_img = Image.fromarray(self.data_list[idx][0]['rgb'][step], mode = 'RGB')
        input_img = fn.resize(input_img, size=[320,480])


        
        
        img = normalize_rgb(input_img)

        
        last_input_img = Image.fromarray(self.data_list[idx][0]['rgb'][-1], mode = 'RGB')
        last_input_img = fn.resize(last_input_img, size=[320,480])
        
        last_img = normalize_rgb(last_input_img)

        total_objects = 10 # set this for all corresponding sornet problem. 
        self.action_1 = []
        for i in range(total_objects):
            self.action_1.append(0)
        self.action_1[0] = 1
        #print(self.action_1)
        for i in range(3):
            # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
            # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
            self.action_1.append(self.data_list[idx][1]['behavior_params']['']['behaviors']['push']['target_pose'][i] - self.data_list[idx][1]['behavior_params']['']['behaviors']['approach']['target_pose'][i])
        self.action_1[-1] = 0 #self.action_1[-1] - 0.02 # 0 with yixuan test
        self.action_1 = [self.action_1]

        ground_truth_relations = get_ground_truth_relations(self.data_list[idx][1]['objects']['block_1']['position'], self.data_list[idx][1]['objects']['block_1']['extents'], self.data_list[idx][1]['objects']['block_2']['position'], self.data_list[idx][1]['objects']['block_2']['extents'])


        #img = torch.zeros(3,320,480)
        #print(img.shape)
        
        obj_patches = []
        for i in range(self.max_nobj):
            obj_patches.append(torch.zeros(3,32,32))
        obj_patches = torch.stack(obj_patches)

        # relations, mask = [], []
        # ids = np.arange(self.max_nobj)
        # for relation in scene['relations']:
        #     for k in range(1, self.max_nobj):
        #         for i, j in zip(ids, np.roll(ids, -k)):
        #             if i >= len(objects) or j >= len(objects):
        #                 relations.append(0)
        #                 mask.append(0)
        #             else:
        #                 relations.append(relation[i][j])
        #                 mask.append(relation[i][j] != -1)
        relations, mask = [], []
        for i in range(360):
            relations.append(1)
            mask.append(1)

        relations = torch.tensor(relations).float()
        mask = torch.tensor(mask).float()

        return img, obj_patches, relations, mask


class LeonardoDataset(Dataset):
    def __init__(
        self, data_dir, split, predicates, obj_file, colors=None,
        randpatch=True, view=1, randview=True, gripper=False,
        img_size=(224,224)
    ):
        with open(f'{data_dir}/{split}_nframes.json') as f:
            n_frames = json.load(f)
            self.sequences = list(n_frames.keys())
            n_frames = list(n_frames.values())
            self.cum_n_frames = np.cumsum(n_frames)
        with h5py.File(f'{data_dir}/{split}.h5') as data:
            all_predicates = data['predicates'][()].decode().split('|')
            pred_ids = {pred: i for i, pred in enumerate(all_predicates)}
            self.pred_ids = [pred_ids[pred] for pred in predicates]

        self.data_dir = data_dir
        self.split = split
        self.h5 = None

        self.colors = colors
        self.obj_file = obj_file
        self.obj_h5 = None
        self.randpatch = randpatch

        self.view = view
        self.randview = randview

        self.gripper = gripper
        self.img_size = img_size

    def __len__(self):
        return self.cum_n_frames[-1]

    def load_h5(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(f'{self.data_dir}/{self.split}.h5', 'r')
        if self.obj_h5 is None:
            self.obj_h5 = h5py.File(f'{self.data_dir}/{self.obj_file}', 'r')
        # Get H5 file index and frame index
        file_idx = np.argmax(idx < self.cum_n_frames)
        data = self.h5[self.sequences[file_idx]]
        frame_idx = idx
        if file_idx > 0:
            frame_idx = idx - self.cum_n_frames[file_idx - 1]
        return data, frame_idx

    def get_rgb(self, data, idx):
        v = torch.randint(self.view, ()).item() if self.randview else self.view
        rgb = Image.open(BytesIO(data[f'rgb{v}'][idx])).convert('RGB')
        return normalize_rgb(rgb.resize(self.img_size))

    def get_patches(self, colors):
        obj_patches = []
        for color in colors:
            patch_idx = 0
            if self.randpatch:
                patch_idx = torch.randint(len(self.obj_h5[color]), ()).item()
            patch = Image.open(BytesIO(self.obj_h5[color][patch_idx]))
            obj_patches.append(normalize_rgb(patch))
        return torch.stack(obj_patches)

    def get_gripper(self, data, idx):
        gripper = data['gripper'][idx].astype('float32')
        gripper = torch.from_numpy(gripper).reshape(1, 1, 1)
        return normalize_gripper(gripper).squeeze()

    def __getitem__(self, idx):
        data, frame_idx = self.load_h5(idx)

        # Load predicates from H5 file
        predicates = data['logical'][frame_idx][self.pred_ids]

        # Load RGB from H5 file
        rgb = self.get_rgb(data, frame_idx)

        # Load object patches
        colors = self.colors
        if colors is None:
            colors = data['colors'][()].decode().split(',')
        obj_patches = self.get_patches(colors)

        # Load gripper state from H5 file
        gripper = self.get_gripper(data, frame_idx) if self.gripper else 0
        
        return rgb, obj_patches, gripper, predicates


class RegressionDataset(LeonardoDataset):
    def __init__(
        self, data_dir, split, obj_file, colors=None, objects=None,
        randpatch=True, view=1, randview=True, ee=False, dist=False,
        img_size=(224,224)
    ):
        with open(f'{data_dir}/{split}_nframes.json') as f:
            n_frames = json.load(f)
            self.sequences = list(n_frames.keys())
            n_frames = list(n_frames.values())
            self.cum_n_frames = np.cumsum(n_frames)

        self.data_dir = data_dir
        self.split = split
        self.h5 = None

        self.colors = colors
        self.objects = objects
        self.obj_file = obj_file
        self.obj_h5 = None
        self.randpatch = randpatch

        self.view = view
        self.randview = randview

        self.ee = ee
        self.dist = dist
        self.img_size = img_size

    def __len__(self):
        return self.cum_n_frames[-1]

    def get_ee_obj_xyz(self, data, idx, objects):
        xyz = torch.stack([torch.from_numpy(
            data[f'{obj}_pose'][idx][:3, 3] - data['ee_pose'][idx][:3, 3]
        ) for obj in objects])
        return xyz

    def get_obj_obj_xyz(self, data, idx, objects):
        obj1 = [obj for _ in range(len(objects) - 1) for obj in objects]
        obj2 = [obj for i in range(1, len(objects)) for obj in np.roll(objects, -i)]
        xyz = torch.stack([torch.from_numpy(
            data[f'{o2}_pose'][idx][:3, 3] - data[f'{o1}_pose'][idx][:3, 3]
        ) for o1, o2 in zip(obj1, obj2)])
        return xyz

    def __getitem__(self, idx):
        data, frame_idx = self.load_h5(idx)

        # Load RGB from H5 file
        rgb = self.get_rgb(data, frame_idx)

        # Load object patches
        colors = self.colors
        if colors is None:
            colors = data['colors'][()].decode().split(',')
        obj_patches = self.get_patches(colors)

        # Load regression targets
        objects = self.objects
        if objects is None:
            objects = [f'object{i:02d}' for i in range(len(colors))]
        if self.ee:
            # End effector to object center
            target = self.get_ee_obj_xyz(data, frame_idx, objects)
        else:
            # Object center to object center
            target = self.get_obj_obj_xyz(data, frame_idx, objects)
        if self.dist:
            target = target.norm(dim=-1, keepdim=True).float()
        else:
            target = torch.nn.functional.normalize(target).float()

        return rgb, obj_patches, target
