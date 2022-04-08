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

from datasets import IssacDataset
from networks import EmbeddingNet, ReadoutNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch
from networks import GNNTrainer, GNNModelOptionalEdge
import torch.optim as optim
from itertools import chain, filterfalse, permutations, product
import numpy as np
from torch_geometric.data import Data, Batch
import torch.nn as nn 


def create_graph(num_nodes, node_inp_size, node_pose, edge_size, edge_feature, action):
    nodes = list(range(num_nodes))
    # Create a completely connected graph
    edges = list(permutations(nodes, 2))
    edge_index = torch.LongTensor(np.array(edges).T)
    x = node_pose#torch.zeros((num_nodes, node_inp_size))#torch.eye(node_inp_size).float()
    edge_attr = edge_feature #torch.rand(len(edges), edge_size)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, action = action)
    # Recreate x as target
    data.y = x
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir')
    parser.add_argument('--split', default='valB')
    parser.add_argument('--max_nobj', type=int, default=10)
    parser.add_argument('--img_h', type=int, default=320)
    parser.add_argument('--img_w', type=int, default=480)
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--d_hidden', type=int, default=512)
    parser.add_argument('--n_relation', type=int, default=4)
    # Evaluation
    parser.add_argument('--checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_worker', type=int, default=1)
    args = parser.parse_args()

    data = IssacDataset(
        args.data_dir
    )
    loader = DataLoader(data, args.batch_size, num_workers=args.n_worker)

    model = EmbeddingNet(
        (args.img_w, args.img_h), args.patch_size, args.max_nobj,
        args.width, args.layers, args.heads
    )
    head = ReadoutNet(args.width, args.d_hidden, 0, args.n_relation)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    head.load_state_dict(checkpoint['head'])
    model = model.cuda().eval()
    head = head.cuda().eval()

    correct = 0
    total = 0

    train = True

    # begin the training for each data loader and get the corresponding sornet embedding for each batch size data. 


    for img, obj_patches, target, mask in tqdm(loader):
        print('enter')
        img = img.cuda()
        obj_patches = obj_patches.cuda()
        with torch.no_grad():
            emb, attn = model(img, obj_patches)
            print(emb.device)
            print(emb.shape)
            logits = head(emb)
            pred = (logits > 0).int().cpu()
        # target = target.int()
        # mask = mask.bool()
        # correct += (pred[mask] == target[mask]).sum().item()
        # total += mask.sum().item()

    
    
    
        node_inp_size = emb.shape[2] # specifically 768
        edge_inp_size = 0
        node_emb_size = emb.shape[2]
        edge_emb_size = emb.shape[2]

        all_classifier = False

        classif_model = GNNModelOptionalEdge(
                    node_inp_size, 
                    edge_inp_size,
                    node_output_size = node_emb_size, 
                    predict_edge_output = True,
                    edge_output_size = edge_emb_size,
                    graph_output_emb_size=16, 
                    node_emb_size=node_emb_size, 
                    edge_emb_size=edge_emb_size,
                    message_output_hidden_layer_size=128,  
                    message_output_size=128, 
                    node_output_hidden_layer_size=64,
                    all_classifier = all_classifier,
                    predict_obj_masks=False,
                    predict_graph_output=False,
                    use_edge_embedding = False,
                )
        classif_model_decoder = GNNModelOptionalEdge(
                    node_emb_size, 
                    edge_emb_size,
                    node_output_size = node_inp_size, 
                    predict_edge_output = True,
                    edge_output_size = edge_inp_size,
                    graph_output_emb_size=16, 
                    node_emb_size=node_emb_size, 
                    edge_emb_size=edge_emb_size,
                    message_output_hidden_layer_size=128,  
                    message_output_size=128, 
                    node_output_hidden_layer_size=64,
                    all_classifier = all_classifier,
                    predict_obj_masks=False,
                    predict_graph_output=False,
                    use_edge_embedding = False,
                    use_edge_input = True
                )
        opt_classif = optim.Adam(classif_model.parameters(), lr=1e-4) # yixuan test

        device = "cuda:0"

        node_pose = emb[0]   # set this to the sornet embedding later. 
        node_pose_goal = emb[0]
        
        num_objects = 10
        
        action_torch = torch.zeros((num_objects,num_objects+3)).to(device)
        # set this temporaly for (5,5)) for the test purpose, I guess I don't use this variable for a long time. Set this for the function parameter purpose 
        
        edge_attributes = torch.zeros((num_objects,num_objects+3)).to(device)
        num_nodes = 10 

        data = create_graph(num_nodes, node_inp_size, node_pose, 0, edge_attributes, action = action_torch)
                
                
        data_next = create_graph(num_nodes, node_inp_size, node_pose_goal, 0, edge_attributes, action = action_torch)

        dynamics_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()
        
        
        if True:
            print(data)
            batch = Batch.from_data_list([data]).to(device)
            print(batch)
            outs = classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                        
                        
            data_1_decoder = create_graph(num_nodes, node_emb_size, outs['pred'], edge_emb_size, outs['pred_edge'], action_torch)
                        
            batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
                        
            outs_decoder = classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)
            
            #outs_decoder = self.classif_model_decoder(outs_embed['pred'], batch.edge_index, outs_embed['pred_edge'], batch.batch, batch.action)

            batch2 = Batch.from_data_list([data_next]).to(device)
            #print(batch)
            outs_2 = classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
            #print(outs['pred'].size())
            
            
            data_2_decoder = create_graph(num_nodes, node_emb_size, outs_2['pred'], edge_emb_size, outs_2['pred_edge'], action_torch)
            batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)
            outs_decoder_2 = classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
            
            
            data_2_decoder_edge = create_graph(num_nodes, node_emb_size, outs['pred_embedding'], edge_emb_size, outs['pred_edge_embed'], action_torch)
            batch_decoder_2_edge = Batch.from_data_list([data_2_decoder_edge]).to(device)
            outs_decoder_2_edge = classif_model_decoder(batch_decoder_2_edge.x, batch_decoder_2_edge.edge_index, batch_decoder_2_edge.edge_attr, batch_decoder_2_edge.batch, batch_decoder_2_edge.action)
            #outs_edge = self.classif_model.forward_decoder(outs['pred_embedding'], batch.edge_index, outs['pred_edge_embed'], batch.batch, batch.action)
            
            total_loss = 0
            total_loss += dynamics_loss(node_pose, outs_decoder['pred']) # node reconstruction loss
            total_loss += dynamics_loss(node_pose_goal, outs_decoder_2['pred'])
            total_loss += dynamics_loss(outs['pred_embedding'], outs_decoder_2['current_embed'])
            
            print(outs_decoder['pred_sigmoid'][:].shape)
            curent_ground_truth_relations = torch.zeros(outs_decoder['pred_sigmoid'][:].shape).to(device)
            total_loss += bce_loss(outs_decoder['pred_sigmoid'][:], curent_ground_truth_relations)
            total_loss += bce_loss(outs_decoder_2['pred_sigmoid'][:], curent_ground_truth_relations)
            total_loss += dynamics_loss(outs['pred_edge_embed'], outs_decoder_2['edge_embed'])

            #print(data_2_decoder_edge['pred_sigmoid'])
            total_loss += bce_loss(outs_decoder_2_edge['pred_sigmoid'][:], curent_ground_truth_relations)
            total_loss += dynamics_loss(node_pose_goal, outs_decoder_2_edge['pred'])
            
            
            print(total_loss)

        if train:
            #self.opt_emb.zero_grad()
            opt_classif.zero_grad()
            total_loss.backward()
            opt_classif.step()
                
    
    print('Total', total)
    print('Accuracy', correct / total * 100)
