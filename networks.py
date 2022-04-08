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

from collections import OrderedDict
import torch
import torch.nn as nn

import numpy as np
from collections import OrderedDict
from itertools import chain, permutations, product
from numbers import Number

import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data, Batch
from torch.utils.data import SubsetRandomSampler

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask[:x.shape[0], :x.shape[0]].to(x.device)
        return self.attn(x, x, x, attn_mask=attn_mask)

    def forward(self, x: tuple):
        x, weights = x
        attn, attn_weights = self.attention(self.ln_1(x))
        if weights is None:
            weights = attn_weights.unsqueeze(1)
        else:
            weights = torch.cat([weights, attn_weights.unsqueeze(1)], dim=1)
        x = x + attn
        x = x + self.mlp(self.ln_2(x))
        return x, weights


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks((x, None))


class EmbeddingNet(nn.Module):
    def __init__(
            self, input_dim: tuple, patch_size: int, n_objects: int,
            width: int, layers: int, heads: int
        ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width, kernel_size=patch_size,
            stride=patch_size, bias=False
        )

        scale = width ** -0.5
        n_patches = (input_dim[0] // patch_size) * (input_dim[1] // patch_size)
        self.positional_embedding = nn.Parameter(scale * torch.randn(n_patches + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        seq_len = n_patches + n_objects
        attn_mask = torch.zeros(seq_len, seq_len)
        attn_mask[:, -n_objects:] = -float("inf")
        attn_mask.fill_diagonal_(0)

        self.transformer = Transformer(width, layers, heads, attn_mask)
        self.ln_post = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, objs: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        batch_size, n_obj = objs.shape[:2]
        objs = objs.reshape(-1, *objs.shape[2:])
        objs = self.conv1(objs)
        objs = objs.reshape(batch_size, n_obj, -1) # shape = [*, n_obj, width]

        x = x + self.positional_embedding[1:]
        objs = objs + self.positional_embedding[:1]
        x = torch.cat([x, objs], dim=1)  # shape = [*, grid ** 2 + n_obj, width]
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, weights = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, -objs.shape[1]:, :])

        return x, weights


class ReadoutNet(nn.Module):
    def __init__(self, d_input, d_hidden, n_unary, n_binary):
        super().__init__()
        self.n_unary = n_unary
        self.n_binary = n_binary
        self.d_hidden = d_hidden
        for i in range(n_unary):
            setattr(self, f'unary{i}', self.get_head(d_input, d_hidden, 1))
        for i in range(n_binary):
            setattr(self, f'binary{i}', self.get_head(d_input, d_hidden, 2))

    def get_head(self, d_input, d_hidden, n_args):
        if d_hidden > 1:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, 1)
            )
        else:
            head = nn.Linear(d_input * n_args, d_hidden)
        return head

    def forward(self, x: torch.Tensor):
        n_obj = x.shape[1]
        y = [getattr(self, f'unary{i}')(x) for i in range(self.n_unary)]
        x1 = x.repeat(1, n_obj - 1, 1)
        x2 = torch.cat([x.roll(-i, dims=1) for i in range(1, n_obj)], dim=1)
        x = torch.cat([x1, x2], dim=-1)
        y += [getattr(self, f'binary{i}')(x) for i in range(self.n_binary)]
        y = torch.cat(y, dim=1).squeeze(-1)
        return y


class GNNModelOptionalEdge(MessagePassing):
    def __init__(self, 
                 in_channels, 
                 edge_inp_size,
                 node_output_size, 
                 graph_output_emb_size=16, 
                 node_emb_size=32, 
                 edge_emb_size=32,
                 message_output_hidden_layer_size=128,  
                 message_output_size=128, 
                 node_output_hidden_layer_size=64,
                 edge_output_size=16,
                 all_classifier = False,
                 predict_obj_masks=False,
                 predict_graph_output=False,
                 use_edge_embedding=False,
                 predict_edge_output=False,
                 use_edge_input=False,
                 node_embedding = False):
        device = 'cuda:0'
        self.relation_output_size = 6 
        # define the relation_output_size by hand for all baselines. 
        # Make sure all the planning stuff keeps the same for all our comparison approaches. 
        super(GNNModelOptionalEdge, self).__init__(aggr='mean')
        # all edge output will be classifier
        self.all_classifier = all_classifier

        # Predict if an object moved or not
        self._predict_obj_masks = predict_obj_masks
        # predict any graph level output
        self._predict_graph_output = predict_graph_output

        action_dim = 13
        self._in_channels = action_dim
        

        self._use_edge_dynamics = True

        self.use_edge_input = use_edge_input
        if use_edge_input == False:
            edge_inp_size = 0
            use_edge_embedding = False
            self._use_edge_dynamics = False
        self._edge_inp_size = edge_inp_size

        self._node_emb_size = node_emb_size
        self.node_embedding = node_embedding
        if self.node_embedding:
            self.node_emb = nn.Sequential(
                nn.Linear(in_channels, self._node_emb_size),
                nn.ReLU(),
                nn.Linear(self._node_emb_size, self._node_emb_size)
            ).to(device)

        self.edge_emb_size = edge_emb_size
        self._use_edge_embedding = use_edge_embedding
        self._test_edge_embedding = False
        if use_edge_embedding:
            self.edge_emb = nn.Sequential(
                nn.Linear(edge_inp_size, edge_emb_size),
                nn.ReLU(inplace=True),
                nn.Linear(edge_emb_size, edge_emb_size)
            ).to(device)

        self._message_layer_size = message_output_hidden_layer_size
        self._message_output_size = message_output_size
        message_inp_size = 2*self._node_emb_size + edge_emb_size if use_edge_embedding else \
            2 * self._node_emb_size + edge_inp_size
        # if use_edge_input == False:
        #     message_inp_size = 2 * self._node_emb_size
        self.message_info_mlp = nn.Sequential(
            nn.Linear(message_inp_size, self._message_layer_size),
            nn.ReLU(),
            # nn.Linear(self._message_layer_size, self._message_layer_size),
            # nn.ReLU(),
            nn.Linear(self._message_layer_size, self._message_output_size)
            ).to(device)

        self._node_output_layer_size = node_output_hidden_layer_size
        self._per_node_output_size = node_output_size
        graph_output_emb_size = 0
        self._per_node_graph_output_size = graph_output_emb_size
        self.node_output_mlp = nn.Sequential(
            nn.Linear(self._node_emb_size + self._message_output_size, self._node_output_layer_size),
            nn.ReLU(),
            nn.Linear(self._node_output_layer_size, node_output_size + graph_output_emb_size)
        ).to(device)

        action_dim = self._in_channels
        self.action_dim = action_dim
        self.dynamics =  nn.Sequential(
            nn.Linear(self._in_channels+action_dim, 128),  # larger value
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self._in_channels)
        ).to(device)

        if self._use_edge_dynamics:
            self.edge_dynamics =  nn.Sequential(
                nn.Linear(self._edge_inp_size+action_dim, 128),  # larger value
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self._edge_inp_size)
            ).to(device)

        
        self.graph_dynamics = nn.Sequential(
            nn.Linear(node_output_size+action_dim, 128),  # larger value
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, node_output_size)
        ).to(device)

        self.graph_edge_dynamics = nn.Sequential(
            nn.Linear(edge_output_size+action_dim, 128),  # larger value
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, edge_output_size)
        ).to(device)

        if self._predict_graph_output:
            self._graph_pred_mlp = nn.Sequential(
                nn.Linear(graph_output_emb_size, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            ).to(device)
        
        self._should_predict_edge_output = predict_edge_output
        if predict_edge_output:
            self._edge_output_size = edge_output_size
            # TODO: Add edge attributes as well, should be easy
            if use_edge_embedding:
                if self.all_classifier:
                    self.all_classifier_list = []
                    for all_classifier_id in range((int)(edge_output_size/2)):
                        self.all_classifier_list.append(nn.Sequential(
                        nn.Linear(edge_emb_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 2),
                        nn.ReLU()
                    ).to(device))
                self._edge_output_mlp = nn.Sequential(
                    nn.Linear(edge_emb_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, edge_output_size)
                ).to(device)

                self._edge_output_sigmoid = nn.Sequential(
                    nn.Linear(edge_emb_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.relation_output_size),
                    nn.Sigmoid()
                ).to(device)
            else:
                self._edge_output_mlp = nn.Sequential(
                    nn.Linear(edge_inp_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, edge_output_size)
                ).to(device)
                self._edge_output_sigmoid = nn.Sequential(
                    nn.Linear(edge_inp_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.relation_output_size),
                    nn.Sigmoid()
                ).to(device)
            self._pred_edge_output = None


    def forward(self, x, edge_index, edge_attr, batch, action):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_x has shape [E, edge_features]

        # Get node embeddings for input features
        # print(x.shape)
        # print(self.node_emb)
        self._test_edge_embedding = False
        if self.node_embedding:
            x = self.node_emb(x)

        print(edge_index.device)
        print(x.device)
        #print(edge_attr.device)
        # Begin the message passing scheme
        total_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        #print(total_out)

        # Get outputs for every ndoe vs overall graph
        node_out_index = torch.arange(self._per_node_output_size).to(x.device)
        graph_out_index = torch.arange(
            self._per_node_output_size, 
            self._per_node_output_size+self._per_node_graph_output_size).to(x.device)

        # Get node level outputs, that is [0..node_out_index-1] values from total_out
        out = torch.index_select(total_out, dim=1, index=node_out_index)
        #import pdb; pdb.set_trace()
        if self._predict_obj_masks:
            mask_index = [out.size(1) - 1]
            state_pred_index = [i for i in range(out.size(1)-1)]

            state_pred_out = torch.index_select(out, 1, torch.LongTensor(state_pred_index).to(x.device))
            mask_out = torch.index_select(out, 1, torch.LongTensor(mask_index).to(x.device))[:, 0]
        else:
            state_pred_out = out
            mask_out = None

        # Get graph level outputs, i.e., [node_out_index, end] values from total_out
        if self._predict_graph_output:
            graph_out = torch.index_select(total_out, dim=1, index=graph_out_index)
            graph_out = global_add_pool(graph_out, batch)
            graph_preds = self._graph_pred_mlp(graph_out)
        else:
            graph_preds = None

        #print(state_pred_out.shape)
        # print(state_pred_out.shape)
        # print(action.shape)
        # state_action = torch.cat((state_pred_out, action), axis = 1)
        # # print(state_action.shape)
        # # print(self.dynamics)
        # pred_state = self.dynamics(state_action)

        # # print(self._pred_edge_output.shape)
        # # print(action.shape)
        # edge_action = torch.zeros((self._pred_edge_output.shape[0], self._pred_edge_output.shape[1] + self.action_dim))
        # edge_action[:,:self._pred_edge_output.shape[1]] = self._pred_edge_output
        # edge_action[:,self._pred_edge_output.shape[1]:] = action[0]
        # edge_action = edge_action.to(x.device)
        # #print(edge_action)

        # #edge_action = torch.cat((self._pred_edge_output, action), axis = 1)
        # #print(state_action.shape)
        
        # if self._use_edge_dynamics:
        #     dynamics_edge = self.edge_dynamics(edge_action)

        print(state_pred_out.shape)
        print(action.shape)
        # print(self.graph_dynamics)
        # print(self.node_output_mlp)
        # print(self._per_node_output_size)
        graph_node_action = torch.cat((state_pred_out, action), axis = 1)
        print(graph_node_action.shape)
        print(self.graph_dynamics)
        pred_node_embedding = self.graph_dynamics(graph_node_action)

        #edge_action = torch.stack([action[0][:], action[0][:], action[0][:], action[0][:], action[0][:], action[0][:]])
        edge_num = self._pred_edge_output.shape[0]
        edge_action_list = []
        for _ in range(edge_num):
            edge_action_list.append(action[0][:])
        edge_action = torch.stack(edge_action_list)
        graph_edge_node_action = torch.cat((self._pred_edge_output, edge_action), axis = 1)
        pred_graph_edge_embedding = self.graph_edge_dynamics(graph_edge_node_action)
        return_dict = {'pred': state_pred_out,
        'current_embed': x, 'pred_embedding':pred_node_embedding, 'edge_embed': self._edge_inp, 'pred_edge_embed': pred_graph_edge_embedding}
        if self._should_predict_edge_output:
            return_dict['pred_edge'] = self._pred_edge_output
            #print(self._pred_edge_output_sigmoid)
            return_dict['pred_sigmoid'] = self._pred_edge_output_sigmoid
        if self.all_classifier:
            return_dict['pred_edge_classifier'] = self._pred_edge_classifier
        # if self._use_edge_dynamics:
        #     return_dict['dynamics_edge'] = dynamics_edge

        return return_dict

    def forward_decoder(self, x, edge_index, edge_attr, batch, action):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_x has shape [E, edge_features]

        # Get node embeddings for input features
        # print(x.shape)
        # print(self.node_emb)
        #x = self.node_emb(x)
        

        # Begin the message passing scheme
        self._test_edge_embedding = True
        total_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        #print(total_out)

        # Get outputs for every ndoe vs overall graph
        node_out_index = torch.arange(self._per_node_output_size).to(x.device)
        graph_out_index = torch.arange(
            self._per_node_output_size, 
            self._per_node_output_size+self._per_node_graph_output_size).to(x.device)

        # Get node level outputs, that is [0..node_out_index-1] values from total_out
        out = torch.index_select(total_out, dim=1, index=node_out_index)
        #import pdb; pdb.set_trace()
        if self._predict_obj_masks:
            mask_index = [out.size(1) - 1]
            state_pred_index = [i for i in range(out.size(1)-1)]

            state_pred_out = torch.index_select(out, 1, torch.LongTensor(state_pred_index).to(x.device))
            mask_out = torch.index_select(out, 1, torch.LongTensor(mask_index).to(x.device))[:, 0]
        else:
            state_pred_out = out
            mask_out = None

        # Get graph level outputs, i.e., [node_out_index, end] values from total_out
        if self._predict_graph_output:
            graph_out = torch.index_select(total_out, dim=1, index=graph_out_index)
            graph_out = global_add_pool(graph_out, batch)
            graph_preds = self._graph_pred_mlp(graph_out)
        else:
            graph_preds = None

        #print(state_pred_out.shape)
        # print(state_pred_out.shape)
        # print(action.shape)
        state_action = torch.cat((state_pred_out, action), axis = 1)
        #print(state_action.shape)
        pred_state = self.dynamics(state_action)

        # print(self._pred_edge_output.shape)
        # print(action.shape)
        edge_action = torch.zeros((self._pred_edge_output.shape[0], self._pred_edge_output.shape[1] + self.action_dim))
        edge_action[:,:self._pred_edge_output.shape[1]] = self._pred_edge_output
        edge_action[:,self._pred_edge_output.shape[1]:] = action[0]
        edge_action = edge_action.to(x.device)
        #print(edge_action)

        #edge_action = torch.cat((self._pred_edge_output, action), axis = 1)
        #print(state_action.shape)
        
        if self._use_edge_dynamics:
            dynamics_edge = self.edge_dynamics(edge_action)

        graph_node_action = torch.cat((x, action), axis = 1)
        pred_node_embedding = self.graph_dynamics(graph_node_action)

        #edge_action = torch.stack([action[0][:], action[0][:], action[0][:], action[0][:], action[0][:], action[0][:]])
        edge_num = self._edge_inp.shape[0]
        edge_action_list = []
        for _ in range(edge_num):
            edge_action_list.append(action[0][:])
        edge_action = torch.stack(edge_action_list)
        graph_edge_node_action = torch.cat((self._edge_inp, edge_action), axis = 1)
        pred_graph_edge_embedding = self.graph_edge_dynamics(graph_edge_node_action)
        return_dict = {'pred': state_pred_out, 'object_mask': mask_out, 'graph_pred': graph_preds, 'pred_state': pred_state, 
        'current_embed': x, 'pred_embedding':pred_node_embedding, 'edge_embed': self._edge_inp, 'pred_edge_embed': pred_graph_edge_embedding}
        if self._should_predict_edge_output:
            return_dict['pred_edge'] = self._pred_edge_output
            return_dict['pred_sigmoid'] = self._pred_edge_output_sigmoid
        if self.all_classifier:
            return_dict['pred_edge_classifier'] = self._pred_edge_classifier
        if self._use_edge_dynamics:
            return_dict['dynamics_edge'] = dynamics_edge

        return return_dict

    
    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # edge_attr is the edge attribute between x_i and x_j

        # x_i is the central node that aggregates information
        # x_j is the neighboring node that passes on information.

        # Concatenate features for sender node (x_j) and receiver x_i and get the message from them
        # Maybe there is a better way to get this message information?

        if self._test_edge_embedding:
            edge_inp = edge_attr
        else:
            if self._use_edge_embedding:
                assert self.edge_emb is not None, "Edge embedding model cannot be none"
                # print(edge_attr.shape)
                # print(self.edge_emb)
                edge_inp = self.edge_emb(edge_attr)
            else:
                edge_inp = edge_attr
        self._edge_inp = edge_inp
        #print('edge in GNN', self._edge_inp)

        #print(edge_inp.shape)
        if self.use_edge_input:
            x_ij = torch.cat([x_i, x_j, edge_inp], dim=1)
            # print(x_ij.shape)
            # print(self.message_info_mlp)
            out = self.message_info_mlp(x_ij)
        else:
            print(x_i.shape)
            print(x_j.shape)
            x_ij = torch.cat([x_i, x_j], dim=1)
            print(x_ij.shape)
            print(self.message_info_mlp)
            out = self.message_info_mlp(x_ij)
        # print('out', out.shape)
        # print(out)
        return out

    def update(self, x_ij_aggr, x, edge_index, edge_attr):
        # We can transform the node embedding, or use the transformed embedding directly as well.
        print([x.shape, x_ij_aggr.shape])
        
        inp = torch.cat([x, x_ij_aggr], dim=1)
        if self._should_predict_edge_output:
            source_node_idxs, target_node_idxs = edge_index[0, :], edge_index[1, :]
            if self.use_edge_input:
                edge_inp = torch.cat([
                    self._edge_inp,
                    x[source_node_idxs], x[target_node_idxs],
                    x_ij_aggr[source_node_idxs], x_ij_aggr[target_node_idxs]], dim=1)
            else:
                edge_inp = torch.cat([
                    x[source_node_idxs], x[target_node_idxs],
                    x_ij_aggr[source_node_idxs], x_ij_aggr[target_node_idxs]], dim=1)
            # print(edge_inp.shape)
            # print(self._edge_output_sigmoid)
            # print(self._edge_output_mlp)
            self._pred_edge_output = self._edge_output_mlp(edge_inp)
            self._pred_edge_output_sigmoid = self._edge_output_sigmoid(edge_inp)
            #print(self._pred_edge_output_sigmoid)
            if self.all_classifier:
                self._pred_edge_classifier = []
                for pred_classifier in self.all_classifier_list:
                    pred_classifier = pred_classifier.to(x.device)
                    self._pred_edge_classifier.append(F.softmax(pred_classifier(edge_inp), dim = 1))
        # print('x, x_ij_aggr', [x.shape, x_ij_aggr.shape])
        # print(x_ij_aggr)
        return self.node_output_mlp(inp)

    def edge_decoder_result(self):
        if self._should_predict_edge_output:
            return self._pred_edge_output
        else:
            return None

class GNNTrainer(object):
    def __init__(self, node_inp_size=3, edge_inp_size=2, graph_output_size=2, predict_graph_output=False, node_output_size=3, predict_edge_output=False, edge_output_size=2, use_edge_input = False):
        self._model = GNNModel(
            node_inp_size, 
            edge_inp_size,
            node_output_size, 
            predict_edge_output = predict_edge_output,
            edge_output_size = edge_output_size,
            graph_output_emb_size=16, 
            node_emb_size = 16, 
            message_output_hidden_layer_size=128,  
            message_output_size=128, 
            node_output_hidden_layer_size=64,
            predict_obj_masks=False,
            predict_graph_output=False,
            use_edge_input = use_edge_input,
        )
        self._opt = None

    def get_parameters(self):
        return self._model.parameters()
    
    def create_optimizer(self):
        self._opt = self.configure_optimizer()

    def configure_optimizer(self):
        return torch.optim.Adam(self.get_parameters(), lr=1e-4)
    
    def forward(self, geom_batch):
        outs = self._model(geom_batch.x, geom_batch.edge_index, geom_batch.edge_attr, geom_batch.batch, geom_batch.action)
        return outs
    def get_state_dict(self):
        return {
            'gnn_model': self._model.state_dict(),
        }
    def save_checkpoint(self):
        cp_filepath = "/home/yixuan/Desktop/mohit_code/saved_model/torch_result/test"
        torch.save(self.get_state_dict(), cp_filepath)
        #print(bcolors.c_red("Save checkpoint: {}".format(cp_filepath)))

    def load_checkpoint(self, checkpoint_path):
        cp_models = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self._model.load_state_dict(cp_models['gnn_model'])
        #self.classif_model.load_state_dict(cp_models['classif_model'])



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
