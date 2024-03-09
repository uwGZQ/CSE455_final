import torch
import torch.nn as nn
from collections import OrderedDict
from utils import split_first_dim_linear
import math
import numpy as np
from itertools import combinations 

from torch.autograd import Variable

import torchvision.models as models

NUM_SAMPLES=1

# np.random.seed(3483)
# torch.manual_seed(3483)
# torch.cuda.manual_seed(3483)
# torch.cuda.manual_seed_all(3483)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        # pe is of shape max_len(5000) x 2048(last layer of FC)
        pe = torch.zeros(max_len, d_model)
        # position is of shape 5000 x 1
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        # pe contains a vector of shape 1 x 5000 x 2048
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)

class DistanceLoss(nn.Module):
    "Compute the Query-class similarity on the patch-enriched features."
    def __init__(self, args, temporal_set_size=3):
        super(DistanceLoss, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.dropout = nn.Dropout(p = 0.1)

        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        # self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples = [torch.tensor(comb) for comb in frame_combinations]

        self.tuples_len = len(self.tuples) # 28 for tempset_2

        # nn.Linear(4096, 1024)
        self.clsW = nn.Linear(self.args.trans_linear_in_dim * self.temporal_set_size, self.args.trans_linear_in_dim//2)
        self.relu = torch.nn.ReLU() 


    def forward(self, support_set, support_labels, queries):
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        n_queries = queries.shape[0] #20
        n_support = support_set.shape[0] #25
        
        # Add a dropout before creating tuples
        support_set = self.dropout(support_set) # 25 x 8 x 2048
        queries = self.dropout(queries) # 20 x 8 x 2048

        # construct new queries and support set made of tuples of images after pe
        # Support set s = number of tuples(28 for 2/56 for 3) stacked in a list form containing elements of form 25 x 4096(2 x 2048 - (2 frames stacked))
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]

        support_set = torch.stack(s, dim=-2).to(device) # 25 x 28 x 4096
        queries = torch.stack(q, dim=-2) # 20 x 28 x 4096
        support_labels = support_labels.to(device)
        unique_labels = torch.unique(support_labels) # 5

        query_embed = self.clsW(queries.view(-1, self.args.trans_linear_in_dim*self.temporal_set_size)) # 560[20x28] x 1024

        # Add relu after clsW
        query_embed = self.relu(query_embed) # 560 x 1024        

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        dist_all = torch.zeros(n_queries, self.args.way) # 20 x 5

        for label_idx, c in enumerate(unique_labels):
            # Select keys corresponding to this class from the support set tuples
            class_k = torch.index_select(support_set, 0, self._extract_class_indices(support_labels, c)) # 5 x 28 x 4096

            # Reshaping the selected keys
            class_k = class_k.view(-1, self.args.trans_linear_in_dim*self.temporal_set_size) # 140 x 4096

            # Get the support set projection from the current class
            support_embed = self.clsW(class_k.to(queries.device))  # 140[5 x 28] x1024

            # Add relu after clsW
            support_embed = self.relu(support_embed) # 140 x 1024

            # Calculate p-norm distance between the query embedding and the support set embedding
            distmat = torch.cdist(query_embed, support_embed) # 560[20 x 28] x 140[28 x 5]

            # Across the 140 tuples compared against, get the minimum distance for each of the 560 queries
            min_dist = distmat.min(dim=1)[0].reshape(n_queries, self.tuples_len) # 20[5-way x 4-queries] x 28

            # Average across the 28 tuples
            query_dist = min_dist.mean(dim=1)  # 20

            # Make it negative as this has to be reduced.
            distance = -1.0 * query_dist
            c_idx = c.long()
            dist_all[:,c_idx] = distance # Insert into the required location.

        return_dict = {'logits': dist_all}
        
        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
       
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        # self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples = [torch.tensor(comb) for comb in frame_combinations]

        self.tuples_len = len(self.tuples) #28
    
    def forward(self, support_set, support_labels, queries):
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        n_queries = queries.shape[0] #20
        n_support = support_set.shape[0] #25
        
        # static pe after adding the position embedding
        support_set = self.pe(support_set) # Support set is of shape 25 x 8 x 2048 -> 25 x 8 x 2048
        queries = self.pe(queries) # Queries is of shape 20 x 8 x 2048 -> 20 x 8 x 2048

        # construct new queries and support set made of tuples of images after pe
        # Support set s = number of tuples(28 for 2/56 for 3) stacked in a list form containing elements of form 25 x 4096(2 x 2048 - (2 frames stacked))
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]

        support_set = torch.stack(s, dim=-2) # 25 x 28 x 4096
        queries = torch.stack(q, dim=-2) # 20 x 28 x 4096

        # apply linear maps for performing self-normalization in the next step and the key map's output
        '''
            support_set_ks is of shape 25 x 28 x 1152, where 1152 is the dimension of the key = query head. converting the 5-way*5-shot x 28(tuples).
            query_set_ks is of shape 20 x 28 x 1152 covering 4 query/sample*5-way x 28(number of tuples)
        '''
        support_set_ks = self.k_linear(support_set) # 25 x 28 x 1152
        queries_ks = self.k_linear(queries) # 20 x 28 x 1152
        support_set_vs = self.v_linear(support_set) # 25 x 28 x 1152
        queries_vs = self.v_linear(queries) # 20 x 28 x 1152
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks).to(device) # 25 x 28 x 1152
        mh_queries_ks = self.norm_k(queries_ks).to(device) # 20 x 28 x 1152
        support_labels = support_labels.to(device)
        mh_support_set_vs = support_set_vs.to(device) # 25 x 28 x 1152
        mh_queries_vs = queries_vs.to(device) # 20 x 28 x 1152
        
        unique_labels = torch.unique(support_labels) # 5

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        all_distances_tensor = torch.zeros(n_queries, self.args.way) # 20 x 5

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class 
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c)) # 5 x 28 x 1152
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c)) # 5 x 28 x 1152
            k_bs = class_k.shape[0] # 5

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim) # 20 x 5 x 28 x 28

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3) # 20 x 28 x 5 x 28 
            
            # [For the 20 queries' 28 tuple pairs, find the best match against the 5 selected support samples from the same class
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1) # 20 x 28 x 140
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)] # list(20) x 28 x 140
            class_scores = torch.cat(class_scores) # 560 x 140 - concatenate all the scores for the tuples
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len) # 20 x 28 x 5 x 28
            class_scores = class_scores.permute(0,2,1,3) # 20 x 5 x 28 x 28
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v) # 20 x 5 x 28 x 1152 
            query_prototype = torch.sum(query_prototype, dim=1).to(device) # 20 x 28 x 1152 -> Sum across all the support set values of the corres. class
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype # 20 x 28 x 1152
            norm_sq = torch.norm(diff, dim=[-2,-1])**2 # 20 
            distance = torch.div(norm_sq, self.tuples_len) # 20
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance # 20
        
        return_dict = {'logits': all_distances_tensor}
        
        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

class Token_Perceptron(torch.nn.Module):
    '''
        2-layer Token MLP
    '''
    def __init__(self, in_dim):
        super(Token_Perceptron, self).__init__()
        # in_dim 8
        self.inp_fc = nn.Linear(in_dim, in_dim)
        self.out_fc = nn.Linear(in_dim, in_dim)
        self.relu = torch.nn.ReLU() 

    def forward(self, x):

        # Applying the linear layer on the input
        output = self.inp_fc(x) # B x 2048 x 8

        # Apply the relu non-linearity
        output = self.relu(output) # B x 2048 x 8

        # Apply the 2nd linear layer
        output = self.out_fc(output)
        
        return output

class Bottleneck_Perceptron_2_layer(torch.nn.Module):
    '''
        2-layer Bottleneck MLP
    '''
    def __init__(self, in_dim):
        # in_dim 2048
        super(Bottleneck_Perceptron_2_layer, self).__init__()
        self.inp_fc = nn.Linear(in_dim, in_dim)
        self.out_fc = nn.Linear(in_dim, in_dim)
        self.relu = torch.nn.ReLU() 

    def forward(self, x):
        output = self.relu(self.inp_fc(x))
        output = self.out_fc(output)
        
        return output 

class Bottleneck_Perceptron_3_layer_res(torch.nn.Module):
    '''
        3-layer Bottleneck MLP followed by a residual layer
    '''
    def __init__(self, in_dim):
        # in_dim 2048
        super(Bottleneck_Perceptron_3_layer_res, self).__init__()
        self.inp_fc = nn.Linear(in_dim, in_dim//2)
        self.hid_fc = nn.Linear(in_dim//2, in_dim//2)
        self.out_fc = nn.Linear(in_dim//2, in_dim)
        self.relu = torch.nn.ReLU() 

    def forward(self, x):
        output = self.relu(self.inp_fc(x))
        output = self.relu(self.hid_fc(output)) 
        output = self.out_fc(output)
        
        return output + x # Residual output

class Self_Attn_Bot(nn.Module):
    """ Self attention Layer
        Attention-based frame enrichment
    """
    def __init__(self,in_dim, seq_len):
        super(Self_Attn_Bot,self).__init__()
        self.chanel_in = in_dim # 2048
        
        # Using Linear projections for Key, Query and Value vectors
        self.key_proj = nn.Linear(in_dim, in_dim)
        self.query_proj = nn.Linear(in_dim, in_dim)
        self.value_conv = nn.Linear(in_dim, in_dim)

        self.softmax  = nn.Softmax(dim=-1) #
        self.gamma = nn.Parameter(torch.zeros(1))
        self.Bot_MLP = Bottleneck_Perceptron_3_layer_res(in_dim)
        max_len = int(seq_len * 1.5)
        self.pe = PositionalEncoding(in_dim, 0.1, max_len)

    def forward(self, x):

        """
            inputs :
                x : input feature maps( B X C X W )[B x 16 x 2048]
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width)
        """

        # Add a position embedding to the 16 patches
        x = self.pe(x) # B x 16 x 2048

        m_batchsize,C,width = x.size() # m = 200/160, C = 2048, width = 16

        # Save residual for later use
        residual = x # B x 16 x 2048

        # Perform query projection
        proj_query  = self.query_proj(x) # B x 16 x 2048

        # Perform Key projection
        proj_key = self.key_proj(x).permute(0, 2, 1) # B x 2048  x 16

        energy = torch.bmm(proj_query,proj_key) # transpose check B x 16 x 16
        attention = self.softmax(energy) #  B x 16 x 16

        # Get the entire value in 2048 dimension 
        proj_value = self.value_conv(x).permute(0, 2, 1) # B x 2048 x 16

        # Element-wise multiplication of projected value and attention: shape is x B x C x N: 1 x 2048 x 8
        out = torch.bmm(proj_value,attention.permute(0,2,1)) # B x 2048 x 16

        # Reshaping before passing through MLP
        out = out.permute(0, 2, 1) # B x 16 x 2048

        # Passing via gamma attention
        out = self.gamma*out + residual # B x 16 x 2048

        # Pass it via a 3-layer Bottleneck MLP with Residual Layer defined within MLP
        out = self.Bot_MLP(out)  # B x 16 x 2048

        return out

class MLP_Mix_Enrich(nn.Module):
    """ 
        Pure Token-Bottleneck MLP-based enriching features mechanism
    """
    def __init__(self,in_dim, seq_len):
        super(MLP_Mix_Enrich,self).__init__()
        # in_dim = 2048
        self.Tok_MLP = Token_Perceptron(seq_len) # seq_len = 8 frames
        self.Bot_MLP = Bottleneck_Perceptron_2_layer(in_dim)

        max_len = int(seq_len * 1.5) # seq_len = 8
        self.pe = PositionalEncoding(in_dim, 0.1, max_len)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W ) # B(25/20) x 8 x 2048
            returns :
                out : self MLP-enriched value + input feature 
        """

        # Add a position embedding to the 8 frames
        x = self.pe(x) # B x 8 x 2048

        # Store the residual for use later
        residual1 = x # B x 8 x 2048

        # Pass it via a 2-layer Token MLP followed by Residual Layer
        # Permuted before passing into the MLP: B x 2048 x 8 
        out = self.Tok_MLP(x.permute(0, 2, 1)).permute(0, 2, 1) + residual1 # B x 8 x 2048

        # Storing a residual 
        residual2 = out # B x 8 x 2048
        
        # Pass it via 2-layer Bottleneck MLP defined on Channel(2048) features
        out = self.Bot_MLP(out) + residual2 # B x 8 x 2048

        return out

class CNN_STRM(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """

    def __init__(self, args):
        super(CNN_STRM, self).__init__()

        self.train()
        self.args = args

        # Using ResNet Backbone
        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        self.num_patches = 16

        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))

        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set]) 

        # New-distance metric for post patch-level enriched features
        self.new_dist_loss_post_pat = [DistanceLoss(args, s) for s in args.temp_set]

        # Linear-based patch-level attention over the 16 patches
        self.attn_pat = Self_Attn_Bot(self.args.trans_linear_in_dim, self.num_patches)

        # MLP-mixing frame-level enrichment over the 8 frames.
        self.fr_enrich = MLP_Mix_Enrich(self.args.trans_linear_in_dim, self.args.seq_len)

    def forward(self, context_images, context_labels, target_images):

        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        '''
        context_features = self.resnet(context_images) # 200 x 2048 x 7 x 7
        target_features = self.resnet(target_images) # 160 x 2048 x 7 x 7

        # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(context_features) # 200 x 2048 x 4 x 4
        target_features = self.adap_max(target_features) # 160 x 2048 x 4 x 4

        # Reshape before averaging across all the patches
        context_features = context_features.reshape(-1, self.args.trans_linear_in_dim, self.num_patches) # 200 x 2048 x 16
        target_features = target_features.reshape(-1, self.args.trans_linear_in_dim, self.num_patches) # 160 x 2048 x 16       

        # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1) # 200 x 16 x 2048
        target_features = target_features.permute(0, 2, 1) # 160 x 16 x 2048

        # Performing self-attention across the 16 patches
        context_features = self.attn_pat(context_features) # 200 x 16 x 2048 
        target_features = self.attn_pat(target_features) # 160 x 16 x 2048

        # Average across the patches 
        context_features = torch.mean(context_features, dim = 1) # 200 x 2048
        target_features = torch.mean(target_features, dim = 1) # 160 x 2048

        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        target_features = target_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 20 x 8 x 2048

        print("Context Features: ", context_features.shape)
        print("Target Features: ", target_features.shape)

        # Compute logits using the new loss before applying frame-level attention
        all_logits_post_pat = [n(context_features, context_labels, target_features)['logits'] for n in self.new_dist_loss_post_pat]
        all_logits_post_pat = torch.stack(all_logits_post_pat, dim=-1) # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]

        # Combing the patch and frame-level logits
        sample_logits_post_pat = all_logits_post_pat
        sample_logits_post_pat = torch.mean(sample_logits_post_pat, dim=[-1]) # 20 x 5

        # Perform self-attention across the 8 frames
        context_features_fr = self.fr_enrich(context_features) # 25 x 8 x 2048
        target_features_fr = self.fr_enrich(target_features) # 20 x 8 x 2048

        print("Context Features Frame level: ", context_features_fr.shape)
        print("Target Features Frame level: ", target_features_fr.shape)

        '''
            For different temporal lengths(2, 3, ...) get the final logits and perform mean.
        '''

        # Frame-level logits
        all_logits_fr = [t(context_features_fr, context_labels, target_features_fr)['logits'] for t in self.transformers]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1) # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1]) # 20 x 5

        return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]), 
                    'logits_post_pat': split_first_dim_linear(sample_logits_post_pat, [NUM_SAMPLES, target_features.shape[0]])}

        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)
            self.new_dist_loss_post_pat = [n.cuda(0) for n in self.new_dist_loss_post_pat]

            self.attn_pat.cuda(0)
            self.attn_pat = torch.nn.DataParallel(self.attn_pat, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.fr_enrich.cuda(0)
            self.fr_enrich = torch.nn.DataParallel(self.fr_enrich, device_ids=[i for i in range(0, self.args.num_gpus)])



class RGB_Strm_Backbone(nn.Module):
    
    '''
    Input: 
    support_set, 和query set
    Return: 
    S: 25*8*2048; Q: 20*8*2048
    '''
    def __init__(self, args, ):
        super(RGB_Strm_Backbone, self).__init__()
        # self.train()
        self.args = args
        resnet = models.resnet50(pretrained=True)  
        
        
        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        self.num_patches = 16

        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))

        # Temporal Cross Transformer for modelling temporal relations

        # New-distance metric for post patch-level enriched features

        # Linear-based patch-level attention over the 16 patches
        self.attn_pat = Self_Attn_Bot(self.args.trans_linear_in_dim, self.num_patches)

        # MLP-mixing frame-level enrichment over the 8 frames.
        self.fr_enrich = MLP_Mix_Enrich(self.args.trans_linear_in_dim, self.args.seq_len)

                   
    def forward(self, context_feature,context_labels, target_feature):
            
        context_features = self.resnet(context_feature) # 200 x 2048 x 7 x 7
        target_features = self.resnet(target_feature) # 160 x 2048 x 7 x 7
            # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(context_features) # 200 x 2048 x 4 x4
        target_features = self.adap_max(target_features)    # 160 x 2048 x 4x 4
            # Reshape before averaging across all the patches
        context_features = context_features.reshape(-1, 2048, self.num_patches) # 200 x 2048 x 16  
        target_features = target_features.reshape(-1, 2048, self.num_patches)   
        
            # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1) # 200 x 16 x 2048
        target_features = target_features.permute(0, 2, 1)
            # Average across the patches 
            
        context_features = self.attn_pat(context_features) # 200 x 16 x 2048 
        target_features = self.attn_pat(target_features) # 160 x 16 x 2048
        
        context_features = torch.mean(context_features, dim = 1) # 200 x 2048 
        target_features = torch.mean(target_features, dim = 1) # 200 x 2048
        

        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        target_features = target_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048


        
        
        context_features = self.fr_enrich(context_features) # 25 x 8 x 2048
        target_features = self.fr_enrich(target_features) # 20 x 8 x 2048
        
        target_features = target_features.mean(dim=1) # 20 x 2048
        context_features = context_features.mean(dim=1) # 25 x 2048
        print("Context Features video level: ", context_features.shape)
        print("Target Features video level: ", target_features.shape)
        # 我要把每个class对应的vieo的特征做平均，作为class的特征
        # todo, based on context_labels的值，把context_features分成5类，然后求平均
        context_group = zip(context_features, context_labels)
        context_group = list(context_group)
        context_group = sorted(context_group, key=lambda x: x[1])
        context_features, context_labels = zip(*context_group)
        context_features = torch.stack(context_features)
        context_labels = torch.stack(context_labels)
        print("context_features zip shape", context_features.shape)
        context_features = context_feature.reshape(self.args.way, -1, self.args.trans_linear_in_dim)
        context_features = context_features.mean(dim=1)
        # context_group = list(zip(context_features, context_labels))
        # import random
        # random.shuffle(context_group)
        # context_features, context_labels = zip(*context_group)

# python run.py -c checkpoint_dir_ucf/ --query_per_class 4 --shot 5 --way 5 --trans_linear_out_dim 1152 --test_iters 10000 --dataset ucf --split 3 -lr 0.0001 --img_size 224 --scratch new --num_gpus 0 --method resnet50 --save_freq 10000 --print_freq 1 --training_iterations 20010 --temp_set 2

        return {'context_features': context_features, 
                    'target_features': target_features}
    
    
    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=list(range(self.args.num_gpus)))
            self.attn_pat = torch.nn.DataParallel(self.attn_pat, device_ids=list(range(self.args.num_gpus)))
            self.fr_enrich = torch.nn.DataParallel(self.fr_enrich, device_ids=list(range(self.args.num_gpus)))

class Backbone(nn.Module):
    def __init__(self, args):
        pass
        
    def forward(self, x):
        pass

class AAS(nn.Module):
    def __init__(self, args):
        super(AAS, self).__init__()
        self.args = args
        self.rgb_Posterior = ModalitySpecificPosterior(args)
        self.flow_Posterior = ModalitySpecificPosterior(args)
    def forward(self, x):
        # context rgb features: torch.Size([self.way, 2048]) -> 5, 2048
        # context flow features: torch.Size([self.way, 1024])
        # target rgb features: torch.Size([self.query_per_class * self.way, 2048]) -> 20, 2048
        # target flow features: torch.Size([self.query_per_class * self.way, 1024])
        context_rgb_features = x['context_rgb_features']
        context_flow_features = x['context_flow_features']
        target_rgb_features = x['target_rgb_features']
        target_flow_features = x['target_flow_features']
        p_r, c_r, h_r = self.rgb_Posterior(context_rgb_features, target_rgb_features)
        p_f, c_f, h_f = self.flow_Posterior(context_flow_features, target_flow_features)
        # For each query sample, if a specific modality achieves high absolute certainty and relative certainty, this modality is re- liable enough to express discriminative action characteris- tic in the few-shot task. Conversely, if the certainty is low, the modality is probably unreliable to identify actions in the few-shot task. To facilitate the exploring of cross-modal complementarity, we select query samples with large differ- ences in the reliability of two modalities and organize them into two group
        # To facilitate the exploring of cross-modal complementarity, we select query samples with large differ- ences in the reliability of two modalities and organize them into two groups:
        # Q_r = {(x_i^r, x_i^f) | h_i^r > h_i^f, c_i^r > c_i^f}
        # Q_f = {(x_i^r, x_i^f) | h_i^r < h_i^f, c_i^r < c_i^f}
        Q_r = []
        Q_f = []

        for i in range(len(c_r)):
            if h_r[i] > h_f[i] and c_r[i] > c_f[i]:
                Q_r.append((target_rgb_features[i], target_flow_features[i]))
            else:
                Q_f.append((target_rgb_features[i], target_flow_features[i]))
        return {"Q_r": Q_r, "Q_f": Q_f, "p_r": p_r, "c_r": c_r, "h_r": h_r, "p_f": p_f, "c_f": c_f, "h_f": h_f}

class ModalitySpecificPosterior(nn.Module):
    def __init__(self, args):
        super(ModalitySpecificPosterior, self).__init__()
        self.args = args
        self.psi = nn.PairwiseDistance(p=2)
        
    def forward(self, class_prototypes,query_features ):
        """
        Compute the modality-specific posterior distribution for each query sample.
        Args:
            query_features: Tensor of shape (num_queries, feature_dim) containing the query features.
            class_prototypes: Tensor of shape (num_classes, feature_dim) containing the class prototypes.
        Returns:
            posterior: Tensor of shape (num_queries, num_classes) containing the posterior distribution for each query sample.
        """
        print("query_features", query_features)
        print("class_prototypes", class_prototypes)

        num_queries = query_features.size(0)
        num_classes = class_prototypes.size(0)
        posterior = torch.zeros(num_queries, num_classes)

        
        for i in range(num_queries):
            for k in range(num_classes):
                distance = self.psi(query_features[i], class_prototypes[k])
                posterior[i, k] = torch.exp(-distance)
        posterior = posterior / torch.sum(posterior, dim=1, keepdim=True)
        
        # We define the absolute certainty c_m^i as the maximum element of the modality-specific posterior distribution:
        c = torch.max(posterior, dim=1)[0]

        print("c", c)
        print("posterior", posterior)
        # we define the relative certainty h_m^i as the nega- tive self-entropy of the modality-specific posterior distribu- tion:
        h = -torch.sum(posterior * torch.log(posterior), dim=1)
        return posterior, c, h




class AMD(nn.Module):
    def __init__(self, args):
        super(AMD, self).__init__()
        self.args = args
        self.KLDivLoss = torch.nn.KLDivLoss()

    def forward(self, x):
        Q_r = x['Q_r']
        Q_f = x['Q_f']
        p_r = x['p_r']
        c_r = x['c_r']
        p_f = x['p_f']
        c_f = x['c_f']
        L_f_r = 0
        L_r_f = 0
        for i in range(len(Q_f)):
            L_f_r += c_f[i] * self.KLDivLoss(p_r[i], p_f[i])
        for i in range(len(Q_r)):
            L_r_f += c_r[i] * self.KLDivLoss(p_f[i], p_r[i])
        print("pf shape", p_f.shape)
        print("pr shape", p_r.shape)
        print("cr shape", c_r.shape)
        print("cf shape", c_f.shape)
        print("Q_f shape", len(Q_f))
        print("Q_r shape", len(Q_r))
        print("c_f", c_f)
        print("c_r", c_r)

        L_f_r = L_f_r / torch.sum(c_f)
        L_r_f = L_r_f / torch.sum(c_r)
        print("L_f_r", L_f_r)
        print("L_r_f", L_r_f)
        # import sys
        # sys.exit()

        return L_f_r, L_r_f
        

class AMI(nn.Module):
    def __init__(self, args):
        super(AMI, self).__init__()
        self.args = args
    def forward(self, x, output_AAS):
        context_rgb_features = x['context_rgb_features']
        context_flow_features = x['context_flow_features']
        target_rgb_features = x['target_rgb_features']
        target_flow_features = x['target_flow_features']
        c_r = output_AAS['c_r']
        c_f = output_AAS['c_f']
        w_r = c_r / (c_r + c_f)
        w_f = c_f / (c_r + c_f)
        num_queries = target_rgb_features.size(0)
        num_classes = context_rgb_features.size(0)
        posterior = torch.zeros(num_queries, num_classes)
        print("shapes")
        print("target_rgb_features", target_rgb_features.shape)
        print("context_rgb_features", context_rgb_features.shape)
        print("target_flow_features", target_flow_features.shape)
        print("context_flow_features", context_flow_features.shape)
        print("w_r", w_r.shape)
        print("w_f", w_f.shape)
        for i in range(num_queries):
            for k in range(num_classes):
                distance_rgb = torch.nn.PairwiseDistance(p=2)(target_rgb_features[i], context_rgb_features[k])
                distance_flow = torch.nn.PairwiseDistance(p=2)(target_flow_features[i], context_flow_features[k])
                print("distance_rgb", distance_rgb)
                print("distance_flow", distance_flow)
                print("i",i)
                print("w_r[i]", w_r[i])
                print("w_f[i]", w_f[i])

                print("torch.exp(-distance_rgb)", torch.exp(-distance_rgb))
                print("torch.exp(-distance_flow)", torch.exp(-distance_flow))
                posterior[i, k] = w_r[i] * torch.exp(-distance_rgb) + w_f[i] * torch.exp(-distance_flow)
        posterior = posterior / torch.sum(posterior, dim=1, keepdim=True)
        return posterior


class AMFAR(nn.Module):
    def __init__(self, args):
        super(AMFAR, self).__init__()
        self.args = args
        self.AAS = AAS(args)
        self.AMD = AMD(args)
        self.AMI = AMI(args)
    def forward(self, x):
        output_AAS = self.AAS(x)
        P_r = output_AAS['p_r']
        P_f = output_AAS['p_f']
        L_f_r, L_r_f = self.AMD(output_AAS)
        posterior = self.AMI(x,output_AAS)
        return {"L_f_r": L_f_r, "L_r_f": L_r_f, "P_f": P_f, "P_r": P_r, "posterior": posterior}
        # return L_f_r, L_r_f, P_f, P_r, posterior



if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 2048
            self.trans_linear_out_dim = 128

            self.way = 5
            self.shot = 5
            self.query_per_class = 4
            self.trans_dropout = 0.1
            self.seq_len = 8 
            self.img_size = 224
            self.method = "resnet50"
            self.num_gpus = 1
            self.temp_set = [2,3]
    # args = ArgsObject()
    # # torch.manual_seed(CNN_STRM(args))
    # # model = CNN_STRM(args)
    # model = RGB_Strm_Backbone(args)
    # support_imgs = torch.rand(args.way * args.shot * args.seq_len,3, args.img_size, args.img_size)
    # target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len ,3, args.img_size, args.img_size)
    # support_labels = torch.tensor([0,1,2,3,4])

    # out = model(support_imgs, support_labels, target_imgs)
    # print("returns context feature shape ",out['context_features'].shape)
    # print("returns target feature shape ",out['target_features'].shape)
    # print("STRM returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(out['logits'].shape))
    # print("STRM returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(out['logits_post_pat'].shape))
    # example use for ModalitySpecificPosterior
    args = ArgsObject()
    model = AMFAR(args)
    context_rgb_features = torch.rand(5, 2048)
    context_flow_features = torch.rand(5, 1024)
    target_rgb_features = torch.rand(20, 2048)
    target_flow_features = torch.rand(20, 1024)
    print("context_rgb_features", context_rgb_features)
    print("context_flow_features", context_flow_features)
    print("target_rgb_features", target_rgb_features)
    print("target_flow_features", target_flow_features)
    out = model({"context_rgb_features": context_rgb_features, "context_flow_features": context_flow_features, "target_rgb_features": target_rgb_features, "target_flow_features": target_flow_features})
    print("shape of out", out['L_f_r'].shape, out['L_r_f'].shape, out['P_f'].shape, out['P_r'].shape, out['posterior'].shape)
    print("Loss L_f_r", out['L_f_r'])
    print("Loss L_r_f", out['L_r_f'])
    # Outputs are: the loss L_f-r, the loss L_r-f, the flow posterior P_f, the rgb posterior P_r, and the posterior distribution for each query sample
