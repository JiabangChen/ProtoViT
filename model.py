from xml.etree.ElementPath import xpath_tokenizer_re
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from tools.deit_features import deit_tiny_patch_features, deit_small_patch_features
from tools.cait_features import cait_xxs24_224_features

base_architecture_to_features = {'deit_small_patch16_224': deit_small_patch_features,
                                 'deit_tiny_patch16_224': deit_tiny_patch_features,
                                 #'deit_base_patch16_224':deit_base_patch16_224,
                                 'cait_xxs24_224': cait_xxs24_224_features,}

class PPNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 num_classes, init_weights=True,
                 prototype_activation_function='log',
                 sig_temp = 1.0,
                 radius = 3,
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape # p, d, n_p
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes # 10
        self.epsilon = 1e-4
        self.normalizer = nn.Softmax(dim=1)
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function
        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)
        # 和传统P方法一样，也建立一个matrix，表示每一个P所属的类别，其所属的类别置为1，不所属的类别置为0
 
        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        #self.proto_layer_rf_info = proto_layer_rf_info

        self.features = features

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        # 和传统P 方法一样，这里需要把它设置成可学习的模型参数形式，这样在model.parameter时才会显示P向量，P向量也才会被优化器更新，才会
        # 和模型中其他可学习参数一起放到cuda上/保存在权重的参数字典里
        self.radius = radius # 即，在radius范围内的patch被认为是邻居
        # initializations for adaptive subpatch
        #self.patch_select_init = torch.zeros(prototype_shape[0],1, prototype_shape[-1]) # 2000,1, 4 
        self.patch_select = nn.Parameter(torch.ones(1, prototype_shape[0], prototype_shape[-1])*0.1, 
                                         requires_grad=True) # （1 2000 4）的矩阵，每一行代表一个P的4个子组件的选择权重，即指示函数中的那个v。
        # 初始值设置为0.1，且是可学习的参数，这样在训练过程中就会根据损失函数的梯度更新这个v，从而实现对每个P的子组件的选择，即对指示函数的输出的调控.
        # 设置为0.1对应于一开始要把指示函数的值设置为接近于1，见论文p6，为的是让每一个P在第一次joint training时都能很好的去学习
        self.temp = sig_temp # 理解是指示函数中的那个tau
        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias
        features_name = str(features).upper()
        #print(features_name)
        if features_name.startswith('VISION'):
            self.arc = 'deit'
        elif features_name.startswith('CAIT'):
            self.arc = 'cait'
        if init_weights:
            self._initialize_weights() # 这里是只初始化FC的权重，backbone的权重用预训练权重做初始化，P的数值用随机初始化，
            # 指示函数中那个可学习的 vector v用全是0.1的矩阵来初始化。

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        this version is to reuse the cls-token 
        in computation of feature representation 

        patch_emb_new = patch_emb - cls_token_emb (a focal similarity style)
        
        '''
        x = self.features.patch_embed(x) # (B, 196, 384)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)# 将CLS token从（1，1，384）扩展为（B，1，384），与batch维度适配
        if self.arc == 'deit':
            '''
            forward feature from Deit backbone 
            '''
            x = torch.cat((cls_token, x), dim=1) 
            x = self.features.pos_drop(x + self.features.pos_embed)
            x = self.features.blocks(x)
            x = self.features.norm(x) # bsz, 197, 384
            # 这里之所以不用self.features(x)是因为这会直接调用DeiT的forward函数，而这个函数中是需要让CLS token经过head的，但我的head已经删去，
            # 所以这样写会报错。另一种写法可以避免分类头层的计算，即只计算DeiT的backbone。是self.features.forward_features(x),这确实
            # 不报错了，但他往往只返回CLS token，而无法返回最后的patch token，因此也无法去做CLS和patch token的减法，因此这里就是自行写了
            # DeiT的forward函数。为了把patch token和CLS token都拿到。Jiabang's alert

        elif self.arc == 'cait':
            """
            forward feature from cait backbone 
            """
            x = x + self.features.pos_embed
            x = self.features.pos_drop(x)
            for i , blk in enumerate(self.features.blocks):
                x = blk(x)
            for i , blk in enumerate(self.features.blocks_token_only):
                cls_token = blk(x, cls_token)
            x = torch.cat((cls_token, x), dim=1)
            x = self.features.norm(x) # bsz, 197, dim

        # patch_emb that adds global info 
        x_2 = x[:, 1:] - x[:, 0].unsqueeze(1) # bsz, 196, dim，即patch token减去CLS token
        #x = x[:,1:] # bsz, 196, dim
        fea_len =x_2.shape[1] # 一共有几个patch token，即196
        B, fea_width, fea_height = x_2.shape[0],int(fea_len ** (1/2)), int(fea_len ** (1/2))
        feature_emb = x_2.permute(0,2,1).reshape(B, -1, fea_width, fea_height)
        # （B(即bsz),196,384）->(B,384,196)->(B,384,14,14)
        #print(feature_emb.shape)
        return feature_emb
    
    def _cosine_convolution(self, x):

        x = F.normalize(x,p=2,dim=1)
        now_prototype_vectors = F.normalize(self.prototype_vectors,p=2,dim=1)
        distances = F.conv2d(input=x, weight=now_prototype_vectors)#, stride=2)
        distances = -distances

        return distances
    
    def _project2basis(self,x):
        # essentially the same 
        x = F.normalize(x,p=2,dim=1)
        now_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=1)
        distances = F.conv2d(input=x, weight=now_prototype_vectors)#, stride=2)
        #distances*= 10 # enables a larger gradient
        return distances
    
    def prototype_distances(self, x):

        conv_features = self.conv_features(x)
        cosine_distances = self._cosine_convolution(conv_features)
        project_distances = self._project2basis(conv_features)
        return project_distances,cosine_distances
    
    def global_min_pooling(self,distances):

        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        return min_distances

    def global_max_pooling(self,distances):

        max_distances = F.max_pool2d(distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        max_distances = max_distances.view(-1, self.num_prototypes)

        return max_distances
    
    def subpatch_dist(self, x):
        """
        Input: data x 
        output: conv_features, activation map for each subpatch concat into one tensor 
        dist_all: bsz, num_proto, 14*14, 4 
        (14*14): flatten number of activation map (vary by prototype size)
        """
        #slots = torch.sigmoid(self.patch_select*self.temp) # temp set large enough to approximate step functions 
        #factor = ((slots.sum(-1))).unsqueeze(-1)# 1, 2000, 1, 1
        dist_all = torch.FloatTensor().cuda() # 创建一个空的 float tensor，然后放到 GPU 上，它的形状是长度为0的一维向量，
        # 它可以当作是后面级联的起点。即与dist_i(形状是[B, 2000, 196, 1])级联时，torch.cat([empty_1d, dist_i]) 会把这个空 tensor
        # 当作没有内容，直接返回后面的 dist_i。因此第一个级联后dist_all = dist_i，后续级联时dist_all会不断增加维度，最终形成（B, 2000, 196, 4）的形状。
        conv_feature = self.conv_features(x) # 即backbone输出的patch token和CLS token相减后再resize成了（B，384，14，14）
        conv_features_normed = F.normalize(conv_feature,p=2,dim=1)#/factor （B，384，14，14）
        now_prototype_vectors= F.normalize(self.prototype_vectors,p=2,dim=1)#/factor （2000，384，4）
        # 上面两个语句的意思是对每一个patch token和prototype各自做normalisation，使得每一个向量的长度等于1，而且这样做不会改变P本身，即P本身还是没有经过归一化的
        # 只是这里用一个新的变量来做归一化，然后来算cosine。其实也可以理解为就是直接用patch token和P在做cosine，只不过不是向量乘积之后再除以
        # 模长，而是先除以模长，再相乘。可见其实无论是P还是patch token都不需要在求cosine similarity时确保这俩的模长等于1，做不做归一化在数学上没有区别。
        # 但值得注意的是，这里其实可以写成conv_features = F.normalize(conv_feature,p=2,dim=1)，因为对于backbone而言，模型求导的是
        # 其中的weight和bias，这样写并不会打破计算图，导数和权重更新是正常的。但不能写成self.prototype_vectors = nn.Parameter(F.normalize(...)，requires_grad=True)
        # 因为这样会使得模型对P求导时的所用的P是forward中新建立的P，而optimizer更新的却是没有做forward前的P。导致新的P不会被原 optimizer 正常更新。
        # 而且梯度也只会停留在这个新建立的P上，不会沿着nn.Parameter()内部的normalize对之前建立的P求导，因为此时的新P是一个新的叶子参数 Jiabang's alert
        now_prototype_vectors = now_prototype_vectors#*slots/factor
        n_p = self.prototype_shape[-1] # prototypical parts的数量。是4
        for i in range(n_p):
            proto_i = now_prototype_vectors[:,:, i].unsqueeze(-1).unsqueeze(-1) # （2000，384，1，1），即每一个小P
            dist_i = F.conv2d(input=conv_features_normed, weight =proto_i).flatten(2).unsqueeze(-1)
            # （B，2000，14，14）-》（B，2000，196）-》（B，2000，196，1） bsz, n_p, 196,1
            #dist_i_standardized = dist_i
            dist_all = torch.cat([dist_all, dist_i], dim=-1)
            # bsz, 2000, 196, 4，最终形状
        return conv_feature, dist_all
    
    def neigboring_mask(self, center_indices):
        """
        This function add a radius by radius matrix center on the 
        selected top patches to encourage adjacency of the prototypes 
        Input center_indices: max_patch_id shape bsz, 2000, 1
        Some hardcoding is here (lazy)
        return a neighboring mask: bsz 2000 196 
        0: means non-adjacent (not included)
        1: adjacent (included)
        """
        # add padding to the original target size 
        large_padded = (14+self.radius*2)**2 # 16^2
        large_matrix = torch.zeros(center_indices.shape[0], self.num_prototypes, large_padded).cuda() # (B,2000,16^2)全0
        small_total = (2*self.radius + 1)**2 # 3^2
        small_matrix = torch.ones(center_indices.shape[0], self.num_prototypes, small_total).cuda() # （B，2000，3^2）全1
        batch_size, num_points, _ = center_indices.shape
        small_size = int(small_matrix.shape[-1]**0.5) # 3
        large_size = int(large_matrix.shape[-1]**0.5) # 16
        # Reshape center_indices for broadcasting, and convert to 2D indices
        # Unfortunately divmod doesn't work on torch 
        #center_row, center_col = divmod(center_indices.squeeze(-1).cpu().numpy(), 14)
        center_row, center_col = center_indices.squeeze(-1)//14, center_indices.squeeze(-1)%14
        # 即对于找出来的与P最相似的那个patch token的idx，计算出其所在行和列，形状是（B，2000）
        # Calculate the top-left corner for the rxr addition
        # relative location in the padded matrix to the original shape 
        start_row = torch.tensor(center_row+self.radius - small_size // 2)
        start_col = torch.tensor(center_col+self.radius - small_size // 2)
        # 这里其实start_row/col等于center_row/col的值，但start_row/col是针对于那个16 x 16的大matrix而言的，所以在大matrix上，它的位置
        # 就是center_row/col位置的左上角，形状是（B，2000）
        # Handle boundaries (padding might be required if indices go negative)
        start_row = torch.clamp(start_row, 0, large_size - small_size)
        start_col = torch.clamp(start_col, 0, large_size - small_size)
        # 即希望start_row/col的大小需要在0-13之间，但这里其实是冗余的，因为start_row/col在数值上是等于center_row/col的，那么也就是一定在0-13之间
        # Iterate through each possible position in the rxr matrix
        for i in range(small_size):
            for j in range(small_size):
                # adjacent 矩阵是3x3的，因此做这个for循环
                # Determine the corresponding position in the 14x14 matrix
                large_row = start_row + i
                large_col = start_col + j
                # 在大matrix上画出这个adjacent矩阵的每一个元素，形状是（B，2000）
                # Convert 2D indices back to 1D indices for both matrices
                large_idx = large_row * large_size + large_col
                # 以大matrix的形状为标准，把2维idx转为1维的idx，形状是（B，2000）
                small_idx = i * small_size + j
                # 同样，以adjacent matrix的形状（3x3）为标准，把2维idx转为1维idx，形状是（B，2000）
                # Add the small matrix values to the large matrix
                large_matrix.view(batch_size, num_points, -1)[torch.arange(batch_size)[:, None], 
                                                                torch.arange(num_points), large_idx] += small_matrix[..., small_idx]
                # large_matrix.view(batch_size, num_points, -1)的形状还是(B,2000,16^2)，torch.arange(batch_size)生成一个一维的有B长度的张量
                # 后面加上[:,None]的意思是在第一维加一个维度，使得输出结果形状变成（B，1）。torch.arange(num_points)输出形状是（2000），
                # large_idx的形状是（B，2000）.因此首先索引内的三个元素会一同broadcast成（B，2000）。然后这样的索引方式会使得索引结果也是
                # （B，2000），且索引结果的每一个元素是result[b, p] = large_matrix[idx0[b, p],idx1[b, p],idx2[b, p]],这里idx0和idx1
                # 是broadcast后的B和P，因此这里前半句的意思就是生成的result的形状是（B，2000），其中每一个值就是large_idx中的值对应在large_matrix中的值
                # ...的意思是张量中的所有其他维度，因此small_matrix[..., small_idx]=small_matrix[：，：, small_idx]。所以这个语句的意思
                # 就是对于每一个sample，每一个P而言，large_matrix上第large_idx的值加上用small_matrix的small_idx的值，就是赋1，其实就是
                # 在大matrix上确定好start_row/col的位置之后，在这个位置上画出一个3x3的矩阵，然后这个矩阵中每一个元素在大matrix上的位置置为1
        large_matrix_reshape = large_matrix.view(batch_size, num_points,large_size,large_size)
        large_matrix_unpad = large_matrix_reshape[:,:, self.radius: -self.radius,  self.radius:-self.radius] # bsz, 2000, 14,14
        # 这里就是把大matrix变成（B，2000，16，16）的形状之后，从中扣除padding，即只选择中间的（B，2000，14，14），然后再reshape成（B，2000，196）
        large_matrix_unpad = large_matrix_unpad.reshape(batch_size,num_points,-1) # bdz, 2000, 196
        return large_matrix_unpad
    

    def greedy_distance(self, x, get_f = False):
        """
        This function implements greedy matching algorithm 
        takes input image and returns the similarity scores 
        by greedy match, the designed mindistances, 
        and corresponding patch index. 
        mask identity: 1 kept, 0 removed 

        Similarity score is caculated as a sum of scores for all
        sub-component of prototypes 

        X: input from sample batches 
        get_f: bool indicate if we want to return conv_features
        """
        conv_features, dist_all = self.subpatch_dist(x)
        # 这一步输出的是backbone输出的patch token和CLS token相减后再resize成了（B，384，14，14）的conv_features（没做归一化的）以及
        # 每一个P的每一个小P与这个batch的backbone输出的每一个patch token做cosine similarity，输出形状是（bsz, 2000, 196, 4），
        slots = torch.sigmoid(self.patch_select*self.temp) # temp set large enough to approximate step functions
        # 求指示函数的输出（1，2000，4）
        factor = ((slots.sum(-1))).unsqueeze(-1) + 1e-10# 1, 2000, 1, avoid 0 division
        #slots = self.soft_round(slots)
        # distance calculation 
        n_p = self.prototype_shape[-1]#P中小P的数量，即4
        # hard-code for now, always reinitialize for each of the calculation 
        # 196 hard code for now 
        mask_act = torch.ones((x.shape[0], self.num_prototypes, dist_all.shape[2])).cuda() # （B, 2000, 196）
        mask_subpatch = torch.ones((x.shape[0], self.num_prototypes, n_p)).cuda() # （B, 2000, 4）
        mask_all = torch.ones((x.shape[0], self.num_prototypes, dist_all.shape[2], n_p)).cuda() # B, 2000, 196, 4
        # initialize adj mask ==> everything is considered adjacent at begining 
        adjacent_mask = torch.ones((x.shape[0], self.num_prototypes, dist_all.shape[2])).cuda()# B, 2000, 196
        indices =  torch.FloatTensor().cuda() # 一个空的 float tensor，放到GPU上
        values =  torch.FloatTensor().cuda() # 一个空的 float tensor，放到GPU上
        # to record the sequence of subpatches being selected for later reordering 
        subpatch_ids = torch.LongTensor().cuda() # 一个空的 long tensor，放到GPU上
        for _ in range(n_p):
            dist_all_masked = dist_all + (1-mask_all*adjacent_mask.unsqueeze(-1))*(-1e5) # 一开始后面一项等于0，dist_all_masked=dist_all
            # 当我获得最相似的小P-token对之后，会产生一个adjacent mask,根据这个mask来把不在adjacent mask区域内的小P和token的相似值置为负1e5
            # 以及上一个token和所有小P的相似值，和上一个小P和所有token的相似值，都置为－1e5.然后拿着操作过的dist_all_masked，继续去寻找最相似的小P-token对
            max_subs, max_subs_id = dist_all_masked.max(2) # bsz, num_proto, num_subpatches
            # 对于每一个小P，找出输入图中最相似的token，max_subs和max_subs_id形状都是（B，2000，4），分别对应最相似值和对应的token的idx，即196个token中第几个与这个小P最相似
            max_sub_act, max_sub_act_id = max_subs.max(-1) # bsz, num_proto
            # 在找到输入图中与各个小P最相似的token之后，找出这些小P与token对相似值最大的那一个，形状是（B,2000），分别对应最相似值和对应的小P的idx，
            # 即4个小P中第几个与输入图中某个token的相似值相比于其他token和小P对的相似值都要高
            max_patch_id = max_subs_id.gather(-1,max_sub_act_id.unsqueeze(-1))
            # 从 max_subs_id 里取出被选中的小 P 对应的 patch index，即最相似的小P-token对中的这个token的idx（B,2000,1）
            adjacent_mask = self.neigboring_mask(max_patch_id)
            # 通过这个最相似token的idx，找出adjacent mask,形状是（B,2000,196）, 其中adjacent mask覆盖的位置置为了1
            mask_act = mask_act.scatter(index = max_patch_id, dim=2, value =0)
            # 这个scatter语句的意思是在mask_act张量的第二维，按照max_patch_id 指定的位置设成 0，即对于每一个图片，每一个P，在我找出相似值
            # 最高的小P-token对之后，把这个token对应位置在mask_act上置为0.
            mask_subpatch = mask_subpatch.scatter(index=max_sub_act_id.unsqueeze(-1), dim=2, value=0)
            # 同样，对于每一个图片，每一个P，在我找出相似值最高的小P-token对之后，在mask_subpatch上把这个小P对应对应位置置为0
            mask_all = mask_all*mask_act.unsqueeze(-1)
            mask_all = mask_all.permute(0,1,3,2) # 结果是（B,2000,4,196）
            mask_all = mask_all*mask_subpatch.unsqueeze(-1)
            mask_all = mask_all.permute(0,1,3,2)# 结果是 bsz, 2000, 196, 4，其中，第三维在最相似的token idx处是0，第四维在最相似的小P idx处是0
            # 如果抛开B，这是一个长方体的张量，那么对于某一个P而言，是一个长方形的张量，那么[:,token_idx]=0,同时[小P_idx,:]=0
            max_sub_act = max_sub_act.unsqueeze(-1) # （B,2000,1）
            subpatch_ids = torch.cat([subpatch_ids, max_sub_act_id.unsqueeze(-1)], dim = -1)
            # 对于每一个张图，每一个P，记录每一次产生最相似小P-TOKEN对的小P的idx，for循环走完结果是（B,2000,4）
            indices = torch.cat([indices, max_patch_id], dim =-1)
            # 对于每一张图，每一个P，记录每一次产生最相似小P-token对的token的idx，for循环走完结果是（B，2000，4）
            values = torch.cat([values, max_sub_act], dim =-1)
            # 对于每一张图，每一个P，记录每一次产生最相似小P-token对的相似值，for循环走完结果是（B，2000，4）
        subpatch_ids = subpatch_ids.to(torch.int64)
        _,sub_indexes = subpatch_ids.sort(-1) # 对于每一张图，每一个P，按照小P的index重排顺序，从小到大排，sub_indexes是每个小P在subpatch_ids中的原来的位置
        values_reordered = torch.gather(values, -1,sub_indexes)
        indices_reordered = torch.gather(indices, -1,sub_indexes)
        # 上述两个是对values和indices重排，使得与每一个小P一一对应，形状其实都是（B，2000，4）
        # standardized values by slots --> used for prediction 
        values_slot = (values_reordered.clone())*(slots*n_p/factor)
        # 即论文2式，在考虑adjacent mask的情况下，把每一个小P和其最相似的token的相似值乘上指示函数的输出值，并且乘上一个系数防止由于指示函数削减了
        # 某几个token-小P的权重从而导致整个P的相似性分数大量减小。这里的clone不加也OK，最后输出形状是（B，2000，4）
        #assert((mask.sum(2) == (196-n_p)).sum() == (mask.shape[0]*mask.shape[1]))
        #max_activations = values.sum(-1) # bsz, 2000/ n_p
        max_activation_slots = values_slot.sum(-1) # （B，2000），即每一个图中每一个P对于这个图的cosine similarity score
        min_distances = n_p -max_activation_slots # 由于一个P有四个小P，因此其对于一张图的最大cosine similarity score是4
        # 这里用最大cosine similarity score减去实际的score来表示某一个P到某一张图的距离 （B，2000）
        if get_f:
            return conv_features, min_distances, indices_reordered
        # 这里输出的conv_features （B，384，14，14）输出的是backbone输出的patch token和CLS token相减后产生的张量
        # indices_reordered （B，2000，4）输出的是在考虑adjacent mask的情况下每一个小P与其最相似的那个token的idx
        return max_activation_slots, min_distances, values_reordered
        # values_reordered形状是（B，2000，4），表示对于每一张图，每一个P的每一个小P在考虑adjacent mask的情况下和这张图的cosine similarity score

    def push_forward_old(self, x):
        conv_output = self.conv_features(x) #[batchsize,128,14,14]
        distances = self._project2basis(conv_output)
        distances = - distances
        return conv_output, distances
    
    def push_forward(self, x):
        """
        This function does not return distance measure for 
        each patch. Instead, it returns the max overall similarity score 
        by the nature of greedy matching
        """
        #conv_output = self.conv_features(x)
        conv_output, min_distances,indices = self.greedy_distance(x, get_f=True)
        return conv_output, min_distances, indices 
        
    def forward(self, x):
        max_activation, min_distances, values = self.greedy_distance(x)
        logits = self.last_layer(max_activation)
        return logits, min_distances, values
    
    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            #'\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          #self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)
        # 这里.data.copy_的意思是直接访问参数底层 tensor（通过.data实现），绕开 autograd 计算图（即此操作与求导无关），然后用.copy_()
        # 原地赋值，即保留原来的nn.Parameter对象，只改里面的数值

    def _initialize_weights(self):

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)



def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 192, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    sig_temp = 1.0,
                    radius = 1,
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained) # 搭建P方法的backbone，使用的是DeiT
    # 且加载好了官方的预训练权重，没有分类头，也不用像传统P方法那样去计算receptive field的信息

    return PPNet(features=features,
                 img_size=img_size, # 224
                 prototype_shape=prototype_shape, # (2000, 384, 4)
                 num_classes=num_classes, # 200
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function, # log
                 radius = radius, # 1
                 sig_temp = sig_temp, # 100
                 add_on_layers_type=add_on_layers_type) # regular
    # 建立P模型，包括backbone，指示函数和贪心算法相关参数，P vector，和FC layer，以及还有一个P的assign matrix

