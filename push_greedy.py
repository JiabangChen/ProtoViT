import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time
import torch.nn.functional as F

from helpers import makedir, find_high_activation_crop

def save_prototype_original_img_with_bbox(dir_for_saving_prototypes, img_dir,prototype_img_filename_prefix,j,
                                          sub_patches,
                                          indices,
                                          bound_box_j, color=(0, 255, 255)):
    """
    a modified bbox function that takes the bound_box_j that contains k patches 
    and return the deformed boudning boxes 
    bound_box_j:(5,4)
    sub_patches:4
    indices: 这四个与小P最相近的token在feature map上的index（xx行xx列）
    color for first selected (from top to bottom):
    Yellow, red, green, blue 
    """
    save_dir = os.path.join(dir_for_saving_prototypes,
                 prototype_img_filename_prefix + 'bbox-original' + str(j) +'.png')
    p_img_bgr = cv2.imread(img_dir) # 重新把原图读进来
    img_bbox = p_img_bgr.copy()
    # cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
    #               color, thickness=2)
    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0,0, 255)] # 不同的小P有不同的颜色，分别为Yellow, red, green, blue
    mask_val = np.ones((14,14))*0.4 # set everything else to be 0.2
    for k in range(sub_patches):
        if bound_box_j[1,k] != -1:
            # 只有指示函数不为0的小P才会被project，才会被可视化
            x,y = indices[0][k], indices[1][k]#提取出与某一个小P最相近的token在feature map上的位置，并在mask_val上置为1
            mask_val[x,y] = 1
            bbox_height_start_k = bound_box_j[1,k]
            bbox_height_end_k = bound_box_j[2,k]
            bbox_width_start_k = bound_box_j[3,k]
            bbox_width_end_k = bound_box_j[4,k]
            color = colors[k]
            #获取这个token在原图上的覆盖位置，以及这个bounding box得用什么颜色
            cv2.rectangle(p_img_bgr, (bbox_width_start_k, bbox_height_start_k), (bbox_width_end_k-1, bbox_height_end_k-1),
                    color, thickness=2) # 画方框
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(save_dir, p_img_rgb,vmin=0.0,vmax=1.0) # 画好方框保存图片
    size = p_img_rgb.shape[1] # 224
    
    img_bbox_rgb = np.clip(img_bbox + 150, 0, 255)# increase the brightness
    img_bbox_rgb = img_bbox[...,::-1]
    img_bbox_rgb = np.float32(img_bbox_rgb) / 255
    width = size//14
    
    #bb_og = p_img_rgb.copy()
    for i in range(0, 196):
        x = i %14
        y = i//14
        img_bbox_rgb[y*width:(y+1)*width, x*width:(x+1)*width]*=mask_val[y,x]
        # 让小 P 对应的 patch 保持原亮度，把其他 patch 变暗。然后保存

    save_dir2 = os.path.join(dir_for_saving_prototypes,
                 prototype_img_filename_prefix + '_vis_' + str(j) +'.png')
    plt.imsave(save_dir2, img_bbox_rgb,vmin=0.0,vmax=1.0)
    
    #save_dir3 = os.path.join(dir_for_saving_prototypes,
                 #prototype_img_filename_prefix + '_vis_bb_' + str(j) +'.png')
    
    #plt.imsave(save_dir3, img_bbox_rgb,vmin=0.0,vmax=1.0)
    #for k in range()
    


def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               pnet,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):
    pnet.eval()
    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input) # push的数据集是没有做normalisation的，因此这里加上normalisation
    with torch.no_grad():
        search_batch = search_batch.cuda()
    # pruned values 
    protoL_input_torch, proto_dist_torch, proto_indices_torch = pnet.push_forward(search_batch) 
    slots_torch_raw = torch.sigmoid(pnet.patch_select*pnet.temp) # 用-200，200的新v重新计算每一个小P的指示函数输出值（1，2000，4）
    slots_torch = torch.round(slots_torch_raw, decimals=1)
    proto_slots = np.copy(slots_torch.detach().cpu().numpy()) # 即做完指示函数之后把 slots_torch_raw 里的每个数四舍五入到 1 位小数
    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy()) # （B，384，14，14）输出的是backbone输出的patch token和CLS token相减后产生的张量
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())
    # 用最大cosine similarity score（4）减去实际的某一个P对某一个图的相似值来表示某一个P到某一张图的距离 （B，2000）
    proto_indice_ = np.copy(proto_indices_torch.detach().cpu().numpy()) # （B，2000，4）输出的是在考虑adjacent mask的情况下每一个小P与其最相似的那个token的idx
    del protoL_input_torch, proto_dist_torch, proto_indices_torch,slots_torch,slots_torch_raw
    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)
    # 产生一个字典，键是200个类别，值是这些类别的图在这个batch中的idx，如果这个batch中没有这个类别的，那么就是空列表
    prototype_shape = pnet.prototype_shape
    n_prototypes = prototype_shape[0] # 2000
    proto_h = prototype_shape[2] # 4
    #proto_w = prototype_shape[3]
    # number of prototypical patches for each prototype 
    n_p = proto_h # 4
    for j in range(n_prototypes):
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(pnet.prototype_class_identity[j]).item()
            # update_prototypes_on_batch这个函数是对每一个batch做遍历，这里就是对每一个P做遍历，找出这个P在这个batch中最短的距离/最相似
            # 的余弦值（只与P所属类的图做比对），是否比当前最短的距离要短，若是，则更新P的project，最短距离，visualisation，heatmap，bound_box等等
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue # 若这个batch中没有P所属类的图，就看下一个P
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j] # 输出形状是（A），A是这个batch中某个P所属类
            # 的图像的数量，输出值代表这几张图像和这个P的余弦距离
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:,j]
        # find the min of the min_distances 
        batch_min_proto_dist_j = np.amin(proto_dist_j) # 这个P在这个batch中最短的距离（只与P所属类的图做比对）
        #batch_min_dist_j_indices = np.argmin(proto_dist_j,keepdims=True)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape)) # 输出一个一维列表，值的意思是proto_dist_j中哪一个图与这个P的距离最小
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                batch_argmin_proto_dist_j, the index of closest img to p_j
                min_j_indice, the indices of the sub-part of p_j on the closet img
                '''
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]
                # 这里是直接找出proto_dist_j中与P距离最小的图在整个batch中的idx
            # retrieve the corresponding feature map
            batch_argmin_j_patch_indices = proto_indice_[batch_argmin_proto_dist_j, j][0] # 根据与P距离最小的图在整个batch中的idx，
            # 找出这个图中与四个小P距离最近的四个token的idx
            #batch_argmin_j_patch_subvalues = protot_subvalues[batch_argmin_proto_dist_j, j][0]
            proto_slots_j = (proto_slots.squeeze())[j] # 这个P的四个小P的指示函数
            min_j_indice = np.unravel_index(batch_argmin_j_patch_indices.astype(int), (14,14))
            # 这是把与四个小P距离最近的四个token的idx从一维idx变成二维idx，即在14 x 14中的第几行第几列是这个token
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            global_min_proto_dist[j] = batch_min_proto_dist_j # 更新最短的距离
            # get the whole image 
            original_img_j = search_batch_input[batch_argmin_proto_dist_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            # 找出这个batch中与这个P距离最短（且比目前最短还要短）的原图
            grid_width = 16
            for k in range(n_p):
                if proto_slots_j[k]!= 0: # 小P的投射与否与其指示函数的值有关
                    # each patch is 1x1 containing 16 x 16 pixels 
                    fmap_height_start_index_k = min_j_indice[0][k]* prototype_layer_stride
                    fmap_height_end_index_k = fmap_height_start_index_k + 1
                    fmap_width_start_index_k = min_j_indice[1][k] * prototype_layer_stride
                    fmap_width_end_index_k = fmap_width_start_index_k + 1

                    batch_min_fmap_patch_j_k = protoL_input_[img_index_in_batch,
                                                        :,
                                                        fmap_height_start_index_k:fmap_height_end_index_k,
                                                        fmap_width_start_index_k:fmap_width_end_index_k]
                    # 找出与小P最相似的patch token （384，1，1）
                    #print(batch_min_fmap_patch_j_k.shape)
                    #print(global_min_fmap_patches[j,:,k].shape)
                    global_min_fmap_patches[j,:,k] = batch_min_fmap_patch_j_k.squeeze(-1).squeeze(-1) # 更新要投射过去的patch token，一个token对一个小P
                    bound_idx_k = np.array([[fmap_height_start_index_k, fmap_height_end_index_k],
                    [fmap_width_start_index_k, fmap_width_end_index_k]]) # 这个token在feature map上的位置[[高度起点, 高度终点],[宽度起点, 宽度终点]]
                    pix_bound_k= bound_idx_k*grid_width # 直接投射到原图上，这个token在原图上的位置
                    # not saving prototype img for now, prototypes are shown in the bbox
                    proto_img_j_k = original_img_j[bound_idx_k[0][0]:bound_idx_k[0][1],
                                            bound_idx_k[1][0]:bound_idx_k[1][1], :]
                    proto_bound_boxes[j, 0, k] = batch_argmin_proto_dist_j[0] + start_index_of_search_batch # 与P最相似的图在整个push dataset上的idx
                    proto_bound_boxes[j, 1, k] = pix_bound_k[0][0]
                    proto_bound_boxes[j, 2, k] = pix_bound_k[0][1]
                    proto_bound_boxes[j, 3, k] = pix_bound_k[1][0]
                    proto_bound_boxes[j, 4, k] = pix_bound_k[1][1] # 这个token在原图上的区域的位置
                    if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                        proto_bound_boxes[j, 5, k] = search_y[batch_argmin_proto_dist_j[0]].item()
            # start saving images 
            if dir_for_saving_prototypes is not None:
                if prototype_img_filename_prefix is not None:
                    original_img_path = os.path.join(dir_for_saving_prototypes,
                            prototype_img_filename_prefix + '-original' + str(j) + '.png')
                    plt.imsave(original_img_path,
                    original_img_j,
                    vmin=0.0,
                    vmax=1.0) # 这里存与P最相似的原图
            # rt = os.path.join(dir_for_saving_prototypes,
            #                 prototype_img_filename_prefix + 'bbox-original' + str(j) +'.png')
            save_prototype_original_img_with_bbox(dir_for_saving_prototypes, original_img_path,prototype_img_filename_prefix,j = j,
                                                  sub_patches = n_p,
                                                  indices = min_j_indice,
                                                  bound_box_j = proto_bound_boxes[j], color=(0, 255, 255))
            # rt_newvis = os.path.join(dir_for_saving_prototypes,
            #                 prototype_img_filename_prefix + '_newvis_' + str(j) +'.png')
            # proto_new_vis(rt_newvis, original_img_path,sub_patches= n_p,
            #                             slots = proto_slots_j,
            #                             indices = min_j_indice,
            #                             bound_box_j = proto_bound_boxes[j], color=(0, 255, 255))
            
    return None 



# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    pnet, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):
    pnet.eval()
    log('\tpush')
    start = time.time()
    prototype_shape = pnet.prototype_shape # 2000 384 4
    n_prototypes = pnet.num_prototypes
    global_min_proto_dist = np.full(n_prototypes, np.inf) # 一个一维numpy数组，长度是2000，值是无穷大
    # saves the patch representation that gives the current smallest distance
    n_p = prototype_shape[2] # 4
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         n_p]) # 2000 384 4的全0数组
    # update the discrete slots approximated by sigmoid 
    slots = torch.sigmoid(pnet.patch_select*pnet.temp).clone() # 指示函数的输出（1，2000，4）
    slots_rounded = slots.round() # 小于0.5的变成0，大于0.5的变成1
    result_tensor = torch.where(slots_rounded == 0, torch.tensor(-1), slots_rounded)*200
    # 查一遍slots_rounded中的元素，如果等于0，就将值改成-1，等于1的不变。然后对整一个张量 x 200
    pnet.patch_select.data.copy_(torch.tensor(result_tensor.detach().cpu().numpy(), dtype=torch.float32).cuda())
    # pnet.patch_select.data就是把指示函数的那个v的张量给找出来（v自身是一个可学习的nn.Parameter参数对象），然后用.copy_原地赋值
    # 赋的值也是一个张量，用result_tensor来赋值，但他是一个不在cuda上的张量，因此得把它也放到cuda上。可见其实上面的slots_rounded只是
    # 用来求出result_tensor，然后通过乘以200的方式让新v变得很大，因此可以让push中的指示函数输出值基本接近与0或者1.
    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    ex_dim:sub_patch component index
    '''
    if save_prototype_class_identity:
        proto_bound_boxes = np.full(shape=[n_prototypes, 5, n_p],
                                            fill_value=-1) # 初始化proto_bound_boxes，形状是（2000，5，4），先把所有值写成-1
    else:
        proto_bound_boxes = np.full(shape=[n_prototypes, 4, n_p],
                                            fill_value=-1)
    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size
    num_classes = pnet.num_classes
    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size # 即给一个batch在整个data loader中确定一个起点
        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   pnet,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)
    
    # print out the stats for slots after push 
    slots_pushed= torch.sigmoid(pnet.patch_select*pnet.temp).squeeze(1).sum(-1) # （1，2000），即每一个P的各个小P的指示函数之和
    unique_elements, counts = torch.unique(slots_pushed, return_counts=True) # 找出slots_pushed中有几个独一无二的值。理解上
    # slots_pushed的值只能有0，1，2，3，4（应该是小数，但与这五个数非常接近），然后counts的意思是这几个值在slots_pushed中出现过几次
    counter= dict(zip(unique_elements.tolist(), counts.tolist())) # 然后把统计结果展示成字典，有哪几个数，以及每个数出现过几次
    log(str(counter))

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    # push prototype to latent feature 
    pnet.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    # 注意！jiabang's alert，这个代码的逻辑就是只跑一次warm-up+joint+indicator训练+push+FC训练。因为在push的时候slots计算中用到的v
    # 会被永久性变更为-200和200.同时，slots=0的小P也会被代替成全0向量。这就明摆着不欢迎后续的训练了
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))
    return None 