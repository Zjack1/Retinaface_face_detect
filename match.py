import torch
print( torch.__version__)#1.4.0

truths=torch.randn(4,4)#[目标，每个边框坐标4个值]
labels=torch.Tensor([[99], [100]])#两个类别，类别数据比较大，容易一眼就看出来
overlaps = torch.tensor([[0., 0., 0.4, 0.5],  [0., 0., 0.5, 0.6, ],  [0.8, 0., 0., 0., ],  [0., 07., 0., 0., ]])
print("overlaps:",overlaps)
# overlaps: \
# tensor([[0.8388, 0.9215, 0.4808, 0.4071, 0.2703],
#         [0.1282, 0.9579, 0.6970, 0.3630, 0.5478]])

#为每个GT框匹配一个IoU最大的先验框，获取 与目标GT框（GTBox） IoU最大的 先验框（PriorBox） 的值和索引
#GTBox 1：PriorBox n
best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
print(overlaps.max(1, keepdim=True))
# torch.return_types.max(
# values=tensor([[0.9715],
#         [0.7579]]),
# indices=tensor([[1],
#         [1]]))

#与上面相反，获取 与先验框（PriorBox） IoU最大的 GT框（GTBox） 的值和索引
#PriorBox 1：GTBox n
best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
print(overlaps.max(0, keepdim=True))
# torch.return_types.max(
# values=tensor([[0.8388, 0.9715, 0.6970, 0.4071, 0.5478]]),
# indices=tensor([[0, 0, 1, 0, 1]]))


best_prior_idx.squeeze_(1)  # best_prior_idx的shape是[num_objects,1],去掉1维，shape变为[num_objects]
best_prior_overlap.squeeze_(1) # best_prior_idx的shape是[num_objects,1],去掉1维，shape变为[num_objects]


best_truth_idx.squeeze_(0)  # best_truth_idx的shape是[1,num_priors],去掉0维，shape变为[num_priors]
best_truth_overlap.squeeze_(0)  # best_truth_overlap的shape是[1,num_priors],去掉0维，shape变为[num_priors]



#GTBox与PriorBox的IoU最大的，确保最好的PriorBox保留下来，设置为2,只要大于阈值就行，设置88也没问题.

best_truth_overlap.index_fill_(0, best_prior_idx, 88)
print("best_truth_overlap:",best_truth_overlap)
# best_truth_overlap: \
# tensor([ 0.8388, 88.0000,  0.6970,  0.4071,  0.5478])

# 保证每一个GTBox匹配它的都是具有最大IoU的PriorBox
for j in range(best_prior_idx.size(0)):
    best_truth_idx[best_prior_idx[j]] = j
# 循环过程
# best_prior_idx.size(0): 2
# j 0
# best_prior_idx[j]: tensor(1)
# j 1
# best_prior_idx[j]: tensor(1)
# best_truth_idx: tensor([0, 1, 1, 0, 1])

print("best_truth_idx:",best_truth_idx)
#best_truth_idx: tensor([0, 1, 1, 0, 1])

matches = truths[best_truth_idx]#每一个PriorBox对应的bbox取出来
print("matches:",matches)
# matches: tensor([[ 0.8280,  0.5840,  0.5258,  1.1107],
#         [-1.4758,  0.3446,  0.3785,  1.8465],
#         [-1.4758,  0.3446,  0.3785,  1.8465],
#         [ 0.8280,  0.5840,  0.5258,  1.1107],
#         [-1.4758,  0.3446,  0.3785,  1.8465]])

print("best_truth_overlap：",best_truth_overlap)#每一个anchor对应的label取出来
#best_truth_overlap： tensor([ 0.8388, 88.0000,  0.6970,  0.4071,  0.5478])

conf = labels[best_truth_idx] #PriorBox对应的gt box的类别
print(conf.shape)#torch.Size([5, 1])
print("conf:",conf)
# conf: tensor([[ 99.],
#         [100.],
#         [100.],
#         [ 99.],
#         [100.]])
threshold=0.1
conf[best_truth_overlap < threshold] = 0  #过滤掉iou太低的,标记为background
print("conf:",conf)
# conf: tensor([[ 99.],
#         [100.],
#         [100.],
#         [ 99.],
#         [100.]])