import math
"""
num_gpu 所有卡的数量
num_layer 模型层数(encoder + decoder)
d_model 模型维度
bsz 单卡上的max_token/wpb
model_size 模型总参数量
data_size 数据token总数
base_wps 根据dummy_mt计算的单卡wps
bdw 通信带宽(bit/s)

"""
def compute_time(node_num, num_gpu, num_layer, dmodel, wpb, share_model_size, data_size, base_wps, bdw, moe_freq, topk, remote_rate, gpu_per_node, fsdp, update_freq):
    computation_time = wpb / base_wps
    global_bsz = wpb * num_gpu
    num_step = data_size / global_bsz
    all2all_forward = 2
    all2all_backward = 2
    fp16 = 16 # 16bit
    scale_factor = 1
    #if node_num == 1:
        #scale_factor = 1
    #else:
        #scale_factor = math.sqrt(node_num-1) # 根据gshard,随着device数量D增加,all2all通信成本按sqrt(D)增加
    all2all_datasize = topk * wpb * dmodel * fp16 * num_layer * all2all_forward * all2all_backward / moe_freq # message in one gpu

    if fsdp:
        # ZERO需要为forward和backward各付出一次all-gather,且随着梯度累积继续线性增加
        single_all_reduce_scatter_datasize = share_model_size * fp16 # reduce-scatter
        single_all_gather_datasize = share_model_size * fp16
        all_reduce_datasize = single_all_reduce_scatter_datasize  + single_all_gather_datasize * 2 * update_freq # reduce-scatter + (forward + backward) * update_freq * all-gather
    else:
        all_reduce_datasize = 2 * share_model_size * fp16 # reduce-scatter + all-gather

    # 不考虑节点内通信,只考虑节点间通讯,不考虑节点间多跳和节点拓扑结构
    all_reduce_time = remote_rate * all_reduce_datasize / bdw
    all2all_time = remote_rate * gpu_per_node * all2all_datasize * scale_factor / bdw
    communication_time = all_reduce_time + all2all_time
    print(f"computation time per step: {computation_time} s, communication time per step: {communication_time} s, remote_rate: {remote_rate}")
    print(f"all reduce time: {all_reduce_time} s, all2all time:{all2all_time} s")
    train_time_per_step = computation_time + communication_time
    train_time = train_time_per_step * num_step / (3600 * 24) # the number of day
    return train_time, num_step

if __name__ == "__main__":

    # Example: 2台4卡v100实践,与真实环境下速度完全一致
    # max_token为3277, 梯度累积5次,不使用fsdp
    """
    node_num = 2 # x台
    gpu_per_node = 4 # y卡
    num_gpu = node_num * gpu_per_node # x台y卡
    remote_rate = 1 - gpu_per_node / num_gpu  # 在本地节点的显卡的比例,仅计算节点间通信成本
    num_layer = 24 # x + y (x encoder - y decoder)
    moe_freq = 2
    topk = 2 # topk gate
    d_model = 1024
    # bsz = 3277 # 单卡上的max_token/wpb
    wpb = 1e5 / 8 # 单卡wpb,以实际log中为准,bsz*update_freq会偏大
    model_size = 4e8 + 4.2e8 # 共享参数+专家参数, 4e共享参数与4.2亿专家参数
    share_model_size = 4e8 # 共享参数,all-reduce操作只针对共享参数
    data_size = 2.8e9 # 70亿句*60token/句
    base_wps = 37000 / 8 # 实验测得8卡wps(单机),无限带宽下的训练速度,按数据并行线性假设,除8得到单卡wps 
    bdw = 10e9 # 10Gbit/s / 1.25GB/s
    fsdp = False
    update_freq = 5
    days, num_step = compute_time(node_num, num_gpu, num_layer, d_model, wpb, share_model_size, data_size, base_wps, bdw, moe_freq, topk, remote_rate, gpu_per_node, fsdp, update_freq)
    print(f"Training needs {days} days, total {num_step} steps")
    """

    # Example: 2台8卡v100实践, 真实: 12s / step, 计算: 11.3s / step
    # 每张v100放2个expert, max_token为2000, 无梯度累积

    # node_num = 2 # x台服务器
    # gpu_per_node = 8 # y张显卡
    # num_gpu = node_num * gpu_per_node # x台y卡
    # remote_rate = 1 - gpu_per_node / num_gpu  # 在本地节点的显卡的比例,仅计算节点间通信成本
    # num_layer = 48 # x + y (x encoder - y decoder)
    # moe_freq = 4 
    # topk = 2 # topk gate
    # d_model = 2048
    # wpb = 15840 / 8 # 单卡wpb: 实验测得n卡wpb(单机), 除n得到单卡wpb,以实际log中为准,bsz*update_freq会偏大
    # model_size = 26.22e8 + 8.06e8 # 共享参数+专家参数,26.22共享参数,8.06e专家参数(2个expert/gpu)
    # share_model_size = 26.22e8 # 共享参数,all-reduce操作只针对共享参数
    # data_size = 4.2e11 # 70亿句*60token/句
    # base_wps = 4000 / 8 # 实验测得n卡wps(单机),即无限带宽下的训练速度,按数据并行线性假设,除n得到单卡wps
    # bdw = 12e9 # 12Gbit/s / 1.5GB/s
    # fsdp = True 
    # update_freq = 1
    # days, num_step = compute_time(node_num, num_gpu, num_layer, d_model, wpb, share_model_size, data_size, base_wps, bdw, moe_freq, topk, remote_rate, gpu_per_node, fsdp, update_freq)
    # print(f"Training needs {days} days, total {num_step} steps")


    # Example: 3台8卡v100实践, 真实: 14s / step, 计算: 13.7s/step
    # 每张v100放2个expert, max_token为2000, 无梯度累积
  
    node_num = 8 # x台服务器
    gpu_per_node = 8 # y张显卡
    num_gpu = node_num * gpu_per_node # x台y卡
    remote_rate = 1 - gpu_per_node / num_gpu  # 在本地节点的显卡的比例,仅计算节点间通信成本
    num_layer = 48 # x + y (x encoder - y decoder)
    moe_freq = 4 
    topk = 2 # topk gate
    d_model = 2048
    wpb = 15840 / 8 # 单卡wpb: 实验测得n卡wpb(单机), 除n得到单卡wpb,以实际log中为准,bsz*update_freq会偏大
    model_size = 26.22e8 + 8.06e8 # 共享参数+专家参数,26.22共享参数,8.06e专家参数(2个expert/gpu)
    share_model_size = 26.22e8 # 共享参数,all-reduce操作只针对共享参数
    data_size = 4.2e11 # 70亿句*60token/句
    base_wps = 4000 / 8 # 实验测得n卡wps(单机),即无限带宽下的训练速度,按数据并行线性假设,除n得到单卡wps
    bdw = 100e9 # 12Gbit/s / 1.5GB/s
    fsdp = True 
    update_freq = 1
    days, num_step = compute_time(node_num, num_gpu, num_layer, d_model, wpb, share_model_size, data_size, base_wps, bdw, moe_freq, topk, remote_rate, gpu_per_node, fsdp, update_freq)
    print(f"Training needs {days} days, total {num_step} steps")
 


    # Example: 1台8卡v100实践, 真实: 5.6s / step
    # 每张v100放2个expert, max_token为1000, 梯度累积=2

    # node_num = 1 # x台服务器
    # gpu_per_node = 8 # y张显卡
    # num_gpu = node_num * gpu_per_node # x台y卡
    # remote_rate = 1 - gpu_per_node / num_gpu  # 在本地节点的显卡的比例,仅计算节点间通信成本
    # num_layer = 48 # x + y (x encoder - y decoder)
    # moe_freq = 4 
    # topk = 2 # topk gate
    # d_model = 2048
    # model_size = 26.22e8 + 8.06e8 # 共享参数+专家参数,26.22共享参数,8.06e专家参数(2个expert/gpu)
    # share_model_size = 26.22e8 # 共享参数,all-reduce操作只针对共享参数
    # data_size = 4.2e11 # 70亿句*60token/句
    # update_freq = 2
    # wpb = update_freq * 7920 / 8 # 单卡wpb: 实验测得n卡wpb(单机), 除n得到单卡wpb,以实际log中为准,bsz*update_freq会偏大
    # base_wps = 2840 / 8 # 实验测得n卡wps(单机),即无限带宽下的训练速度,按数据并行线性假设,除n得到单卡wps
    # bdw = 12e9 # 12Gbit/s / 1.5GB/s
    # fsdp = True 
    
    # days, num_step = compute_time(node_num, num_gpu, num_layer, d_model, wpb, share_model_size, data_size, base_wps, bdw, moe_freq, topk, remote_rate, gpu_per_node, fsdp, update_freq)
    # print(f"Training needs {days} days, total {num_step} steps")


    # Example: 2台8卡v100实践, 真实: 26.6s / step , 计算: 20.8s / step
    # 每张v100放2个expert, max_token为2000, 梯度累积=2

    # node_num = 2 # x台服务器
    # gpu_per_node = 8 # y张显卡
    # num_gpu = node_num * gpu_per_node # x台y卡
    # remote_rate = 1 - gpu_per_node / num_gpu  # 在本地节点的显卡的比例,仅计算节点间通信成本
    # num_layer = 48 # x + y (x encoder - y decoder)
    # moe_freq = 4 
    # topk = 2 # topk gate
    # d_model = 2048
    # model_size = 26.22e8 + 8.06e8 # 共享参数+专家参数,26.22共享参数,8.06e专家参数(2个expert/gpu)
    # share_model_size = 26.22e8 # 共享参数,all-reduce操作只针对共享参数
    # data_size = 4.2e11 # 70亿句*60token/句
    # update_freq = 2
    # wpb = update_freq * 15840 / 8 # 单卡wpb: 实验测得n卡wpb(单机), 除n得到单卡wpb,以实际log中为准,bsz*update_freq会偏大
    # base_wps = 4000 / 8 # 实验测得n卡wps(单机),即无限带宽下的训练速度,按数据并行线性假设,除n得到单卡wps
    # bdw = 12e9 # 12Gbit/s / 1.5GB/s
    # fsdp = True 
    
    # days, num_step = compute_time(node_num, num_gpu, num_layer, d_model, wpb, share_model_size, data_size, base_wps, bdw, moe_freq, topk, remote_rate, gpu_per_node, fsdp, update_freq)
    # print(f"Training needs {days} days, total {num_step} steps")
    

    # Example: 2台8卡v100实践, 真实:  / step , 计算: 39.8s / step
    # 每张v100放2个expert, max_token为2000, 梯度累积=4

    # node_num = 2 # x台服务器
    # gpu_per_node = 8 # y张显卡
    # num_gpu = node_num * gpu_per_node # x台y卡
    # remote_rate = 1 - gpu_per_node / num_gpu  # 在本地节点的显卡的比例,仅计算节点间通信成本
    # num_layer = 48 # x + y (x encoder - y decoder)
    # moe_freq = 4 
    # topk = 2 # topk gate
    # d_model = 2048
    # model_size = 26.22e8 + 8.06e8 # 共享参数+专家参数,26.22共享参数,8.06e专家参数(2个expert/gpu)
    # share_model_size = 26.22e8 # 共享参数,all-reduce操作只针对共享参数
    # data_size = 4.2e11 # 70亿句*60token/句
    # update_freq = 4
    # wpb = update_freq * 15840 / 8 # 单卡wpb: 实验测得n卡wpb(单机), 除n得到单卡wpb,以实际log中为准,bsz*update_freq会偏大
    # base_wps = 4000 / 8 # 实验测得n卡wps(单机),即无限带宽下的训练速度,按数据并行线性假设,除n得到单卡wps
    # bdw = 12e9 # 12Gbit/s / 1.5GB/s
    # fsdp = True 
    
    # days, num_step = compute_time(node_num, num_gpu, num_layer, d_model, wpb, share_model_size, data_size, base_wps, bdw, moe_freq, topk, remote_rate, gpu_per_node, fsdp, update_freq)
    # print(f"Training needs {days} days, total {num_step} steps")
   