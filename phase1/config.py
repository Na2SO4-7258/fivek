DATASET="f" # f=FiveK，p=PPSD
if DATASET not in ("f","p"):raise ValueError("DATASET must be 'f' or 'p'")

FIVEK_ROOT="FiveK" # FiveK 数据集根目录
PPSD_ROOT=r"E:\SOFT3881\ppsd" # PPSD 根目录
ROOT=FIVEK_ROOT if DATASET=="f" else PPSD_ROOT # 当前数据集根目录
CACHE="cache_u8" if DATASET=="f" else "cache_u8/ppsd" # 缓存目录，按数据集隔离
DEVICE="cuda" # 训练设备
AMP=True # 是否启用混合精度
SEED=42 # 随机种子
Z_DIM=3 # 一阶段编辑潜变量维度，同时也是 Flow 输出维度
DIM=32 # 二阶段条件/历史/query 特征维度
IMAGE_SIZE=256 # 缓存与训练图像尺寸
ADAM_BETAS=(.9,.999) # Adam 动量系数
ADAM_EPS=1e-8 # Adam 数值稳定项
WEIGHT_DECAY=1e-4 # 权重衰减
MIN_LR_RATIO=.02 # 余弦退火最低学习率比例
GRAD_CLIP=5. # 梯度裁剪上限
NORM_EPS=1e-6 # 归一化稳定项
PSNR_EPS=1e-12 # PSNR 的 MSE 下限
DOWNLOAD_END=5000 # 下载最大编号
DOWNLOAD_SIZE=256 # 下载图最长边
DOWNLOAD_WORKERS=12 # 下载线程数
DOWNLOAD_TIMEOUT=180 # 下载超时秒数
DOWNLOAD_CHUNK=1024*1024 # 流式下载块大小
DOWNLOAD_POOL_SIZE=32 # HTTP 连接池容量
DOWNLOAD_RETRIES=5 # 下载重试次数

P1_EXPERTS=("A","B","C","D","E") # FiveK 一阶段专家；PPSD 使用所选范围全部用户
P1_TRAIN_END=500 if DATASET=="f" else 200 # 一阶段训练截止编号/用户 index
P1_END=600 if DATASET=="f" else 250 # 一阶段验证截止编号/用户 index
P1_SAVE="phase1/out/model/best.pt" if DATASET=="f" else "phase1/out/ppsd/model/best.pt" # 一阶段最佳权重路径
P1_EPOCHS=500 # 一阶段训练轮数
P1_MICRO_BATCH=1 # 一阶段单次前向 batch
P1_ACCUM_STEPS=8 # 梯度累积步数
P1_VAL_BATCH=1 # 一阶段验证 batch
P1_LR=2e-4 # 一阶段学习率
P1_NUM_WORKERS=2 # 数据加载进程数
P1_PREFETCH=2 # 每个 worker 预取批数
P1_WIDTH=32 # 一阶段编码器基础通道
P1_DEPTHS=(2,3,4,6) # 四级编码器残差块数
P1_ENC_RES=128 # 一阶段编辑编码器输入分辨率
P1_CM=8 # 颜色统计每轴分箱数
P1_CM_HIDDEN=256 # 颜色统计投影隐藏维度
P1_CM_DIM=128 # 颜色统计输出维度
P1_HEAD_HIDDEN=256 # 潜变量预测头隐藏维度
P1_LOGVAR_INIT=-6. # 潜变量 logvar 初始偏置
P1_LOGVAR_RANGE=(-8.,2.) # 潜变量 logvar 截断范围
P1_LUT_SIZE=17 # 3D LUT 每轴网格数

P1_LUT_BASES=8 # LUT 基底数量

P1_LUT_HIDDEN=256 # LUT 系数网络隐藏维度
P1_LUT_INIT_STD=.05 # LUT 基底初始化标准差
P1_LUT_SCALE=.5 # LUT 残差最大幅度
P1_KNOTS=17 # 全局调色曲线节点数
P1_ADJUST_HIDDEN=128 # 全局调色网络隐藏维度
P1_CURVE_SCALE=.25 # 曲线偏移幅度
P1_MATRIX_SCALE=.3 # 颜色矩阵偏移幅度
P1_BIAS_SCALE=.1 # 颜色偏置幅度
P1_SMOOTH_WEIGHT=1e-4 # LUT 平滑正则权重
P1_IDENTITY_WEIGHT=1e-5 # LUT 恒等正则权重
P1_KL_WEIGHT=5e-4 # KL 正则权重
P1_FREE_BITS=.02 # 每维 KL 自由阈值
P1_MSE_WEIGHT=.5 # 重建损失中 MSE 权重
P1_GLOBAL_P=.5 # 使用全局图像概率
P1_FLIP_P=.5 # 水平/垂直翻转概率

P2_EXPERTS=("A","B","C","D","E") # FiveK 二阶段专家；PPSD 按用户分别构建历史
P2_TRAIN_END=753 if DATASET=="f" else 200 # 二阶段训练截止编号/用户 index
P2_END=754 if DATASET=="f" else 250 # 二阶段验证截止编号/用户 index
P2_SAVE="phase2/out/model_hist/best.pt" if DATASET=="f" else "phase2/out/ppsd/model_hist/best.pt" # 二阶段最佳权重路径
P2_LAST="phase2/out/model_hist/last.pt" if DATASET=="f" else "phase2/out/ppsd/model_hist/last.pt" # 二阶段最后权重/续训路径
P2_EPOCHS=500 # 二阶段训练轮数
P2_BATCH_SIZE=6 # 二阶段 batch 大小
P2_LR=2e-4 # 二阶段学习率
P2_HISTORY=8 # 训练历史样本数上限
P2_HIST_MIN=8 # 训练历史样本数下限
P2_HIST_RES=128 # 历史图像编码分辨率
P2_HISTORY_SEED=0 # 固定验证历史随机种子
P2_LAYERS=4 # 条件 Transformer 层数
P2_HEADS=4 # Transformer 注意力头数，DIM 必须可整除
P2_FF_MULT=4 # Transformer FFN 扩展倍数
P2_DROPOUT=0. # Transformer dropout
P2_EMBED_STD=.02 # query/history 类型嵌入初始化标准差
P2_CHANNELS=(32,64,128,192) # 图片编码器四级通道数
P2_DEPTHS=(2,2,3,2) # 图片编码器四级残差块数
P2_CONV_KERNEL=7 # ConvNeXt 深度卷积核
P2_CONV_EXPANSION=4 # ConvNeXt 通道扩展倍数
P2_ENCODER_HIDDEN=256 # 图片编码器输出头隐藏维度
P2_RELATION_EXPANSION=2 # 历史关系网络隐藏倍数
P2_VISUAL_DIM=96 # 全局视觉特征维度
P2_STATS_DIM=32 # 统计特征维度
P2_STATS_HIDDEN=64 # 统计网络隐藏维度
P2_STATS_QUANTILES=(.05,.25,.5,.75,.95) # 亮度分位点
P2_FUSE_HIDDEN=128 # 视觉+统计融合隐藏维度
P2_FLOW_LAYERS=3 # 条件 Flow 层数
P2_PCA_K=min(12,Z_DIM) # PCA/白化使用维度上限，不超过 Z_DIM
P2_FLOW_BINS=8 # RQS 分箱数
P2_FLOW_HIDDEN=64 # Flow 子网络隐藏维度
P2_FLOW_BOUND=6. # RQS 样条边界
P2_FLOW_BLOCKS=2 # Flow 残差块数
P2_FLOW_DROPOUT=0. # Flow dropout
P2_FLOW_BATCH_NORM=False # Flow 是否使用 BatchNorm
P2_MC_SAMPLES=256 # 训练验证 MC 采样数
P2_KEEP_FRAC=.5 # 按概率保留采样比例
P2_N_CLUSTERS=5 # 训练验证 K-means 聚类数
P2_KM_ITERS=10 # 训练验证 K-means 迭代次数
P2_PEAK_CLAMP=6. # 采样/峰值搜索 z 截断幅度
P2_RANK_BY="size" # 聚类排序方式：size 或 logp
P2_VAL_EVERY=1 # 完整渲染验证间隔
P2_VAL_BATCH=8 # 二阶段验证 batch
P2_RENDER_CHUNK=32 # 调用一阶段 decoder 的分块大小
P2_ENCODE_CHUNK=32 # 提取一阶段 z 的分块大小
P2_LOG_EVERY=20 # 训练进度刷新间隔
P2_TEST_EXPERT="A" # FiveK 独立测试专家
P2_TEST_TRAIN_END=1200 if DATASET=="f" else 100 # 独立测试历史池截止编号/用户 index
P2_TEST_END=1250 if DATASET=="f" else 110 # 独立测试截止编号/用户 index
P2_TEST_CSV="phase2/out/test_mc.csv" if DATASET=="f" else "phase2/out/ppsd/test_mc.csv" # 独立测试结果 CSV
P2_TEST_HISTORY=300 # 独立测试历史样本数
P2_PEAK_STARTS=16 # 峰值搜索起点数
P2_PEAK_STEPS=50 # 峰值搜索迭代次数
P2_PEAK_LR=.04 # 峰值搜索学习率
P2_PEAK_MERGE=2. # 峰值合并距离阈值
P2_MAX_PEAKS=8 # 最多保留峰值数
P2_TEST_MC_SAMPLES=512 # 独立测试 MC 采样数
P2_TEST_MC_CLUSTERS=3 # 独立测试聚类数
P2_OUTLIER_MAD=3.5 # 离群过滤 MAD 倍数
P2_OUTLIER_MIN_FRAC=.5 # 过滤后最小样本比例
P2_OUTLIER_FALLBACK_FRAC=.8 # 回退时保留样本比例
P2_TEST_KMEANS_STEPS=50 # 独立测试 K-means 最大迭代次数
P2_KMEANS_TOL=1e-5 # K-means 收敛阈值

