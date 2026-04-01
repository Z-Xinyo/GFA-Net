# GFA-Net

This repo is the official implementation for Learning Efficient Glimpse-Focus Attention for Self-Supervised Skeleton-based Action Recognition via Distribution-Aware Distillation. 

# Illustration

![image-20260401163811291](C:\Users\zhou\Desktop\assets\image-20260401163811291.png)

Comparison with state-of-the-art self-supervised methods across multiple benchmarks and protocols(radar chart: higher area indicates better overall accuracy) and accuracy-efficiency trade-off (scatter plot: top-1 accuracy vs. GFLOPs per action). GFA-Net achieves new state-of-the-art results, with its lightweight variant (GFA-Net-Tiny) delivering competitive performance at substantially reduced computational cost. 

# Framework

![image-20260401164102303](C:\Users\zhou\Desktop\assets\image-20260401164102303.png)

Overall architecture of the proposed Glimpse-Focus Attention Network (GFA-Net) with queue-based distribution-aware distillation. Input skeleton sequences are processed by a GCN-based feature extractor, followed by spatio-temporal decoupling into separate spatial and temporal embeddings. Parallel Glimpse (global attention) and Focus (hierarchical local attention) branches synchronously model long-range holistic dependencies and fine-grained intra-/inter-region dynamics, respectively, with max-pooling for aggregation. During self-supervised pretraining, a heavyweight teacher network generates rich representations to guide a lightweight student (with reduced backbone layers and channel dimensions) by aligning similarity distributions over dynamic queues of historical teacher features. The teacher is used only during training, incurring no inference overhead.

# Visualization

![image-20260401164535491](C:\Users\zhou\Desktop\assets\image-20260401164535491.png)