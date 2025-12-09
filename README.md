# 宠物图像分割系统 - Oxford-IIIT Pet Image Segmentation

基于U-Net架构的宠物图像语义分割系统，使用TensorFlow实现像素级图像分割。

## 核心特性
- **U-Net架构**：编码器-解码器结构，专为图像分割设计
- **迁移学习**：使用预训练的MobileNetV2作为特征提取器
- **实时可视化**：训练过程中实时显示分割效果
- **自动下载**：一键运行自动下载数据集
- **完整训练流程**：数据预处理、增强、训练、评估一体化

## 快速开始

### 系统要求
- Python 3.7+
- TensorFlow 2.8+
- 4GB以上内存
- 推荐使用GPU

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/mingze1-cn/pet-image-segmentation.git
cd pet-image-segmentation

# 安装依赖
pip install -r requirements.txt

# 运行程序（自动下载数据集）
python oxford_pet_segmentation_unet.py
