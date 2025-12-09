"""导入库"""
import matplotlib
import matplotlib.pyplot as plt
import warnings
import tensorflow_datasets as tfds
import tensorflow as tf
from IPython.display import clear_output
from tensorflow_examples.models.pix2pix import pix2pix

# 忽略警告
warnings.filterwarnings('ignore')
# 强制 TkAgg 后端
matplotlib.use('TkAgg')
# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']

# 加载oxford数据集
dataset, info = tfds.load('oxford_iiit_pet', with_info=True, data_dir='data/oxford')

# 数据预处理
def data_dispose(input_image, input_mask):
    # 归一化处理
    input_image = tf.cast(input_image, tf.float32) / 128 - 1
    input_mask -= 1

    return input_image, input_mask

# 加载并处理训练集数据
def load_train_dispose(data):
    # 设置图像和掩码大小
    input_image = tf.image.resize(data['image'], (128, 128))
    input_mask = tf.image.resize(data['segmentation_mask'], (128, 128))

    # 水平翻转
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    
    # 调用预处理函数
    input_image, input_mask = data_dispose(input_image, input_mask)

    return input_image, input_mask

# 加载并处理测试集数据
def load_test_dispose(data):
    # 设置图像和掩码大小
    input_image = tf.image.resize(data['image'], (128, 128))
    input_mask = tf.image.resize(data['segmentation_mask'], (128, 128))
    
    # 调用预处理函数
    input_image, input_mask = data_dispose(input_image, input_mask)

    return input_image, input_mask

# 获取所有训练样本
TRAIN_LENGTH = info.splits['train'].num_examples
# 定义每个批次的样本
BATCH_SIZE = 64
# 定义缓冲区大小
BUFFER_SIZE = 1000
# 计算训练步数
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# 使用map将函数运用到每一项
train = dataset['train'].map(load_train_dispose, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_test_dispose)

# 打乱分批处理
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 预处理测试集数据
test_dataset = test.batch(BATCH_SIZE)

# 定义可视化函数
def display(display_list):
    # 设置画板大小
    plt.figure(figsize=(8, 6))
    # 创建标题列表
    title = ['输入图像', '真实掩码', '预测掩码']

    # 遍历显示列表
    for i in range(len(display_list)):
        # 创建子图
        plt.subplot(1, len(display_list), i + 1)
        # 设置标题
        plt.title(title[i])
        # 设置显示图像
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

# 提取数据点
for image, mask in train.take(1):
    sample_image, sample_mask = image, mask

    # 调用可视化函数
    display([sample_image, sample_mask])

# 设置通道数量
OUTPUT_CHANNELS = 3

# 使用预训练模型
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False)

# 创建层名称列表
layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu',
    'block_16_project'
]

# 从指定层中提取特征图
layers = [base_model.get_layer(name).output for name in layer_names]

# 创建新模型
down_stack = tf.keras.Model(base_model.input, layers, trainable=False)

# 创建上采样层
up_stack = [
    pix2pix.upsample(512, 3),   
    pix2pix.upsample(256, 3),   
    pix2pix.upsample(128, 3),   
    pix2pix.upsample(64, 3)   
]

# 创建U-Net模型构建函数
def unet_model(output_channels):
    # 定义最后一层
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, 2, padding='same', activation='softmax')

    # 定义输出层
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = inputs

    # 降频取样
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # 升频取样
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # 使用最后一层
    x = last(x)

    # 返回一个新的模型
    return tf.keras.Model(inputs=inputs, outputs=x)

# 调用构建函数
model = unet_model(OUTPUT_CHANNELS)
# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

# 将掩码从概率分布转为单通道的整数
def create_mask(pred_mask):
    # 从指定层找到最大索引
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # 在最后一个维度上新增一个轴
    pred_mask = pred_mask[..., tf.newaxis]

    # 返回第一个分割样本
    return pred_mask[0]

# 定义可视化预测函数
def show_prediction(dataset=None, num=1):
    # 判断dataset是否传入
    if dataset:
        # 提取num个数据点
        for image, mask in dataset.take(num):
            # 使用模型预测图像获取预测掩码
            pred_mask = model.predict(image)
            # 调用可视化函数
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])

# 定义回调函数
class DisplayCallbacks(tf.keras.callbacks.Callback):
    # 定义周期结束事件函数
    def on_epoch_end(self, epoch, logs=None):
        # 清除输出
        clear_output(wait=True)
        # 调用可视化预测函数
        show_prediction()
        # 打印周期编码
        print(f'\n第{epoch + 1}个训练周期后的结果\n')

# 定义训练周期
EPOCHS = 2
# 定义测试集分割数
TEST_SPLITS = 5
# 计算验证步数
VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // TEST_SPLITS

# 训练模型
model_history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=test_dataset, callbacks=[DisplayCallbacks()])

# 获取训练损失和验证损失
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

# 定义训练序列
epoch = range(EPOCHS)
# 创建新窗口
plt.figure()
# 绘制损失曲线
plt.plot(epoch, loss, 'r', label='训练损失')
plt.plot(epoch, val_loss, 'bo', label='验证损失')
# 设置标题和横纵标签
plt.title('训练损失和验证损失')
plt.xlabel('训练周期')
plt.ylabel('损失值大小')
# 显示图例
plt.legend()
# 显示图像
plt.show(block=False)
plt.pause(3)
plt.close()

# 调用可视化函数
show_prediction(test_dataset, 3)