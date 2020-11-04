import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from hpu_lap import HierarchicalProbUNet

data_dir = "../data/CelebAMask-HQ"


def generate_free_form_mask(height, width, m1, m2, maxver=70, max_brush_width=30, maxlength=30):
    '''
    The function is to generate a free form mask given the height and weight of the image. the starting vertext must be in m1, and the mask won't go outside of m2
    :param height:
    :param width:
    :param m1:
    :param m2:
    :param maxver:
    :param max_brush_width:
    :param maxlength:
    :return:
    '''
    mask = np.zeros((height, width, 3))
    num_ver = np.random.randint(maxver // 2, maxver)
    start_x = np.random.randint(0, height)
    start_y = np.random.randint(0, width)
    while m1[start_x, start_y] == 0:
        start_x = np.random.randint(0, height)
        start_y = np.random.randint(0, width)
    brush_width = np.random.randint(5, max_brush_width)
    for i in range(num_ver):
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(0, maxlength)
        end_x = int(start_x + length * np.cos(angle))
        end_y = int(start_y + length * np.sin(angle))
        if end_x >= height:
            end_x = height - 1
        elif end_x < 0:
            end_x = 0
        if end_y >= width:
            end_y = width - 1
        elif end_y < 0:
            end_y = 0
        while m2[end_x, end_y] == 0:
            angle = np.random.uniform(0, 2 * np.pi)
            length = np.random.randint(0, maxlength)
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            if end_x >= height:
                end_x = height - 1
            elif end_x < 0:
                end_x = 0
            if end_y >= width:
                end_y = width - 1
            elif end_y < 0:
                end_y = 0
        mask = cv2.line(mask, (start_y, start_x), (end_y, end_x), (255, 255, 255), brush_width)
    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    mask = np.reshape(mask, (height, width))
    return mask


def generate_training(im, mask1, mask2):
    '''
    the function is to generate the training inputs
    :param im:
    :param mask1:
    :param mask2:
    :return:
    '''
    mask = generate_free_form_mask(im.shape[0], im.shape[1], mask1, mask2)
    xx = np.argwhere(mask == 1)
    res = np.zeros((im.shape[0], im.shape[1], 7))
    res[:, :, 0:3] = im
    for x in xx:
        for i in range(3):
            res[x[0], x[1], i] = 1
    res[:, :, 3] = mask
    res[:, :, 4:7] = im
    return res


def get_faces(ind):
    '''
    the function is to get the mask of the faces to constraint the freeform mask
    :param ind:
    :return:
    '''
    sind = str(ind)
    fold = ind // 2000
    while len(sind) != 5:
        sind = '0' + sind
    names = ['skin', 'cloth', 'hair', 'neck']
    c = 0
    for name in names:
        if os.path.exists(os.path.join(data_dir, 'CelebAMask-HQ-mask-anno',
                                       str(fold) + '/' + sind + '_' + name + '.png')):
            if c == 0:
                mask = cv2.resize(cv2.imread(
                    os.path.join(data_dir, 'CelebAMask-HQ-mask-anno', str(fold) + '/' + sind + '_' + name + '.png')),
                    (256, 256))
                m1 = mask.copy()
                c += 1
            else:
                mask += cv2.resize(cv2.imread(
                    os.path.join(data_dir, 'CelebAMask-HQ-mask-anno', str(fold) + '/' + sind + '_' + name + '.png')),
                    (256, 256))
    mask = (mask > 0).astype(np.float)
    m1 = (m1 > 0).astype(np.float)
    mask = mask[:, :, 0]
    m1 = m1[:, :, 0]
    return m1, mask


def load_data_celeb(lis, t='train'):
    x = []
    if t == 'train':
        lb = 0
        hb = 27000
        lim = 800
    else:
        lb = 27000
        hb = 30000
        lim = 1
    for i in range(lim):
        ind = np.random.randint(lb, hb)
        if len(lis) == 0:
            break
        else:
            while ind not in lis:
                ind = np.random.randint(lb, hb)
            lis.remove(ind)
            im = cv2.imread(os.path.join(data_dir, 'CelebA-HQ-img', str(ind) + '.jpg'))
            mask1, mask2 = get_faces(ind)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (256, 256))
            im = im / 255.0
            im = generate_training(im, mask1, mask2)
            x.append(im)
    return np.array(x)


class CeleTrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.len = 27000

    def __getitem__(self, index):
        im = cv2.imread(os.path.join(data_dir, 'CelebA-HQ-img', str(index) + '.jpg'))
        mask1, mask2 = get_faces(index)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (256, 256))
        im = im / 255.0
        im = generate_training(im, mask1, mask2)
        return im

    def __len__(self):
        return self.len


def my_model():
    model = HierarchicalProbUNet(
        num_layers=7,
        num_filters=[64, 128, 256, 512, 1024, 1024, 1024],
        num_prior_layers=4,
        num_filters_prior=[40, 20, 10, 5],
        # 4 x 4, 8 x 8, 16 x 16, 32 x 32
        rec=1.0,
        p=[0, 0, 0, 0.00002, 0],
        s=[0, 0, 0, 0.002, 0],
        tv=0,
        name='ProbUNet',
    )
    return model


def train():
    print(tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    epochs = 30
    out = '../output/naive_inpaint/'
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = my_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01))

    train_dataset = CeleTrainDataset()
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=400, shuffle=True, num_workers=8)

    for epoch in range(epochs):
        # lis = []
        # for i in range(27000):
        #     lis.append(i)
        # while len(lis) != 0:
        #     print(epoch, len(lis))
        #     x = load_data_celeb(lis)
        #     model.fit(x, x, epochs=1, batch_size=16)
        for c, x in enumerate(data_loader):
            print("epoch", epoch, ",", c, "/", 27000//400)
            x = x.numpy()
            model.fit(x, x, epochs=1, batch_size=16)
        model.save_weights(out + str(epoch) + '.h5', save_format='h5')


def continue_train(num):
    print(tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    epochs = 50
    out = '../output/naive_inpaint/'
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = my_model()
        inputs = tf.keras.Input(shape=(256, 256, 7,))
        model(inputs)
        model.load_weights(out + str(num) + '.h5', by_name=True, skip_mismatch=True)
        model.compile()

    train_dataset = CeleTrainDataset()
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=400, shuffle=True, num_workers=8)

    for epoch in range(num + 1, epochs):
        # lis = []
        # for i in range(27000):
        #     lis.append(i)
        # while len(lis) != 0:
        #     print(epoch, len(lis))
        #     x = load_data_celeb(lis)
        #     model.fit(x, x, epochs=1, batch_size=16)
        # model.save_weights(out + str(epoch) + '.h5', save_format='h5')
        for c, x in enumerate(data_loader):
            print("epoch", epoch, ",", c, "/", 27000//400)
            x = x.numpy()
            model.fit(x, x, epochs=1, batch_size=16)
        model.save_weights(out + str(epoch) + '.h5', save_format='h5')


def evaluation(num):
    print(tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    out = '../output/naive_inpaint/'
    # hpu hpu_temp
    model = my_model()
    inputs = tf.keras.Input(shape=(256, 256, 7,))
    model(inputs)
    model.load_weights(out + str(num) + '.h5', by_name=True, skip_mismatch=True)
    lis = []
    for i in range(27000, 30000):
        lis.append(i)
    while len(lis) != 0:
        x = load_data_celeb(lis, 'valid')

        plt.subplot(4, 2, 1)
        plt.imshow(x[0, :, :, 0:3])
        plt.subplot(4, 2, 2)
        plt.imshow(x[0, :, :, 4:7])
        for i in range(6):
            y = model.sample(x[0:1, :, :, 0:4], is_training=False)
            plt.subplot(4, 2, i + 3)
            plt.imshow(y[0, :, :, :])
        plt.show()


def reconstruct(num):
    print(tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    out = '../output/naive_inpaint/'
    # hpu hpu_temp
    model = my_model()
    inputs = tf.keras.Input(shape=(256, 256, 7,))
    model(inputs)
    model.load_weights(out + str(num) + '.h5', by_name=True, skip_mismatch=True)
    lis = []
    for i in range(27000, 30000):
        lis.append(i)
    while len(lis) != 0:
        x = load_data_celeb(lis, 'valid')

        plt.subplot(4, 2, 1)
        plt.imshow(x[0, :, :, 0:3])
        plt.subplot(4, 2, 2)
        plt.imshow(x[0, :, :, 4:7])
        for i in range(6):
            y = model.reconstruct(x[0:1, :, :, 0:7], is_training=False)
            plt.subplot(4, 2, i + 3)
            plt.imshow(y[0, :, :, :])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--start_epoch', type=int, default=0)
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "eval":
        evaluation(args.start_epoch)
    elif args.mode == "reconstruct":
        reconstruct(args.start_epoch)
    elif args.mode == "continue_train":
        continue_train(args.start_epoch)
    else:
        raise NotImplementedError

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
