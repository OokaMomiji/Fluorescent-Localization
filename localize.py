#### extract seq used mask
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import imageio
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from skimage import measure
import imgrvt as rvt

from config import settings

import matplotlib.pyplot as plt
import scienceplots

class LocsExtractor:
    """
    用于图像处理的类，包括读取图像数据和对图像进行预处理。

    方法：
    - `read_tif_multiframe`: 读取多帧 TIFF 图像。
    - `extract_number`: 从文件名中提取数字，适用于排序。
    - `load_images_from_folder_multithreaded`: 使用多线程从文件夹加载图像。
    - `load_images`: 主方法：根据文件路径加载图像，可以是单个文件或文件夹。
    """
    
    def __init__(self, folder=None):
        """
        初始化 ImageProcessor 类。

        :param folder: 图像文件夹路径（可选）
        """
        self.folder = folder

    @staticmethod
    def extract_number(filename):
        """
        从文件名中提取数字（假设文件名格式为 '_数字.tiff'）。

        :param filename: 文件名
        :return: 提取的数字，如果匹配失败则返回 -1
        """
        match = re.search(r'_(\d+)\.tiff$', filename)
        return int(match.group(1)) if match else -1

    
    def load_images_and_compute_locs(self, background=None, start_frame=None, end_frame=None):
        """
        使用多线程从文件夹加载图片，并计算与 mask 的平均值。

        :param background: 与图片相同大小的 numpy 数组，用于去掉背景
        :param start_frame: 起始帧（默认为 None）
        :param end_frame: 结束帧（默认为 None）
        :return: 每帧图片与 mask 相乘后求平均值的数组
        """
        filenames = sorted([f for f in os.listdir(self.folder) if f.endswith(".tiff")], key=self.extract_number)
        
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = len(filenames)
        
        
        rmin = settings.LOCS_CONFIG.rmin
        rmax = settings.LOCS_CONFIG.rmax
        neighborhood_size = settings.LOCS_CONFIG.neighborhood_size 
        threshold = settings.LOCS_CONFIG.threshold 

        # 定义读取并计算均值的函数
        def frame_process_func(filename,
                           background = None,
                           rmin = 1,
                           rmax = 10,
                           neighborhood_size=5,
                           threshold=1,
                           ):
            img = imageio.imread(os.path.join(self.folder, filename)).astype(np.uint16)
            if background is not None:
                img = img - background
            data=rvt.rvt(img,rmin=rmin,rmax=rmax)

            data_max = filters.maximum_filter(data, neighborhood_size)
            maxima = (data == data_max)
            data_min = filters.minimum_filter(data, neighborhood_size)
            diff = ((data_max - data_min) > threshold)
            maxima[diff == 0] = 0

            labeled, num_objects = ndimage.label(maxima)
            slices = ndimage.find_objects(labeled)
            x, y = [], []
            for dy, dx in slices:
                x_center = (dx.start + dx.stop - 1)/2
                x.append(x_center)
                y_center = (dy.start + dy.stop - 1)/2
                y.append(y_center)
            return x,y

        

        # 使用线程池并行读取并计算
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(frame_process_func, filenames[i], background, rmin, rmax, neighborhood_size, threshold) for i in range(start_frame, end_frame)]
            results = [future.result() for future in futures]  # results 是 [(x1, y1), (x2, y2), ...]
    
            xs, ys = [], []
            for x, y in results:
                if len(x)>0:
                    xs.extend(x)
                    ys.extend(y)
            self.xs = np.array(xs)
            self.ys = np.array(ys)
            

    def calculate_frequency(self,):
        image_size = settings.LOCS_CONFIG.image_size
        fthreshold = settings.LOCS_CONFIG.fthreshold
        self.frequency_matrix, xedges, yedges = np.histogram2d(self.xs, self.ys, bins=image_size,range=[[0, image_size], [0, image_size]])
        def frequency_process_func(img,
                           threshold=1,
                           ):
                        # 读取图像
            image = img

            # 二值化图像
            binary_image = image >= threshold  # 你可以调整阈值

            # 查找所有 blobs
            label_image = measure.label(binary_image)

            # 计算每个 blob 的中心
            properties = measure.regionprops(label_image)

            # 存储 blobs 的中心
            centers = []

            for prop in properties:
                centers.append(prop.centroid)
            
            ys, xs = zip(*centers) # 注意xy顺序

            # 转换为 NumPy 数组
            xs = np.array(xs)
            ys = np.array(ys)
            return xs, ys

        self.ex, self.ey = frequency_process_func(self.frequency_matrix,threshold=fthreshold)
    
    def plot_frequency_matrix(self,with_dots= False):
        plt.style.use(['science', 'nature'])
        figsize = tuple(settings.FREQUENCY_PLOT.figsize)
        vmax = settings.FREQUENCY_PLOT.vmax
        # 创建绘图窗口
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.imshow(self.frequency_matrix,vmax=vmax)
        if with_dots:
            ax1.plot(self.ex, self.ey, 'ro', alpha =0.5)
        plt.show()

    def cal_freq_and_plot(self):
        self.calculate_frequency()
        self.plot_frequency_matrix(with_dots=True)

    def extract_sequences(self, start_frame=None, end_frame=None):
        filenames = sorted([f for f in os.listdir(self.folder) if f.endswith(".tiff")], key=self.extract_number)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = len(filenames)
        size = settings.SEQUENCES_CONFIG.extract_size
        def extract_points_single_frame(filename, ex, ey, size):
            img = imageio.imread(os.path.join(self.folder, filename)).astype(np.uint16)
            H, W = img.shape
            results = np.zeros(len(ex))
            for i, (x, y) in enumerate(zip(ex, ey)):
                x_r, y_r = round(x), round(y)
                x_min = np.clip(x_r - size-1, 0, H - 1)
                x_max = np.clip(x_r + size, 0, H - 1)
                y_min = np.clip(y_r - size-1, 0, W - 1)
                y_max = np.clip(y_r + size, 0, W - 1)
                results[i] = np.mean(np.mean(img[x_min:x_max,y_min:y_max],axis=0),axis=0)
            return results
        
        # 使用线程池并行读取并计算
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(extract_points_single_frame, filenames[i], self.ex, self.ey, size) for i in range(start_frame, end_frame)]
            results = np.array([future.result() for future in futures])
            self.sequences = results.T

    def plot_sequences(self,):
        total_plots_number = settings.SEQUENCES_PLOT.total_plots_number
        start = settings.SEQUENCES_PLOT.start
        space = int(len(self.ex)/total_plots_number)
        for i in range(total_plots_number):
            index = i*space + start
            plt.plot(self.sequences[index])
            plt.show()

    