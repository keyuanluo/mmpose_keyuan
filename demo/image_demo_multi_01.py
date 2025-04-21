# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os

import pandas as pd  # 用于保存Excel文件
from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


def main():
    # 指定输入和输出的根目录路径
    input_root_dir = '/media/robert/4TB-SSD/watchped_dataset - 副本/Box'   # 请将此路径替换为您的输入大文件夹路径
    output_root_dir = '/media/robert/4TB-SSD/watchped_dataset - 副本/Pose' # 请将此路径替换为您的输出大文件夹路径

    # 指定配置文件和检查点文件路径
    config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'  # 请将此路径替换为您的配置文件路径
    checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'  # 请将此路径替换为您的检查点文件路径

    # 可选参数，可以根据需要修改
    device = 'cuda:0'  # 用于推理的设备
    draw_heatmap = False  # 是否可视化预测的热力图
    show_kpt_idx = False  # 是否显示关键点的索引
    skeleton_style = 'mmpose'  # 骨架样式选择
    kpt_thr = 0.3  # 关键点可视化的阈值
    radius = 3  # 关键点可视化的半径
    thickness = 1  # 连线的粗细
    alpha = 0.8  # 边界框的透明度
    show = False  # 是否显示图像

    # 根据配置文件和检查点文件构建模型
    if draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        config_file,
        checkpoint_file,
        device=device,
        cfg_options=cfg_options)

    # 初始化可视化器
    model.cfg.visualizer.radius = radius
    model.cfg.visualizer.alpha = alpha
    model.cfg.visualizer.line_width = thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style=skeleton_style)

    # 准备收集关键点数据
    data_list = []

    # 遍历输入根目录下的所有文件和子目录
    for root, dirs, files in os.walk(input_root_dir):
        for file in files:
            # 检查文件是否是图像文件（根据需要可以添加更多的扩展名）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(root, file)
                # 相对于输入根目录的相对路径
                relative_path = os.path.relpath(root, input_root_dir)
                # 构建对应的输出目录路径
                output_subdir = os.path.join(output_root_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                # 输出图片的文件名，在尾部添加 "_pose"
                filename, ext = os.path.splitext(file)
                out_file = os.path.join(output_subdir, f'{filename}_pose{ext}')

                # 对单张图像进行推理
                batch_results = inference_topdown(model, img_path)
                results = merge_data_samples(batch_results)

                # 提取关键点并添加到数据列表
                if results.pred_instances is None or len(results.pred_instances) == 0:
                    print_log(
                        f'未在图像中检测到人体：{img_path}',
                        logger='current',
                        level=logging.INFO)
                    continue
                else:
                    keypoints = results.pred_instances.keypoints  # 已经是 numpy.ndarray 类型
                    for idx, person_keypoints in enumerate(keypoints):
                        keypoints_flat = person_keypoints.flatten()
                        data_row = {
                            'image_name': os.path.join(relative_path, file),
                            'person_id': idx + 1
                        }
                        for i in range(len(keypoints_flat) // 2):
                            data_row[f'x{i+1}'] = keypoints_flat[2 * i]
                            data_row[f'y{i+1}'] = keypoints_flat[2 * i + 1]
                        data_list.append(data_row)

                # 显示结果并保存处理后的图片
                img = imread(img_path, channel_order='rgb')
                visualizer.add_datasample(
                    name='result',
                    image=img,
                    data_sample=results,
                    draw_gt=False,
                    draw_bbox=True,
                    kpt_thr=kpt_thr,
                    draw_heatmap=draw_heatmap,
                    show_kpt_idx=show_kpt_idx,
                    skeleton_style=skeleton_style,
                    show=show,
                    wait_time=0,
                    out_file=out_file)

                print_log(
                    f'已处理图像：{img_path}，结果保存至 {out_file}',
                    logger='current',
                    level=logging.INFO)

    # 将关键点数据保存到 Excel 文件
    if data_list:
        df = pd.DataFrame(data_list)
        excel_file = os.path.join(output_root_dir, 'keypoints.xlsx')
        df.to_excel(excel_file, index=False)
        print_log(
            f'关键点已保存到 {excel_file}',
            logger='current',
            level=logging.INFO)
    else:
        print_log(
            '没有可保存的关键点数据。',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
