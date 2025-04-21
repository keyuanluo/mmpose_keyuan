# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os

import pandas as pd
from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--device', default='cuda:0', help='用于推理的设备')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='是否可视化预测的热力图')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='是否显示关键点的索引')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='骨架样式选择')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='关键点可视化的阈值')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='关键点可视化的半径')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='骨架连线的粗细')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='边界框的透明度')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='是否显示图像')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 在这里指定输入和输出目录的路径
    input_dir = '/home/robert/mmpose/000391_pedestrian_1.jpg'   # 请将此路径替换为您的输入目录路径
    output_dir = '/home/robert/mmpose' # 请将此路径替换为您的输出目录路径

    # 指定配置文件和检查点文件路径
    config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

    # 构建模型
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        config_file,
        checkpoint_file,
        device=args.device,
        cfg_options=cfg_options)

    # 初始化可视化器
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style=args.skeleton_style)

    # 准备收集关键点数据
    data_list = []

    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录下的所有图像文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # 检查文件是否是图像文件
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(root, file)
                # 对单张图像进行推理
                batch_results = inference_topdown(model, img_path)
                results = merge_data_samples(batch_results)

                # 获取关键点
                if results.pred_instances is None or len(results.pred_instances) == 0:
                    print_log(
                        f'未在图像中检测到人体：{img_path}',
                        logger='current',
                        level=logging.INFO)
                    continue

                keypoints = results.pred_instances.keypoints.cpu().numpy()  # 形状：[人数，关键点数，2]

                # 假设每张图片只取第一个检测到的人体的关键点
                person_keypoints = keypoints[0]
                # 将关键点展开为一维
                keypoints_flat = person_keypoints.flatten()
                # 准备数据行
                data_row = {'image_name': file}
                for i in range(len(keypoints_flat) // 2):
                    data_row[f'x{i+1}'] = keypoints_flat[2 * i]
                    data_row[f'y{i+1}'] = keypoints_flat[2 * i + 1]
                data_list.append(data_row)

                # 显示结果并保存带姿态的图像
                img = imread(img_path, channel_order='rgb')
                # 生成输出路径
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                out_file = os.path.join(output_subdir, file)
                visualizer.add_datasample(
                    'result',
                    img,
                    data_sample=results,
                    draw_gt=False,
                    draw_bbox=True,
                    kpt_thr=args.kpt_thr,
                    draw_heatmap=args.draw_heatmap,
                    show_kpt_idx=args.show_kpt_idx,
                    skeleton_style=args.skeleton_style,
                    show=args.show,
                    out_file=out_file,
                    wait_time=0  # 不延迟
                )

                print_log(
                    f'已处理图像：{img_path}，结果保存至 {out_file}',
                    logger='current',
                    level=logging.INFO)

    # 将关键点数据保存到 Excel 文件
    if data_list:
        df = pd.DataFrame(data_list)
        excel_file = os.path.join(output_dir, 'keypoints.xlsx')
        df.to_excel(excel_file, index=False)
        print_log(
            f'关键点数据已保存至 {excel_file}',
            logger='current',
            level=logging.INFO)
    else:
        print_log(
            '没有可保存的关键点数据。',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
