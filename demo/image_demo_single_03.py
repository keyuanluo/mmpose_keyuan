# Copyright (c) OpenMMLab. All rights reserved.
import logging
from argparse import ArgumentParser

import pandas as pd  # 用于保存Excel文件
from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='图片文件')
    parser.add_argument('config', default='td-hm_hrnet-w48_8xb32-210e_coco-256x192.py', help='配置文件')
    parser.add_argument('checkpoint', default='td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth', help='检查点文件')
    parser.add_argument('--out-file', default=None, help='输出图片的路径')
    parser.add_argument(
        '--device', default='cuda:0', help='用于推理的设备')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='可视化预测的热力图')
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
        help='连线的粗细')
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

    # 根据配置文件和检查点文件构建模型
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # 初始化可视化器
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style=args.skeleton_style)

    # 对单张图像进行推理
    batch_results = inference_topdown(model, args.img)
    results = merge_data_samples(batch_results)

    # 提取关键点并保存到Excel
    if results.pred_instances is None or len(results.pred_instances) == 0:
        print_log(
            f'未在图像中检测到人体：{args.img}',
            logger='current',
            level=logging.INFO)
    else:
        keypoints = results.pred_instances.keypoints  # 已经是 numpy.ndarray 类型
        data_list = []
        for idx, person_keypoints in enumerate(keypoints):
            keypoints_flat = person_keypoints.flatten()
            data_row = {'image_name': args.img, 'person_id': idx + 1}
            for i in range(len(keypoints_flat) // 2):
                data_row[f'x{i+1}'] = keypoints_flat[2 * i]
                data_row[f'y{i+1}'] = keypoints_flat[2 * i + 1]
            data_list.append(data_row)
        # 保存到Excel文件
        df = pd.DataFrame(data_list)
        if args.out_file:
            excel_file = args.out_file.rsplit('.', 1)[0] + '.xlsx'
        else:
            excel_file = 'keypoints.xlsx'
        df.to_excel(excel_file, index=False)
        print_log(
            f'关键点已保存到 {excel_file}',
            logger='current',
            level=logging.INFO)

    # 显示结果
    img = imread(args.img, channel_order='rgb')
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
        out_file=args.out_file)

    if args.out_file is not None:
        print_log(
            f'输出图像已保存到 {args.out_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
