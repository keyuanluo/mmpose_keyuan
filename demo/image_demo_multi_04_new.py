import logging
import os

import pandas as pd
from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

def main():
    # ========== 1. 配置路径 ==========
    # 输入根目录：Box_video_new_01（包含 video_0001 ... video_0255）
    input_root_dir = '/media/robert/4TB-SSD/watchped_dataset - 副本/Box_video_new_01'
    # 输出根目录：Pose_video_new_01（对应子目录 video_0001 ... video_0255）
    output_root_dir = '/media/robert/4TB-SSD/watchped_dataset - 副本/Pose_video_new_01'

    # 配置文件和检查点文件路径
    # config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    config_file = "/home/robert/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
    # checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
    checkpoint_file = "/home/robert/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"

    # 推理相关设置
    device = 'cuda:0'  # 或 'cpu'
    draw_heatmap = False
    show_kpt_idx = False
    skeleton_style = 'mmpose'
    kpt_thr = 0.3
    radius = 3
    thickness = 1
    alpha = 0.8
    show = False

    # ========== 2. 初始化模型与可视化器 ==========
    if draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        config_file,
        checkpoint_file,
        device=device,
        cfg_options=cfg_options)

    model.cfg.visualizer.radius = radius
    model.cfg.visualizer.alpha = alpha
    model.cfg.visualizer.line_width = thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta, skeleton_style=skeleton_style)

    # 若目标根目录不存在，先创建
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    # ========== 3. 遍历 video_0001 ... video_0255 ==========
    for i in range(1, 256):
        video_folder_name = f'video_{i:04d}'
        input_video_dir = os.path.join(input_root_dir, video_folder_name)
        output_video_dir = os.path.join(output_root_dir, video_folder_name)

        if not os.path.isdir(input_video_dir):
            # 当前 video 文件夹不存在就跳过
            continue

        # 如果输出目录不存在就创建
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir)

        # ========== 4. 处理当前 video 文件夹下的所有图像 ==========
        data_list = []  # 用于保存关键点信息的列表
        has_image = False

        # 列出该文件夹下所有文件，筛选出图像
        for file in sorted(os.listdir(input_video_dir)):
            file_lower = file.lower()
            if file_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                has_image = True
                img_path = os.path.join(input_video_dir, file)

                # 生成可视化输出的文件名(在文件名后加"_pose")
                filename, ext = os.path.splitext(file)
                output_filename = f'{filename}_pose{ext}'
                out_file = os.path.join(output_video_dir, output_filename)

                # 推理
                batch_results = inference_topdown(model, img_path)
                results = merge_data_samples(batch_results)

                if results.pred_instances is None or len(results.pred_instances) == 0:
                    print_log(
                        f'未在图像中检测到人体：{img_path}',
                        logger='current',
                        level=logging.INFO)
                    continue
                else:
                    keypoints = results.pred_instances.keypoints  # shape: (num_person, num_kpt, 2)
                    for idx_person, person_keypoints in enumerate(keypoints):
                        keypoints_flat = person_keypoints.flatten()  # 展开为一维
                        data_row = {
                            'image_name': output_filename,  # 仅文件名
                            'person_id': idx_person + 1
                        }
                        # 填写每个关键点坐标 x{i}, y{i}
                        num_kpt = len(keypoints_flat) // 2
                        for j in range(num_kpt):
                            data_row[f'x{j+1}'] = keypoints_flat[2*j]
                            data_row[f'y{j+1}'] = keypoints_flat[2*j + 1]
                        data_list.append(data_row)

                # 显示并保存可视化后的图像
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
                    f'[{video_folder_name}] 已处理图像：{img_path}，结果保存至 {out_file}',
                    logger='current',
                    level=logging.INFO)

        # ========== 5. 若有图像并且检测到关键点，则保存 Excel ==========
        if has_image and data_list:
            df = pd.DataFrame(data_list)
            excel_file = os.path.join(output_video_dir, 'keypoints.xlsx')
            df.to_excel(excel_file, index=False)
            print_log(
                f'[{video_folder_name}] 关键点已保存到 {excel_file}',
                logger='current',
                level=logging.INFO)
        elif has_image and not data_list:
            # 有图但没检测到任何关键点
            print_log(
                f'[{video_folder_name}] 没有检测到人体关键点，未生成 Excel 文件。',
                logger='current',
                level=logging.INFO)

    print_log('所有视频处理已完成。', logger='current', level=logging.INFO)


if __name__ == '__main__':
    main()
