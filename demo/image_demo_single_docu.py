import logging
import os
import pandas as pd  # 用于保存Excel文件
from mmcv.image import imread
from mmengine.logging import print_log
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

def process_folder(input_folder, output_folder, config_file, checkpoint_file):
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

    # 初始化模型
    if draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(config_file, checkpoint_file, device=device, cfg_options=cfg_options)

    # 初始化可视化器
    model.cfg.visualizer.radius = radius
    model.cfg.visualizer.alpha = alpha
    model.cfg.visualizer.line_width = thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta, skeleton_style=skeleton_style)

    data_list = []

    # 遍历输入文件夹中的图像
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    has_image = False
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            has_image = True
            img_path = os.path.join(input_folder, file)
            filename, ext = os.path.splitext(file)
            output_filename = f'{filename}_pose{ext}'
            out_file = os.path.join(output_folder, output_filename)

            # 对单张图像进行推理
            batch_results = inference_topdown(model, img_path)
            results = merge_data_samples(batch_results)

            if results.pred_instances is None or len(results.pred_instances) == 0:
                print_log(f'未在图像中检测到人体：{img_path}', logger='current', level=logging.INFO)
                continue

            keypoints = results.pred_instances.keypoints
            for idx, person_keypoints in enumerate(keypoints):
                keypoints_flat = person_keypoints.flatten()
                data_row = {'image_name': output_filename, 'person_id': idx + 1}
                for i in range(len(keypoints_flat) // 2):
                    data_row[f'x{i+1}'] = keypoints_flat[2 * i]
                    data_row[f'y{i+1}'] = keypoints_flat[2 * i + 1]
                data_list.append(data_row)

            # 保存可视化结果
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

            print_log(f'已处理图像：{img_path}，结果保存至 {out_file}', logger='current', level=logging.INFO)

    if has_image and data_list:
        df = pd.DataFrame(data_list)
        excel_file = os.path.join(output_folder, 'keypoints.xlsx')
        df.to_excel(excel_file, index=False)
        print_log(f'关键点已保存到 {excel_file}', logger='current', level=logging.INFO)
    elif has_image and not data_list:
        print_log(f'在目录 {input_folder} 中没有检测到人体关键点，未生成 Excel 文件。', logger='current', level=logging.INFO)

    print_log('子文件夹处理完成。', logger='current', level=logging.INFO)

if __name__ == '__main__':
    input_folder = '/media/robert/4TB-SSD/watchped_dataset - 副本/Box/evening/12_08_2022_18_13_38_scene5_stop_all'
    output_folder = '/media/robert/4TB-SSD/漏网之鱼pose/evening/12_08_2022_18_13_38_scene5_stop_all'
    config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

    process_folder(input_folder, output_folder, config_file, checkpoint_file)
