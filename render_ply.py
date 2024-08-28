import open3d as o3d

# 設置快捷鍵退出的回調函數
def exit_visualizer(vis):
    vis.destroy_window()
    return False

frames = 2
ply_path = './outputs/gaussiandreamer-sd/a_fox@20240828-150958/save/it1200-test-color.ply'

# 使用 VisualizerWithKeyCallback 而不是普通的 Visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# 讀取點雲
pcd = o3d.io.read_point_cloud(ply_path)

# 直接使用 get_render_option 調整點大小
render_option = vis.get_render_option()
render_option.point_size = 1.0  # 調整這個數值來縮小或放大點的尺寸

# 添加幾何體到視窗
vis.add_geometry(pcd)

# 設置按下 'Q' 鍵退出視覺化
vis.register_key_callback(ord("Q"), exit_visualizer)

ctr = vis.get_view_control()
ctr.set_lookat([0, 0, 0])  # 观察中心
ctr.set_front([0.5, -1.0, 0.2])  # 从 Y 轴的负方向开始看
ctr.set_up([0.0, 0.0, 1.0])  # 设置上方向为 Z 轴正方向
ctr.set_zoom(0.8)  # 设置缩放比例

# 初始更新
vis.poll_events()
vis.update_renderer()

# 更新循環
for i in range(1, frames):
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

# 這裡的循環會繼續，直到用戶按下 'Q' 鍵
vis.run()
