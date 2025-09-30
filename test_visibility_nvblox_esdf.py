import numpy as np
import math
import imageio.v3 as iio

import open3d as o3d
import matplotlib.pyplot as plt
import torch
from PIL import Image
import zarr
from tqdm import tqdm
import time
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import copy
import gc
# UniDepth imports
from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
from unidepth.utils.camera import Pinhole
from unidepth.utils import colorize
from im2flow2act.common.imagecodecs_numcodecs import register_codecs
from im2flow2act.common.utility.zarr import parallel_reading

# nvblox imports
from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
from nvblox_torch.mesh import ColorMesh
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.constants import constants


# ========= 사용자 설정 =========
# Zarr 데이터 경로 설정
BUFFER_PATH = "/home/dscho-larr/fast_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/slam_head_mounted_camera_multi_marker_initial_lag"
EPISODE_IDX = 0
FRAME_IDX = 100  # 특정 프레임 선택
DEPTH_SCALE = 0.001        # 깊이 단위 → 미터 변환 (예: mm면 0.001, 이미 m면 1.0)
OFFSET_DISTANCE = 0.0 # for convex part of the constructed mesh

# TSDF 설정
# Mesh 품질 선택: "high_resolution" (5mm), "medium_resolution" (10mm), "low_resolution" (20mm)
MESH_QUALITY = "low_resolution" # "medium_resolution"  # "high_resolution", "medium_resolution", "low_resolution"

if MESH_QUALITY == "high_resolution":
    VOXEL_SIZE = 0.005  # 5mm - 높은 해상도, 조각난 mesh
    
elif MESH_QUALITY == "medium_resolution":
    VOXEL_SIZE = 0.01   # 10mm - 중간 해상도, 균형잡힌 mesh
    
else:  # low_resolution
    VOXEL_SIZE = 0.02   # 20mm - 낮은 해상도, 매끄러운 mesh
    

print(f"Using {MESH_QUALITY}: voxel_size={VOXEL_SIZE}m")

# Depth estimation 설정
USE_MONODEPTH = True  # True: UniDepth 사용, False: raw depth 사용
MODEL_TYPE = "l"  # UniDepth model type: s, b, l

# 이미지 리사이즈 설정
RESIZE = True  # True: 이미지를 256x256으로 리사이즈, False: 원본 크기 사용
RESIZE_SIZE = (256, 256)  # 리사이즈할 크기 (width, height)

# (중요) 카메라 내파라미터: 사용자가 직접 채우세요.
K = np.array([
    [604.682922, 0.0, 328.062561],
    [0.0, 604.898438, 244.393188],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# 새 카메라 뷰포인트들 (월드 좌표, world = 원 카메라 좌표계로 가정)
# CANDIDATE_VIEWPOINTS = [
#     np.array([0.20, 0.00, 0.00], dtype=np.float64),  # 옆으로
#     np.array([0.00, 0.20, 0.00], dtype=np.float64),  # 위로
#     np.array([0.00, 0.00, 0.20], dtype=np.float64),  # 앞으로
#     np.array([0.15, 0.15, 0.00], dtype=np.float64),  # 대각선
#     np.array([0.15, 0.15, 0.15], dtype=np.float64),  # 3D 대각선
# ]
CANDIDATE_VIEWPOINTS = [
    np.array([-0.05, 0.00, 0.00], dtype=np.float64),
    np.array([-0.10, 0.00, 0.00], dtype=np.float64),
    np.array([-0.15, 0.00, 0.00], dtype=np.float64),
    np.array([-0.20, 0.00, 0.00], dtype=np.float64),
    np.array([-0.25, 0.00, 0.00], dtype=np.float64),
]


# 각 viewpoint의 yaw 회전 (deg)
CANDIDATE_YAW_DEGS = [0.0, 0.0, 0.0, 0.0, 0.0]

# M개의 viewpoint set 생성 (예제에서는 M=3으로 설정)
M = 10 # 3  # M개의 viewpoint set
L = len(CANDIDATE_VIEWPOINTS)  # L개의 viewpoints per set



print(f"Creating {M} viewpoint sets, each with {L} viewpoints")
print(f"Total viewpoints: {M} x {L} = {M*L}")

# [M, L] 모양의 candidate viewpoints 배열 생성
# 현재는 예제로 기존 CANDIDATE_VIEWPOINTS를 M번 복사
CANDIDATE_VIEWPOINTS_MATRIX = np.array([CANDIDATE_VIEWPOINTS for _ in range(M)])
print(f"Candidate viewpoints matrix shape: {CANDIDATE_VIEWPOINTS_MATRIX.shape}")  # [M, L, 3]


# ========= 유틸 함수 =========
def resize_image_and_depth(rgb, depth, target_size=(256, 256)):
    """
    RGB 이미지와 depth 이미지를 지정된 크기로 리사이즈 (OpenCV 사용)
    
    Args:
        rgb: RGB 이미지 (H, W, 3)
        depth: depth 이미지 (H, W)
        target_size: 리사이즈할 크기 (width, height)
    
    Returns:
        resized_rgb: 리사이즈된 RGB 이미지
        resized_depth: 리사이즈된 depth 이미지
        scale_factor: 스케일 팩터 (원본 크기 / 리사이즈 크기)
    """
    import cv2
    
    H, W = rgb.shape[:2]
    target_w, target_h = target_size
    
    rgb_uint8 = rgb.astype(np.uint8)
    
    rgb_resized = cv2.resize(rgb_uint8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    
    
    # Depth 이미지 리사이즈 (INTER_NEAREST 사용)
    depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # 스케일 팩터 계산 (x, y 방향 각각)
    scale_factor_x = (target_w-1) / (W-1)
    scale_factor_y = (target_h-1) / (H-1)
    
    return rgb_resized, depth_resized, scale_factor_x, scale_factor_y


def adjust_camera_intrinsics(K, scale_factor_x, scale_factor_y):
    """
    카메라 내부 파라미터를 리사이즈에 맞게 조정
    
    Args:
        K: 원본 카메라 내부 파라미터 (3, 3)
        scale_factor_x: x 방향 스케일 팩터
        scale_factor_y: y 방향 스케일 팩터
    
    Returns:
        K_adjusted: 조정된 카메라 내부 파라미터
    """
    K_adjusted = K.copy()
    K_adjusted[0, 0] *= scale_factor_x  # fx
    K_adjusted[1, 1] *= scale_factor_y  # fy
    K_adjusted[0, 2] *= scale_factor_x  # cx
    K_adjusted[1, 2] *= scale_factor_y  # cy
    
    return K_adjusted


def load_rgbd_from_zarr(buffer_path, episode_idx, frame_idx, depth_scale=0.001):
    """Zarr에서 RGB와 depth 데이터를 불러오는 함수"""
    register_codecs()
    
    # Zarr 버퍼 열기
    buffer = zarr.open(buffer_path, mode="r")
    episode = buffer[f"episode_{episode_idx}"]
    
    # RGB 프레임 로드
    rgb_frames = parallel_reading(
        group=episode["camera_0"],
        array_name="rgb",
    )
    rgb = rgb_frames[frame_idx]  # 특정 프레임 선택
    
    # Depth 프레임 로드
    
    depth_frames = parallel_reading(
        group=episode["camera_0"],
        array_name="depth",
    )
    depth_raw = depth_frames[frame_idx]  # 특정 프레임 선택
    depth_m = depth_raw.astype(np.float32) * depth_scale
    
    
    return rgb, depth_m


def radial_depth_to_z_depth(radial_depth, intrinsics):
    """
    Convert radial depth (distance from camera center) to Z-depth (distance from camera plane).
    """
    H, W = radial_depth.shape
    
    # Create pixel coordinate grid
    pixel_x, pixel_y = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing='xy'
    )
    
    # Convert to normalized camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Normalized coordinates (x/z, y/z)
    norm_x = (pixel_x - cx) / fx
    norm_y = (pixel_y - cy) / fy
    
    # Calculate squared distance from optical center
    squared_distance_from_center = norm_x**2 + norm_y**2
    
    # Convert radial depth to Z-depth
    depth_scaling = np.sqrt(1 + squared_distance_from_center)
    z_depth = radial_depth / depth_scaling
    
    return z_depth


def get_depth_with_unidepth(rgb, intrinsics, model, camera):
    """UniDepth를 사용하여 depth를 추정하고 Z-depth로 변환"""
    # Convert RGB frame to tensor
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    
    # Predict depth
    with torch.no_grad():
        predictions = model.infer(rgb_torch, camera)
    
    # Extract depth prediction (radial depth)
    radial_depth_pred = predictions["depth"].squeeze().cpu().numpy()
    
    # Convert radial depth to Z-depth
    z_depth_pred = radial_depth_to_z_depth(radial_depth_pred, intrinsics)
    
    return z_depth_pred


def scale_depth_with_raw(depth_pred, depth_raw, intrinsics):
    """Raw depth를 사용하여 predicted depth를 스케일링"""
    # Ensure spatial dimensions match
    if depth_raw.shape != depth_pred.shape:
        print(f"Resizing depth_raw from {depth_raw.shape} to {depth_pred.shape}")
        depth_raw_resized = np.array(Image.fromarray(depth_raw.astype(np.float32)).resize(
            (depth_pred.shape[1], depth_pred.shape[0]), Image.NEAREST))
        depth_raw = depth_raw_resized
    
    # Use all valid depth pixels for scaling
    valid_mask = (depth_raw > 0) & (depth_pred > 0)
    
    if np.sum(valid_mask) > 100:  # Require at least 100 valid pixels
        # Use median for robust scaling
        scale_factor = np.median(depth_raw[valid_mask] / depth_pred[valid_mask])
        depth_scaled = depth_pred * scale_factor
        print(f"Applied scale factor {scale_factor:.3f} using raw depth ({np.sum(valid_mask)} valid pixels)")
        return depth_scaled
    else:
        print(f"Insufficient valid depth values ({np.sum(valid_mask)} pixels), using original depth")
        return depth_pred




def pick_query_pixel(depth_m):
    """이미지 중앙에서 오른쪽으로 이미지 크기에 비례한 픽셀 이동한 위치에서 유효한 depth 값을 가진 픽셀을 선택."""
    H, W = depth_m.shape
    cx, cy = W // 2, H // 2
    
    # 이미지 크기에 비례한 오프셋 계산 (원본 640x480에서 30픽셀이었으므로)
    base_offset = 30
    base_width = 640
    offset = max(int(base_offset * W / base_width), 5)  # 최소 5픽셀
    
    # 중앙에서 오른쪽으로 비례한 픽셀 이동
    target_x = cx + offset
    target_y = cy
    
    # 타겟 위치가 유효한지 확인
    if 0 <= target_x < W and 0 <= target_y < H and depth_m[target_y, target_x] > 0:
        return (target_x, target_y)
    
    # 타겟 위치가 유효하지 않으면 주변에서 유효한 픽셀 찾기
    radius = max(H, W) // 20
    ys, xs = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = xs*xs + ys*ys <= radius*radius
    candidates = np.argwhere(mask) + np.array([target_y-radius, target_x-radius])
    for yy, xx in candidates:
        if 0 <= yy < H and 0 <= xx < W and depth_m[yy, xx] > 0:
            return int(xx), int(yy)
    
    # 여전히 찾지 못하면 원래 중앙에서 찾기
    if depth_m[cy, cx] > 0:
        return (cx, cy)
    
    # 마지막 fallback: 유효한 픽셀 중 하나 선택
    ys, xs = np.where(depth_m > 0)
    if len(ys) == 0:
        raise RuntimeError("유효 깊이 픽셀이 없습니다.")
    i = len(ys) // 2
    return int(xs[i]), int(ys[i])



def create_mapper_with_nvblox(depth_image, rgb_image, K, voxel_size=0.005, max_integration_distance=5.0, mapper=None):
    """
    nvblox를 사용하여 mapper만 생성 (mesh 없이)
    
    Args:
        depth_image: (H, W) 깊이 이미지 (미터)
        rgb_image: (H, W, 3) RGB 이미지
        K: (3, 3) 카메라 내부 파라미터
        voxel_size: voxel 크기 (미터)
        max_integration_distance: 최대 통합 거리 (미터)
        mapper: 기존 Mapper 객체 재사용 (None이면 새로 생성)
    
    Returns:
        mapper: nvblox Mapper 객체
    """
    print("Creating mapper with nvblox...")
    start = time.time()
    # 데이터를 torch tensor로 변환
    H, W = depth_image.shape

    # Depth 이미지를 torch tensor로 변환
    depth_tensor = torch.from_numpy(depth_image.astype(np.float32)).cuda()
    
    # RGB 이미지를 torch tensor로 변환 (H, W, 3)
    rgb_tensor = torch.from_numpy(rgb_image).cuda()
    
    # 카메라 내부 파라미터를 torch tensor로 변환
    K_tensor = torch.from_numpy(K).float().cpu()
    
    # 카메라 pose (identity)
    pose_tensor = torch.eye(4).float().cpu()
    
    # Mapper 생성 또는 재사용
    if mapper is None:
        mapper = Mapper(voxel_sizes_m=voxel_size)
    
    # TSDF layer에 depth frame 추가
    mapper.add_depth_frame(depth_tensor, pose_tensor, K_tensor)
    
    # Color layer에 RGB frame 추가
    mapper.add_color_frame(rgb_tensor, pose_tensor, K_tensor)
    
    # 메시 업데이트
    mapper.update_color_mesh()

    # ESDF layer 업데이트 (TSDF 기반으로 ESDF 생성)
    mapper.update_esdf()
    
    print(f"nvblox mapper created in {time.time() - start:.4f} seconds")
    
    return mapper


# ========= nvblox ESDF 기반 Visibility 체크 =========
def check_visibility_with_esdf_ray_marching(mapper, origin, target_point, max_distance, num_samples=100):
    """
    ESDF layer를 사용한 ray marching으로 visibility 체크
    
    Args:
        mapper: nvblox Mapper 객체
        origin: 레이 시작점 (월드 좌표)
        target_point: 목표 점 (월드 좌표)
        max_distance: 최대 거리
        num_samples: ray marching 샘플 수
        mapper_id: mapper ID
    
    Returns:
        (visible, hit_distance, hit_point)
        - visible: True if point is visible
        - hit_distance: distance to first hit (if any)
        - hit_point: hit point coordinates (if any)
    """
    start_time = time.time()
    
    # 방향 벡터 계산
    direction = target_point - origin
    distance = np.linalg.norm(direction)
    if distance == 0:
        return True, 0, None
    
    direction = direction / distance
    
    # Ray marching을 위한 샘플 포인트들 생성
    sample_distances = np.linspace(0, min(distance, max_distance), num_samples)
    sample_points = origin + direction.reshape(1, 3) * sample_distances.reshape(-1, 1)
    
    # ESDF layer에서 distance 값들 쿼리
    sample_points_tensor = torch.from_numpy(sample_points.astype(np.float32)).cuda()
    
    # ESDF 쿼리 수행
    esdf_results = mapper.query_differentiable_layer(
        query_type=QueryType.ESDF,
        query=sample_points_tensor
    )
    
    # ESDF 결과에서 distance 추출
    esdf_distances = esdf_results.cpu().numpy()  # [num_samples]
    
    # Visibility 판단 로직
    # ESDF distance < 0이면 물체 내부, distance > 0이면 물체 외부
    # Ray가 물체에 부딪히는 첫 번째 지점을 찾음
    hit_mask = (esdf_distances < 0)  # ESDF는 음수면 물체 내부
    
    if np.any(hit_mask):
        # 첫 번째 hit 지점 찾기
        first_hit_idx = np.argmax(hit_mask)
        hit_distance = sample_distances[first_hit_idx]
        hit_point = sample_points[first_hit_idx]
        
        # Hit point가 target point보다 가까우면 occluded
        if hit_distance < distance - 0.01:  # 작은 tolerance
            visible = False
        else:
            visible = True
            hit_distance = distance
            hit_point = target_point
    else:
        # Hit이 없으면 visible
        visible = True
        hit_distance = distance
        hit_point = target_point
    
    end_time = time.time()
    print(f"    ESDF ray marching visibility check: {end_time - start_time:.4f}s")
    
    return visible, hit_distance, hit_point


def batch_esdf_visibility_check(mapper, origins, target_point, max_distances, num_samples=100):
    """
    ESDF layer를 사용한 batch visibility 체크
    
    Args:
        mapper: nvblox Mapper 객체
        origins: 레이 시작점들 [M, 3]
        target_point: 목표 점 (월드 좌표)
        max_distances: 최대 거리들 [M]
        num_samples: ray marching 샘플 수
        mapper_id: mapper ID
    
    Returns:
        (visible_array, hit_distances_array)
        - visible_array: [M] boolean array, True if visible
        - hit_distances_array: [M] float array, hit distances
    """
    total_start_time = time.time()
    
    M = len(origins)
    visible_array = np.zeros(M, dtype=bool)
    hit_distances_array = np.zeros(M, dtype=np.float64)
    
    # 각 ray에 대해 개별적으로 처리 (batch 처리 최적화는 나중에)
    for i in range(M):
        visible, hit_distance, _ = check_visibility_with_esdf_ray_marching(
            mapper, origins[i], target_point, max_distances[i], num_samples
        )
        visible_array[i] = visible
        hit_distances_array[i] = hit_distance
    
    total_time = time.time() - total_start_time
    print(f"    Batch ESDF visibility check ({M} rays): {total_time:.4f}s total")
    
    return visible_array, hit_distances_array
    





    
    






# 기존 mesh 기반 raycasting 함수들은 TSDF 기반 함수로 교체됨





def create_ray_based_esdf_3d_visualization_plotly(mapper, query_point, candidate_viewpoints, results, save_path, 
                                                  robot_mask=None, robot_points_original=None, robot_points_transformed=None,
                                                  voxel_size=0.02, max_distance=1.0, num_ray_samples=50):
    """
    Ray 기반 ESDF 3D 시각화 생성 (RGB point cloud + ray 색상으로 ESDF 표시)
    
    Args:
        mapper: nvblox Mapper 객체
        query_point: 쿼리 포인트
        candidate_viewpoints: 후보 viewpoint들
        results: visibility 결과들
        save_path: 저장 경로
        robot_mask: 로봇 마스크
        robot_points_original: 원본 로봇 포인트들
        robot_points_transformed: 변환된 로봇 포인트들
        voxel_size: ESDF 시각화용 voxel 크기
        max_distance: 최대 거리 (미터)
        num_ray_samples: ray당 샘플 수
    """
    print("Creating Ray-based ESDF 3D visualization...")
    
    # ESDF layer에서 거리 정보 쿼리를 위한 그리드 생성
    def get_aabb_voxel_center_grid(layer, voxel_size_m):
        """ESDF layer의 AABB를 커버하는 voxel center 그리드 생성"""
        # Get the limits of the mapped space.
        min_block_idx, max_block_idx = layer.get_block_limits()
        aabb_min_vox = min_block_idx * layer.block_dim_in_voxels
        aabb_max_vox = (max_block_idx + 1) * layer.block_dim_in_voxels
        
        # Create a 3D grid of points.
        x_linspace = torch.linspace(aabb_min_vox[0], aabb_max_vox[0], 
                                   aabb_max_vox[0] - aabb_min_vox[0] + 1, dtype=torch.int)
        y_linspace = torch.linspace(aabb_min_vox[1], aabb_max_vox[1], 
                                   aabb_max_vox[1] - aabb_min_vox[1] + 1, dtype=torch.int)
        z_linspace = torch.linspace(aabb_min_vox[2], aabb_max_vox[2], 
                                   aabb_max_vox[2] - aabb_min_vox[2] + 1, dtype=torch.int)
        
        x_grid, y_grid, z_grid = torch.meshgrid(x_linspace, y_linspace, z_linspace, indexing='ij')
        query_grid_xyz_vox = torch.stack([x_grid, y_grid, z_grid], dim=-1)
        
        # Voxel units to meters.
        query_grid_xyz_m = (query_grid_xyz_vox + 0.5) * layer.voxel_size()
        query_grid_xyz_m = query_grid_xyz_m.cuda()
        return query_grid_xyz_m
    
    # ESDF layer에서 거리 정보 쿼리
    try:
        # TSDF layer를 사용해서 AABB 범위를 얻음
        tsdf_layer = mapper.tsdf_layer_view()
        query_grid_xyz_m = get_aabb_voxel_center_grid(tsdf_layer, voxel_size)
        
        # ESDF 쿼리 수행
        print(f"Querying ESDF at {query_grid_xyz_m.numel() // 3} points...")
        sdf_values = mapper.query_differentiable_layer(
            QueryType.ESDF, 
            query_grid_xyz_m.reshape(-1, 3)
        )
        sdf_values = sdf_values.reshape(query_grid_xyz_m.shape[:-1])
        
        # 유효한 쿼리 마스크 생성
        from nvblox_torch.constants import constants
        valid_mask = torch.logical_not(sdf_values == constants.esdf_unknown_distance())
        
        # 거리 범위 필터링 (너무 먼 거리는 제외)
        distance_mask = torch.abs(sdf_values) <= max_distance
        valid_mask = valid_mask & distance_mask
        
        # 유효한 포인트들만 추출
        valid_points = query_grid_xyz_m[valid_mask].cpu().numpy()
        valid_distances = sdf_values[valid_mask].cpu().numpy()
        
        print(f"ESDF visualization: {len(valid_points)} valid points with distances in range [{valid_distances.min():.3f}, {valid_distances.max():.3f}]")
        
    except Exception as e:
        print(f"Failed to query ESDF: {e}")
        print("Falling back to mesh visualization...")
        return create_3d_visualization_plotly_mesh_fallback(mapper, query_point, candidate_viewpoints, results, save_path, 
                                                           robot_mask, robot_points_original, robot_points_transformed)
    
    # Plotly 시각화 생성
    fig = go.Figure()
    
    # 1. ESDF 거리 필드 시각화 (voxel grid)
    if len(valid_points) > 0:
        # 거리 값에 따른 색상 매핑
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # 거리 범위 정규화 (-max_distance ~ +max_distance)
        normalized_distances = (valid_distances + max_distance) / (2 * max_distance)
        normalized_distances = np.clip(normalized_distances, 0, 1)
        
        # 색상 맵 생성 (음수=빨간색, 0=노란색, 양수=파란색)
        colors = []
        for dist in normalized_distances:
            if dist < 0.5:  # 음수 거리 (물체 내부)
                # 빨간색에서 노란색으로
                r = 1.0
                g = dist * 2
                b = 0.0
            else:  # 양수 거리 (물체 외부)
                # 노란색에서 파란색으로
                r = 2.0 - dist * 2
                g = 2.0 - dist * 2
                b = (dist - 0.5) * 2
            
            colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
        
        # ESDF voxel grid 시각화
        fig.add_trace(go.Scatter3d(
            x=valid_points[:, 0],
            y=valid_points[:, 1],
            z=valid_points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
                opacity=0.6,
                line=dict(width=0)
            ),
            name='ESDF Distance Field',
            showlegend=True,
            hovertemplate='<b>ESDF Distance</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<br>' +
                         'Distance: %{customdata:.3f}m<br>' +
                         '<extra></extra>',
            customdata=valid_distances
        ))
    
    # 2. 원본 로봇 포인트들 표시 (빨간색)
    if robot_points_original is not None and len(robot_points_original) > 0:
        robot_original_sampled = robot_points_original[::max(1, len(robot_points_original)//1000)]  # 샘플링
        
        fig.add_trace(go.Scatter3d(
            x=robot_original_sampled[:, 0],
            y=robot_original_sampled[:, 1],
            z=robot_original_sampled[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color='red',
                opacity=0.7,
                symbol='circle'
            ),
            name='Original Robot Points',
            showlegend=True
        ))
    
    # 3. 변환된 로봇 포인트들 표시 (주황색)
    if robot_points_transformed is not None and len(robot_points_transformed) > 0:
        robot_transformed_sampled = robot_points_transformed[::max(1, len(robot_points_transformed)//1000)]  # 샘플링
        
        fig.add_trace(go.Scatter3d(
            x=robot_transformed_sampled[:, 0],
            y=robot_transformed_sampled[:, 1],
            z=robot_transformed_sampled[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color='orange',
                opacity=0.7,
                symbol='square'
            ),
            name='Transformed Robot Points',
            showlegend=True
        ))
    
    # 4. 쿼리 포인트 표시
    fig.add_trace(go.Scatter3d(
        x=[query_point[0]],
        y=[query_point[1]],
        z=[query_point[2]],
        mode='markers',
        marker=dict(
            size=12,
            color='red',
            symbol='diamond',
            line=dict(width=2, color='darkred')
        ),
        name='Query Point',
        showlegend=True
    ))
    
    # 5. 각 viewpoint와 Line of Sight 표시
    for i, (viewpoint, result) in enumerate(zip(candidate_viewpoints, results)):
        # Viewpoint 표시
        viewpoint_color = 'green' if result['visible_robot'] else 'red'
        fig.add_trace(go.Scatter3d(
            x=[viewpoint[0]],
            y=[viewpoint[1]],
            z=[viewpoint[2]],
            mode='markers',
            marker=dict(
                size=8,
                color=viewpoint_color,
                symbol='circle',
                line=dict(width=2, color='darkgreen' if result['visible_robot'] else 'darkred')
            ),
            name=f'Viewpoint {i+1} ({result["visible_robot"] and "VISIBLE" or "OCCLUDED"})',
            showlegend=True
        ))
        
        # Line of Sight 표시
        los_color = 'green' if result['visible_robot'] else 'red'
        los_style = 'solid' if result['visible_robot'] else 'dash'
        
        fig.add_trace(go.Scatter3d(
            x=[viewpoint[0], query_point[0]],
            y=[viewpoint[1], query_point[1]],
            z=[viewpoint[2], query_point[2]],
            mode='lines',
            line=dict(
                color=los_color,
                width=6,
                dash=los_style
            ),
            name=f'LoS {i+1} ({result["visible_robot"] and "VISIBLE" or "OCCLUDED"})',
            showlegend=True
        ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text='ESDF-based 3D Distance Field Visualization',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='lightgray'
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # HTML 파일로 저장
    import plotly.offline as pyo
    pyo.plot(fig, filename=save_path, auto_open=False)
    print(f"ESDF-based 3D visualization saved to: {save_path}")


def create_ray_based_esdf_3d_visualization_plotly_new(mapper, query_point, candidate_viewpoints, results, save_path, 
                                                      robot_mask=None, robot_points_original=None, robot_points_transformed=None,
                                                      voxel_size=0.02, max_distance=1.0, num_ray_samples=50):
    """
    새로운 Ray 기반 ESDF 3D 시각화 생성 (RGB point cloud + ray 색상으로 ESDF 표시)
    """
    print("Creating NEW Ray-based ESDF 3D visualization...")
    
    # Plotly 3D 시각화 생성
    fig = go.Figure()
    
    # 1. RGB Point Cloud 시각화 (원본 RGB 이미지 + depth로 3D point cloud 생성)
    try:
        print("Creating RGB point cloud from original images...")
        
        # 원본 이미지 로드 (전역 변수에서 가져오기)
        # 이 부분은 메인 함수에서 전역 변수로 설정되어야 함
        if 'rgb_image_original' in globals() and 'depth_image_original' in globals():
            rgb_img = rgb_image_original
            depth_img = depth_image_original
            
            # 원본 카메라 내부 파라미터 사용 (전역 변수에서 가져오기)
            if 'camera_intrinsics_original' in globals():
                K_original = camera_intrinsics_original
                fx, fy = K_original[0, 0], K_original[1, 1]
                cx, cy = K_original[0, 2], K_original[1, 2]
            
            
            # 이미지 크기
            img_height, img_width = rgb_img.shape[:2]
            
            # 3D point cloud 생성 (모든 픽셀 계산, 다운샘플링 없음)
            points_3d = []
            colors_3d = []
            
            for v in range(img_height):
                for u in range(img_width):
                    depth = depth_img[v, u]
                    
                    # 유효한 depth 값만 처리
                    if depth > 0 and depth < 5.0:
                        # 픽셀 좌표를 3D 좌표로 변환
                        x = (u - cx) * depth / fx
                        y = (v - cy) * depth / fy
                        z = depth
                        
                        points_3d.append([x, y, z])
                        
                        # RGB 색상 (0-255 범위를 0-1로 정규화)
                        r, g, b = rgb_img[v, u]
                        colors_3d.append([r/255.0, g/255.0, b/255.0])
            
            if points_3d:
                points_3d = np.array(points_3d)
                colors_3d = np.array(colors_3d)
                
                # RGB point cloud 추가
                fig.add_trace(go.Scatter3d(
                    x=points_3d[:, 0],
                    y=points_3d[:, 1],
                    z=points_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=colors_3d,
                        opacity=0.6
                    ),
                    name='RGB Point Cloud',
                    showlegend=True
                ))
                print(f"RGB point cloud added: {len(points_3d)} points")
            else:
                print("No valid points found for RGB point cloud")
        else:
            print("Original RGB and depth images not available in global scope")
            
    except Exception as e:
        print(f"RGB point cloud extraction failed: {e}")
        print("Continuing without RGB point cloud...")
    
    # 2. Ray 기반 ESDF 시각화
    try:
        print("Generating ESDF-colored rays...")
        
        for i, (viewpoint, is_visible) in enumerate(zip(candidate_viewpoints, results)):
            # Ray 방향 계산
            direction = query_point - viewpoint
            distance = np.linalg.norm(direction)
            if distance == 0:
                continue
            
            direction = direction / distance
            
            # Ray를 따라 샘플링
            ray_points = []
            ray_colors = []
            ray_esdf_values = []  # 실제 ESDF 값 저장
            
            for j in range(num_ray_samples):
                t = (j / (num_ray_samples - 1)) * distance
                sample_point = viewpoint + t * direction
                
                # ESDF 쿼리
                try:
                    esdf_value = mapper.query_differentiable_layer(
                        QueryType.ESDF, 
                        torch.tensor(sample_point, dtype=torch.float32).unsqueeze(0).cuda()
                    )[0].item()
                    
                    # ESDF unknown 값 처리
                    if abs(esdf_value - constants.esdf_unknown_distance()) < 1e-6:  # ESDF unknown
                        color = [0.0, 0.0, 0.0]  # 검은색 (unknown)
                    else:
                        # 유효한 ESDF 값 범위로 색상 매핑 (ray 거리 기반 정규화)
                        # min=0, max=distance로 정규화
                        if esdf_value < 0:  # 물체 내부 (음수)
                            # 음수: 빨간색에서 노란색으로 (0에서 distance까지)
                            normalized = min(1.0, -esdf_value / distance)
                            color = [1.0, normalized, 0.0]
                        else:  # 물체 외부 (양수)
                            # 양수: 노란색에서 파란색으로 (0에서 distance까지)
                            normalized = min(1.0, esdf_value / distance)
                            color = [1.0 - normalized, 1.0 - normalized, normalized]
                    
                    ray_points.append(sample_point)
                    ray_colors.append(color)
                    ray_esdf_values.append(esdf_value)  # 실제 ESDF 값 저장
                    
                except:
                    # ESDF 쿼리 실패시 중립 색상
                    ray_points.append(sample_point)
                    ray_colors.append([0.5, 0.5, 0.5])
                    ray_esdf_values.append(0.0)  # 기본값
            
            if ray_points:
                ray_points = np.array(ray_points)
                ray_colors = np.array(ray_colors)
                
                # Ray 시각화 (선분으로)
                for j in range(len(ray_points) - 1):
                    fig.add_trace(go.Scatter3d(
                        x=[ray_points[j][0], ray_points[j+1][0]],
                        y=[ray_points[j][1], ray_points[j+1][1]],
                        z=[ray_points[j][2], ray_points[j+1][2]],
                        mode='lines',
                        line=dict(
                            color=ray_colors[j],
                            width=4
                        ),
                        showlegend=False,
                        hovertemplate=f'<b>Ray {i+1}</b><br>' +
                                     'ESDF Distance: %{text}<br>' +
                                     '<extra></extra>',
                        text=[f'{ray_esdf_values[j]:.3f}m' for _ in range(2)]
                    ))
        
        print("ESDF-colored rays generated successfully")
        
    except Exception as e:
        print(f"Ray-based ESDF visualization failed: {e}")
    
    # 3. Query Point 표시
    fig.add_trace(go.Scatter3d(
        x=[query_point[0]],
        y=[query_point[1]],
        z=[query_point[2]],
        mode='markers',
        marker=dict(
            size=12,
            color='red',
            symbol='diamond'
        ),
        name='Query Point',
        hovertemplate='<b>Query Point</b><br>' +
                     'X: %{x:.3f}<br>' +
                     'Y: %{y:.3f}<br>' +
                     'Z: %{z:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    # 4. Robot Segmentation Mask Point Cloud 표시
    if robot_mask is not None and robot_points_original is not None:
        try:
            # 원본 robot points
            robot_points_3d = []
            robot_colors_3d = []
            
            # robot_points_original이 numpy 배열인지 리스트인지 확인
            if hasattr(robot_points_original, 'shape'):  # numpy 배열
                for i in range(len(robot_points_original)):
                    point = robot_points_original[i]
                    if len(point) >= 3:  # x, y, z 좌표가 있는 경우
                        robot_points_3d.append([point[0], point[1], point[2]])
                        robot_colors_3d.append([1.0, 0.0, 1.0])  # 마젠타색
            else:  # 리스트
                for point in robot_points_original:
                    if len(point) >= 3:  # x, y, z 좌표가 있는 경우
                        robot_points_3d.append([point[0], point[1], point[2]])
                        robot_colors_3d.append([1.0, 0.0, 1.0])  # 마젠타색
            
            if robot_points_3d:
                fig.add_trace(go.Scatter3d(
                    x=[p[0] for p in robot_points_3d],
                    y=[p[1] for p in robot_points_3d],
                    z=[p[2] for p in robot_points_3d],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=robot_colors_3d,
                        opacity=0.7
                    ),
                    name='Robot Points (Original)',
                    showlegend=True
                ))
                print(f"Robot Points (Original) added: {len(robot_points_3d)} points")
        except Exception as e:
            print(f"Robot Points (Original) visualization failed: {e}")
    
    # 5. SE(3) Transform된 Robot Point Cloud 표시
    if robot_mask is not None and robot_points_transformed is not None:
        try:
            # 변환된 robot points
            transformed_points_3d = []
            transformed_colors_3d = []
            
            # robot_points_transformed가 numpy 배열인지 리스트인지 확인
            if hasattr(robot_points_transformed, 'shape'):  # numpy 배열
                for i in range(len(robot_points_transformed)):
                    point = robot_points_transformed[i]
                    if len(point) >= 3:  # x, y, z 좌표가 있는 경우
                        transformed_points_3d.append([point[0], point[1], point[2]])
                        transformed_colors_3d.append([0.0, 1.0, 1.0])  # 시안색
            else:  # 리스트
                for point in robot_points_transformed:
                    if len(point) >= 3:  # x, y, z 좌표가 있는 경우
                        transformed_points_3d.append([point[0], point[1], point[2]])
                        transformed_colors_3d.append([0.0, 1.0, 1.0])  # 시안색
            
            if transformed_points_3d:
                fig.add_trace(go.Scatter3d(
                    x=[p[0] for p in transformed_points_3d],
                    y=[p[1] for p in transformed_points_3d],
                    z=[p[2] for p in transformed_points_3d],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=transformed_colors_3d,
                        opacity=0.7
                    ),
                    name='Robot Points (Transformed)',
                    showlegend=True
                ))
                print(f"Robot Points (Transformed) added: {len(transformed_points_3d)} points")
        except Exception as e:
            print(f"Robot Points (Transformed) visualization failed: {e}")
    
    # 6. Hitting Points 표시 (ESDF 값이 양수에서 음수로 바뀌는 지점)
    hitting_points = []
    try:
        for i, (viewpoint, is_visible) in enumerate(zip(candidate_viewpoints, results)):
            if is_visible:  # visible한 경우에만 hitting point 계산
                direction = query_point - viewpoint
                distance = np.linalg.norm(direction)
                direction = direction / distance
                
                # ESDF 기반 hitting point 찾기
                hit_point = None
                prev_esdf_value = None
                
                for j in range(num_ray_samples):
                    t = (j / (num_ray_samples - 1)) * distance
                    sample_point = viewpoint + t * direction
                    
                    try:
                        esdf_value = mapper.query_differentiable_layer(
                            QueryType.ESDF, 
                            torch.tensor(sample_point, dtype=torch.float32).unsqueeze(0).cuda()
                        )[0].item()
                        
                        # ESDF unknown 값이 아닌 경우에만 처리
                        if abs(esdf_value - constants.esdf_unknown_distance()) > 1e-6:
                            if prev_esdf_value is not None:
                                # 양수에서 음수로 바뀌는 지점 찾기
                                if prev_esdf_value > 0 and esdf_value < 0:
                                    hit_point = sample_point
                                    break
                            prev_esdf_value = esdf_value
                    except:
                        continue
                
                if hit_point is not None:
                    hitting_points.append(hit_point)
    except Exception as e:
        print(f"Hitting points calculation failed: {e}")
        hitting_points = []
    
    if hitting_points:
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in hitting_points],
            y=[p[1] for p in hitting_points],
            z=[p[2] for p in hitting_points],
            mode='markers',
            marker=dict(
                size=6,
                color='orange',
                symbol='diamond'
            ),
            name='Hitting Points',
            showlegend=True
        ))
    
    # 4. Viewpoints 표시
    for i, (viewpoint, is_visible) in enumerate(zip(candidate_viewpoints, results)):
        viewpoint_color = 'green' if is_visible else 'red'
        fig.add_trace(go.Scatter3d(
            x=[viewpoint[0]],
            y=[viewpoint[1]],
            z=[viewpoint[2]],
            mode='markers',
            marker=dict(
                size=8,
                color=viewpoint_color,
                symbol='circle'
            ),
            name=f'Viewpoint {i+1} ({is_visible and "VISIBLE" or "OCCLUDED"})',
            showlegend=True
        ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text='Ray-based ESDF 3D Visualization',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='lightgray'
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # HTML 파일로 저장
    import plotly.offline as pyo
    pyo.plot(fig, filename=save_path, auto_open=False)
    print(f"Ray-based ESDF 3D visualization saved to: {save_path}")


def create_3d_visualization_plotly_mesh_fallback(mapper, query_point, candidate_viewpoints, results, save_path, 
                                                robot_mask=None, robot_points_original=None, robot_points_transformed=None):
    """
    ESDF 쿼리 실패시 mesh 기반 fallback 시각화
    """
    print("Using mesh-based fallback visualization...")
    
    # 기존 mesh 기반 시각화 함수 호출
    mesh = mapper.get_color_mesh()
    return create_3d_visualization_plotly(mesh, query_point, candidate_viewpoints, results, save_path, 
                                         robot_mask, robot_points_original, robot_points_transformed)


def create_3d_visualization_plotly(mesh, query_point, candidate_viewpoints, results, save_path, 
                                   robot_mask=None, robot_points_original=None, robot_points_transformed=None):
    """
    Plotly를 사용한 인터랙티브 3D 시각화 생성 (색상 정보 + 로봇 변환 정보 포함)
    """
    # 메시 데이터 추출 (nvblox ColorMesh 또는 Open3D mesh 모두 지원)
    if hasattr(mesh, 'vertices') and callable(mesh.vertices):
        # nvblox ColorMesh
        vertices = mesh.vertices().cpu().numpy()
        faces = mesh.triangles().cpu().numpy()
        vertex_colors = mesh.vertex_colors().cpu().numpy() if hasattr(mesh, 'vertex_colors') and callable(mesh.vertex_colors) else None
    else:
        # Open3D mesh (fallback)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        vertex_colors = np.asarray(mesh.vertex_colors) if hasattr(mesh, 'vertex_colors') and len(mesh.vertex_colors) > 0 else None
    
    # 디버깅: mesh 정보 출력
    print(f"=== Mesh Visualization Debug ===")
    print(f"Mesh vertices: {len(vertices)}")
    print(f"Mesh faces: {len(faces)}")
    print(f"Robot points original: {len(robot_points_original) if robot_points_original is not None else 0}")
    print(f"Robot points transformed: {len(robot_points_transformed) if robot_points_transformed is not None else 0}")
    
    # Robot transform으로 인한 mesh 변화 확인
    if robot_points_original is not None and robot_points_transformed is not None:
        print(f"Robot transformation applied: {len(robot_points_original)} -> {len(robot_points_transformed)} points")
        # 변환된 robot points가 mesh에 포함되어 있는지 확인
        if len(robot_points_transformed) > 0:
            # 변환된 robot points의 범위 확인
            robot_min = np.min(robot_points_transformed, axis=0)
            robot_max = np.max(robot_points_transformed, axis=0)
            print(f"Transformed robot bounds: min={robot_min}, max={robot_max}")
            
            # Mesh vertices 범위 확인
            mesh_min = np.min(vertices, axis=0)
            mesh_max = np.max(vertices, axis=0)
            print(f"Mesh bounds: min={mesh_min}, max={mesh_max}")
            
            # Robot points가 mesh 범위 내에 있는지 확인
            robot_in_mesh = np.all(robot_min >= mesh_min - 0.1) and np.all(robot_max <= mesh_max + 0.1)
            print(f"Robot points within mesh bounds: {robot_in_mesh}")
    
    # 디버깅 정보 출력
    print(f"Mesh data: {len(vertices)} vertices, {len(faces)} faces")
    if len(faces) > 0:
        max_face_idx = np.max(faces)
        min_face_idx = np.min(faces)
        print(f"Face indices range: {min_face_idx} to {max_face_idx}")
        print(f"Vertex indices range: 0 to {len(vertices)-1}")
        
        # Face 인덱스가 vertex 범위를 벗어나는지 확인
        if max_face_idx >= len(vertices):
            print(f"WARNING: Face indices exceed vertex range! Max face idx: {max_face_idx}, Max vertex idx: {len(vertices)-1}")
            # 잘못된 face들을 필터링
            valid_faces = faces[np.all(faces < len(vertices), axis=1)]
            print(f"Filtered faces: {len(valid_faces)} valid faces out of {len(faces)}")
            faces = valid_faces
    
    # 3D 시각화 생성
    fig = go.Figure()
    
    # 1. 메시 표시 (실제 mesh로 시각화)
    if len(faces) > 0 and len(vertices) > 0:
        # 간단한 샘플링: 면의 수만 제한하고 vertex는 그대로 유지
        if len(faces) > 50000:
            # 면의 수를 제한 (vertex는 그대로 유지) - 더 많은 face 사용
            face_indices = np.random.choice(len(faces), 50000, replace=False)
            faces_sampled = faces[face_indices]
            print(f"Sampled {len(faces_sampled)} faces from {len(faces)} total faces")
        else:
            faces_sampled = faces
        
        # Mesh3d로 실제 mesh 표시 (vertex 인덱스는 원본 그대로 사용)
        if vertex_colors is not None and len(vertex_colors) > 0:
            # RGB 색상 정보가 있는 경우 - 실제 RGB 색상 사용
            if vertex_colors.max() <= 1.0:
                # Open3D 형식 (0-1 범위)
                colors_rgb = vertex_colors
            else:
                # nvblox 형식 (0-255 범위) - 0-1로 정규화
                colors_rgb = vertex_colors / 255.0
            
            # RGB 색상을 hex 문자열로 변환
            colors_hex = []
            for color in colors_rgb:
                r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                colors_hex.append(f'rgb({r},{g},{b})')
            
            # RGB 색상을 사용한 mesh 시각화
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces_sampled[:, 0],
                j=faces_sampled[:, 1],
                k=faces_sampled[:, 2],
                facecolor=colors_hex,  # 실제 RGB 색상 사용
                opacity=0.8,
                name='3D Mesh (RGB Colored)',
                showlegend=True
            ))
        else:
            # 색상 정보가 없는 경우 (기본 색상)
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces_sampled[:, 0],
                j=faces_sampled[:, 1],
                k=faces_sampled[:, 2],
                color='lightblue',
                opacity=0.6,
                name='3D Mesh',
                showlegend=True
            ))
    else:
        # Fallback: point cloud로 표시
        print("Warning: No faces found, falling back to point cloud visualization")
        fig.add_trace(go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1], 
            z=vertices[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='lightblue',
                opacity=0.6
            ),
            name='3D Scene (Point Cloud)',
            showlegend=True
        ))
    
    # 1.5. 원본 로봇 포인트들을 mesh로 표시 (빨간색)
    if robot_points_original is not None and len(robot_points_original) > 0:
        # 로봇 포인트들을 샘플링 (너무 많으면)
        if len(robot_points_original) > 2000:
            robot_indices = np.random.choice(len(robot_points_original), 2000, replace=False)
            robot_original_sampled = robot_points_original[robot_indices]
        else:
            robot_original_sampled = robot_points_original
        
        # Robot points를 mesh로 변환하여 표시
        try:
            # Open3D PointCloud 생성
            robot_pcd = o3d.geometry.PointCloud()
            robot_pcd.points = o3d.utility.Vector3dVector(robot_original_sampled)
            
            # Convex hull을 사용하여 mesh 생성
            robot_hull, _ = robot_pcd.compute_convex_hull()
            robot_vertices = np.asarray(robot_hull.vertices)
            robot_faces = np.asarray(robot_hull.triangles)
            
            if len(robot_faces) > 0:
                # Robot mesh를 시각화 (빨간색, 반투명)
                fig.add_trace(go.Mesh3d(
                    x=robot_vertices[:, 0],
                    y=robot_vertices[:, 1],
                    z=robot_vertices[:, 2],
                    i=robot_faces[:, 0],
                    j=robot_faces[:, 1],
                    k=robot_faces[:, 2],
                    color='red',
                    opacity=0.3,
                    name='Original Robot Mesh',
                    showlegend=True
                ))
                print(f"Original robot mesh created: {len(robot_vertices)} vertices, {len(robot_faces)} faces")
            else:
                # Fallback: point cloud로 표시
                fig.add_trace(go.Scatter3d(
                    x=robot_original_sampled[:, 0],
                    y=robot_original_sampled[:, 1],
                    z=robot_original_sampled[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='red',
                        opacity=0.7,
                        symbol='circle'
                    ),
                    name='Original Robot Points',
                    showlegend=True
                ))
        except Exception as e:
            print(f"Failed to create original robot mesh: {e}")
            # Fallback: point cloud로 표시
            fig.add_trace(go.Scatter3d(
                x=robot_original_sampled[:, 0],
                y=robot_original_sampled[:, 1],
                z=robot_original_sampled[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color='red',
                    opacity=0.7,
                    symbol='circle'
                ),
                name='Original Robot Points',
                showlegend=True
            ))
    
    # 1.6. 변환된 로봇 포인트들을 mesh로 표시 (주황색)
    if robot_points_transformed is not None and len(robot_points_transformed) > 0:
        # 로봇 포인트들을 샘플링 (너무 많으면)
        if len(robot_points_transformed) > 2000:
            robot_indices = np.random.choice(len(robot_points_transformed), 2000, replace=False)
            robot_transformed_sampled = robot_points_transformed[robot_indices]
        else:
            robot_transformed_sampled = robot_points_transformed
        
        # Robot points를 mesh로 변환하여 표시
        try:
            # Open3D PointCloud 생성
            robot_pcd = o3d.geometry.PointCloud()
            robot_pcd.points = o3d.utility.Vector3dVector(robot_transformed_sampled)
            
            # Convex hull을 사용하여 mesh 생성
            robot_hull, _ = robot_pcd.compute_convex_hull()
            robot_vertices = np.asarray(robot_hull.vertices)
            robot_faces = np.asarray(robot_hull.triangles)
            
            if len(robot_faces) > 0:
                # Robot mesh를 시각화 (주황색, 반투명)
                fig.add_trace(go.Mesh3d(
                    x=robot_vertices[:, 0],
                    y=robot_vertices[:, 1],
                    z=robot_vertices[:, 2],
                    i=robot_faces[:, 0],
                    j=robot_faces[:, 1],
                    k=robot_faces[:, 2],
                    color='orange',
                    opacity=0.4,
                    name='Transformed Robot Mesh',
                    showlegend=True
                ))
                print(f"Robot mesh created: {len(robot_vertices)} vertices, {len(robot_faces)} faces")
            else:
                # Fallback: point cloud로 표시
                fig.add_trace(go.Scatter3d(
                    x=robot_transformed_sampled[:, 0],
                    y=robot_transformed_sampled[:, 1],
                    z=robot_transformed_sampled[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='orange',
                        opacity=0.7,
                        symbol='square'
                    ),
                    name='Transformed Robot Points',
                    showlegend=True
                ))
        except Exception as e:
            print(f"Failed to create robot mesh: {e}")
            # Fallback: point cloud로 표시
            fig.add_trace(go.Scatter3d(
                x=robot_transformed_sampled[:, 0],
                y=robot_transformed_sampled[:, 1],
                z=robot_transformed_sampled[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color='orange',
                    opacity=0.7,
                    symbol='square'
                ),
                name='Transformed Robot Points',
                showlegend=True
            ))
        
        # 변환 벡터 표시 (화살표)
        if robot_points_original is not None and len(robot_points_original) > 0:
            # 몇 개의 대표적인 포인트에 대해서만 변환 벡터 표시
            num_arrows = min(10, len(robot_points_original), len(robot_points_transformed))
            arrow_indices = np.random.choice(min(len(robot_points_original), len(robot_points_transformed)), 
                                           num_arrows, replace=False)
            
            for idx in arrow_indices:
                start = robot_points_original[idx]
                end = robot_points_transformed[idx]
                
                # 화살표를 여러 선분으로 표현
                fig.add_trace(go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode='lines',
                    line=dict(
                        color='purple',
                        width=3
                    ),
                    name='Robot Transformation' if idx == arrow_indices[0] else None,
                    showlegend=True if idx == arrow_indices[0] else False
                ))
    
    # 2. 쿼리 포인트 표시 (더 크고 명확하게)
    fig.add_trace(go.Scatter3d(
        x=[query_point[0]],
        y=[query_point[1]],
        z=[query_point[2]],
        mode='markers',
        marker=dict(
            size=12,
            color='red',
            symbol='diamond',
            line=dict(width=2, color='darkred')
        ),
        name='Query Point',
        showlegend=True
    ))
    
    # 3. 원점 (카메라 위치) 표시
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            symbol='circle',
            line=dict(width=2, color='darkblue')
        ),
        name='Camera Origin',
        showlegend=True
    ))
    
    # 4. 각 viewpoint와 Line of Sight 표시
    colors = ['green', 'blue', 'orange', 'purple', 'brown']
    
    for i, (viewpoint, result) in enumerate(zip(candidate_viewpoints, results)):
        # Viewpoint 표시 (더 크고 명확하게)
        viewpoint_color = 'green' if result['visible_robot'] else 'red'
        fig.add_trace(go.Scatter3d(
            x=[viewpoint[0]],
            y=[viewpoint[1]],
            z=[viewpoint[2]],
            mode='markers',
            marker=dict(
                size=8,
                color=viewpoint_color,
                symbol='circle',
                line=dict(width=2, color='darkgreen' if result['visible_robot'] else 'darkred')
            ),
            name=f'Viewpoint {i+1} ({result["visible_robot"] and "VISIBLE" or "OCCLUDED"})',
            showlegend=True
        ))
        
        # Line of Sight 표시 (더 두껍고 명확하게)
        los_color = 'green' if result['visible_robot'] else 'red'
        los_style = 'solid' if result['visible_robot'] else 'dash'
        
        fig.add_trace(go.Scatter3d(
            x=[viewpoint[0], query_point[0]],
            y=[viewpoint[1], query_point[1]],
            z=[viewpoint[2], query_point[2]],
            mode='lines',
            line=dict(
                color=los_color,
                width=6,
                dash=los_style
            ),
            name=f'LoS {i+1} ({result["visible_robot"] and "VISIBLE" or "OCCLUDED"})',
            showlegend=True
        ))
        
        # Hit point 표시 (OCCLUDED인 경우)
        if not result['visible_robot']:
            hit_distance = result['hit_distance_robot']
            direction = query_point - viewpoint
            direction = direction / np.linalg.norm(direction)
            hit_point = viewpoint + direction * hit_distance
            
            fig.add_trace(go.Scatter3d(
                x=[hit_point[0]],
                y=[hit_point[1]],
                z=[hit_point[2]],
                mode='markers',
                marker=dict(
                    size=6,
                    color='orange',
                    symbol='x'
                ),
                name=f'Hit Point {i+1}',
                showlegend=True
            ))
            
            # Hit point에서 쿼리 포인트까지의 선 (가려진 부분)
            fig.add_trace(go.Scatter3d(
                x=[hit_point[0], query_point[0]],
                y=[hit_point[1], query_point[1]],
                z=[hit_point[2], query_point[2]],
                mode='lines',
                line=dict(
                    color='red',
                    width=3,
                    dash='dot'
                ),
                name=f'Occluded Part {i+1}',
                showlegend=True
            ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text='3D Line of Sight Analysis: Scene Mesh + Robot Transform Visualization',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='lightgray'
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # HTML 파일로 저장
    pyo.plot(fig, filename=save_path, auto_open=False)
    print(f"3D visualization saved to: {save_path}")



def create_esdf_multi_viewpoint_set_3d_visualization(mappers, query_point, viewpoint_matrix, visibility_results, final_rewards, save_path, voxel_size=0.02, max_distance=1.0):
    """
    M개의 viewpoint set을 모두 보여주는 ESDF 기반 3D 시각화
    
    Args:
        mappers: L개의 mapper 리스트
        query_point: 쿼리 포인트
        viewpoint_matrix: [M, L, 3] 모양의 viewpoint 매트릭스
        visibility_results: [M, L] 모양의 visibility 결과
        final_rewards: [M] 모양의 최종 reward 배열
        save_path: 저장 경로
        voxel_size: ESDF 시각화용 voxel 크기
        max_distance: 최대 거리 (미터)
    """
    M, L = viewpoint_matrix.shape[:2]
    
    # 서브플롯 생성 (M개의 viewpoint set을 각각 표시)
    fig = make_subplots(
        rows=1, cols=M,
        specs=[[{'type': 'scatter3d'} for _ in range(M)]],
        subplot_titles=[f'Viewpoint Set {i+1} (Reward: {final_rewards[i]:.1f})' for i in range(M)],
        horizontal_spacing=0.05
    )
    
    # 각 viewpoint set에 대해 ESDF 기반 시각화
    for set_idx in range(M):
        # 첫 번째 mapper를 대표로 사용 (실제로는 각 set마다 다른 mapper를 사용해야 함)
        mapper = mappers[0]  # 간단히 첫 번째 mapper 사용
        
        try:
            # TSDF layer를 사용해서 AABB 범위를 얻음
            tsdf_layer = mapper.tsdf_layer_view()
            
            # 간단한 그리드 생성 (성능을 위해 해상도 제한)
            min_block_idx, max_block_idx = tsdf_layer.get_block_limits()
            aabb_min_vox = min_block_idx * tsdf_layer.block_dim_in_voxels
            aabb_max_vox = (max_block_idx + 1) * tsdf_layer.block_dim_in_voxels
            
            # 해상도 제한 (너무 많은 포인트 방지)
            max_resolution = 20
            x_step = max(1, (aabb_max_vox[0] - aabb_min_vox[0]) // max_resolution)
            y_step = max(1, (aabb_max_vox[1] - aabb_min_vox[1]) // max_resolution)
            z_step = max(1, (aabb_max_vox[2] - aabb_min_vox[2]) // max_resolution)
            
            x_linspace = torch.arange(aabb_min_vox[0], aabb_max_vox[0] + 1, x_step, dtype=torch.int)
            y_linspace = torch.arange(aabb_min_vox[1], aabb_max_vox[1] + 1, y_step, dtype=torch.int)
            z_linspace = torch.arange(aabb_min_vox[2], aabb_max_vox[2] + 1, z_step, dtype=torch.int)
            
            x_grid, y_grid, z_grid = torch.meshgrid(x_linspace, y_linspace, z_linspace, indexing='ij')
            query_grid_xyz_vox = torch.stack([x_grid, y_grid, z_grid], dim=-1)
            query_grid_xyz_m = (query_grid_xyz_vox + 0.5) * tsdf_layer.voxel_size()
            query_grid_xyz_m = query_grid_xyz_m.cuda()
            
            # ESDF 쿼리 수행
            sdf_values = mapper.query_differentiable_layer(
                QueryType.ESDF, 
                query_grid_xyz_m.reshape(-1, 3)
            )
            sdf_values = sdf_values.reshape(query_grid_xyz_m.shape[:-1])
            
            # 유효한 쿼리 마스크 생성
            from nvblox_torch.constants import constants
            valid_mask = torch.logical_not(sdf_values == constants.esdf_unknown_distance())
            distance_mask = torch.abs(sdf_values) <= max_distance
            valid_mask = valid_mask & distance_mask
            
            # 유효한 포인트들만 추출
            valid_points = query_grid_xyz_m[valid_mask].cpu().numpy()
            valid_distances = sdf_values[valid_mask].cpu().numpy()
            
            # ESDF 거리 필드 시각화
            if len(valid_points) > 0:
                # 거리 값에 따른 색상 매핑
                normalized_distances = (valid_distances + max_distance) / (2 * max_distance)
                normalized_distances = np.clip(normalized_distances, 0, 1)
                
                colors = []
                for dist in normalized_distances:
                    if dist < 0.5:  # 음수 거리 (물체 내부)
                        r, g, b = 1.0, dist * 2, 0.0
                    else:  # 양수 거리 (물체 외부)
                        r, g, b = 2.0 - dist * 2, 2.0 - dist * 2, (dist - 0.5) * 2
                    colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
                
                # ESDF voxel grid 시각화
                fig.add_trace(go.Scatter3d(
                    x=valid_points[:, 0],
                    y=valid_points[:, 1],
                    z=valid_points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=colors,
                        opacity=0.4,
                        line=dict(width=0)
                    ),
                    name=f'ESDF {set_idx+1}',
                    showlegend=False
                ), row=1, col=set_idx+1)
            
        except Exception as e:
            print(f"Failed to create ESDF visualization for set {set_idx+1}: {e}")
            # Fallback: 빈 시각화
            pass
        
        # 쿼리 포인트 표시
        fig.add_trace(go.Scatter3d(
            x=[query_point[0]], y=[query_point[1]], z=[query_point[2]],
            mode='markers',
            marker=dict(size=6, color='red', symbol='diamond'),
            name=f'Query {set_idx+1}',
            showlegend=False
        ), row=1, col=set_idx+1)
        
        # 현재 set의 viewpoints 표시
        for viewpoint_idx in range(L):
            viewpoint = viewpoint_matrix[set_idx, viewpoint_idx]
            visible = visibility_results[set_idx, viewpoint_idx]
            
            viewpoint_color = 'green' if visible else 'red'
            fig.add_trace(go.Scatter3d(
                x=[viewpoint[0]], y=[viewpoint[1]], z=[viewpoint[2]],
                mode='markers',
                marker=dict(size=4, color=viewpoint_color, symbol='circle'),
                name=f'VP{viewpoint_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
            
            # Line of Sight 표시
            los_color = 'green' if visible else 'red'
            los_style = 'solid' if visible else 'dash'
            
            fig.add_trace(go.Scatter3d(
                x=[viewpoint[0], query_point[0]],
                y=[viewpoint[1], query_point[1]],
                z=[viewpoint[2], query_point[2]],
                mode='lines',
                line=dict(color=los_color, width=2, dash=los_style),
                name=f'LoS{viewpoint_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text=f'ESDF-based Multi-Viewpoint Set Analysis (M={M}, L={L})',
            x=0.5,
            font=dict(size=18)
        ),
        width=400 * M,
        height=600,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    # 각 서브플롯의 scene 설정
    for i in range(1, M+1):
        fig.update_scenes(
            xaxis_title='X (m)',
            yaxis_title='Y (m)', 
            zaxis_title='Z (m)',
            aspectmode='data',
            row=1, col=i
        )
    
    # HTML 파일로 저장
    import plotly.offline as pyo
    pyo.plot(fig, filename=save_path, auto_open=False)
    print(f"ESDF-based multi-viewpoint set 3D visualization saved to: {save_path}")


def create_multi_viewpoint_set_3d_visualization(meshes, query_point, viewpoint_matrix, visibility_results, final_rewards, save_path):
    """
    M개의 viewpoint set을 모두 보여주는 3D 시각화
    
    Args:
        meshes: L개의 mesh 리스트
        query_point: 쿼리 포인트
        viewpoint_matrix: [M, L, 3] 모양의 viewpoint 매트릭스
        visibility_results: [M, L] 모양의 visibility 결과
        final_rewards: [M] 모양의 최종 reward 배열
        save_path: 저장 경로
    """
    M, L = viewpoint_matrix.shape[:2]
    
    # 서브플롯 생성 (M개의 viewpoint set을 각각 표시)
    fig = make_subplots(
        rows=1, cols=M,
        specs=[[{'type': 'scatter3d'} for _ in range(M)]],
        subplot_titles=[f'Viewpoint Set {i+1} (Reward: {final_rewards[i]:.1f})' for i in range(M)],
        horizontal_spacing=0.05
    )
    
    # 각 viewpoint set에 대해 시각화
    for set_idx in range(M):
        # 첫 번째 mesh를 대표로 사용 (실제로는 각 set마다 다른 mesh를 사용해야 함)
        mesh = meshes[0]  # 간단히 첫 번째 mesh 사용
        
        # 메시 데이터 추출
        if hasattr(mesh, 'vertices') and callable(mesh.vertices):
            vertices = mesh.vertices().cpu().numpy()
            faces = mesh.triangles().cpu().numpy()
        else:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
        
        # 메시를 샘플링 (성능을 위해) - 면의 수만 제한
        if len(faces) > 5000:
            # 면의 수를 제한 (vertex는 그대로 유지)
            face_indices = np.random.choice(len(faces), 5000, replace=False)
            faces_sampled = faces[face_indices]
        else:
            faces_sampled = faces
        
        # 메시 표시 (실제 mesh로)
        if len(faces_sampled) > 0:
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces_sampled[:, 0],
                j=faces_sampled[:, 1],
                k=faces_sampled[:, 2],
                color='lightblue',
                opacity=0.3,
                name=f'Scene {set_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
        else:
            # Fallback: point cloud
            fig.add_trace(go.Scatter3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                mode='markers',
                marker=dict(size=2, color='lightblue', opacity=0.3),
                name=f'Scene {set_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
        
        # 쿼리 포인트 표시
        fig.add_trace(go.Scatter3d(
            x=[query_point[0]], y=[query_point[1]], z=[query_point[2]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name=f'Query {set_idx+1}',
            showlegend=False
        ), row=1, col=set_idx+1)
        
        # 현재 set의 viewpoints 표시
        for viewpoint_idx in range(L):
            viewpoint = viewpoint_matrix[set_idx, viewpoint_idx]
            visible = visibility_results[set_idx, viewpoint_idx]
            
            viewpoint_color = 'green' if visible else 'red'
            fig.add_trace(go.Scatter3d(
                x=[viewpoint[0]], y=[viewpoint[1]], z=[viewpoint[2]],
                mode='markers',
                marker=dict(size=6, color=viewpoint_color, symbol='circle'),
                name=f'VP{viewpoint_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
            
            # Line of Sight 표시
            los_color = 'green' if visible else 'red'
            los_style = 'solid' if visible else 'dash'
            
            fig.add_trace(go.Scatter3d(
                x=[viewpoint[0], query_point[0]],
                y=[viewpoint[1], query_point[1]],
                z=[viewpoint[2], query_point[2]],
                mode='lines',
                line=dict(color=los_color, width=2, dash=los_style),
                name=f'LoS{viewpoint_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text=f'Multi-Viewpoint Set Analysis (M={M}, L={L})',
            x=0.5,
            font=dict(size=18)
        ),
        width=400 * M,
        height=600,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    # 각 서브플롯의 scene 설정
    for i in range(1, M+1):
        fig.update_scenes(
            xaxis_title='X (m)',
            yaxis_title='Y (m)', 
            zaxis_title='Z (m)',
            aspectmode='data',
            row=1, col=i
        )
    
    # HTML 파일로 저장
    pyo.plot(fig, filename=save_path, auto_open=False)
    print(f"Multi-viewpoint set 3D visualization saved to: {save_path}")


# ========= 로봇 변환을 고려한 nvblox 메시 생성 =========
def create_mapper_with_robot_transformation_nvblox(rgb, depth, robot_mask, robot_se3_transform, 
                                               camera_intrinsics, voxel_size=0.005, max_integration_distance=5.0, mapper=None):
    """
    로봇의 SE(3) 변환을 고려한 nvblox mapper 생성 (mesh 없이)
    
    Args:
        rgb: RGB 이미지 (H, W, 3)
        depth: 깊이 이미지 (H, W) 미터 단위
        robot_mask: 로봇 segmentation mask (H, W) boolean array
        robot_se3_transform: 로봇의 SE(3) 변환 행렬 (4, 4)
        camera_intrinsics: 카메라 내부 파라미터 (3, 3)
        voxel_size: voxel 크기 (미터)
        max_integration_distance: 최대 통합 거리 (미터)
        mapper: 기존 mapper (None이면 새로 생성)
    
    Returns:
        mapper: nvblox Mapper
        robot_points_transformed: 변환된 로봇 포인트들
    """
    print("Creating nvblox mapper with robot transformation...")
    
    # 1) 현재 RGBD에서 로봇 부분 제거
    rgb_without_robot, depth_without_robot = remove_robot_from_rgbd_direct(rgb, depth, robot_mask)
    
    # 2) 로봇 포인트들을 SE(3) 변환 (robot_points_original은 이미 외부에서 생성됨)
    robot_points_transformed = transform_robot_points_direct(depth, robot_mask, robot_se3_transform, camera_intrinsics)
    
    # 3) 변환된 로봇을 새로운 RGBD에 추가
    rgb_future, depth_future = add_transformed_robot_to_rgbd_direct(
        rgb_without_robot, depth_without_robot, robot_points_transformed, camera_intrinsics
    )
    
    # 4) Transform된 RGBD 이미지 저장 (디버깅용)
    import os
    debug_dir = "robot_transform_debug"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Transform 정보를 파일명에 포함
    transform_translation = robot_se3_transform[:3, 3]
    transform_str = f"t{transform_translation[0]:.2f}_{transform_translation[1]:.2f}_{transform_translation[2]:.2f}"
    
    # 4x2 그리드로 모든 단계 저장
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 원본 RGB
    axes[0,0].imshow(rgb)
    axes[0,0].set_title('Original RGB')
    axes[0,0].axis('off')
    
    # 원본 Depth
    im1 = axes[0,1].imshow(depth, cmap='magma')
    axes[0,1].set_title('Original Depth')
    axes[0,1].axis('off')
    plt.colorbar(im1, ax=axes[0,1], shrink=0.7)
    
    # Robot 제거된 RGB
    axes[0,2].imshow(rgb_without_robot)
    axes[0,2].set_title('RGB without Robot')
    axes[0,2].axis('off')
    
    # Robot 제거된 Depth
    im2 = axes[0,3].imshow(depth_without_robot, cmap='magma')
    axes[0,3].set_title('Depth without Robot')
    axes[0,3].axis('off')
    plt.colorbar(im2, ax=axes[0,3], shrink=0.7)
    
    # Transform된 RGB
    axes[1,0].imshow(rgb_future)
    axes[1,0].set_title(f'Transformed RGB (t={transform_translation})')
    axes[1,0].axis('off')
    
    # Transform된 Depth
    im3 = axes[1,1].imshow(depth_future, cmap='magma')
    axes[1,1].set_title(f'Transformed Depth (t={transform_translation})')
    axes[1,1].axis('off')
    plt.colorbar(im3, ax=axes[1,1], shrink=0.7)
    
    # Robot points 비교 (빈 공간 활용)
    axes[1,2].text(0.5, 0.5, f'Robot Transform\nTranslation: {transform_translation}\n\nTransformed Robot Center:\n{np.mean(robot_points_transformed, axis=0)}', 
                   ha='center', va='center', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1,2].set_title('Transform Info')
    axes[1,2].axis('off')
    
    # Robot points 개수 정보
    axes[1,3].text(0.5, 0.5, f'Robot Points Count:\nTransformed: {len(robot_points_transformed)}\n\nDepth Pixels:\nOriginal: {np.sum(depth > 0)}\nWithout Robot: {np.sum(depth_without_robot > 0)}\nWith Transform: {np.sum(depth_future > 0)}', 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1,3].set_title('Statistics')
    axes[1,3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, f"robot_transform_{transform_str}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved robot transform visualization: robot_transform_{transform_str}.png")
    
    # 5) nvblox mapper 생성 (mesh 없이)
    print(f"  Creating mapper from RGBD with {np.sum(depth_future > 0)} valid depth pixels")
    mapper = create_mapper_with_nvblox(depth_future, rgb_future, camera_intrinsics, voxel_size, max_integration_distance, mapper=mapper)
    
    # 디버깅: 생성된 mapper 정보
    if mapper is not None:
        print(f"  Generated mapper successfully")
    else:
        print("  ERROR: Failed to generate mapper!")
    
    return mapper, robot_points_transformed
    

def remove_robot_from_rgbd_direct(rgb, depth, robot_mask):
    """
    RGB와 depth에서 로봇 부분을 직접 제거 (numpy array 기반)
    """
    # Deep copy로 원본 보존
    rgb_clean = copy.deepcopy(rgb)
    depth_clean = copy.deepcopy(depth)
    
    # 로봇 마스크가 True인 부분을 0으로 설정
    rgb_clean[robot_mask] = 0
    depth_clean[robot_mask] = 0
    
    return rgb_clean, depth_clean


def transform_robot_points_direct(depth, robot_mask, robot_se3_transform, camera_intrinsics):
    """
    로봇 포인트들을 SE(3) 변환 (numpy array 기반)
    """
    # 로봇 마스크가 True인 픽셀들의 3D 좌표 계산
    H, W = robot_mask.shape
    robot_pixels = np.argwhere(robot_mask)
    
    if len(robot_pixels) == 0:
        print("Warning: No robot pixels found in mask")
        return np.array([])
    
    # 픽셀 좌표를 3D 좌표로 변환
    robot_points_3d = []
    valid_depth_count = 0
    
    for y, x in robot_pixels:
        depth_val = depth[y, x]
        if depth_val > 0:
            valid_depth_count += 1
            # 픽셀을 3D로 unproject
            fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
            cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
            
            z = depth_val
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy
            
            robot_points_3d.append([x_3d, y_3d, z])
    
    if len(robot_points_3d) == 0:
        print("Warning: No valid depth values found for robot pixels")
        return np.array([])
    
    robot_points_3d = np.array(robot_points_3d)
    print(f"  Found {len(robot_pixels)} robot pixels, {valid_depth_count} with valid depth")
    print(f"  Robot transform: translation={robot_se3_transform[:3, 3]}")
    
    # SE(3) 변환 적용
    robot_points_homo = np.hstack([robot_points_3d, np.ones((len(robot_points_3d), 1))])
    robot_points_transformed_homo = (robot_se3_transform @ robot_points_homo.T).T
    robot_points_transformed = robot_points_transformed_homo[:, :3]
    
    # 변환 전후 비교
    original_center = np.mean(robot_points_3d, axis=0)
    transformed_center = np.mean(robot_points_transformed, axis=0)
    print(f"  Original robot center: {original_center}")
    print(f"  Transformed robot center: {transformed_center}")
    print(f"  Translation applied: {transformed_center - original_center}")
    
    return robot_points_transformed


def add_transformed_robot_to_rgbd_direct(rgb_without_robot, depth_without_robot, robot_points_transformed, camera_intrinsics):
    """
    변환된 로봇 포인트들을 새로운 RGBD에 직접 추가 (numpy array 기반)
    더 매끄러운 mesh를 위해 주변 픽셀도 함께 채움
    """
    if len(robot_points_transformed) == 0:
        print("Warning: No transformed robot points to add")
        return rgb_without_robot, depth_without_robot
    
    # Deep copy로 원본 보존
    rgb_future = copy.deepcopy(rgb_without_robot) # value: [0, 255]
    depth_future = copy.deepcopy(depth_without_robot)
    
    # 변환된 로봇 포인트들을 이미지 좌표로 project
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    added_pixels = 0
    valid_points = 0
    
    # 매끄러운 mesh를 위한 주변 픽셀 채우기 설정
    fill_radius = 2  # 주변 2픽셀 반경까지 채움
    
    for point in robot_points_transformed:
        x_3d, y_3d, z_3d = point
        
        if z_3d > 0:
            valid_points += 1
            # 3D를 픽셀로 project
            x_pixel = int(fx * x_3d / z_3d + cx)
            y_pixel = int(fy * y_3d / z_3d + cy)
            
            # 이미지 범위 내에 있는지 확인
            if 0 <= x_pixel < depth_future.shape[1] and 0 <= y_pixel < depth_future.shape[0]:
                # 주변 픽셀들도 함께 채우기 (매끄러운 mesh를 위해)
                for dy in range(-fill_radius, fill_radius + 1):
                    for dx in range(-fill_radius, fill_radius + 1):
                        nx, ny = x_pixel + dx, y_pixel + dy
                        
                        # 이미지 범위 내 확인
                        if 0 <= nx < depth_future.shape[1] and 0 <= ny < depth_future.shape[0]:
                            # 거리에 따른 depth 보정 (원근감 고려)
                            distance_factor = np.sqrt(dx*dx + dy*dy) / fill_radius
                            adjusted_depth = z_3d + distance_factor * 0.01  # 1cm씩 증가
                            
                            # 기존 depth가 0이거나 robot이 더 가까우면 업데이트
                            if depth_future[ny, nx] == 0 or depth_future[ny, nx] > adjusted_depth:
                                depth_future[ny, nx] = adjusted_depth
                                # 로봇 색상 (회색)으로 설정 [0, 255] 범위
                                rgb_future[ny, nx] = [128, 128, 128]  # 0.5 * 255 = 127.5 ≈ 128
                                added_pixels += 1
                            else:
                                # 기존 depth가 더 가까워도 robot 영역임을 표시하기 위해 색상만 변경 [0, 255] 범위
                                rgb_future[ny, nx] = [128, 128, 128]  # 0.5 * 255 = 127.5 ≈ 128
                                added_pixels += 1
    
    print(f"  Added {added_pixels} robot pixels from {valid_points} valid transformed points")
    
    # 매끄러운 mesh를 위한 morphological operations 적용
    if added_pixels > 0:
        try:
            from scipy import ndimage
            
            # Robot 영역 마스크 생성 (회색 픽셀들)
            robot_mask = np.all(rgb_future == [128, 128, 128], axis=2)
            
            # Morphological closing (구멍 메우기)
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            robot_mask_closed = ndimage.binary_closing(robot_mask, structure=kernel)
            
            # Closing으로 추가된 영역에 depth 값 보간
            new_robot_pixels = robot_mask_closed & ~robot_mask
            if np.any(new_robot_pixels):
                # 새로운 robot 픽셀들에 대해 주변 depth 값으로 보간
                for y, x in np.argwhere(new_robot_pixels):
                    # 주변 유효한 depth 값들 찾기
                    y_min, y_max = max(0, y-2), min(depth_future.shape[0], y+3)
                    x_min, x_max = max(0, x-2), min(depth_future.shape[1], x+3)
                    neighborhood = depth_future[y_min:y_max, x_min:x_max]
                    valid_depths = neighborhood[neighborhood > 0]
                    
                    if len(valid_depths) > 0:
                        # 주변 depth 값의 평균으로 보간
                        interpolated_depth = np.mean(valid_depths)
                        depth_future[y, x] = interpolated_depth
                        rgb_future[y, x] = [128, 128, 128]
                        added_pixels += 1
                
                print(f"  Morphological closing added {np.sum(new_robot_pixels)} pixels")
        except ImportError:
            print("  scipy not available, skipping morphological operations")
    
    # 디버깅: 변환된 robot points의 범위와 추가된 픽셀 분포 확인
    if len(robot_points_transformed) > 0:
        robot_min = np.min(robot_points_transformed, axis=0)
        robot_max = np.max(robot_points_transformed, axis=0)
        print(f"  Transformed robot 3D bounds: min={robot_min}, max={robot_max}")
        
        # 추가된 픽셀들의 depth 범위 확인
        if added_pixels > 0:
            added_depth_values = depth_future[depth_future != depth_without_robot]
            if len(added_depth_values) > 0:
                print(f"  Added depth range: {np.min(added_depth_values):.3f} to {np.max(added_depth_values):.3f}m")
            else:
                print("  WARNING: No depth values were actually added!")
        else:
            print("  WARNING: No robot pixels were added to RGBD!")
    
    return rgb_future, depth_future


def create_robot_segmentation_mask_demo(rgb_shape, robot_bbox=None):
    """
    데모용 로봇 segmentation mask 생성
    실제로는 SAM이나 다른 segmentation 모델을 사용해야 함
    """
    H, W = rgb_shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    
    if robot_bbox is None:
        # 데모용: 이미지 크기에 비례하는 로봇 크기 설정
        # 원본 640x480에서 60x60이었으므로, 비례적으로 조정
        base_robot_size = 60
        base_image_size = 640  # 원본 이미지의 가로 크기
        
        # 현재 이미지 크기에 비례하여 로봇 크기 계산
        robot_size = max(int(base_robot_size * W / base_image_size), 10)  # 최소 10픽셀
        center_x, center_y = W//2, H//2  # 이미지 중앙
        # 정사각형의 오른쪽 변 중심이 이미지 중앙이 되도록 조정
        robot_bbox = (center_x - robot_size//2, center_y - robot_size//2, 
                     center_x + robot_size//2, center_y + robot_size//2)
    
    x1, y1, x2, y2 = robot_bbox
    # 경계 체크
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    mask[y1:y2, x1:x2] = True
    
    return mask


def create_robot_se3_transforms_demo(candidate_viewpoints):
    """
    데모용 로봇 SE(3) 변환들을 생성 (각 viewpoint별로)
    실제로는 로봇의 다음 스텝 위치를 예측해야 함
    
    Args:
        candidate_viewpoints: viewpoint들의 리스트
    
    Returns:
        se3_transforms: 각 viewpoint에 대응하는 SE(3) 변환 행렬들의 리스트
    """
    se3_transforms = []
    
    # 시작점과 끝점 정의 (linear interpolation을 위해)
    start_translation = np.array([0.0, 0.0, 0.0])  # 시작점 (원점)
    end_translation = np.array([0.1, 0.0, -0.2])   # 끝점 (+x 10cm, -z 20cm)
    
    
    for i, viewpoint in enumerate(candidate_viewpoints):
        # Linear interpolation between start and end
        alpha = i / (len(candidate_viewpoints) - 1) if len(candidate_viewpoints) > 1 else 0
        translation = start_translation + alpha * (end_translation - start_translation)
        
        # 회전 없음 (단순 이동만)
        rotation_matrix = np.eye(3)
        
        # SE(3) 변환 행렬 생성
        se3_transform = np.eye(4)
        se3_transform[:3, :3] = rotation_matrix
        se3_transform[:3, 3] = translation
        
        se3_transforms.append(se3_transform)
    
    return se3_transforms


# ========= 메인 =========
def main():
    print("=== ESDF-based Line-of-Sight Visibility Test with Robot Transformation ===")
    print("Using nvblox ESDF layer for ray marching visibility check")
    
    # 전체 실행 시간 측정
    total_start_time = time.time()
    
    # 1) 입력 로드
    print("\n=== 1. Loading RGB-D Data ===")
    load_start_time = time.time()
    rgb, depth_raw = load_rgbd_from_zarr(BUFFER_PATH, EPISODE_IDX, FRAME_IDX, DEPTH_SCALE)
    H, W = depth_raw.shape
    load_end_time = time.time()
    print(f"Loaded RGB: {rgb.shape}, Depth: {depth_raw.shape}")
    print(f"Data loading time: {load_end_time - load_start_time:.4f} seconds")
    
    # 전역 변수로 설정 (RGB point cloud 생성용)
    global rgb_image_original, depth_image_original, camera_intrinsics_original
    rgb_image_original = rgb
    depth_image_original = depth_raw
    camera_intrinsics_original = K  # 원본 카메라 내부 파라미터

    # 1.5) 이미지 리사이즈 (옵션) - UniDepth 사용시에는 RGB만 리사이즈
    if RESIZE:
        print(f"\n=== 1.5. Resizing RGB to {RESIZE_SIZE} ===")
        resize_start_time = time.time()
        rgb_resized, _, scale_factor_x, scale_factor_y = resize_image_and_depth(rgb, depth_raw, RESIZE_SIZE)
        K_adjusted = adjust_camera_intrinsics(K, scale_factor_x, scale_factor_y)
        resize_end_time = time.time()
        print(f"Resized RGB: {rgb_resized.shape}, Original depth: {depth_raw.shape}")
        print(f"Scale factors: x={scale_factor_x:.4f}, y={scale_factor_y:.4f}")
        print(f"Adjusted camera intrinsics: fx={K_adjusted[0,0]:.2f}, fy={K_adjusted[1,1]:.2f}, cx={K_adjusted[0,2]:.2f}, cy={K_adjusted[1,2]:.2f}")
        print(f"RGB resize time: {resize_end_time - resize_start_time:.4f} seconds")
    else:
        K_adjusted = K
        rgb_resized = rgb

    if K is None:
        raise ValueError("K (intrinsics)가 None 입니다. 코드 상단의 K를 사용자의 카메라 파라미터로 채워주세요.")

    # 2) Depth 처리
    print("\n=== 2. Depth Processing ===")
    depth_start_time = time.time()
    
    if USE_MONODEPTH:
        print("Using UniDepth for depth estimation...")
        
        # UniDepth 모델 로드
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        name = f"unidepth-v2-vit{MODEL_TYPE}14"
        model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
        model.interpolation_mode = "bilinear"
        model = model.to(device).eval()
        
        # 카메라 설정 (원본 크기로 UniDepth 실행)
        intrinsics_torch = torch.from_numpy(K).float()
        camera = Pinhole(K=intrinsics_torch.unsqueeze(0))
        if isinstance(model, (UniDepthV2old, UniDepthV1)):
            camera = camera.K.squeeze(0)
        
        # Depth 추정 (원본 크기 RGB 사용)
        print(f"Running UniDepth on original RGB: {rgb.shape}")
        depth_pred_original = get_depth_with_unidepth(rgb, K, model, camera)
        
        # Raw depth로 스케일링 (원본 크기)
        depth_m_original = scale_depth_with_raw(depth_pred_original, depth_raw, K)
        
        # 리사이즈가 필요한 경우 depth만 리사이즈
        if RESIZE:
            print(f"Resizing depth from {depth_m_original.shape} to {RESIZE_SIZE}")
            import cv2
            depth_m = cv2.resize(depth_m_original, RESIZE_SIZE, interpolation=cv2.INTER_LINEAR)
        else:
            depth_m = depth_m_original
        
        print(f"Final depth shape: {depth_m.shape}")
    else:
        print("Using raw depth data...")
        if RESIZE:
            import cv2
            depth_m = cv2.resize(depth_raw, RESIZE_SIZE, interpolation=cv2.INTER_LINEAR)
        else:
            depth_m = depth_raw
    
    depth_end_time = time.time()
    print(f"Depth processing time: {depth_end_time - depth_start_time:.4f} seconds")

    # 3) 월드 좌표계 = 원 카메라0 좌표계로 가정
    R0 = np.eye(3, dtype=np.float64)
    t0 = np.zeros(3, dtype=np.float64)

    # 4) 쿼리 픽셀 & 3D점
    print("\n=== 3. Query Point Selection ===")
    query_start_time = time.time()
    
    uq, vq = pick_query_pixel(depth_m)
    dq = float(depth_m[vq, uq])
    fx, fy, cx, cy = K_adjusted[0,0], K_adjusted[1,1], K_adjusted[0,2], K_adjusted[1,2]
    xq = (uq - cx) * dq / fx
    yq = (vq - cy) * dq / fy
    Xc0_q = np.array([xq, yq, dq], dtype=np.float64)
    Xw_q = R0.T @ (Xc0_q - t0)

    query_end_time = time.time()
    print(f"[Query] pixel=({uq},{vq}), depth={dq:.4f} m, world_pos={Xw_q}")
    print(f"Query point selection time: {query_end_time - query_start_time:.4f} seconds")

    # 5) 각 mapper별로 로봇 변환을 고려한 nvblox mapper 생성
    print("\n=== 4. nvblox Mapper Generation with Robot Transformation ===")
    mapper_start_time = time.time()
    
    # 데모용 로봇 segmentation mask 생성
    robot_mask = create_robot_segmentation_mask_demo(rgb_resized.shape)
    print(f"Robot mask created: {np.sum(robot_mask)} pixels")
    
    # L개의 로봇 SE(3) 변환 생성 (각 column index i에 대응)
    robot_se3_transforms = create_robot_se3_transforms_demo(CANDIDATE_VIEWPOINTS)
    print(f"Generated {len(robot_se3_transforms)} robot SE(3) transforms")
    for i, transform in enumerate(robot_se3_transforms):
        print(f"  Transform {i+1}: translation={transform[:3, 3]}")
    
    # L개의 mapper 생성 (각 column index i에 대응하는 robot transform으로)
    mappers_with_robot = []
    robot_points_transformed_list = []
    
    # robot_points_original을 한번만 생성 (모든 mapper에서 공통으로 사용)
    print(f"  Creating robot_points_original once for all mappers...")
    robot_points_original = transform_robot_points_direct(depth_m, robot_mask, np.eye(4), K_adjusted)
    print(f"  robot_points_original created: {len(robot_points_original)} points")
    
    # 각 mapper마다 독립적인 mapper 생성 (robot transform이 제대로 반영되도록)
    for i, robot_transform in enumerate(robot_se3_transforms):
        print(f"  Creating mapper {i+1} with robot transform: translation={robot_transform[:3, 3]}")
        start = time.time()
        mapper_robot, robot_points_transformed = create_mapper_with_robot_transformation_nvblox(
            rgb_resized, depth_m, robot_mask, robot_transform, K_adjusted, voxel_size=VOXEL_SIZE, max_integration_distance=5.0, mapper=None  # 독립적인 mapper 사용
        )
        print(f"  nvblox robot transformed mapper creation time: {time.time() - start:.4f} seconds")
        mappers_with_robot.append(mapper_robot)
        robot_points_transformed_list.append(robot_points_transformed)
        print(f"  Mapper {i+1}: Created successfully")
        
        # 디버깅: 변환된 로봇 포인트들의 중심점 확인
        if len(robot_points_transformed) > 0:
            transformed_center = np.mean(robot_points_transformed, axis=0)
            print(f"  Transformed robot center: {transformed_center}")
        
        
    # Mapper 객체들 디버깅: 각각 다른 인스턴스인지 확인
    print(f"\n=== Mapper Objects Debugging ===")
    print(f"Total mappers created: {len(mappers_with_robot)}")
    
    for i, mapper in enumerate(mappers_with_robot):
        print(f"  Mapper {i+1}:")
        print(f"    - Object ID: {id(mapper)}")
        print(f"    - Type: {type(mapper)}")
        
        # Mapper가 갖고 있는 depth 정보 확인 (TSDF layer에서)
        try:
            # TSDF layer의 voxel 정보 확인
            tsdf_layer = mapper.tsdf_layer_view()
            print(f"    - TSDF Layer: {tsdf_layer}")
            print(f"    - TSDF Layer ID: {id(tsdf_layer)}")
            
            # ESDF layer 정보 확인 (esdf_layer_view는 존재하지 않음)
            print(f"    - ESDF Layer: Not directly accessible via esdf_layer_view()")
            print(f"    - ESDF Layer ID: N/A")
            
        except Exception as e:
            print(f"    - Layer access error: {e}")
    
    # Mapper 객체들이 서로 다른 인스턴스인지 확인
    print(f"\n=== Mapper Instance Comparison ===")
    for i in range(len(mappers_with_robot)):
        for j in range(i+1, len(mappers_with_robot)):
            mapper1 = mappers_with_robot[i]
            mapper2 = mappers_with_robot[j]
            is_different = id(mapper1) != id(mapper2)
            print(f"  Mapper {i+1} vs Mapper {j+1}: {'DIFFERENT' if is_different else 'SAME'} instances (IDs: {id(mapper1)} vs {id(mapper2)})")
            
            # TSDF layer도 다른 인스턴스인지 확인
            try:
                tsdf1 = mapper1.tsdf_layer_view()
                tsdf2 = mapper2.tsdf_layer_view()
                tsdf_different = id(tsdf1) != id(tsdf2)
                print(f"    TSDF Layers: {'DIFFERENT' if tsdf_different else 'SAME'} instances (IDs: {id(tsdf1)} vs {id(tsdf2)})")
            except Exception as e:
                print(f"    TSDF Layer comparison error: {e}")
            
            # ESDF layer 비교는 직접 접근 불가능
            print(f"    ESDF Layers: Cannot compare directly (no esdf_layer_view method)")
    
    print(f"=== End Mapper Debugging ===\n")
        
    mapper_end_time = time.time()
    print(f"\nAll robot-transformed nvblox mappers created successfully in {mapper_end_time - mapper_start_time:.4f} seconds")
    
    # 비교를 위해 원본 nvblox mapper도 생성
    print("\nCreating original nvblox mapper for comparison...")
    original_mapper_start_time = time.time()
    mapper_original = create_mapper_with_nvblox(depth_m, rgb_resized, K_adjusted, voxel_size=VOXEL_SIZE, max_integration_distance=5.0)
    original_mapper_end_time = time.time()
    print(f"Original nvblox mapper created in {original_mapper_end_time - original_mapper_start_time:.4f} seconds")
    
    # 6) M개의 viewpoint set에 대해 각 mapper별로 ESDF 기반 visibility 체크
    print("\n=== 5. nvblox ESDF Visibility Check ===")
    visibility_start_time = time.time()
    
    print(f"Processing {M} viewpoint sets, each with {L} viewpoints...")
    print(f"Total viewpoints to check: {M} x {L} = {M*L}")
    
    # [M, L] 모양의 visibility 결과 저장
    visibility_results = np.zeros((M, L), dtype=bool)  # True if visible, False if occluded
    hit_distances = np.zeros((M, L), dtype=np.float64)  # Hit distances for each viewpoint
    
    # ESDF 기반 visibility 체크를 위한 설정
    num_samples = 100  # Ray marching 샘플 수
    print(f"Using ESDF ray marching with {num_samples} samples per ray")
    
    denoising_steps = 2 # dscho temporary debug for SVDD
    for diff_step in range(denoising_steps):
        print('\n------------Denoising step ', diff_step, '------------\n')
        # 각 mapper별로 (L개의 mapper) visibility 체크
        for mapper_idx in range(L):  # mapper_idx는 column index i에 해당
            print(f"\n=== Checking mapper {mapper_idx + 1} (column {mapper_idx}) ===")
            current_mapper = mappers_with_robot[mapper_idx]
            
            # 현재 mapper_idx에 해당하는 column의 viewpoints들을 모음: CANDIDATE_VIEWPOINTS_MATRIX[:, mapper_idx]
            viewpoints_for_this_mapper = CANDIDATE_VIEWPOINTS_MATRIX[:, mapper_idx]  # [M, 3] 모양
            print(f"  Viewpoints for this mapper: {viewpoints_for_this_mapper.shape} (M viewpoints)")
            
            # ESDF 기반 batch visibility 체크
            print(f"  Performing ESDF ray marching for {M} viewpoints...")
            
            # 각 viewpoint에서 query point까지의 거리 계산
            origins = viewpoints_for_this_mapper  # [M, 3] 모양
            distances = np.linalg.norm(Xw_q - origins, axis=1)  # [M] 모양 - 각 ray의 거리
            max_distances = distances - OFFSET_DISTANCE  # [M] 모양
            
            # ESDF 기반 batch visibility 체크 수행
            batch_visible, batch_hit_distances = batch_esdf_visibility_check(
                current_mapper, origins, Xw_q, max_distances, num_samples
            )
            
            # 결과 저장
            for set_idx in range(M):
                visibility_results[set_idx, mapper_idx] = batch_visible[set_idx]
                hit_distances[set_idx, mapper_idx] = batch_hit_distances[set_idx]
                
                # Query point까지의 실제 거리 계산
                query_distance = distances[set_idx]
                
                # Query point에서 카메라 원점까지의 거리 계산
                camera_origin = np.array([0.0, 0.0, 0.0])
                query_to_camera_distance = np.linalg.norm(Xw_q - camera_origin)
                
                print(f"    Set {set_idx + 1}: {viewpoints_for_this_mapper[set_idx]} -> {'VISIBLE' if batch_visible[set_idx] else 'OCCLUDED'} (hit: {batch_hit_distances[set_idx]:.3f}m, query_dist: {query_distance:.3f}m, query_to_camera: {query_to_camera_distance:.3f}m)")
            
    
    visibility_end_time = time.time()
    print(f"\nnvblox ESDF visibility check time: {visibility_end_time - visibility_start_time:.4f} seconds")
    
    # 7) Reward 계산 (visible=1, occluded=0)
    print("\n=== 6. Reward Calculation ===")
    reward_start_time = time.time()
    
    # [M, L] 모양의 reward 매트릭스 생성 (visible=1, occluded=0)
    reward_matrix = visibility_results.astype(np.float64)
    
    print("Reward matrix (M x L):")
    print("  M = viewpoint set index (row)")
    print("  L = viewpoint index (column)")
    print("  Value: 1.0 = visible, 0.0 = occluded")
    print()
    
    for set_idx in range(M):
        print(f"Viewpoint set {set_idx + 1}: {reward_matrix[set_idx]}")
    
    # 각 viewpoint set별로 최종 reward 계산 (L개의 viewpoint change에 따른 reward 합계)
    final_rewards = np.sum(reward_matrix, axis=1)  # [M] 모양의 배열
    print(f"\nFinal rewards for each viewpoint set:")
    for set_idx in range(M):
        print(f"  Set {set_idx + 1}: {final_rewards[set_idx]:.1f} (sum of {L} viewpoints)")
    
    print(f"\nFinal rewards array: {final_rewards}")
    print(f"Total reward sum: {np.sum(final_rewards):.1f}")
    
    reward_end_time = time.time()
    print(f"Reward calculation time: {reward_end_time - reward_start_time:.4f} seconds")
    
    # 결과 정리 (기존 코드와 호환성을 위해)
    results = []
    for i, (viewpoint, yaw_deg) in enumerate(zip(CANDIDATE_VIEWPOINTS, CANDIDATE_YAW_DEGS)):
        # 첫 번째 viewpoint set의 결과를 사용 (기존 시각화 코드와 호환)
        visible_robot = visibility_results[0, i] if i < L else False
        hit_distance_robot = hit_distances[0, i] if i < L else 0.0
        
        # 각 viewpoint에서 query point까지의 거리 계산
        distance_to_query = np.linalg.norm(Xw_q - viewpoint)
        
        result = {
            'viewpoint': viewpoint,
            'yaw_deg': yaw_deg,
            'visible_robot': visible_robot,
            'visible_original': visible_robot,  # 호환성을 위해 동일하게 설정
            'hit_distance_robot': hit_distance_robot,
            'hit_distance_original': hit_distance_robot,  # 호환성을 위해 동일하게 설정
            'distance_to_query': distance_to_query
        }
        results.append(result)
    
    # 결과 출력
    print("\n=== Visibility Check Results ===")
    for i, result in enumerate(results):
        print(f"\nViewpoint {i+1}: {result['viewpoint']}, yaw={result['yaw_deg']}°")
        print(f"  Robot-transformed mesh: {'✓ VISIBLE' if result['visible_robot'] else '✗ OCCLUDED'} (hit_distance: {result['hit_distance_robot']:.3f}m)")
        print(f"  Original mesh: {'✓ VISIBLE' if result['visible_original'] else '✗ OCCLUDED'} (hit_distance: {result['hit_distance_original']:.3f}m)")
        
        # 로봇 변환 효과 분석
        if result['visible_original'] != result['visible_robot']:
            print(f"  🔄 Robot transformation changed visibility: {'✓ VISIBLE' if result['visible_original'] else '✗ OCCLUDED'} → {'✓ VISIBLE' if result['visible_robot'] else '✗ OCCLUDED'}")
        if abs(result['hit_distance_original'] - result['hit_distance_robot']) > 0.01:
            print(f"  📏 Hit distance changed: {result['hit_distance_original']:.3f}m → {result['hit_distance_robot']:.3f}m")

    # 8) 시각화
    print("\n=== 7. Creating Visualizations ===")
    viz_start_time = time.time()
    
    import os
    save_path = "visibility_test_output_esdf"
    os.makedirs(save_path, exist_ok=True)
    
    # 2D 시각화 (segmentation mask 포함)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
    # RGB with Query Point
    ax[0,0].imshow(rgb_resized)
    ax[0,0].scatter([uq], [vq], c='cyan', s=40, marker='x', label='Query Point')
    ax[0,0].set_title('RGB with Query Point')
    ax[0,0].axis('off')
    ax[0,0].legend()
    
    # Depth with Query Point
    im = ax[0,1].imshow(depth_m, cmap='magma')
    ax[0,1].scatter([uq], [vq], c='cyan', s=40, marker='x', label='Query Point')
    ax[0,1].set_title('Depth (m)')
    ax[0,1].axis('off')
    ax[0,1].legend()
    fig.colorbar(im, ax=ax[0,1], shrink=0.7, label='meters')
    
    # Robot Segmentation Mask
    ax[1,0].imshow(rgb_resized)
    robot_mask_overlay = np.zeros_like(rgb_resized)
    robot_mask_overlay[robot_mask] = [255, 0, 0]  # 빨간색
    ax[1,0].imshow(robot_mask_overlay, alpha=0.3)
    ax[1,0].scatter([uq], [vq], c='cyan', s=40, marker='x', label='Query Point')
    ax[1,0].set_title('Robot Segmentation Mask (Red)')
    ax[1,0].axis('off')
    ax[1,0].legend()
    
    # Robot mask만 표시
    ax[1,1].imshow(robot_mask, cmap='Reds', alpha=0.8)
    ax[1,1].scatter([uq], [vq], c='cyan', s=40, marker='x', label='Query Point')
    ax[1,1].set_title('Robot Mask Only')
    ax[1,1].axis('off')
    ax[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "rgb_depth_query_with_robot_mask.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("2D visualization with robot mask saved to: ", os.path.join(save_path, "rgb_depth_query_with_robot_mask.png"))
    
    # Reward 매트릭스 시각화
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Reward 매트릭스 히트맵
    im1 = ax[0].imshow(reward_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax[0].set_title('Reward Matrix (M x L)\nGreen=Visible(1), Red=Occluded(0)')
    ax[0].set_xlabel('Viewpoint Index (L)')
    ax[0].set_ylabel('Viewpoint Set Index (M)')
    
    # 컬러바 추가
    cbar1 = plt.colorbar(im1, ax=ax[0], shrink=0.8)
    cbar1.set_label('Reward (1.0=Visible, 0.0=Occluded)')
    
    # 각 셀에 값 표시
    for i in range(M):
        for j in range(L):
            text = ax[0].text(j, i, f'{reward_matrix[i, j]:.0f}',
                            ha="center", va="center", color="black", fontweight='bold')
    
    # 최종 reward 막대 그래프
    bars = ax[1].bar(range(1, M+1), final_rewards, color=['skyblue', 'lightgreen', 'lightcoral'][:M])
    ax[1].set_title('Final Rewards per Viewpoint Set')
    ax[1].set_xlabel('Viewpoint Set Index (M)')
    ax[1].set_ylabel('Total Reward')
    ax[1].set_ylim(0, L + 0.5)
    
    # 각 막대 위에 값 표시
    for i, reward in enumerate(final_rewards):
        ax[1].text(i+1, reward + 0.1, f'{reward:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "reward_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Reward analysis visualization saved to: ", os.path.join(save_path, "reward_analysis.png"))

    # 3D 시각화 - 각 mapper별로 독립적인 시각화 생성 (ESDF 기반)
    print("Creating individual 3D visualizations for each mapper...")
    
    # 각 mapper별로 시각화 생성
    for mapper_idx in range(L):
        print(f"Creating visualization for mapper {mapper_idx + 1}...")
        
        # 현재 mapper와 관련 데이터
        current_mapper = mappers_with_robot[mapper_idx]  # ESDF 쿼리를 위한 mapper
        current_robot_points_original = robot_points_original  # 공통으로 사용
        current_robot_points_transformed = robot_points_transformed_list[mapper_idx]
        
        # 현재 mapper에 대한 visibility 결과만 추출 (모든 viewpoint set에서)
        current_mapper_results = []
        for set_idx in range(M):
            visible = visibility_results[set_idx, mapper_idx]
            hit_distance = hit_distances[set_idx, mapper_idx]
            viewpoint = CANDIDATE_VIEWPOINTS_MATRIX[set_idx, mapper_idx]
            
            # 각 viewpoint에서 query point까지의 거리 계산
            distance_to_query = np.linalg.norm(Xw_q - viewpoint)
            
            result = {
                'viewpoint': viewpoint,
                'yaw_deg': 0.0,  # 기본값
                'visible_robot': visible,
                'visible_original': visible,  # ESDF 기반이므로 동일
                'hit_distance_robot': hit_distance,
                'hit_distance_original': hit_distance,  # ESDF 기반이므로 동일
                'distance_to_query': distance_to_query
            }
            current_mapper_results.append(result)
        
        # 현재 mapper에 대한 Ray 기반 ESDF 시각화 생성
        create_ray_based_esdf_3d_visualization_plotly_new(
            current_mapper, Xw_q, CANDIDATE_VIEWPOINTS_MATRIX[:, mapper_idx], current_mapper_results,
            os.path.join(save_path, f"3d_ray_esdf_visualization_mapper_{mapper_idx + 1}.html"),
            robot_mask, current_robot_points_original, current_robot_points_transformed,
            voxel_size=VOXEL_SIZE, max_distance=1.0, num_ray_samples=50
        )
    
    # 원본 mapper와 비교를 위한 시각화 생성
    print("Creating original mapper visualization for comparison...")
    original_results = []
    for i, viewpoint in enumerate(CANDIDATE_VIEWPOINTS):
        # 원본 mapper에서는 모든 viewpoint가 visible하다고 가정
        original_results.append({
            'viewpoint': viewpoint,
            'yaw_deg': 0.0,
            'visible_robot': True,  # 원본에서는 가정
            'visible_original': True,  # ESDF 기반이므로 동일
            'hit_distance_robot': 0.0,
            'hit_distance_original': 0.0,  # ESDF 기반이므로 동일
            'distance_to_query': np.linalg.norm(Xw_q - viewpoint)
        })
    
    create_ray_based_esdf_3d_visualization_plotly_new(
        mapper_original, Xw_q, CANDIDATE_VIEWPOINTS, original_results,
        os.path.join(save_path, "3d_ray_esdf_visualization_original.html"),
        robot_mask, robot_points_original, None,
        voxel_size=VOXEL_SIZE, max_distance=1.0, num_ray_samples=50
    )
    
    # M개의 viewpoint set을 모두 보여주는 3D 시각화 생성 (ESDF 기반)
    create_esdf_multi_viewpoint_set_3d_visualization(mappers_with_robot, Xw_q, CANDIDATE_VIEWPOINTS_MATRIX, 
                                                    visibility_results, final_rewards,
                                                    os.path.join(save_path, "3d_multi_viewpoint_sets_esdf.html"),
                                                    voxel_size=VOXEL_SIZE, max_distance=1.0)
    
    
    
    viz_end_time = time.time()
    print(f"Visualization time: {viz_end_time - viz_start_time:.4f} seconds")

    # 전체 실행 시간
    total_end_time = time.time()
    print(f"\n=== Total Execution Time: {total_end_time - total_start_time:.4f} seconds ===")

    # 결과 요약
    print("\n=== Summary ===")
    print(f"Configuration: M={M} viewpoint sets, L={L} viewpoints per set")
    print(f"Total viewpoints processed: {M} x {L} = {M*L}")
    
    # 각 viewpoint set별 요약
    print(f"\nViewpoint Set Analysis:")
    for set_idx in range(M):
        visible_count = np.sum(visibility_results[set_idx])
        total_reward = final_rewards[set_idx]
        print(f"  Set {set_idx + 1}: {visible_count}/{L} visible viewpoints, reward = {total_reward:.1f}")
    
    # 전체 통계
    total_visible = np.sum(visibility_results)
    total_possible = M * L
    overall_visibility_rate = total_visible / total_possible * 100
    
    print(f"\nOverall Statistics:")
    print(f"  Total visible viewpoints: {total_visible}/{total_possible} ({overall_visibility_rate:.1f}%)")
    print(f"  Average reward per set: {np.mean(final_rewards):.2f}")
    print(f"  Best performing set: Set {np.argmax(final_rewards) + 1} (reward: {np.max(final_rewards):.1f})")
    print(f"  Worst performing set: Set {np.argmin(final_rewards) + 1} (reward: {np.min(final_rewards):.1f})")
    
    # Reward 매트릭스 요약
    print(f"\nReward Matrix Summary:")
    print(f"  Matrix shape: {reward_matrix.shape}")
    print(f"  Total reward sum: {np.sum(final_rewards):.1f}")
    print(f"  Reward variance: {np.var(final_rewards):.2f}")
    
    print("\nDetailed results (first viewpoint set only):")
    for i, result in enumerate(results):
        status_robot = "VISIBLE" if result['visible_robot'] else "OCCLUDED"
        print(f"  Viewpoint {i+1}: {status_robot} (hit_distance: {result['hit_distance_robot']:.3f}m)")


if __name__ == "__main__":
    main()
