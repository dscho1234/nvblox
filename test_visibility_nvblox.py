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


# ========= 사용자 설정 =========
# Zarr 데이터 경로 설정
BUFFER_PATH = "/home/dscho-larr/fast_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/slam_head_mounted_camera_multi_marker_initial_lag"
EPISODE_IDX = 0
FRAME_IDX = 100  # 특정 프레임 선택
DEPTH_SCALE = 0.001        # 깊이 단위 → 미터 변환 (예: mm면 0.001, 이미 m면 1.0)

# TSDF 설정
VOXEL_SIZE = 0.005 # 0.005  # 5mm voxel size (이미지에서 권장하는 3-5mm 범위)
TSDF_TRUNCATION = 0.02  # 2cm truncation distance (voxel_size * 4)

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
CANDIDATE_VIEWPOINTS = [
    np.array([0.20, 0.00, 0.00], dtype=np.float64),  # 옆으로
    np.array([0.00, 0.20, 0.00], dtype=np.float64),  # 위로
    np.array([0.00, 0.00, 0.20], dtype=np.float64),  # 앞으로
    np.array([0.15, 0.15, 0.00], dtype=np.float64),  # 대각선
    np.array([0.15, 0.15, 0.15], dtype=np.float64),  # 3D 대각선
]

# 각 viewpoint의 yaw 회전 (deg)
CANDIDATE_YAW_DEGS = [0.0, 0.0, 0.0, 0.0, 0.0]

# M개의 viewpoint set 생성 (예제에서는 M=3으로 설정)
M = 10 # 3  # M개의 viewpoint set
L = len(CANDIDATE_VIEWPOINTS)  # L개의 viewpoints per set

# Batch raycasting 사용 여부 설정
BATCH_RAY_CASTING = True  # True: batch raycasting, False: sequential raycasting

print(f"Creating {M} viewpoint sets, each with {L} viewpoints")
print(f"Total viewpoints: {M} x {L} = {M*L}")
print(f"Batch raycasting: {'ENABLED' if BATCH_RAY_CASTING else 'DISABLED'}")

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


def depth_to_vertices(K, depth_m):
    """K: (3,3), depth_m: (H,W) in meters -> (H,W,3) in camera-0 coords (OpenCV 픽셀 기준)"""
    H, W = depth_m.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    z = depth_m
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    V = np.stack([x, y, z], axis=-1)
    return V  # (H,W,3)


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


# ========= nvblox 기반 메시 생성 =========
def create_mesh_with_nvblox(depth_image, rgb_image, K, voxel_size=0.005, max_integration_distance=5.0, return_mapper=False, mapper=None):
    """
    nvblox를 사용하여 3D 메시 생성
    
    Args:
        depth_image: (H, W) 깊이 이미지 (미터)
        rgb_image: (H, W, 3) RGB 이미지
        K: (3, 3) 카메라 내부 파라미터
        voxel_size: voxel 크기 (미터)
        max_integration_distance: 최대 통합 거리 (미터)
        return_mapper: True면 (mesh, mapper) 튜플 반환, False면 mesh만 반환
        mapper: 기존 Mapper 객체 재사용 (None이면 새로 생성)
    
    Returns:
        mesh: nvblox ColorMesh 객체 또는 (mesh, mapper) 튜플
    """
    print("Creating mesh with nvblox...")
    start = time.time()
    # 데이터를 torch tensor로 변환
    H, W = depth_image.shape
    
    # Depth 이미지를 torch tensor로 변환 (GPU)
    depth_tensor = torch.from_numpy(depth_image.astype(np.float32)).cuda()
    
    # RGB 이미지를 torch tensor로 변환 (GPU, uint8)
    rgb_tensor = torch.from_numpy((rgb_image * 255).astype(np.uint8)).cuda()
    
    # 카메라 내부 파라미터를 torch tensor로 변환 (CPU)
    intrinsics_tensor = torch.from_numpy(K.astype(np.float32)).cpu()
    
    # 카메라 포즈 (identity matrix, CPU)
    pose_tensor = torch.eye(4, dtype=torch.float32).cpu()
    
    # Mapper 재사용 또는 새로 생성
    if mapper is None:
        # nvblox Mapper 설정
        projective_integrator_params = ProjectiveIntegratorParams()
        projective_integrator_params.projective_integrator_max_integration_distance_m = max_integration_distance
        mapper_params = MapperParams()
        mapper_params.set_projective_integrator_params(projective_integrator_params)
        
        # Mapper 생성
        mapper = Mapper(
            voxel_sizes_m=voxel_size,
            # integrator_types=ProjectiveIntegratorType.TSDF,
            mapper_parameters=mapper_params
        )
    
    
    # 데이터 통합
    mapper.add_depth_frame(depth_tensor, pose_tensor, intrinsics_tensor)
    mapper.add_color_frame(rgb_tensor, pose_tensor, intrinsics_tensor)
    
    # 메시 업데이트
    mapper.update_color_mesh()
    
    # 메시 가져오기
    color_mesh = mapper.get_color_mesh()
    
    print(f"nvblox mesh created with {color_mesh.vertices().shape[0]} vertices and {color_mesh.triangles().shape[0]} triangles")
    print('color mesh update time: ', time.time() - start)
    
    
    if return_mapper:
        return color_mesh, mapper
    else:
        return color_mesh




# ========= nvblox ESDF 기반 Visibility 체크 =========
def nvblox_mesh_to_open3d(nvblox_mesh):
    """
    nvblox ColorMesh를 Open3D TriangleMesh로 변환
    
    Args:
        nvblox_mesh: nvblox ColorMesh 객체
    
    Returns:
        o3d.geometry.TriangleMesh: Open3D 메시 객체
    """
    
    # nvblox mesh에서 vertices와 triangles 추출
    vertices = nvblox_mesh.vertices().cpu().numpy()
    triangles = nvblox_mesh.triangles().cpu().numpy()
    
    if len(vertices) == 0 or len(triangles) == 0:
        return None
    
    # Open3D 메시 생성
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # 법선 계산 (raycasting에 필요)
    mesh.compute_vertex_normals()
    
    return mesh
    



def check_visibility_with_scene(scene, origin, target_point, max_distance, num_samples=50):
    """
    미리 생성된 Open3D RaycastingScene을 사용한 visibility 체크
    
    Args:
        scene: 미리 생성된 o3d.t.geometry.RaycastingScene 객체
        origin: 레이 시작점 (월드 좌표)
        target_point: 목표 점 (월드 좌표)
        max_distance: 최대 거리
        num_samples: 사용되지 않음 (Open3D raycasting은 샘플링 불필요)
    
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
    
    if scene is None:
        print("    Scene is None, assuming visible")
        return True, max_distance, None
    
    # Open3D tensor 기반 ray casting 사용
    visible, hit_distance, hit_point = ray_casting_with_scene(
        scene, origin, direction, max_distance
    )
    
    end_time = time.time()
    print(f"    nvblox mesh raycast visibility check: {end_time - start_time:.4f}seconds")
    
    return visible, hit_distance, hit_point


def check_visibility_with_nvblox_mesh_raycast(mesh, origin, target_point, max_distance, num_samples=50):
    """
    nvblox mesh를 Open3D로 변환하여 tensor 기반 ray casting으로 visibility 체크
    (기존 호환성을 위한 wrapper 함수)
    
    Args:
        mesh: nvblox ColorMesh 객체
        origin: 레이 시작점 (월드 좌표)
        target_point: 목표 점 (월드 좌표)
        max_distance: 최대 거리
        num_samples: 사용되지 않음 (Open3D raycasting은 샘플링 불필요)
    
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
    
    
    # nvblox mesh를 Open3D로 변환
    open3d_mesh = nvblox_mesh_to_open3d(mesh)
    
    if open3d_mesh is None:
        print("    Failed to convert mesh, assuming visible")
        return True, max_distance, None
    
    print(f"    Mesh has {len(open3d_mesh.vertices)} vertices, {len(open3d_mesh.triangles)} triangles")
    
    # Open3D tensor 기반 ray casting 사용
    visible, hit_distance, hit_point = ray_casting_with_open3d_mesh(
        open3d_mesh, origin, direction, max_distance
    )
    
    end_time = time.time()
    print(f"    nvblox mesh raycast visibility check: {end_time - start_time:.4f}seconds")
    
    return visible, hit_distance, hit_point
    
    


def ray_casting_with_scene(scene, origin, direction, max_distance):
    """
    미리 생성된 Open3D RaycastingScene을 사용한 단일 ray casting
    
    Args:
        scene: 미리 생성된 o3d.t.geometry.RaycastingScene 객체
        origin: 레이 시작점 (월드 좌표)
        direction: 레이 방향 (정규화된 벡터)
        max_distance: 최대 거리
    
    Returns:
        (visible, hit_distance, hit_point)
        - visible: True if point is visible
        - hit_distance: distance to first hit (if any)
        - hit_point: hit point coordinates (if any)
    """
    if scene is None:
        print("    Scene is None, assuming visible")
        return True, max_distance, None
    
    start_time = time.time()
    
    # 레이 생성
    ray = o3d.core.Tensor([origin.tolist() + direction.tolist()], dtype=o3d.core.Dtype.Float32)
    
    # 레이와 메시의 교차점 계산
    ans = scene.cast_rays(ray)
    
    # 결과 분석
    hit = ans['t_hit'].numpy()[0]
    
    end_time = time.time()
    print(f"    Ray casting: {end_time - start_time:.4f}seconds")
    
    if np.isinf(hit) or hit > max_distance:
        # 교차점이 없거나 너무 멀리 있음
        return True, max_distance, None
    else:
        # 교차점이 있음
        hit_point = origin + hit * direction
        return False, hit, hit_point


def ray_casting_with_open3d_mesh(mesh, origin, direction, max_distance):
    """
    Open3D 메시를 사용한 ray casting으로 line-of-sight 체크 (비교용)
    (기존 호환성을 위한 wrapper 함수)
    
    Args:
        mesh: Open3D TriangleMesh 객체
        origin: 레이 시작점 (월드 좌표)
        direction: 레이 방향 (정규화된 벡터)
        max_distance: 최대 거리
    
    Returns:
        (visible, hit_distance, hit_point)
        - visible: True if point is visible
        - hit_distance: distance to first hit (if any)
        - hit_point: hit point coordinates (if any)
    """
    start_time = time.time()
    
    # 레이 생성
    ray = o3d.core.Tensor([origin.tolist() + direction.tolist()], dtype=o3d.core.Dtype.Float32)
    
    # 메시를 tensor로 변환
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    
    # Ray casting 수행
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)
    
    # 레이와 메시의 교차점 계산
    ans = scene.cast_rays(ray)
    
    # 결과 분석
    hit = ans['t_hit'].numpy()[0]
    
    end_time = time.time()
    print(f"    Ray casting: {end_time - start_time:.4f}seconds")
    
    if np.isinf(hit) or hit > max_distance:
        # 교차점이 없거나 너무 멀리 있음
        return True, max_distance, None
    else:
        # 교차점이 있음
        hit_point = origin + hit * direction
        return False, hit, hit_point


def create_raycasting_scene(mesh):
    """
    nvblox mesh로부터 Open3D RaycastingScene을 미리 생성
    
    Args:
        mesh: nvblox ColorMesh 객체
    
    Returns:
        scene: o3d.t.geometry.RaycastingScene 객체
    """
    print("    Creating raycasting scene...")
    start_time = time.time()
    
    # 1. nvblox mesh를 Open3D로 변환
    open3d_mesh = nvblox_mesh_to_open3d(mesh)
    
    if open3d_mesh is None:
        print("    Failed to convert mesh, returning None")
        return None
    
    # 2. 메시를 tensor로 변환
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(open3d_mesh)
    
    # 3. Ray casting scene 생성
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)
    
    end_time = time.time()
    print(f"    Scene creation time: {end_time - start_time:.4f}s")
    
    return scene


def batch_raycasting_with_scene(scene, origins, directions, max_distances):
    """
    미리 생성된 Open3D RaycastingScene을 사용한 batch raycasting
    
    Args:
        scene: 미리 생성된 o3d.t.geometry.RaycastingScene 객체
        origins: 레이 시작점들 [M, 3]
        directions: 레이 방향들 (정규화된 벡터) [M, 3]
        max_distances: 최대 거리들 [M]
    
    Returns:
        (visible_array, hit_distances_array)
        - visible_array: [M] boolean array, True if visible
        - hit_distances_array: [M] float array, hit distances
    """
    if scene is None:
        print("    Scene is None, assuming all visible")
        return np.ones(len(origins), dtype=bool), max_distances.copy()
    
    total_start_time = time.time()
    
    # 1. Batch ray 생성
    step1_start = time.time()
    rays = np.hstack([origins, directions])
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    step1_time = time.time() - step1_start
    print(f"      Step 1 - Ray preparation: {step1_time:.4f}s")
    
    # 2. Batch raycasting 수행
    n_iter = 1
    step2_start = time.time()
    for _ in range(n_iter):
        ans = scene.cast_rays(rays_tensor)
    step2_time = (time.time() - step2_start) / n_iter
    print(f"      Step 2 - Ray casting: {step2_time:.4f}s")
    
    # 3. 결과 분석
    step3_start = time.time()
    hit_distances = ans['t_hit'].numpy()  # [M]
    
    # Visibility 판단: hit이 inf이거나 max_distance보다 크면 visible
    visible = np.logical_or(np.isinf(hit_distances), hit_distances > max_distances)
    
    # Hit distance 조정: visible한 경우 max_distance로 설정
    hit_distances_adjusted = np.where(visible, max_distances, hit_distances)
    step3_time = time.time() - step3_start
    print(f"      Step 3 - Result processing: {step3_time:.4f}s")
    
    total_time = time.time() - total_start_time
    print(f"    Batch raycasting ({len(origins)} rays): {total_time:.4f}s total")
    
    # 각 단계별 시간 요약
    print(f"      Time breakdown: rays={step1_time:.4f}s, casting={step2_time:.4f}s, processing={step3_time:.4f}s")
    
    return visible, hit_distances_adjusted


def batch_raycasting_with_open3d(mesh, origins, directions, max_distances):
    """
    Open3D tensor 기반 batch raycasting으로 여러 viewpoint에 대해 병렬로 visibility 체크
    (기존 호환성을 위한 wrapper 함수)
    
    Args:
        mesh: nvblox ColorMesh 객체
        origins: 레이 시작점들 [M, 3]
        directions: 레이 방향들 (정규화된 벡터) [M, 3]
        max_distances: 최대 거리들 [M]
    
    Returns:
        (visible_array, hit_distances_array)
        - visible_array: [M] boolean array, True if visible
        - hit_distances_array: [M] float array, hit distances
    """
    # Scene을 생성하고 바로 raycasting 수행
    scene = create_raycasting_scene(mesh)
    return batch_raycasting_with_scene(scene, origins, directions, max_distances)





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
    
    # 메시를 샘플링해서 너무 많은 점을 줄임 (성능 향상)
    if len(vertices) > 10000:
        # 랜덤 샘플링
        indices = np.random.choice(len(vertices), 10000, replace=False)
        vertices = vertices[indices]
    
    # 3D 시각화 생성
    fig = go.Figure()
    
    # 1. 메시 표시 (색상 정보 포함)
    if vertex_colors is not None and len(vertex_colors) > 0:
        # 색상 정보가 있는 경우
        colors = vertex_colors
        if len(colors) > 10000:
            colors = colors[indices]
        
        # RGB 색상을 hex로 변환 (nvblox는 0-255 범위, Open3D는 0-1 범위)
        if colors.max() <= 1.0:
            # Open3D 형식 (0-1 범위)
            color_hex = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors]
        else:
            # nvblox 형식 (0-255 범위)
            color_hex = [f'rgb({int(c[0])},{int(c[1])},{int(c[2])})' for c in colors]
        
        fig.add_trace(go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1], 
            z=vertices[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=color_hex,
                opacity=0.8
            ),
            name='3D Scene (Colored)',
            showlegend=True
        ))
    else:
        # 색상 정보가 없는 경우 (기본 색상)
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
            name='3D Scene',
            showlegend=True
        ))
    
    # 1.5. 원본 로봇 포인트들 표시 (빨간색)
    if robot_points_original is not None and len(robot_points_original) > 0:
        # 로봇 포인트들을 샘플링 (너무 많으면)
        if len(robot_points_original) > 1000:
            robot_indices = np.random.choice(len(robot_points_original), 1000, replace=False)
            robot_original_sampled = robot_points_original[robot_indices]
        else:
            robot_original_sampled = robot_points_original
            
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
    
    # 1.6. 변환된 로봇 포인트들 표시 (주황색)
    if robot_points_transformed is not None and len(robot_points_transformed) > 0:
        # 로봇 포인트들을 샘플링 (너무 많으면)
        if len(robot_points_transformed) > 1000:
            robot_indices = np.random.choice(len(robot_points_transformed), 1000, replace=False)
            robot_transformed_sampled = robot_points_transformed[robot_indices]
        else:
            robot_transformed_sampled = robot_points_transformed
            
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
            text='3D Line of Sight Visibility Analysis with Robot Transformation',
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


def create_comprehensive_3d_visualization(mesh, query_point, candidate_viewpoints, results, save_path):
    """
    더 상세한 3D 시각화 (메시 + 점들 + LoS)
    """
    # 메시 데이터 추출 (nvblox ColorMesh 또는 Open3D mesh 모두 지원)
    if hasattr(mesh, 'vertices') and callable(mesh.vertices):
        # nvblox ColorMesh
        vertices = mesh.vertices().cpu().numpy()
        faces = mesh.triangles().cpu().numpy()
    else:
        # Open3D mesh (fallback)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
    
    # 메시를 샘플링 (성능을 위해)
    if len(vertices) > 5000:
        indices = np.random.choice(len(vertices), 5000, replace=False)
        vertices = vertices[indices]
    
    # 서브플롯 생성 (2x2)
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Full 3D Scene', 'Viewpoints Only', 'Line of Sight Analysis', 'Hit Points Detail'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. 전체 3D 장면 (좌상)
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        mode='markers',
        marker=dict(size=1, color='lightblue', opacity=0.3),
        name='Scene',
        showlegend=False
    ), row=1, col=1)
    
    # 2. Viewpoints만 (우상)
    for i, (viewpoint, result) in enumerate(zip(candidate_viewpoints, results)):
        viewpoint_color = 'green' if result['visible_robot'] else 'red'
        fig.add_trace(go.Scatter3d(
            x=[viewpoint[0]], y=[viewpoint[1]], z=[viewpoint[2]],
            mode='markers',
            marker=dict(size=8, color=viewpoint_color, symbol='circle'),
            name=f'VP{i+1}',
            showlegend=False
        ), row=1, col=2)
    
    # 3. Line of Sight 분석 (좌하)
    fig.add_trace(go.Scatter3d(
        x=[query_point[0]], y=[query_point[1]], z=[query_point[2]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Query',
        showlegend=False
    ), row=2, col=1)
    
    for i, (viewpoint, result) in enumerate(zip(candidate_viewpoints, results)):
        los_color = 'green' if result['visible_robot'] else 'red'
        fig.add_trace(go.Scatter3d(
            x=[viewpoint[0], query_point[0]],
            y=[viewpoint[1], query_point[1]],
            z=[viewpoint[2], query_point[2]],
            mode='lines',
            line=dict(color=los_color, width=3),
            name=f'LoS{i+1}',
            showlegend=False
        ), row=2, col=1)
    
    # 4. Hit Points 상세 (우하)
    for i, (viewpoint, result) in enumerate(zip(candidate_viewpoints, results)):
        if not result['visible_robot']:
            hit_distance = result['hit_distance_robot']
            direction = query_point - viewpoint
            direction = direction / np.linalg.norm(direction)
            hit_point = viewpoint + direction * hit_distance
            
            fig.add_trace(go.Scatter3d(
                x=[hit_point[0]], y=[hit_point[1]], z=[hit_point[2]],
                mode='markers',
                marker=dict(size=6, color='orange', symbol='x'),
                name=f'Hit{i+1}',
                showlegend=False
            ), row=2, col=2)
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text='Comprehensive 3D Line of Sight Analysis',
            x=0.5,
            font=dict(size=18)
        ),
        width=1400,
        height=1000,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    # 각 서브플롯의 scene 설정
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_scenes(
                xaxis_title='X (m)',
                yaxis_title='Y (m)', 
                zaxis_title='Z (m)',
                aspectmode='data',
                row=i, col=j
            )
    
    # HTML 파일로 저장
    pyo.plot(fig, filename=save_path, auto_open=False)
    print(f"Comprehensive 3D visualization saved to: {save_path}")


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
        else:
            vertices = np.asarray(mesh.vertices)
        
        # 메시를 샘플링 (성능을 위해)
        if len(vertices) > 2000:
            indices = np.random.choice(len(vertices), 2000, replace=False)
            vertices = vertices[indices]
        
        # 메시 표시
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
def create_mesh_with_robot_transformation_nvblox(rgb, depth, robot_mask, robot_se3_transform, 
                                               camera_intrinsics, voxel_size=0.005, max_integration_distance=5.0, mapper=None):
    """
    로봇의 SE(3) 변환을 고려한 nvblox 메시 생성
    
    Args:
        rgb: RGB 이미지 (H, W, 3)
        depth: 깊이 이미지 (H, W) 미터 단위
        robot_mask: 로봇 segmentation mask (H, W) boolean array
        robot_se3_transform: 로봇의 SE(3) 변환 행렬 (4, 4)
        camera_intrinsics: 카메라 내부 파라미터 (3, 3)
        voxel_size: voxel 크기 (미터)
        max_integration_distance: 최대 통합 거리 (미터)
        return_mapper: True면 (mesh, mapper, robot_points_original, robot_points_transformed) 튜플 반환
    
    Returns:
        mesh: 변환된 로봇을 고려한 nvblox ColorMesh
        robot_points_original: 원본 로봇 포인트들
        robot_points_transformed: 변환된 로봇 포인트들
        mapper: nvblox Mapper (return_mapper=True일 때만)
    """
    print("Creating nvblox mesh with robot transformation...")
    
    # 1) 현재 RGBD에서 로봇 부분 제거
    rgb_without_robot, depth_without_robot = remove_robot_from_rgbd_direct(rgb, depth, robot_mask)
    
    # 2) 로봇 포인트들을 SE(3) 변환
    robot_points_original = transform_robot_points_direct(rgb, depth, robot_mask, np.eye(4), camera_intrinsics)
    robot_points_transformed = transform_robot_points_direct(rgb, depth, robot_mask, robot_se3_transform, camera_intrinsics)
    
    # 3) 변환된 로봇을 새로운 RGBD에 추가
    rgb_future, depth_future = add_transformed_robot_to_rgbd_direct(
        rgb_without_robot, depth_without_robot, robot_points_transformed, camera_intrinsics
    )
    
    # 4) nvblox 메시 생성
    mesh, mapper = create_mesh_with_nvblox(depth_future, rgb_future, camera_intrinsics, voxel_size, max_integration_distance, return_mapper=True, mapper=mapper)
    return mesh, robot_points_original, robot_points_transformed, mapper
    

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


def transform_robot_points_direct(rgb, depth, robot_mask, robot_se3_transform, camera_intrinsics):
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
    """
    if len(robot_points_transformed) == 0:
        print("Warning: No transformed robot points to add")
        return rgb_without_robot, depth_without_robot
    
    # Deep copy로 원본 보존
    rgb_future = copy.deepcopy(rgb_without_robot)
    depth_future = copy.deepcopy(depth_without_robot)
    
    # 변환된 로봇 포인트들을 이미지 좌표로 project
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    added_pixels = 0
    valid_points = 0
    
    for point in robot_points_transformed:
        x_3d, y_3d, z_3d = point
        
        if z_3d > 0:
            valid_points += 1
            # 3D를 픽셀로 project
            x_pixel = int(fx * x_3d / z_3d + cx)
            y_pixel = int(fy * y_3d / z_3d + cy)
            
            # 이미지 범위 내에 있는지 확인
            if 0 <= x_pixel < depth_future.shape[1] and 0 <= y_pixel < depth_future.shape[0]:
                # 기존 depth보다 가까우면 업데이트
                if depth_future[y_pixel, x_pixel] == 0 or depth_future[y_pixel, x_pixel] > z_3d:
                    depth_future[y_pixel, x_pixel] = z_3d
                    # 로봇 색상 (회색)으로 설정
                    rgb_future[y_pixel, x_pixel] = [0.5, 0.5, 0.5]
                    added_pixels += 1
    
    print(f"  Added {added_pixels} robot pixels from {valid_points} valid transformed points")
    
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
    print("=== Direct Mesh-based Line-of-Sight Visibility Test with Robot Transformation ===")
    print("Using TriangleMesh.create_from_depth_image for faster mesh generation")
    
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

    # 5) 각 mesh별로 로봇 변환을 고려한 nvblox 메시 생성
    print("\n=== 4. nvblox Mesh Generation with Robot Transformation ===")
    mesh_start_time = time.time()
    
    # 데모용 로봇 segmentation mask 생성
    robot_mask = create_robot_segmentation_mask_demo(rgb_resized.shape)
    print(f"Robot mask created: {np.sum(robot_mask)} pixels")
    
    # L개의 로봇 SE(3) 변환 생성 (각 column index i에 대응)
    robot_se3_transforms = create_robot_se3_transforms_demo(CANDIDATE_VIEWPOINTS)
    print(f"Generated {len(robot_se3_transforms)} robot SE(3) transforms")
    for i, transform in enumerate(robot_se3_transforms):
        print(f"  Transform {i+1}: translation={transform[:3, 3]}")
    
    # L개의 mesh 생성 (각 column index i에 대응하는 robot transform으로)
    meshes_with_robot = []
    mappers_with_robot = []
    robot_points_original_list = []
    robot_points_transformed_list = []
    
    # Mapper 재사용을 위한 공유 mapper 생성
    shared_mapper = None
    
    for i, robot_transform in enumerate(robot_se3_transforms):
        print(f"  Creating mesh {i+1} with robot transform: translation={robot_transform[:3, 3]}")
        start = time.time()
        mesh_with_robot, robot_points_original, robot_points_transformed, mapper_robot = create_mesh_with_robot_transformation_nvblox(
            rgb_resized, depth_m, robot_mask, robot_transform, K_adjusted, voxel_size=VOXEL_SIZE, max_integration_distance=5.0,  mapper=shared_mapper
        )
        print(f"  nvblox robot transformed mesh creation time: {time.time() - start:.4f} seconds")
        meshes_with_robot.append(mesh_with_robot)
        mappers_with_robot.append(mapper_robot)
        robot_points_original_list.append(robot_points_original)
        robot_points_transformed_list.append(robot_points_transformed)
        print(f"  Mesh {i+1}: {mesh_with_robot.vertices().shape[0]} vertices, {mesh_with_robot.triangles().shape[0]} triangles")
        
        # 첫 번째 iteration에서 shared_mapper 설정
        if shared_mapper is None:
            shared_mapper = mapper_robot
        
        # 디버깅: 변환된 로봇 포인트들의 중심점 확인
        if len(robot_points_transformed) > 0:
            transformed_center = np.mean(robot_points_transformed, axis=0)
            print(f"  Transformed robot center: {transformed_center}")
        
        
    mesh_end_time = time.time()
    print(f"\nAll robot-transformed nvblox meshes created successfully in {mesh_end_time - mesh_start_time:.4f} seconds")
    
    # 비교를 위해 원본 nvblox 메시도 생성 (mapper도 함께 반환)
    print("\nCreating original nvblox mesh for comparison...")
    original_mesh_start_time = time.time()
    mesh_original, mapper_original = create_mesh_with_nvblox(depth_m, rgb_resized, K_adjusted, voxel_size=VOXEL_SIZE, max_integration_distance=5.0, return_mapper=True)
    original_mesh_end_time = time.time()
    print(f"Original nvblox mesh created in {original_mesh_end_time - original_mesh_start_time:.4f} seconds")
    print(f"Original mesh has {mesh_original.vertices().shape[0]} vertices and {mesh_original.triangles().shape[0]} triangles")
    
    # 6) M개의 viewpoint set에 대해 각 mesh별로 visibility 체크
    print("\n=== 5. nvblox ESDF Visibility Check ===")
    visibility_start_time = time.time()
    
    print(f"Processing {M} viewpoint sets, each with {L} viewpoints...")
    print(f"Total viewpoints to check: {M} x {L} = {M*L}")
    
    # [M, L] 모양의 visibility 결과 저장
    visibility_results = np.zeros((M, L), dtype=bool)  # True if visible, False if occluded
    hit_distances = np.zeros((M, L), dtype=np.float64)  # Hit distances for each viewpoint
    
    # 각 mesh별로 RaycastingScene을 미리 생성 (성능 최적화)
    print(f"\n=== Pre-creating RaycastingScenes for {L} meshes ===")
    scene_creation_start_time = time.time()
    raycasting_scenes = []
    
    for mesh_idx in range(L):
        print(f"  Creating scene for mesh {mesh_idx + 1}...")
        scene = create_raycasting_scene(meshes_with_robot[mesh_idx])
        raycasting_scenes.append(scene)
    
    scene_creation_end_time = time.time()
    print(f"All RaycastingScenes created in {scene_creation_end_time - scene_creation_start_time:.4f} seconds")
    
    denoising_steps = 10
    for diff_step in range(denoising_steps):
        print('\n------------Denoising step ', diff_step, '------------\n')
        # 각 mesh별로 (L개의 mesh) visibility 체크
        for mesh_idx in range(L):  # mesh_idx는 column index i에 해당
            print(f"\n=== Checking mesh {mesh_idx + 1} (column {mesh_idx}) ===")
            current_mesh = meshes_with_robot[mesh_idx]
            current_scene = raycasting_scenes[mesh_idx]
            
            # 현재 mesh_idx에 해당하는 column의 viewpoints들을 모음: CANDIDATE_VIEWPOINTS_MATRIX[:, mesh_idx]
            viewpoints_for_this_mesh = CANDIDATE_VIEWPOINTS_MATRIX[:, mesh_idx]  # [M, 3] 모양
            print(f"  Viewpoints for this mesh: {viewpoints_for_this_mesh.shape} (M viewpoints)")
            
            if BATCH_RAY_CASTING:
                # Batch raycasting 방식 (미리 생성된 scene 사용)
                print(f"  Performing batch raycasting for {M} viewpoints...")
                
                # Batch raycasting을 위한 ray 생성
                origins = viewpoints_for_this_mesh  # [M, 3] 모양
                directions = Xw_q - origins  # [M, 3] 모양 - 각 viewpoint에서 query point로의 방향
                distances = np.linalg.norm(directions, axis=1)  # [M] 모양 - 각 ray의 거리
                directions = directions / distances[:, np.newaxis]  # 정규화된 방향 벡터 [M, 3]
                
                # 미리 생성된 scene을 사용한 batch raycasting 수행
                batch_visible, batch_hit_distances = batch_raycasting_with_scene(
                    current_scene, origins, directions, distances
                )
                
                # 결과 저장
                for set_idx in range(M):
                    visibility_results[set_idx, mesh_idx] = batch_visible[set_idx]
                    hit_distances[set_idx, mesh_idx] = batch_hit_distances[set_idx]
                    
                    print(f"    Set {set_idx + 1}: {viewpoints_for_this_mesh[set_idx]} -> {'VISIBLE' if batch_visible[set_idx] else 'OCCLUDED'} (hit: {batch_hit_distances[set_idx]:.3f}m)")
            else:
                # 순차적 raycasting 방식 (미리 생성된 scene 사용)
                print(f"  Performing sequential raycasting for {M} viewpoints...")
                
                for set_idx in range(M):  # set_idx는 row index에 해당
                    viewpoint = viewpoints_for_this_mesh[set_idx]  # [3] 모양
                    
                    # 해당 viewpoint에서 query point까지의 거리 계산
                    distance = np.linalg.norm(Xw_q - viewpoint)
                    
                    # 미리 생성된 scene으로 visibility 체크
                    visible, hit_distance, hit_point = check_visibility_with_scene(
                        current_scene, viewpoint, Xw_q, distance
                    )
                    
                    # 결과 저장 (set_idx, mesh_idx 위치에 저장)
                    visibility_results[set_idx, mesh_idx] = visible
                    hit_distances[set_idx, mesh_idx] = hit_distance
                    
                    print(f"    Set {set_idx + 1}: {viewpoint} -> {'VISIBLE' if visible else 'OCCLUDED'} (hit: {hit_distance:.3f}m)")
    
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
    os.makedirs("visibility_test_output", exist_ok=True)
    
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
    plt.savefig("visibility_test_output/rgb_depth_query_with_robot_mask.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("2D visualization with robot mask saved to: visibility_test_output/rgb_depth_query_with_robot_mask.png")
    
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
    plt.savefig("visibility_test_output/reward_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Reward analysis visualization saved to: visibility_test_output/reward_analysis.png")

    # 3D 시각화
    print("Creating 3D visualization...")
    
    
    representative_mesh = meshes_with_robot[-1] 
    representative_robot_points_original = robot_points_original_list[-1]
    representative_robot_points_transformed = robot_points_transformed_list[-1]
    
    # Plotly 기반 인터랙티브 3D 시각화 생성 (로봇 변환 고려)
    create_3d_visualization_plotly(representative_mesh, Xw_q, CANDIDATE_VIEWPOINTS, results, 
                                   "visibility_test_output/3d_visualization_with_robot.html",
                                   robot_mask, representative_robot_points_original, representative_robot_points_transformed)
    
    # 상세한 3D 시각화 생성 (로봇 변환 고려)
    create_comprehensive_3d_visualization(representative_mesh, Xw_q, CANDIDATE_VIEWPOINTS, results,
                                         "visibility_test_output/3d_comprehensive_with_robot.html")
    
    # 원본과 비교를 위한 시각화도 생성
    create_3d_visualization_plotly(mesh_original, Xw_q, CANDIDATE_VIEWPOINTS, results, 
                                   "visibility_test_output/3d_visualization_original.html",
                                   robot_mask, representative_robot_points_original, None)
    
    # M개의 viewpoint set을 모두 보여주는 3D 시각화 생성
    create_multi_viewpoint_set_3d_visualization(meshes_with_robot, Xw_q, CANDIDATE_VIEWPOINTS_MATRIX, 
                                               visibility_results, final_rewards,
                                               "visibility_test_output/3d_multi_viewpoint_sets.html")
    
    
    
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
