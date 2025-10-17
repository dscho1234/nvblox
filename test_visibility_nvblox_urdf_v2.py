import numpy as np
import math
import imageio.v3 as iio
import os
import sys

import open3d as o3d
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transforms3d.quaternions import quat2mat
import zarr
from tqdm import tqdm
import time
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import copy
import gc
from urdf_parser_py.urdf import URDF
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import trimesh
import cv2
# Unitree Z1 SDK 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "unitree_ros_to_real", "unitree_legged_sdk", "lib"))
sys.path.append("/home/dscho1234/Workspace/z1_sdk/lib")
import unitree_arm_interface

# UniDepth imports
from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
from unidepth.utils.camera import Pinhole
from unidepth.utils import colorize
from im2flow2act.common.imagecodecs_numcodecs import register_codecs
from im2flow2act.common.utility.zarr import parallel_reading

# nvblox imports
from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams, TsdfDecayIntegratorParams, MeshIntegratorParams

# ========= Verbose logging control =========
VERBOSE = False

def vprint(*args, **kwargs) -> None:
    """Print only when VERBOSE is True."""
    if VERBOSE:
        print(*args, **kwargs)


#!/usr/bin/env python3
"""
Unconstrained Inverse Kinematics Solver

This script provides methods to solve inverse kinematics without workspace constraints,
focusing only on achieving the target pose regardless of singularities or workspace limits.
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize
import time

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import unitree_arm_interface


    

# ========= Reward Tracking Class =========
class RewardTracker:
    """
    매 프레임마다 visibility reward를 추적하고 저장하는 클래스
    기존 카메라와 active camera의 reward를 모두 추적
    """
    def __init__(self):
        # 기존 카메라 reward 데이터
        self.current_frame_rewards = []  # 각 프레임의 기존 카메라 reward 저장
        self.current_frame_indices = []  # 프레임 인덱스 저장
        self.current_total_visible_points = []  # 각 프레임에서 보이는 포인트 수
        self.current_total_query_points = []  # 각 프레임의 총 쿼리 포인트 수
        
        # Active 카메라 reward 데이터
        self.active_frame_rewards = []  # 각 프레임의 active 카메라 reward 저장
        self.active_frame_indices = []  # 프레임 인덱스 저장
        self.active_total_visible_points = []  # 각 프레임에서 보이는 포인트 수
        self.active_total_query_points = []  # 각 프레임의 총 쿼리 포인트 수
        
    def add_current_camera_reward(self, frame_idx, visibility_results):
        """
        기존 카메라의 visibility 결과를 reward로 저장
        
        Args:
            frame_idx (int): 프레임 인덱스
            visibility_results (np.array): visibility 결과 배열 (boolean)
        """
        if visibility_results is not None:
            visible_count = np.sum(visibility_results)
            total_count = len(visibility_results)
            reward = visible_count / total_count if total_count > 0 else 0.0
            
            self.current_frame_rewards.append(reward)
            self.current_frame_indices.append(frame_idx)
            self.current_total_visible_points.append(visible_count)
            self.current_total_query_points.append(total_count)
            
            print(f"Frame {frame_idx} [Current Camera]: Reward = {reward:.4f} ({visible_count}/{total_count} points visible)")
        else:
            # visibility_results가 None인 경우 0 reward
            self.current_frame_rewards.append(0.0)
            self.current_frame_indices.append(frame_idx)
            self.current_total_visible_points.append(0)
            self.current_total_query_points.append(0)
            print(f"Frame {frame_idx} [Current Camera]: Reward = 0.0000 (no visibility data)")
    
    def add_active_camera_reward(self, frame_idx, visibility_results):
        """
        Active 카메라의 visibility 결과를 reward로 저장
        
        Args:
            frame_idx (int): 프레임 인덱스
            visibility_results (np.array): visibility 결과 배열 (boolean)
        """
        if visibility_results is not None:
            visible_count = np.sum(visibility_results)
            total_count = len(visibility_results)
            reward = visible_count / total_count if total_count > 0 else 0.0
            
            self.active_frame_rewards.append(reward)
            self.active_frame_indices.append(frame_idx)
            self.active_total_visible_points.append(visible_count)
            self.active_total_query_points.append(total_count)
            
            print(f"Frame {frame_idx} [Active Camera]: Reward = {reward:.4f} ({visible_count}/{total_count} points visible)")
        else:
            # visibility_results가 None인 경우 0 reward
            self.active_frame_rewards.append(0.0)
            self.active_frame_indices.append(frame_idx)
            self.active_total_visible_points.append(0)
            self.active_total_query_points.append(0)
            print(f"Frame {frame_idx} [Active Camera]: Reward = 0.0000 (no visibility data)")
    
    def add_frame_reward(self, frame_idx, visibility_results):
        """
        기존 호환성을 위한 메서드 (기존 카메라로 처리)
        """
        self.add_current_camera_reward(frame_idx, visibility_results)
    
    def plot_reward_changes(self, save_path="visibility_test_output"):
        """
        프레임에 따른 reward 변화를 plot하고 저장 (기존 카메라와 active 카메라 비교)
        
        Args:
            save_path (str): 저장할 경로
        """
        has_current_data = len(self.current_frame_rewards) > 0
        has_active_data = len(self.active_frame_rewards) > 0
        
        if not has_current_data and not has_active_data:
            print("No reward data to plot")
            return
            
        # matplotlib을 사용한 plot 생성
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Reward 변화 그래프 (비교)
        if has_current_data:
            ax1.plot(self.current_frame_indices, self.current_frame_rewards, 'b-', 
                    linewidth=2, marker='o', markersize=4, label='Current Camera', alpha=0.8)
        if has_active_data:
            ax1.plot(self.active_frame_indices, self.active_frame_rewards, 'r-', 
                    linewidth=2, marker='s', markersize=4, label='Active Camera', alpha=0.8)
        
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Reward (Visibility Ratio)')
        ax1.set_title('Visibility Reward Comparison: Current vs Active Camera')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.legend()
        
        # 2. 보이는 포인트 수 변화 (비교)
        if has_current_data:
            ax2.plot(self.current_frame_indices, self.current_total_visible_points, 'b-', 
                    linewidth=2, marker='o', markersize=4, label='Current Camera', alpha=0.8)
        if has_active_data:
            ax2.plot(self.active_frame_indices, self.active_total_visible_points, 'r-', 
                    linewidth=2, marker='s', markersize=4, label='Active Camera', alpha=0.8)
        
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Number of Visible Points')
        ax2.set_title('Visible Points Count Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Reward 분포 히스토그램 (비교)
        if has_current_data and has_active_data:
            ax3.hist(self.current_frame_rewards, bins=20, alpha=0.6, color='blue', 
                    edgecolor='black', label='Current Camera', density=True)
            ax3.hist(self.active_frame_rewards, bins=20, alpha=0.6, color='red', 
                    edgecolor='black', label='Active Camera', density=True)
            ax3.legend()
        elif has_current_data:
            ax3.hist(self.current_frame_rewards, bins=20, alpha=0.7, color='blue', 
                    edgecolor='black', label='Current Camera')
        elif has_active_data:
            ax3.hist(self.active_frame_rewards, bins=20, alpha=0.7, color='red', 
                    edgecolor='black', label='Active Camera')
        
        ax3.set_xlabel('Reward Value')
        ax3.set_ylabel('Density' if has_current_data and has_active_data else 'Frequency')
        ax3.set_title('Reward Distribution Comparison')
        ax3.grid(True, alpha=0.3)
        
        # 4. 누적 평균 reward (비교)
        if has_current_data:
            current_cumulative_avg = np.cumsum(self.current_frame_rewards) / np.arange(1, len(self.current_frame_rewards) + 1)
            ax4.plot(self.current_frame_indices, current_cumulative_avg, 'b-', 
                    linewidth=2, marker='o', markersize=4, label='Current Camera', alpha=0.8)
        if has_active_data:
            active_cumulative_avg = np.cumsum(self.active_frame_rewards) / np.arange(1, len(self.active_frame_rewards) + 1)
            ax4.plot(self.active_frame_indices, active_cumulative_avg, 'r-', 
                    linewidth=2, marker='s', markersize=4, label='Active Camera', alpha=0.8)
        
        ax4.set_xlabel('Frame Index')
        ax4.set_ylabel('Cumulative Average Reward')
        ax4.set_title('Cumulative Average Reward Comparison')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, "reward_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 통계 정보 출력
        print(f"\n=== Reward Analysis Summary ===")
        if has_current_data:
            print(f"Current Camera - Total frames: {len(self.current_frame_rewards)}")
            print(f"Current Camera - Average reward: {np.mean(self.current_frame_rewards):.4f}")
            print(f"Current Camera - Max reward: {np.max(self.current_frame_rewards):.4f}")
            print(f"Current Camera - Min reward: {np.min(self.current_frame_rewards):.4f}")
            print(f"Current Camera - Std deviation: {np.std(self.current_frame_rewards):.4f}")
            print(f"Current Camera - Final reward: {self.current_frame_rewards[-1]:.4f}")
        
        if has_active_data:
            print(f"Active Camera - Total frames: {len(self.active_frame_rewards)}")
            print(f"Active Camera - Average reward: {np.mean(self.active_frame_rewards):.4f}")
            print(f"Active Camera - Max reward: {np.max(self.active_frame_rewards):.4f}")
            print(f"Active Camera - Min reward: {np.min(self.active_frame_rewards):.4f}")
            print(f"Active Camera - Std deviation: {np.std(self.active_frame_rewards):.4f}")
            print(f"Active Camera - Final reward: {self.active_frame_rewards[-1]:.4f}")
        
        if has_current_data and has_active_data:
            current_avg = np.mean(self.current_frame_rewards)
            active_avg = np.mean(self.active_frame_rewards)
            improvement = ((active_avg - current_avg) / current_avg * 100) if current_avg > 0 else 0
            print(f"\nComparison:")
            print(f"Active camera improvement: {improvement:+.2f}%")
            print(f"Reward difference: {active_avg - current_avg:+.4f}")
        
        print(f"Reward analysis plot saved to: {plot_path}")
        
        return plot_path

# ========= Z1 Robot Visualizer Class =========
class Z1RobotVisualizer:
    def __init__(self, urdf_path, mesh_base_path):
        """
        Z1 로봇 시각화 클래스 초기화
        
        Args:
            urdf_path: URDF 파일 경로
            mesh_base_path: 메시 파일들이 있는 기본 경로
        """
        self.urdf_path = urdf_path
        self.mesh_base_path = mesh_base_path
        self.robot = None
        self.joint_angles = np.zeros(6)  # 6개 조인트 각도
        self.link_transforms = {}  # 각 링크의 변환 행렬 저장
        self.meshes = {}  # 로드된 메시들 저장
        # color

        # self.link_colors = {
        #     'link00': [0.8, 0.2, 0.2],      # 빨간색 (베이스)
        #     'link01': [0.2, 0.8, 0.2],         # 초록색 (링크1)
        #     'link02': [0.2, 0.2, 0.8],         # 파란색 (링크2)
        #     'link03': [0.8, 0.8, 0.2],         # 노란색 (링크3)
        #     'link04': [0.8, 0.2, 0.8],         # 자홍색 (링크4)
        #     'link05': [0.2, 0.8, 0.8],         # 청록색 (링크5)
        #     'link06': [0.5, 0.5, 0.5],         # 회색 (링크6)
        #     'z1_GripperMover': [0.9, 0.5, 0.1],   # 주황색 (그리퍼)
        #     'z1_GripperStator': [0.5, 0.9, 0.1],  # 연두색 (그리퍼)
        # }
        # silver 
        self.link_colors = {
            'link00': [0.8, 0.8, 0.8],    
            'link01': [0.8, 0.8, 0.8],         
            'link02': [0.8, 0.8, 0.8],         
            'link03': [0.8, 0.8, 0.8],         
            'link04': [0.8, 0.8, 0.8],         
            'link05': [0.8, 0.8, 0.8],         
            'link06': [0.8, 0.8, 0.8],         
            'z1_GripperMover': [0.8, 0.8, 0.8],  
            'z1_GripperStator': [0.8, 0.8, 0.8],
        }

        self.joint_limits = [
            (-2.618, 2.618),   # J1: ±150°
            (0, 3.142),        # J2: 0—180°
            (-2.879, 0),       # J3: -165°—0
            (-1.396, 1.396),   # J4: ±80°
            (-1.484, 1.484),   # J5: ±85°
            (-2.793, 2.793)    # J6: ±160°
        ]
        

        # Unitree Z1 SDK 초기화
        try:
            self.arm_interface = unitree_arm_interface.ArmInterface(hasGripper=True)
            vprint("Unitree Z1 SDK 초기화 성공")
        except Exception as e:
            vprint(f"Unitree Z1 SDK 초기화 실패: {e}")
            self.arm_interface = None
        
        # URDF 파싱
        self.parse_urdf()
        
        # 메시 로딩
        self.load_meshes()
        
    def parse_urdf(self):
        """URDF 파일을 파싱하여 로봇 구조 정보 추출"""
        try:
            self.robot = URDF.from_xml_file(self.urdf_path)
            vprint(f"URDF 파싱 완료: {len(self.robot.joints)} 개 조인트, {len(self.robot.links)} 개 링크")
            
            # 조인트 정보 출력
            for i, joint in enumerate(self.robot.joints):
                vprint(f"Joint {i+1}: {joint.name}, Type: {joint.type}, Axis: {joint.axis}")
                
        except Exception as e:
            vprint(f"URDF 파싱 오류: {e}")
            
    def load_meshes(self):
        """각 링크의 메시 파일을 로드"""
        # URDF에서 메시 파일 경로 추출
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        for link in root.findall('link'):
            link_name = link.get('name')
            visual = link.find('visual')
            
            if visual is not None:
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh_filename = mesh.get('filename')
                        if mesh_filename and 'package://' in mesh_filename:
                            # package:// 경로를 실제 파일 경로로 변환
                            mesh_path = mesh_filename.replace('package://z1_description/', self.mesh_base_path)
                            
                            # 먼저 STL 파일 시도
                            stl_path = mesh_path.replace('.dae', '.STL').replace('visual/', 'collision/')
                            if os.path.exists(stl_path):
                                # STL 파일 로드
                                mesh_obj = o3d.io.read_triangle_mesh(stl_path)
                                if len(mesh_obj.vertices) > 0:
                                    # 메시 단순화 (raycasting 성능 향상)
                                    original_vertices = len(mesh_obj.vertices)
                                    original_triangles = len(mesh_obj.triangles)
                                    
                                    # 삼각형 개수를 50%로 줄이기 (target_triangle_count 설정)
                                    if link_name in ['link00', 'link01', 'link02', 'link03', 'link04']:
                                        target_triangle_count = 100  # 최소 100개 삼각형 유지
                                    else:
                                        target_triangle_count = max(100, original_triangles // 4)  # 최소 100개 삼각형 유지
                                    
                                    # 메시 단순화
                                    mesh_obj = mesh_obj.simplify_quadric_decimation(target_triangle_count)
                                    
                                    # vertices도 최적화 (중복 제거 및 정리)
                                    mesh_obj.remove_degenerate_triangles()
                                    mesh_obj.remove_duplicated_triangles()
                                    mesh_obj.remove_duplicated_vertices()
                                    mesh_obj.remove_unreferenced_vertices()
                                    
                                    # 메시가 너무 단순화되었는지 확인
                                    if len(mesh_obj.vertices) < 10:
                                        # 너무 단순화된 경우 원본 메시 사용
                                        mesh_obj = o3d.io.read_triangle_mesh(stl_path)
                                        print(f"메시가 너무 단순화됨, 원본 사용: {link_name}")
                                    else:
                                        print(f"메시 단순화 완료: {link_name} - {original_vertices}→{len(mesh_obj.vertices)} vertices, {original_triangles}→{len(mesh_obj.triangles)} triangles")
                                    
                                    mesh_obj.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array(self.link_colors[link_name])[None, :], (len(mesh_obj.vertices), 1)))
                                    self.meshes[link_name] = mesh_obj
                                    vprint(f"STL 메시 로드 성공: {link_name} -> {stl_path} ({len(mesh_obj.vertices)} vertices)")
                                    continue
                            
                            # DAE 파일 시도
                            trimesh_obj = trimesh.load(mesh_path)
                            
                            # Open3D 메시로 변환
                            if hasattr(trimesh_obj, 'vertices') and hasattr(trimesh_obj, 'faces'):
                                mesh_obj = o3d.geometry.TriangleMesh()
                                mesh_obj.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices)
                                mesh_obj.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces)
                                
                                if len(mesh_obj.vertices) > 0:
                                    self.meshes[link_name] = mesh_obj
                                    vprint(f"DAE 메시 로드 성공: {link_name} -> {mesh_path} ({len(mesh_obj.vertices)} vertices)")
                                else:
                                    vprint(f"빈 메시: {link_name}")
                            else:
                                vprint(f"메시 형식 오류: {link_name}")
                        
        
        # Gripper 메시도 로드 (URDF에 정의되지 않았지만 메시 파일이 있음)
        self.load_gripper_meshes()
                                
    def load_gripper_meshes(self):
        """Gripper 메시 파일들을 로드"""
        gripper_meshes = ['z1_GripperMover', 'z1_GripperStator']
        
        for gripper_name in gripper_meshes:
            # DAE 파일 경로
            dae_path = os.path.join(self.mesh_base_path, 'meshes', 'visual', f'{gripper_name}.dae')
            # STL 파일 경로
            stl_path = os.path.join(self.mesh_base_path, 'meshes', 'collision', f'{gripper_name}.STL')
            
            try:
                # 먼저 STL 파일 시도
                if os.path.exists(stl_path):
                    mesh_obj = o3d.io.read_triangle_mesh(stl_path)
                    if len(mesh_obj.vertices) > 0:
                        # 메시 단순화 (raycasting 성능 향상)
                        original_vertices = len(mesh_obj.vertices)
                        original_triangles = len(mesh_obj.triangles)
                        
                        # 삼각형 개수를 50%로 줄이기 (target_triangle_count 설정)
                        target_triangle_count = max(100, original_triangles // 4)  # 최소 100개 삼각형 유지
                        
                        # 메시 단순화
                        mesh_obj = mesh_obj.simplify_quadric_decimation(target_triangle_count)
                        
                        # vertices도 최적화 (중복 제거 및 정리)
                        mesh_obj.remove_degenerate_triangles()
                        mesh_obj.remove_duplicated_triangles()
                        mesh_obj.remove_duplicated_vertices()
                        mesh_obj.remove_unreferenced_vertices()
                        
                        # 메시가 너무 단순화되었는지 확인
                        if len(mesh_obj.vertices) < 10:
                            # 너무 단순화된 경우 원본 메시 사용
                            mesh_obj = o3d.io.read_triangle_mesh(stl_path)
                            print(f"Gripper 메시가 너무 단순화됨, 원본 사용: {gripper_name}")
                        else:
                            print(f"Gripper 메시 단순화 완료: {gripper_name} - {original_vertices}→{len(mesh_obj.vertices)} vertices, {original_triangles}→{len(mesh_obj.triangles)} triangles")
                        
                        mesh_obj.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array(self.link_colors[gripper_name])[None, :], (len(mesh_obj.vertices), 1)))            
                        self.meshes[gripper_name] = mesh_obj
                        vprint(f"Gripper STL 메시 로드 성공: {gripper_name} -> {stl_path} ({len(mesh_obj.vertices)} vertices)")
                        continue
                
                # DAE 파일 시도
                if os.path.exists(dae_path):
                    trimesh_obj = trimesh.load(dae_path)
                    if hasattr(trimesh_obj, 'vertices') and hasattr(trimesh_obj, 'faces'):
                        mesh_obj = o3d.geometry.TriangleMesh()
                        mesh_obj.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices)
                        mesh_obj.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces)
                        
                        if len(mesh_obj.vertices) > 0:
                            self.meshes[gripper_name] = mesh_obj
                            vprint(f"Gripper DAE 메시 로드 성공: {gripper_name} -> {dae_path} ({len(mesh_obj.vertices)} vertices)")
                            continue
                            
            except Exception as e:
                vprint(f"Gripper 메시 로드 실패: {gripper_name} - {e}")
                                
    
                                
    def set_joint_angles(self, angles):
        """조인트 각도 설정"""
        self.joint_angles = np.array(angles)
        vprint(f"조인트 각도 설정: {self.joint_angles}")
    
    
    
    def solve_ik_optimization(self, target_T, initial_guess=None):
        """
        Solve IK using optimization-based method with joint limits.
        
        Args:
            target_T: Target 4x4 transformation matrix
            initial_guess: Initial joint angle guess
            
        Returns:
            tuple: (success, joint_angles, final_error)
        """
        if initial_guess is None:
            initial_guess = np.zeros(6)
        
        # Use class joint limits
        
        def objective(q):
            """Objective function: pose error."""
            try:
                current_T = self.arm_interface._ctrlComp.armModel.forwardKinematics(q, 6)
                error = self._compute_pose_error(target_T, current_T)
                return np.sum(error**2)
            except:
                return 1e6  # Large error if FK fails
        
        # Use scipy optimization with bounds
        result = minimize(
            objective, 
            initial_guess, 
            method='L-BFGS-B',
            bounds=self.joint_limits,
            options={'maxiter': 1000, 'gtol': 1e-8}
        )
        
        if result.success:
            final_error = np.sqrt(result.fun)
            return True, result.x, final_error
        else:
            return False, result.x, np.sqrt(result.fun)
    
    def solve_ik_multiple_guesses(self, target_T, initial_guess=None, num_guesses=10, noise_std=0.1):
        """
        Solve IK using multiple random initial guesses within joint limits.
        
        Args:
            target_T: Target 4x4 transformation matrix
            num_guesses: Number of random initial guesses to try
            
        Returns:
            tuple: (success, best_joint_angles, best_error)
        """
        # Use class joint limits
        
        best_error = float('inf')
        best_q = None
        best_success = False
        
        for i in range(num_guesses):
            # Generate random initial guess within joint limits
            if initial_guess is not None and i == 0:
                # First guess: use the provided initial_guess
                guess = initial_guess.copy()
            # elif initial_guess is not None:
            #     # Subsequent guesses: add noise around initial_guess
            #     noise = np.random.normal(0, noise_std, 6)
            #     guess = initial_guess + noise
                
            #     # Ensure the guess is within joint limits
            #     guess = np.clip(guess, 
            #                   [limit[0] for limit in self.joint_limits], 
            #                   [limit[1] for limit in self.joint_limits])
            else:
                # No initial_guess provided: generate random guess within joint limits
                guess = np.array([
                    np.random.uniform(self.joint_limits[j][0], self.joint_limits[j][1]) 
                    for j in range(6)
                ])
            
            # Try optimization method
            success, q, error = self.solve_ik_optimization(target_T, guess)
            
            if success and error < best_error:
                best_error = error
                best_q = q
                best_success = True
            
            if error < 0.001:
                break
        
        return best_success, best_q, best_error
    
    def _compute_pose_error(self, target_T, current_T):
        """
        Compute pose error between target and current transformation matrices.
        
        Args:
            target_T: Target 4x4 transformation matrix
            current_T: Current 4x4 transformation matrix
            
        Returns:
            np.array: 6D error vector [position_error(3), orientation_error(3)]
        """
        # Position error
        pos_error = target_T[:3, 3] - current_T[:3, 3]
        
        # Orientation error using axis-angle representation
        R_rel = target_T[:3, :3].T @ current_T[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
        
        if angle > 1e-6:
            axis = np.array([R_rel[2,1] - R_rel[1,2], 
                            R_rel[0,2] - R_rel[2,0], 
                            R_rel[1,0] - R_rel[0,1]]) / (2 * np.sin(angle))
            rot_error = angle * axis
        else:
            rot_error = np.zeros(3)
        
        # Combine position and rotation errors
        error = np.concatenate([pos_error, rot_error])
        
        return error
    

    def solve_inverse_kinematics(self, target_pose):
        """Inverse Kinematics를 사용하여 목표 pose에서 조인트 각도 계산"""
        if self.arm_interface is None:
            vprint("Unitree Z1 SDK가 초기화되지 않았습니다.")
            return False
            
        try:
            # 현재 조인트 각도를 초기 추정값으로 사용
            current_q = self.joint_angles.copy()
            vprint(f"초기 조인트 각도: {current_q}")
            if (current_q == np.zeros(6)).all():
                initial_guess = None
            else:
                initial_guess = current_q

            

            # Inverse Kinematics 계산 (it often outputs fail even though the target_pose and initial_guess are identical. Maybe due to its internal logic to consider singularities, workspace limits, etc.)
            # success, q_forward = self.arm_interface._ctrlComp.armModel.inverseKinematics(
            #     target_pose, current_q, True  # checkInWorkSpace=True
            # )

            # custom inverse kinematics
            success, q_forward, error = self.solve_ik_multiple_guesses(target_pose, initial_guess=initial_guess, num_guesses=10, noise_std=0.3)
            
            # 계산된 조인트 각도로 업데이트
            self.joint_angles = q_forward
            print(f'Custom IK error: {error:.6f}')
            vprint(f"계산 후 조인트 각도: {q_forward}")
            
            if success:
                
                vprint(f"Inverse Kinematics 성공: {self.joint_angles}")
                
                # Forward Kinematics로 검증
                fk_result = self.arm_interface._ctrlComp.armModel.forwardKinematics(q_forward, 6)
                vprint(f"Forward Kinematics 검증 결과:\n{fk_result}")
                
                return True
            else:
                vprint("Inverse Kinematics 실패: 목표 pose가 작업 공간 밖에 있습니다.")
                return False
                
        except Exception as e:
            vprint(f"Inverse Kinematics 오류: {e}")
            return False
        
    def compute_forward_kinematics(self, gripper_angle=0.0):
        """전진기구학 계산 - URDF 기반 (Unitree SDK는 절대 위치를 반환하므로 부적합)"""
        # Unitree Z1 SDK는 절대 위치를 반환하므로 URDF 기반 계산 사용
        self.compute_forward_kinematics_urdf(gripper_angle)
    
    def compute_forward_kinematics_urdf(self, gripper_angle=0.0):
        """URDF 기반 전진기구학 계산 (폴백)"""
        # 조인트 정보를 딕셔너리로 저장
        joint_info = {}
        for joint in self.robot.joints:
            joint_info[joint.name] = joint
            
        # 링크 정보를 딕셔너리로 저장
        link_info = {}
        for link in self.robot.links:
            link_info[link.name] = link
            
        # 변환 행렬 초기화
        self.link_transforms = {}
        
        # world -> link00 (고정 조인트)
        self.link_transforms['world'] = np.eye(4)
        self.link_transforms['link00'] = np.eye(4)
        
        # 각 조인트에 대해 변환 행렬 계산
        joint_angle_idx = 0
        for joint in self.robot.joints:
            if joint.type == 'revolute' and joint_angle_idx < len(self.joint_angles):
                # 조인트 각도
                angle = self.joint_angles[joint_angle_idx]
                joint_angle_idx += 1
                
                # 조인트 축
                axis = np.array(joint.axis)
                
                # 조인트 원점
                origin = joint.origin
                if origin is not None:
                    xyz = np.array(origin.xyz) if origin.xyz else np.zeros(3)
                    rpy = np.array(origin.rpy) if origin.rpy else np.zeros(3)
                else:
                    xyz = np.zeros(3)
                    rpy = np.zeros(3)
                
                # 회전 행렬 계산 (RPY)
                if np.any(rpy):
                    rot_matrix = R.from_euler('xyz', rpy).as_matrix()
                else:
                    rot_matrix = np.eye(3)
                
                # 조인트 회전 행렬 (축 주위 회전)
                joint_rot_matrix = R.from_rotvec(axis * angle).as_matrix()
                
                # 변환 행렬 구성
                T_joint = np.eye(4)
                T_joint[:3, :3] = rot_matrix @ joint_rot_matrix
                T_joint[:3, 3] = xyz
                
                # 부모 링크의 변환 행렬과 결합
                parent_transform = self.link_transforms.get(joint.parent, np.eye(4))
                child_transform = parent_transform @ T_joint
                
                self.link_transforms[joint.child] = child_transform
                
            elif joint.type == 'fixed':
                # 고정 조인트
                origin = joint.origin
                if origin is not None:
                    xyz = np.array(origin.xyz) if origin.xyz else np.zeros(3)
                    rpy = np.array(origin.rpy) if origin.rpy else np.zeros(3)
                else:
                    xyz = np.zeros(3)
                    rpy = np.zeros(3)
                
                # 회전 행렬 계산
                if np.any(rpy):
                    rot_matrix = R.from_euler('xyz', rpy).as_matrix()
                else:
                    rot_matrix = np.eye(3)
                
                # 변환 행렬 구성
                T_fixed = np.eye(4)
                T_fixed[:3, :3] = rot_matrix
                T_fixed[:3, 3] = xyz
                
                # 부모 링크의 변환 행렬과 결합
                parent_transform = self.link_transforms.get(joint.parent, np.eye(4))
                child_transform = parent_transform @ T_fixed
                
                self.link_transforms[joint.child] = child_transform
        
        # Gripper 위치 계산 (URDF 기반 정확한 오프셋 사용)
        if 'link06' in self.link_transforms:
            # gripperStator는 link06에서 xyz="0.051 0.0 0.0" 오프셋으로 연결
            gripper_stator_offset = np.array([0.051, 0.0, 0.0])  # URDF에서 정의된 오프셋
            gripper_stator_transform = self.link_transforms['link06'].copy()
            gripper_stator_transform[:3, 3] += gripper_stator_transform[:3, :3] @ gripper_stator_offset
            
            self.link_transforms['z1_GripperStator'] = gripper_stator_transform
            
            # gripperMover는 gripperStator에서 xyz="0.049 0.0 0" 오프셋으로 연결
            gripper_mover_offset = np.array([0.049, 0.0, 0.0])  # URDF에서 정의된 오프셋
            gripper_mover_transform = gripper_stator_transform.copy()
            gripper_mover_transform[:3, 3] += gripper_mover_transform[:3, :3] @ gripper_mover_offset
            
            # gripper 회전 적용 (Y축 주위 회전, URDF에서 axis xyz="0 1 0")
            if gripper_angle != 0.0:
                gripper_rotation = R.from_rotvec([0, gripper_angle, 0]).as_matrix()
                gripper_mover_transform[:3, :3] = gripper_mover_transform[:3, :3] @ gripper_rotation
            
            self.link_transforms['z1_GripperMover'] = gripper_mover_transform
                
    
    
    def transform_mesh_to_world(self, mesh, transform):
        """메시를 월드 좌표계로 변환"""
        
        # 1. 정점을 직접 변환 (메시 복사 없이)
        vertices = np.asarray(mesh.vertices)
        
        # 2. 변환 행렬을 3x3 회전과 3x1 이동으로 분해하여 더 효율적인 연산
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        
        # 3. 벡터화된 변환 (더 효율적)
        vertices_world = vertices @ rotation.T + translation
        
        # 4. 새로운 메시 생성 (최소한의 복사)
        transformed_mesh = o3d.geometry.TriangleMesh()
        transformed_mesh.triangles = mesh.triangles
        transformed_mesh.vertex_colors = mesh.vertex_colors
        
        # 5. 변환된 정점을 직접 할당
        transformed_mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
        
        return transformed_mesh
        
    
    def get_robot_meshes_in_specified_transform(self, gripper_angle=0.0, T_target=None):
        """로봇의 모든 링크 메시를 지정된 좌표계로 변환하여 반환
        
        Args:
            gripper_angle: 그리퍼 각도
            T_target: 목표 좌표계 변환 행렬 (4x4). None이면 월드 좌표계 사용
        
        Returns:
            transformed_meshes: 지정된 좌표계로 변환된 메시 딕셔너리
        """
        # 전진기구학 계산
        self.compute_forward_kinematics(gripper_angle)
        
        transformed_meshes = {}
        for link_name, mesh in self.meshes.items():
            
            if link_name in self.link_transforms:
                if T_target is not None:
                    # 월드 좌표계에서 목표 좌표계로 변환
                    world_transform = self.link_transforms[link_name]
                    target_transform = T_target @ world_transform
                    
                    # 메시를 목표 좌표계로 변환
                    target_mesh = self.transform_mesh_to_world(mesh, target_transform)
                    transformed_meshes[link_name] = target_mesh
                else:
                    # T_target이 None이면 월드 좌표계 사용
                    world_mesh = self.transform_mesh_to_world(mesh, self.link_transforms[link_name])
                    transformed_meshes[link_name] = world_mesh
            
        return transformed_meshes

    
    


# ========= 사용자 설정 =========
# Zarr 데이터 경로 설정
# BUFFER_PATH = "/home/dscho1234/fast_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/single_marker_bottle_under_table_v3"
BUFFER_PATH = "/home/dscho1234/fast_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/single_marker_static_bottle_hamer"
USE_DROID = False # True
EPISODE_IDX = 0
FRAME_IDX = 0  # 특정 프레임 선택
DEPTH_SCALE = 0.001        # 깊이 단위 → 미터 변환 (예: mm면 0.001, 이미 m면 1.0)
OFFSET_DISTANCE = 0.05 # for convex part of the constructed mesh



# Mesh 품질 선택: "high_resolution" (5mm), "medium_resolution" (10mm), "low_resolution" (20mm)
MESH_QUALITY = "medium_resolution" # "medium_resolution"  # "high_resolution", "medium_resolution", "low_resolution"

if MESH_QUALITY == "high_resolution":
    VOXEL_SIZE = 0.005  # 5mm - 높은 해상도, 조각난 mesh
    
elif MESH_QUALITY == "medium_resolution":
    VOXEL_SIZE = 0.01   # 10mm - 중간 해상도, 균형잡힌 mesh
    
else:  # low_resolution
    VOXEL_SIZE = 0.02   # 20mm - 낮은 해상도, 매끄러운 mesh
    

vprint(f"Using {MESH_QUALITY}: voxel_size={VOXEL_SIZE}m")

# Depth estimation 설정
USE_MONODEPTH = False # True  # True: UniDepth 사용, False: raw depth 사용
MODEL_TYPE = "l"  # UniDepth model type: s, b, l


USE_FAKE_DEPTH = True
FAKE_DEPTH_VALUE = 0.0  # nvblox는 depth <= 0을 무효한 depth로 처리함

# 이미지 리사이즈 설정
RESIZE = True  # True: 이미지를 256x256으로 리사이즈, False: 원본 크기 사용
RESIZE_SIZE = (256, 256)  # 리사이즈할 크기 (width, height)
IMAGE_SIZE = (640, 480)
# (중요) 카메라 내파라미터: 사용자가 직접 채우세요.
K = np.array([
    [604.682922, 0.0, 328.062561],
    [0.0, 604.898438, 244.393188],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

T_B_M = np.array([
            [-0.0211232, 0.00961882, -0.99973061, 0.87401688],
            [ 0.00934025, -0.99990818, -0.00981788,  0.0965125 ],
            [-0.99973325, -0.00954512,  0.02103141,  0.54736933],
            [ 0.,          0.,          0.,          1.        ]
        ])
        


from scipy.ndimage import distance_transform_edt

def fill_depth_zeros(depth_img):
    # 0이거나 무효한 값들을 마스크로 처리
    mask = (depth_img == 0) | (depth_img <= 0) | np.isnan(depth_img)
    if not np.any(mask):
        return depth_img
    distance, indices = distance_transform_edt(mask, return_indices=True)
    filled = depth_img[tuple(indices)]
    return filled
    
def convert_to_relative(se3_matrices):
    # SE(3) matrices [B, H, 4, 4]
    
    # Get inverse of first transformation for each batch [B, 4, 4]
    first_transforms = se3_matrices[:, 0]  # [B, 4, 4]
    first_inv = np.linalg.inv(first_transforms)  # [B, 4, 4]
    
    # Apply relative transformation: T_rel = T_first_inv @ T_current
    # Broadcasting: [B, 4, 4] @ [B, H, 4, 4] -> [B, H, 4, 4]
    relative_se3 = np.einsum('bij,bhjk->bhik', first_inv, se3_matrices)
    
    # Set first index to identity
    relative_se3[:, 0] = np.eye(4)
    
    return relative_se3


# ========= 카메라 Frustum 체크 함수 =========
def is_point_in_camera_frustum(point_3d, camera_position, camera_rotation_rpy, camera_intrinsics, image_size):
    """
    3D 포인트가 카메라 frustum 내에 있는지 체크
    
    Args:
        point_3d: 3D 포인트 [x, y, z] (월드 좌표)
        camera_position: 카메라 위치 [x, y, z] (월드 좌표)
        camera_rotation_rpy: 카메라 회전 [roll, pitch, yaw] (degrees)
        camera_intrinsics: 카메라 내부 파라미터 (3x3)
        image_size: 이미지 크기 (width, height)
    
    Returns:
        bool: 포인트가 frustum 내에 있으면 True
    """
    # 1. 월드 좌표를 카메라 좌표로 변환
    point_3d = np.array(point_3d, dtype=np.float64)
    camera_position = np.array(camera_position, dtype=np.float64)
    
    # 카메라 회전 행렬 생성 (RPY 순서)
    roll, pitch, yaw = np.deg2rad(camera_rotation_rpy)
    R_cam = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    
    # 월드 좌표를 카메라 좌표로 변환
    point_cam = R_cam.T @ (point_3d - camera_position)
    
    # 2. 카메라 좌표에서 이미지 평면으로 투영
    if point_cam[2] <= 0:  # 카메라 뒤쪽에 있으면 보이지 않음
        return False
    
    # 투영
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    u = fx * point_cam[0] / point_cam[2] + cx
    v = fy * point_cam[1] / point_cam[2] + cy
    
    # 3. 이미지 경계 내에 있는지 체크
    width, height = image_size
    return 0 <= u < width and 0 <= v < height



def create_camera_frustum_visualization(camera_position, camera_rotation_rpy, camera_intrinsics, image_size, max_distance=0.5):
    """
    카메라 frustum을 시각화하기 위한 8개 꼭짓점 생성
    
    Args:
        camera_position: 카메라 위치 [x, y, z]
        camera_rotation_rpy: 카메라 회전 [roll, pitch, yaw] (degrees)
        camera_intrinsics: 카메라 내부 파라미터 (3x3)
        image_size: 이미지 크기 (width, height)
        max_distance: frustum의 최대 거리
    
    Returns:
        frustum_vertices: frustum의 8개 꼭짓점 (8, 3)
    """
    # 카메라 회전 행렬 생성
    roll, pitch, yaw = np.deg2rad(camera_rotation_rpy)
    R_cam = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    
    
    # 카메라 내부 파라미터
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    width, height = image_size
    
    # 이미지 모서리 4개 점 (픽셀 좌표)
    corners_2d = np.array([
        [0, 0],           # 좌상단
        [width, 0],       # 우상단
        [width, height],  # 우하단
        [0, height]       # 좌하단
    ])
    
    # 각 거리에서 frustum 꼭짓점 계산
    frustum_vertices = []
    
    # 카메라 위치 (근거리)
    frustum_vertices.append(camera_position)
    
    # 원거리 frustum 꼭짓점들
    for i, corner_2d in enumerate(corners_2d):
        u, v = corner_2d
        
        # 카메라 좌표에서 3D 방향 계산 (Z축이 앞쪽을 향함)
        x_cam = (u - cx) * max_distance / fx
        y_cam = (v - cy) * max_distance / fy
        z_cam = max_distance
        
        # 월드 좌표로 변환
        point_cam = np.array([x_cam, y_cam, z_cam])
        point_world = camera_position + R_cam @ point_cam
        
        frustum_vertices.append(point_world)
        
    
    return np.array(frustum_vertices)


# ========= 유틸 함수 =========
def resize_image_and_depth(rgb, depth, target_size=(256, 256), fake_depth_mask=None):
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
    
    
    H, W = rgb.shape[:2]
    target_w, target_h = target_size
    
    rgb_uint8 = rgb.astype(np.uint8)
    
    rgb_resized = cv2.resize(rgb_uint8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    
    
    # Depth 이미지 리사이즈 (INTER_NEAREST 사용)
    depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # 스케일 팩터 계산 (x, y 방향 각각)
    scale_factor_x = (target_w-1) / (W-1)
    scale_factor_y = (target_h-1) / (H-1)

    if fake_depth_mask is not None:
        fake_depth_mask_resized = cv2.resize(fake_depth_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        fake_depth_mask_resized = None
    
    return rgb_resized, depth_resized, scale_factor_x, scale_factor_y, fake_depth_mask_resized


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


def apply_fake_depth_to_mask(depths, mask, fake_value):
    """
    Apply fake depth (0) to depth values where mask=1
    
    Args:
        depths: numpy array of shape [t, h, w] with depth values
        mask: numpy array of shape [t, h, w] with binary values (0 or 1)
    
    Returns:
        Modified depths with 0 values in masked regions
    """    
    # Create a copy to avoid modifying original data
    modified_depths = depths.copy()
    
    # Apply 0 depth to masked regions
    modified_depths = np.where(mask == 1, fake_value, modified_depths)
    
    return modified_depths

def load_data_from_zarr(buffer_path, episode_idx, droid=False):
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
    
    # Depth 프레임 로드    
    depth_frames = parallel_reading(
        group=episode["camera_0"],
        array_name="depth",
    )
    if droid:
        T_mc_transformation = episode["T_mc_opt_droid"][:].copy() # [T, 4, 4]
    else:
        T_mc_transformation = episode["T_mc_opt"][:].copy() # [T, 4, 4]
    relative_poses = convert_to_relative(T_mc_transformation[None])[0] # [T, 4, 4]

    action = episode["action"][:].copy() # [T, 7]
    tracking_3d = np.transpose(episode["dift_point_tracking_sequence"][:, :, :3], (1, 0, 2)) # [T, N, 3(or 4)]
    assert rgb_frames.shape[0] == action.shape[0] == tracking_3d.shape[0] == T_mc_transformation.shape[0], f"rgb_frames.shape: {rgb_frames.shape}, action.shape: {action.shape}, tracking_3d.shape: {tracking_3d.shape}, T_mc_transformation.shape: {T_mc_transformation.shape}"


    combined_mask = None
    if USE_FAKE_DEPTH:
        # assert cfg.use_video, "When you use HaMeR related outputs, you should consider that some frames are cut, and the logging should start from the first hand-detected frames"
        mask = episode["sam_mask_sequence_multi_obj"][:] # [t, num_obj(obj, hand), h, w]
        
        use_video = True
        if not use_video: # when you load from raw png files
            all_detected_frame_index = episode["all_detected_frame_index"][()] # get scalar
            assert rgb_frames[all_detected_frame_index:].shape[0] == mask.shape[0], f"rgb_frames shape: {rgb_frames[all_detected_frame_index:].shape}, mask shape: {mask.shape}"
        else: # when you load from parallel_reading
            all_detected_frame_index = 0
        
        print(f"mask.shape: {mask.shape}, all_detected_frame_index: {all_detected_frame_index}")

        assert rgb_frames[all_detected_frame_index:].shape[0] == mask.shape[0], f"rgb_frames shape: {rgb_frames[all_detected_frame_index:].shape}, mask shape: {mask.shape}"
        
        # Combine all objects into a single mask [t, h, w]
        # Any pixel that is 1 in any object becomes 1 in the combined mask
        combined_mask = np.any(mask == 1, axis=1)  # [t, h, w]
        print(f"combined_mask.shape: {combined_mask.shape}")


    print(f"rgb_frames.shape: {rgb_frames.shape}")
    print(f"depth_frames.shape: {depth_frames.shape}")
    print(f"relative_poses.shape: {relative_poses.shape}")
    return rgb_frames, depth_frames, relative_poses, combined_mask, T_mc_transformation, action, tracking_3d

def get_data(rgb_frames, depth_frames, relative_poses, combined_mask, T_mc_transformation, action, tracking_3d, frame_idx, depth_scale=0.001):
    

    fake_depth_mask = None
    if USE_FAKE_DEPTH:
        fake_depth_mask = combined_mask[frame_idx]

            

    rgb = rgb_frames[frame_idx]  # 특정 프레임 선택
    depth_raw = depth_frames[frame_idx]  # 특정 프레임 선택
    depth_m = depth_raw.astype(np.float32) * depth_scale
    relative_pose = relative_poses[frame_idx]
    T_mc = T_mc_transformation[frame_idx]
    act = action[frame_idx]    
    track_3d = tracking_3d[frame_idx]
    
            
    
    return rgb, depth_m, fake_depth_mask, relative_pose, T_mc, act, track_3d


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



def rgbd_to_point_cloud(rgb_frame, depth_frame, camera_intrinsics, max_depth=5.0, depth_scale=1.0):
    """
    RGB-D 프레임을 point cloud로 변환
    
    Args:
        rgb_frame: RGB 이미지 (H, W, 3)
        depth_frame: Depth 이미지 (H, W)
        camera_intrinsics: 카메라 내부 파라미터 (3, 3)
        max_depth: 최대 깊이 값 (이보다 큰 값은 제외)
        depth_scale: 깊이 스케일 팩터
    
    Returns:
        o3d.geometry.PointCloud: Open3D PointCloud 객체
    """
    
    H, W = depth_frame.shape
    
    # 유효한 깊이 값만 선택 (0보다 크고 max_depth보다 작은 값)
    valid_mask = (depth_frame > 0) & (depth_frame < max_depth)
    
    if not np.any(valid_mask):
        print("Warning: No valid depth values found")
        return o3d.geometry.PointCloud()
    
    # 픽셀 좌표 그리드 생성
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 유효한 픽셀만 선택
    valid_u = u[valid_mask]
    valid_v = v[valid_mask]
    valid_depth = depth_frame[valid_mask] * depth_scale
    
    # 카메라 내부 파라미터 추출
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    # 3D 좌표 계산
    x = (valid_u - cx) * valid_depth / fx
    y = (valid_v - cy) * valid_depth / fy
    z = valid_depth
    
    # 3D 포인트 스택
    points_3d = np.stack([x, y, z], axis=1)
    
    # RGB 색상 추출 (유효한 픽셀에 대해서만)
    colors = rgb_frame[valid_mask] / 255.0  # 0-1 범위로 정규화
    
    # Open3D PointCloud 생성
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    
    
    return point_cloud


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
    vprint(f"      Step 1 - Ray preparation: {step1_time:.4f}s")
    
    # 2. Batch raycasting 수행
    step2_start = time.time()
    ans = scene.cast_rays(rays_tensor)
    step2_time = (time.time() - step2_start)
    vprint(f"      Step 2 - Ray casting: {step2_time:.4f}s")
    
    # 3. 결과 분석
    step3_start = time.time()
    hit_distances = ans['t_hit'].numpy()  # [M]
    
    # Visibility 판단: hit이 inf이거나 max_distance보다 크면 visible
    visible = np.logical_or(np.isinf(hit_distances), hit_distances > max_distances)
    
    
    step3_time = time.time() - step3_start
    vprint(f"      Step 3 - Result processing: {step3_time:.4f}s")
    
    total_time = time.time() - total_start_time
    vprint(f"    Batch raycasting ({len(origins)} rays): {total_time:.4f}s total")
    
    # 각 단계별 시간 요약
    vprint(f"      Time breakdown: rays={step1_time:.4f}s, casting={step2_time:.4f}s, processing={step3_time:.4f}s")
    
    return visible, hit_distances


from nvblox_torch.examples.utils.visualization import Visualizer
class CustomVisualizer(Visualizer):
        """
        Custom visualizer that combines robot mesh with scene mesh in _visualize_nvblox_mesh
        and adds Line of Sight (LOS) visualization functionality
        
        Camera Control:
        - follow_camera_view=False: Static camera view (uses initial camera pose)
        - follow_camera_view=True: Dynamic camera view (follows each frame's camera pose)
        - active_camera=False: Use current camera pose for visualization
        - active_camera=True: Use active_camera_pose_list for multiple camera frustum visualization
        - video_save: Enable/disable video recording
        """
        def __init__(self, deep_feature_embedding_dim=None, video_save=False, video_path="output_video.mp4", follow_camera_view=False, active_camera=False, vis_window=True):
            super().__init__(deep_feature_embedding_dim)
            self.los_geometries = {}  # Store LOS geometries for each visualizer
            self.camera_frustum_geometries = {}  # Store camera frustum geometries for each visualizer
            self.active_camera_frustum_geometries = {}  # Store active camera frustum geometries for each visualizer
            self.query_points = None
            self.camera_center = None
            self.visibility_results = None
            self.camera_intrinsics = None
            self.image_size = None
            self.pending_camera_updates = {}  # Store pending camera updates for each visualizer
            
            # Video recording settings
            self.video_save = video_save
            self.video_path = video_path
            self.captured_frames_mesh = []  # Store captured frames for mesh video
            self.captured_frames_point_cloud = []  # Store captured frames for point cloud video
            self.initial_camera_pose = None  # Store initial camera pose
            
            # Camera following settings
            self.follow_camera_view = follow_camera_view
            self.current_camera_pose = None  # Store current camera pose for following
            self.visualizers = {}
            
            # Active camera settings
            self.active_camera = active_camera
            self.active_camera_pose_list = None  # Store list of active camera poses
            
            # Visualization window settings
            self.vis_window = vis_window  # True: show real-time windows, False: hide windows
            
        
        def _create_visualizer(self, window_name: str):
            """Create visualizer with custom settings for LOS visualization"""
            
            visualizer = o3d.visualization.VisualizerWithKeyCallback()
            
            if self.vis_window:
                # 윈도우를 화면에 표시
                visualizer.create_window(width=800, height=600, window_name=window_name)
            else:
                # 윈도우를 숨김 (오프스크린 렌더링)
                visualizer.create_window(width=800, height=600, window_name=window_name, visible=False)
            
            visualizer.get_render_option().line_width = 5  # LOS 선을 더 두껍게
            visualizer.get_render_option().point_size = 3
            visualizer.get_render_option().background_color = np.asarray([0, 0, 0])
            visualizer.register_key_callback(ord(' '), lambda vis: self._toggle_pause(vis))
            return visualizer
        
        def _loop_while_paused(self):
            """Override to capture camera position continuously while paused"""
            
            while self.pause:
                for visualizer_name, visualizer in self.visualizers.items():
                    visualizer.poll_events()
                    visualizer.update_renderer()
                    
                    # Continuously capture camera position while paused
                    view_control = visualizer.get_view_control()
                    camera_params = view_control.convert_to_pinhole_camera_parameters()
                    view_status = visualizer.get_view_status()
                    
                    # Store camera update for later application
                    self.pending_camera_updates[visualizer_name] = {
                        'camera_params': camera_params,
                        'view_status': view_status,
                    }
                    
                
                
                time.sleep(0.001)
        
        def set_initial_camera_pose(self, camera_pose):
            """Set the initial camera pose for the visualizer"""
            self.initial_camera_pose = camera_pose.copy()
        
        def set_current_camera_pose(self, camera_pose):
            """Set the current camera pose for following mode"""
            self.current_camera_pose = camera_pose.cpu().numpy().copy()
        
        def _set_camera_pose_from_matrix(self, visualizer, camera_pose_matrix):
            """Set camera pose from 4x4 transformation matrix"""
            
            
            # Convert torch tensor to numpy if needed
            if isinstance(camera_pose_matrix, torch.Tensor):
                camera_pose_matrix = camera_pose_matrix.cpu().numpy()
            
            # Extract camera position and orientation
            camera_position = camera_pose_matrix[:3, 3]
            camera_rotation = camera_pose_matrix[:3, :3]
            
            # Calculate lookat point (camera position + front direction)
            front_direction = -camera_rotation[:, 2]  # Negative Z axis is front
            lookat_point = camera_position + front_direction * 0.0  # Look 2 units ahead
            
            # Set camera parameters
            view_control = visualizer.get_view_control()
            view_control.set_lookat(lookat_point)
            view_control.set_up(-camera_rotation[:, 1])  # Y axis is up
            view_control.set_front(front_direction)
            view_control.set_zoom(0.2)
            
            # Move camera to position
            # view_control.camera_local_translate(0, 0, 1.0)
        
        def _capture_frame(self, visualizer, visualizer_name):
            """Capture current frame for video recording"""
            if self.video_save:
            
                # Capture screen image
                image = visualizer.capture_screen_float_buffer(do_render=True)
                # Convert to numpy array and scale to 0-255
                image_np = np.asarray(image)
                image_np = (image_np * 255).astype(np.uint8)
                
                # Store frames based on visualizer type
                if 'color_mesh' == visualizer_name:
                    self.captured_frames_mesh.append(image_np)
                elif 'point_cloud' == visualizer_name:
                    self.captured_frames_point_cloud.append(image_np)
                else:
                    raise ValueError(f"Invalid visualizer name: {visualizer_name}")
        
        def _update_visualization(self, visualizer: o3d.visualization.VisualizerWithKeyCallback, visualizer_name: str) -> None:
            visualizer.poll_events()
            visualizer.update_renderer()
            
            # Capture frame for video if enabled
            self._capture_frame(visualizer, visualizer_name)
            if visualizer_name in self.pending_camera_updates:
                view_control = visualizer.get_view_control()
                camera_params = view_control.convert_to_pinhole_camera_parameters()
                view_status = visualizer.get_view_status()
                
                # Store camera update for later application
                self.pending_camera_updates[visualizer_name] = {
                    'camera_params': camera_params,
                    'view_status': view_status,
                }
                time.sleep(0.001)



        def _visualize_nvblox_mesh(self, color_mesh, name='color_mesh'):
            if name not in self.visualizers:
                self.visualizers[name] = self._create_visualizer(name)
            self.visualizers[name].clear_geometries()
            self.visualizers[name].add_geometry(color_mesh)
            self.visualizers[name].update_renderer()
        
        def _visualize_nvblox_point_cloud(self, point_cloud, name='point_cloud'):
            """
            nvblox point cloud을 시각화하는 함수
            
            Args:
                point_cloud: 시각화할 point cloud (numpy array 또는 Open3D PointCloud 객체)
                name: visualizer 이름 (기본값: 'point_cloud')
            """
            
            # visualizer가 없으면 생성
            if name not in self.visualizers:
                self.visualizers[name] = self._create_visualizer(name)
            
            # 기존 geometry 제거
            self.visualizers[name].clear_geometries()
            
            # point_cloud가 numpy array인 경우 Open3D PointCloud로 변환
            if isinstance(point_cloud, np.ndarray):
                o3d_point_cloud = o3d.geometry.PointCloud()
                o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
                
                # 색상이 없는 경우 기본 색상 설정 (흰색)
                if len(point_cloud) > 0:
                    colors = np.ones((len(point_cloud), 3)) * 0.8  # 밝은 회색
                    o3d_point_cloud.colors = o3d.utility.Vector3dVector(colors)
                
                point_cloud = o3d_point_cloud
            
            # point cloud를 visualizer에 추가
            self.visualizers[name].add_geometry(point_cloud)
            self.visualizers[name].update_renderer()
            
            
        
        def create_multiple_lines_of_sight(self, camera_center, target_points, colors=None):
            """
            카메라 중심에서 여러 타겟 포인트까지의 line of sight들을 생성
            
            Args:
                camera_center: 카메라 중심 좌표
                target_points: 타겟 포인트들의 리스트
                colors: 각 선의 색깔 리스트 (None이면 기본 색깔 사용)
            
            Returns:
                o3d.geometry.LineSet: 모든 line of sight를 포함하는 LineSet
            """
            
            
            if colors is None:
                colors = [[1, 0, 0]] * len(target_points)  # 기본적으로 빨간색
            
            points = [camera_center]
            lines = []
            line_colors = []
            
            for i, target_point in enumerate(target_points):
                points.append(target_point)
                lines.append([0, i + 1])  # 카메라 중심(0)에서 각 타겟 포인트로
                line_colors.append(colors[i])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(points))
            line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
            line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))
            
            return line_set
        
        def create_camera_frustum_geometry(self, camera_pose, camera_intrinsics, image_size, max_distance=0.2, color=None):
            """
            카메라 frustum을 Open3D geometry로 생성
            
            Args:
                camera_pose: 카메라 pose (4x4 변환 행렬, torch.Tensor 또는 numpy.ndarray)
                camera_intrinsics: 카메라 내부 파라미터 (3x3)
                image_size: 이미지 크기 (width, height)
                max_distance: frustum의 최대 거리
                color: frustum 색상 [R, G, B] (None이면 기본 색상 사용)
            
            Returns:
                o3d.geometry.LineSet: 카메라 frustum을 나타내는 LineSet
            """
            
            
            # 카메라 위치와 회전 추출 (torch.Tensor 또는 numpy.ndarray 모두 지원)
            if isinstance(camera_pose, torch.Tensor):
                camera_position = camera_pose[:3, 3].cpu().numpy()
                camera_rotation_matrix = camera_pose[:3, :3].cpu().numpy()
            else:
                camera_position = camera_pose[:3, 3]
                camera_rotation_matrix = camera_pose[:3, :3]
            camera_rotation_rpy = R.from_matrix(camera_rotation_matrix).as_euler('xyz', degrees=True)
            
            # frustum 꼭짓점 생성
            frustum_vertices = create_camera_frustum_visualization(
                camera_position, camera_rotation_rpy, camera_intrinsics, image_size, max_distance
            )
            
            # frustum을 LineSet으로 변환
            # 8개 꼭짓점: 0번은 카메라 중심, 1-4번은 원거리 꼭짓점들
            points = frustum_vertices
            lines = [
                # 카메라 중심에서 각 원거리 꼭짓점으로
                [0, 1], [0, 2], [0, 3], [0, 4],
                # 원거리 꼭짓점들을 연결 (사각형)
                [1, 2], [2, 3], [3, 4], [4, 1]
            ]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
            
            # 색상 설정 (기본값: 청록색, 지정된 색상이 있으면 사용)
            if color is None:
                color = [0, 1, 1]  # 청록색 (기본)
            line_set.colors = o3d.utility.Vector3dVector(np.array([color] * len(lines)))
            
            return line_set
        
        def _visualize_camera_pose(self, camera_pose):
            """
            카메라 frustum을 시각화 (기존 _visualize_camera_pose 오버라이드)
            """
            if self.camera_intrinsics is None or self.image_size is None:
                print("Warning: Camera intrinsics or image size not set, skipping camera frustum visualization")
                return
            
            # 모든 visualizer에 카메라 frustum 추가
            for name, visualizer in self.visualizers.items():
                # 기존 카메라 frustum geometry 제거
                if name in self.camera_frustum_geometries:
                    visualizer.remove_geometry(self.camera_frustum_geometries[name])
                
                # 새로운 카메라 frustum 생성 및 추가
                frustum_geometry = self.create_camera_frustum_geometry(
                    camera_pose, self.camera_intrinsics, self.image_size
                )
                self.camera_frustum_geometries[name] = frustum_geometry
                visualizer.add_geometry(frustum_geometry)
                
                
        def _visualize_current_camera_poses(self, current_camera_pose_list, selected_indices=None):
            """
            current_camera_pose_list의 선택된 인덱스 카메라 frustum만 시각화
            """
            if self.camera_intrinsics is None or self.image_size is None:
                print("Warning: Camera intrinsics or image size not set, skipping current camera frustum visualization")
                return
            
            if current_camera_pose_list is None or len(current_camera_pose_list) == 0:
                print("Warning: No current camera poses provided")
                return
            
            # 선택된 인덱스가 없으면 모든 인덱스 사용
            if selected_indices is None:
                selected_indices = list(range(len(current_camera_pose_list)))
            
            # 모든 visualizer에 선택된 current camera frustum들만 추가
            for name, visualizer in self.visualizers.items():
                # 기존 current camera frustum geometries 제거
                if name in self.camera_frustum_geometries:
                    if isinstance(self.camera_frustum_geometries[name], list):
                        for geometry in self.camera_frustum_geometries[name]:
                            visualizer.remove_geometry(geometry)
                    else:
                        visualizer.remove_geometry(self.camera_frustum_geometries[name])
                    self.camera_frustum_geometries[name] = []
                
                # 선택된 current camera pose들에 대해서만 frustum 생성 및 추가
                current_frustum_geometries = []
                for i in selected_indices:
                    if i < len(current_camera_pose_list):
                        camera_pose = current_camera_pose_list[i]
                        # current camera는 청록색으로 표시 (기본 색상)
                        frustum_geometry = self.create_camera_frustum_geometry(
                            camera_pose, self.camera_intrinsics, self.image_size, color=[0, 1, 1]  # 청록색
                        )
                        current_frustum_geometries.append(frustum_geometry)
                        visualizer.add_geometry(frustum_geometry)
                
                self.camera_frustum_geometries[name] = current_frustum_geometries
                
        def _visualize_active_camera_poses(self, active_camera_pose_list, selected_indices=None):
            """
            active_camera_pose_list의 선택된 인덱스 카메라 frustum만 시각화
            """
            if self.camera_intrinsics is None or self.image_size is None:
                print("Warning: Camera intrinsics or image size not set, skipping active camera frustum visualization")
                return
            
            if active_camera_pose_list is None or len(active_camera_pose_list) == 0:
                print("Warning: No active camera poses provided")
                return
            
            # 선택된 인덱스가 없으면 모든 인덱스 사용
            if selected_indices is None:
                selected_indices = list(range(len(active_camera_pose_list)))
            
            # 모든 visualizer에 선택된 active camera frustum들만 추가
            for name, visualizer in self.visualizers.items():
                # 기존 active camera frustum geometries 제거
                if name in self.active_camera_frustum_geometries:
                    for geometry in self.active_camera_frustum_geometries[name]:
                        visualizer.remove_geometry(geometry)
                    self.active_camera_frustum_geometries[name] = []
                
                # 선택된 active camera pose들에 대해서만 frustum 생성 및 추가
                active_frustum_geometries = []
                for i in selected_indices:
                    if i < len(active_camera_pose_list):
                        camera_pose = active_camera_pose_list[i]
                        # active camera는 노란색으로 표시 (기존 카메라와 구분)
                        frustum_geometry = self.create_camera_frustum_geometry(
                            camera_pose, self.camera_intrinsics, self.image_size, color=[1, 1, 0]  # 노란색
                        )
                        active_frustum_geometries.append(frustum_geometry)
                        visualizer.add_geometry(frustum_geometry)
                
                self.active_camera_frustum_geometries[name] = active_frustum_geometries
                
        
        def visualize(self, color_mesh=None, feature_mesh=None, point_cloud=None,camera_pose=None, 
                     query_points=None, visibility_results=None, camera_intrinsics=None, image_size=None,
                     current_camera_pose_list=None, current_visibility_results_dict=None,
                     active_camera_pose_list=None, active_visibility_results_dict=None,
                     selected_indices=None):
            """
            LOS를 포함한 시각화 (원래 visualize 함수 오버라이드)
            
            Args:
                color_mesh: Color mesh to visualize
                feature_mesh: Feature mesh to visualize  
                camera_pose: Camera pose to visualize
                query_points: Query points for LOS visualization (옵션)
                visibility_results: Visibility results for each query point (옵션)
                camera_intrinsics: Camera intrinsics for frustum visualization (옵션)
                image_size: Image size for frustum visualization (옵션)
                current_camera_pose_list: List of current camera poses (옵션)
                current_visibility_results_dict: Dictionary of visibility results for multiple current cameras (옵션)
                active_camera_pose_list: List of active camera poses for multiple frustum visualization (옵션)
                active_visibility_results_dict: Dictionary of visibility results for multiple active cameras (옵션)
                selected_indices: List of selected indices for visualization (옵션)
            """
            # Store camera intrinsics and image size for frustum visualization
            if camera_intrinsics is not None:
                self.camera_intrinsics = camera_intrinsics
            if image_size is not None:
                self.image_size = image_size
            
            # Store active camera pose list
            if active_camera_pose_list is not None:
                self.active_camera_pose_list = active_camera_pose_list
            
            # Update current camera pose if following mode is enabled
            if self.follow_camera_view and camera_pose is not None:
                self.set_current_camera_pose(camera_pose)

            # Store current views before updating

            if color_mesh is not None:
                self._visualize_nvblox_mesh(color_mesh)

            if point_cloud is not None:
                self._visualize_nvblox_point_cloud(point_cloud)

            
            # LOS 시각화 추가 (query_points와 visibility_results가 제공된 경우)
            if query_points is not None and visibility_results is not None:
                # 기존 카메라의 LOS 시각화 (선택된 인덱스만)
                if current_camera_pose_list is not None and current_visibility_results_dict is not None and selected_indices is not None:
                    for name, visualizer in self.visualizers.items():
                        for cam_idx, cam_visibility_results in current_visibility_results_dict.items():
                            if cam_idx in selected_indices and cam_idx < len(current_camera_pose_list):
                                current_camera_pose = current_camera_pose_list[cam_idx]
                                current_camera_center = current_camera_pose[:3, 3].cpu().numpy() if isinstance(current_camera_pose, torch.Tensor) else current_camera_pose[:3, 3]
                                
                                # 기존 카메라 LOS geometry 제거
                                current_los_key = f"{name}_current_los_{cam_idx}"
                                if current_los_key in self.los_geometries:
                                    visualizer.remove_geometry(self.los_geometries[current_los_key])
                                
                                # 가시성에 따른 색깔 결정 (기존 카메라: 초록/빨강)
                                colors = []
                                for visible in cam_visibility_results:
                                    if visible:
                                        colors.append([0, 1, 0])  # 초록색 (visible)
                                    else:
                                        colors.append([1, 0, 0])  # 빨간색 (occluded)
                                
                                # LOS 생성 및 추가
                                line_set = self.create_multiple_lines_of_sight(current_camera_center, query_points, colors)
                                self.los_geometries[current_los_key] = line_set
                                visualizer.add_geometry(line_set)
                                
                
                # active_camera 모드인 경우 선택된 active camera들에 대한 LOS 시각화
                if self.active_camera and self.active_camera_pose_list is not None and len(self.active_camera_pose_list) > 0 and active_visibility_results_dict is not None and selected_indices is not None:
                    # 각 선택된 active camera에 대해 LOS 시각화
                    for name, visualizer in self.visualizers.items():
                        for cam_idx, cam_visibility_results in active_visibility_results_dict.items():
                            if cam_idx in selected_indices and cam_idx < len(self.active_camera_pose_list):
                                active_camera_pose = self.active_camera_pose_list[cam_idx]
                                active_camera_center = active_camera_pose[:3, 3].cpu().numpy() if isinstance(active_camera_pose, torch.Tensor) else active_camera_pose[:3, 3]
                                
                                # active camera LOS geometry 제거
                                active_los_key = f"{name}_active_los_{cam_idx}"
                                if active_los_key in self.los_geometries:
                                    visualizer.remove_geometry(self.los_geometries[active_los_key])
                                
                                # active camera용 가시성에 따른 색깔 결정 (기존과 동일: 초록/빨강)
                                active_colors = []
                                for visible in cam_visibility_results:
                                    if visible:
                                        active_colors.append([0, 1, 0])  # 초록색 (visible)
                                    else:
                                        active_colors.append([1, 0, 0])  # 빨간색 (occluded)
                                
                                # active camera LOS 생성 및 추가
                                active_line_set = self.create_multiple_lines_of_sight(active_camera_center, query_points, active_colors)
                                self.los_geometries[active_los_key] = active_line_set
                                visualizer.add_geometry(active_line_set)
                                
            
            # for name, visualizer in self.visualizers.items():
            #     visualizer.update_renderer()


            if feature_mesh is not None:
                self._visualize_nvblox_feature_mesh(feature_mesh)

            # 카메라 frustum 시각화
            # 기존 카메라 pose 시각화 (선택된 인덱스만)
            if current_camera_pose_list is not None and selected_indices is not None:
                self._visualize_current_camera_poses(current_camera_pose_list, selected_indices)
                
            
            # active_camera 모드인 경우 선택된 active camera pose list의 frustum 시각화
            if self.active_camera and self.active_camera_pose_list is not None and selected_indices is not None:
                self._visualize_active_camera_poses(self.active_camera_pose_list, selected_indices)
                

            # Restore views and update
            for name, visualizer in self.visualizers.items():
                if name in self.pending_camera_updates:
                    visualizer.set_view_status(self.pending_camera_updates[name]['view_status'])
                elif self.follow_camera_view:
                    # Follow camera mode: determine which camera to follow
                    if self.active_camera and self.active_camera_pose_list is not None and len(self.active_camera_pose_list) > 0:
                        # active_camera 모드: 첫 번째 active camera를 follow (인덱스 0)
                        first_active_camera_pose = self.active_camera_pose_list[0]
                        if isinstance(first_active_camera_pose, torch.Tensor):
                            first_active_camera_pose = first_active_camera_pose.cpu().numpy()
                        self._set_camera_pose_from_matrix(self.visualizers[name], first_active_camera_pose)
                        
                    elif self.current_camera_pose is not None:
                        # 기존 방식: 현재 카메라 pose를 follow
                        self._set_camera_pose_from_matrix(self.visualizers[name], self.current_camera_pose)
                        
                elif self.initial_camera_pose is not None:
                    # Static mode: use initial camera pose
                    self._set_camera_pose_from_matrix(self.visualizers[name], self.initial_camera_pose)
                    
                self._update_visualization(visualizer, name)

            # Handle pausing
            if self.pause:
                self._loop_while_paused()
        
        def save_video(self):
            """Save captured frames as separate MP4 videos for mesh and point cloud"""
            if not self.video_save:
                print("Video saving disabled")
                return
            
            # Save mesh video
            if len(self.captured_frames_mesh) > 0:
                mesh_video_path = self.video_path.replace('.mp4', '_mesh.mp4')
                self._save_frames_to_video(self.captured_frames_mesh, mesh_video_path, "mesh")
            else:
                print("No mesh frames captured")
            
            # Save point cloud video
            if len(self.captured_frames_point_cloud) > 0:
                point_cloud_video_path = self.video_path.replace('.mp4', '_point_cloud.mp4')
                self._save_frames_to_video(self.captured_frames_point_cloud, point_cloud_video_path, "point cloud")
            else:
                print("No point cloud frames captured")
        
        def _save_frames_to_video(self, frames, video_path, video_type):
            """Helper method to save frames to video file"""
            if len(frames) == 0:
                print(f"No {video_type} frames to save")
                return
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
            
            # Write frames
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            # Release everything
            out.release()
            print(f"{video_type.capitalize()} video saved to: {video_path}")
            print(f"Total {video_type} frames: {len(frames)}")
            

def get_scene_mesh(rgb_frame, depth_frame, K_frame, relative_pose, voxel_size, mapper):
    # Depth 이미지를 torch tensor로 변환 (GPU)
    depth_tensor = torch.from_numpy(depth_frame.astype(np.float32)).cuda()
    
    # RGB 이미지를 torch tensor로 변환 (GPU)
    rgb_tensor = torch.from_numpy(rgb_frame).cuda()
    
    # 카메라 내부 파라미터를 torch tensor로 변환 (CPU)
    intrinsics_tensor = torch.from_numpy(K_frame.astype(np.float32)).cpu()
    
    # 카메라 포즈
    if relative_pose is None:
        pose_tensor = torch.eye(4, dtype=torch.float32).cpu()
    else:
        pose_tensor = torch.from_numpy(relative_pose).float().cpu()
    
    # nvblox에 데이터 추가 (sun3d.py 방식)
    mapper.add_depth_frame(depth_tensor, pose_tensor, intrinsics_tensor)
    mapper.add_color_frame(rgb_tensor, pose_tensor, intrinsics_tensor)
    
    # 메시 업데이트 (각 프레임마다)
    mapper.update_color_mesh()
    color_mesh = mapper.get_color_mesh()
    
    # nvblox ColorMesh
    scene_open3d = color_mesh.to_open3d()
    
    
    
    scene_mesh = scene_open3d

    # # dscho NOTE: for debug (only for scene meshes, robot meshes doesn't need to be processed)
    mesh_postprocess_start = time.time()
    
    scene_mesh.remove_duplicated_vertices()
    scene_mesh.remove_duplicated_triangles()
    scene_mesh.remove_degenerate_triangles()
    # scene_mesh.remove_non_manifold_edges()
    print(f"  In multiframe example, Mesh postprocess time: {time.time() - mesh_postprocess_start:.4f} seconds") #  0.04s
    
    
    triangle_postprocess_start = time.time()
    
    # # # 작은 컴포넌트 제거(삼각형 개수 기준) (seems to be more effective for small meshes than the above removing duplicated things)
    # tri_clusters, cluster_n_tri, _ = scene_mesh.cluster_connected_triangles()
    # tri_clusters = np.asarray(tri_clusters)
    # keep = [i for i,cnt in enumerate(cluster_n_tri) if cnt >= 800]  # 임계치 튜닝
    # mask = np.isin(tri_clusters, keep)
    # scene_mesh.remove_triangles_by_mask(~mask)
    # scene_mesh.remove_unreferenced_vertices()
    

    # # 절대 면적 기준
    # tri_clusters, cluster_n_tri, cluster_area = scene_mesh.cluster_connected_triangles()
    # cluster_area = np.asarray(cluster_area)  # 각 군집의 총 면적

    # min_area_m2 = 0.02  # 예: 0.02 m^2 미만 군집은 제거 (장면/스케일에 맞게 조절)
    # keep_ids = np.where(cluster_area >= min_area_m2)[0]

    # mask = np.isin(tri_clusters, keep_ids)  # 남길 군집 = True
    # scene_mesh.remove_triangles_by_mask(~mask)
    # scene_mesh.remove_unreferenced_vertices()
    
    # 상대 면적 기준
    tri_clusters, _, cluster_area = scene_mesh.cluster_connected_triangles()
    cluster_area = np.asarray(cluster_area)

    largest = float(cluster_area.max()) if len(cluster_area) else 0.0
    alpha = 0.005   # 예: 최대 군집의 0.5% 미만은 제거
    abs_floor = 800 * 0.5 * (voxel_size ** 2)  # 안전 바닥(아래 설명)
    thresh = max(alpha * largest, abs_floor)

    keep_ids = np.where(cluster_area >= thresh)[0]
    mask = np.isin(tri_clusters, keep_ids)
    scene_mesh.remove_triangles_by_mask(~mask)
    scene_mesh.remove_unreferenced_vertices()

    print(f"  In multiframe example, Mesh remove triangles time: {time.time() - triangle_postprocess_start:.4f} seconds") # 0.04s
    return scene_mesh, pose_tensor

    

def create_active_camera_pose_list(base_camera_pose, num_cameras=10, translation_step=-0.05):
    """
    현재 카메라 pose에서 y방향으로 translation_step씩 이동한 카메라 pose 리스트 생성
    
    Args:
        base_camera_pose: 기준 카메라 pose (4x4 변환 행렬)
        num_cameras: 생성할 카메라 개수
        translation_step: y방향 이동 거리 (미터)
    
    Returns:
        active_camera_pose_list: 생성된 카메라 pose 리스트
    """
    active_camera_pose_list = []
    
    for i in range(num_cameras):
        # 기준 카메라 pose 복사
        camera_pose = base_camera_pose.copy()
        
        # y방향으로 translation_step * (i+1)만큼 이동
        camera_pose[1, 3] += translation_step * (i+1)
        
        active_camera_pose_list.append(camera_pose)
    
    return active_camera_pose_list

def create_current_camera_pose_list(base_camera_pose, num_cameras=10):
    """
    현재 카메라 pose를 복사한 카메라 pose 리스트 생성 (translation 없이)
    
    Args:
        base_camera_pose: 기준 카메라 pose (4x4 변환 행렬)
        num_cameras: 생성할 카메라 개수
    
    Returns:
        current_camera_pose_list: 생성된 카메라 pose 리스트
    """
    current_camera_pose_list = []
    
    for i in range(num_cameras):
        # 기준 카메라 pose 복사 (translation 없이)
        camera_pose = base_camera_pose.copy()
        current_camera_pose_list.append(camera_pose)
    
    return current_camera_pose_list

def create_multiframe_nvblox(save_path, rgb_frames_list, depth_frames_list, depth_frames_wo_fake_depth_list, relative_poses_list, K_adjusted_list, T_mc_list, action_list, tracking_3d_list, robot_viz=None, voxel_size=0.01, image_size=None, follow_camera_view=False, active_camera=False, vis_window=True, droid=False):
    """
    sun3d.py 방식을 차용한 nvblox 기반 multiframe 시각화 함수
    - Mapper를 한 번 생성하고 모든 프레임을 순차적으로 처리
    - 각 프레임마다 mesh를 업데이트하여 애니메이션 생성
    - robot_meshes_list가 제공되면 첫 번째 로봇 메시를 씬과 결합
    - follow_camera_view=True이면 카메라가 각 프레임의 pose를 따라감
    """
    
    
    # 프레임 수 확인
    num_frames = len(rgb_frames_list)
    if num_frames == 0:
        print("No frames provided")
        return
    
    # Reward tracking 초기화
    reward_tracker = RewardTracker()
    
    print(f"Processing {num_frames} frame(s) for nvblox multiframe visualization")
    
    # nvblox Mapper 설정 (sun3d.py 방식)
    projective_integrator_params = ProjectiveIntegratorParams()
    projective_integrator_params.projective_integrator_max_integration_distance_m = 3.0 # default:5.0

    # dscho NOTE: for debug (to remove tiny mesh)
    mesh_param = MeshIntegratorParams()
    mesh_param.mesh_integrator_min_weight = 1.0 # default 1e-4
    

    mapper_params = MapperParams()
    mapper_params.set_projective_integrator_params(projective_integrator_params)
    mapper_params.set_mesh_integrator_params(mesh_param)



    
    # Mapper 생성
    mapper = Mapper(
        voxel_sizes_m=voxel_size,
        mapper_parameters=mapper_params,
    )
    
    

    if droid:
        video_path = os.path.join(save_path, "nvblox_visualization_droid.mp4")
    else:
        video_path = os.path.join(save_path, "nvblox_visualization.mp4")
    visualizer = CustomVisualizer(
        video_save=True,
        video_path=video_path,
        follow_camera_view=follow_camera_view,
        active_camera=active_camera,
        vis_window=vis_window
    )
    
    # 첫 번째 프레임의 카메라 포즈를 초기 뷰로 설정 (follow_camera_view가 False일 때만)
    if not follow_camera_view and len(relative_poses_list) > 0:
        first_camera_pose = relative_poses_list[0]
        visualizer.set_initial_camera_pose(first_camera_pose)
        print("Set initial camera pose from first frame (static mode)")
    elif follow_camera_view:
        print("Camera following mode enabled - camera will follow each frame's pose")

    # 각 프레임을 순차적으로 처리 (sun3d.py 방식)
    ik_success = False
    for frame_idx in range(num_frames):
        print(f"Processing frame {frame_idx + 1}/{num_frames}")
        start = time.time()
        
        # 현재 프레임의 RGB-D 데이터와 intrinsic 가져오기
        rgb_frame = rgb_frames_list[frame_idx]
        depth_frame = depth_frames_list[frame_idx]
        depth_frame_wo_fake_depth = depth_frames_wo_fake_depth_list[frame_idx]
        K_frame = K_adjusted_list[frame_idx]
        relative_pose = relative_poses_list[frame_idx]
        T_mc = T_mc_list[frame_idx]
        action = action_list[frame_idx] # [7], camera coordinate
        tracking_3d = tracking_3d_list[frame_idx] # [N, 3]
        T_mw = T_mc_list[0].copy()
        T_W_B = np.linalg.inv(T_B_M @ T_mw)
        T_B_C = T_B_M @ T_mc

        
        
        scene_mesh, pose_tensor = get_scene_mesh(rgb_frame, depth_frame, K_frame, relative_pose, voxel_size, mapper)

        # action should be robot's base coordinate
        action_se3 = np.eye(4)
        action_se3[:3, 3] = action[:3]
        # action_se3[:3, 3] = np.array([0.7, 0.05, 0.15]) # NOTE: dscho debug (straight forward configuration)
        action_se3[:3, :3] = R.from_euler('xyz', action[3:6], degrees=False).as_matrix()
        gripper_action = action[6]

        # convert to robot's base coordinate if action is in camera coordinate   
        if T_B_C is not None:
            action_se3 = T_B_C @ action_se3
        
        

        # dscho debug
        # NOTE: assume gripper_action is in [0 (open), 1 (close)]
        # [-1, 0] -> [-np.pi/2, 0]
        gripper_angle = (gripper_action-1)*np.pi/2 # unit : (radian)
        
        # Inverse Kinematics로 조인트 각도 계산
        ik_success = robot_viz.solve_inverse_kinematics(action_se3)
        robot_mesh_start = time.time()
        robot_meshes = robot_viz.get_robot_meshes_in_specified_transform(gripper_angle=gripper_angle, T_target=T_W_B) # np.linalg.inv(T_B_C)
        print(f" IK : {ik_success}, In multiframe example, Robot meshes time: {time.time() - robot_mesh_start:.4f} seconds")

        
        # 3. 각 로봇 링크 메시를 결합
        for link_name, robot_mesh in robot_meshes.items():
            if robot_mesh is not None and len(robot_mesh.vertices) > 0:
                scene_mesh = scene_mesh + robot_mesh
                vprint(f"    Added robot link {link_name}: {len(robot_mesh.vertices)} vertices")
        
        if len(scene_mesh.vertices) == 0:
            print("  WARNING: Scene mesh has no vertices!")
        if len(scene_mesh.triangles) == 0:
            print("  WARNING: Scene mesh has no triangles!")

        
        
        # 쿼리 포인트를 월드 좌표계로 변환
        T_W_C = np.linalg.inv(T_mw) @ T_mc
        query_point_in_world_coordinate = []
        for query_idx in range(tracking_3d.shape[0]): 
            # assume it is in camera coordinate
            query_point_homo = np.concatenate([tracking_3d[query_idx], [1]])[:, None] # [4, 1]
            # convert to world coordinate
            query_point = (T_W_C @ query_point_homo)[:3, 0] # [3]
            query_point_in_world_coordinate.append(query_point.copy())
        
        mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(scene_mesh)
        
        # 5. Ray casting scene 생성
        current_scene = o3d.t.geometry.RaycastingScene()
        current_scene.add_triangles(mesh_tensor)
        
        # 카메라 pose list 생성 (기존 카메라와 active camera 모두)
        current_camera_pose_np = pose_tensor.cpu().numpy() if isinstance(pose_tensor, torch.Tensor) else pose_tensor
        
        # 기존 카메라 pose list 생성 (현재 카메라 pose 복사)
        current_camera_pose_list = create_current_camera_pose_list(current_camera_pose_np, num_cameras=10)
        
        
        # 선택된 인덱스 정의 (카메라 리스트 길이에 관계없이 고정)
        selected_indices = [0, 5, 9]  # 고정된 인덱스 사용
        selected_indices = sorted(list(set(selected_indices)))
        
        # 각 선택된 기존 카메라에 대한 visibility 계산
        current_visibility_results_dict = {}
        
        for cam_idx in selected_indices:
            if cam_idx < len(current_camera_pose_list):
                current_camera_pose = current_camera_pose_list[cam_idx]
                current_viewpoint = current_camera_pose[:3, 3]
                current_viewdirection = R.from_matrix(current_camera_pose[:3, :3]).as_euler('xyz', degrees=True)
                
                
                
                # 현재 카메라에 대한 visibility 계산
                current_visibility_results = np.zeros((tracking_3d.shape[0]), dtype=bool)
                current_hit_distances = np.zeros((tracking_3d.shape[0]), dtype=np.float64)
                
                for query_idx in range(tracking_3d.shape[0]):
                    query_point = query_point_in_world_coordinate[query_idx]
                    
                    # 1. LOS 체크 (기존 카메라에서)
                    direction = query_point - current_viewpoint
                    distance = np.linalg.norm(direction)
                    direction_normalized = direction / distance
                    
                    # 단일 ray에 대한 raycasting
                    origins_single = current_viewpoint.reshape(1, 3)
                    directions_single = direction_normalized.reshape(1, 3)
                    distances_single = np.array([distance - OFFSET_DISTANCE])
                    
                    los_visible, los_hit_distances = batch_raycasting_with_scene(
                        current_scene, origins_single, directions_single, distances_single
                    )
                    los_visible = los_visible[0]
                    los_hit_distance = los_hit_distances[0]
                    
                    # 2. 카메라 frustum 체크 (기존 카메라에서)
                    frustum_visible = is_point_in_camera_frustum(
                        query_point, current_viewpoint, current_viewdirection, K_frame, image_size
                    )
                    
                    # 3. 최종 visibility: LOS 체크와 frustum 체크를 모두 통과해야 함
                    final_visible = los_visible and frustum_visible
                    
                    # 결과 저장
                    current_visibility_results[query_idx] = final_visible
                    current_hit_distances[query_idx] = los_hit_distance
                    
                    # 상세 로그 출력
                    los_status = "LOS_OK" if los_visible else "LOS_BLOCKED"
                    frustum_status = "FRUSTUM_OK" if frustum_visible else "FRUSTUM_OUT"
                    final_status = "VISIBLE" if final_visible else "OCCLUDED"
                    
                    vprint(f"      Current Camera {cam_idx} Query {query_idx + 1}: {final_status} (LOS: {los_status}, Frustum: {frustum_status}, dist(without offset): {distance:.3f}m, hit: {los_hit_distance:.3f}m) query_point: {query_point}")
                
                # 결과를 딕셔너리에 저장
                current_visibility_results_dict[cam_idx] = current_visibility_results
                vprint(f"Current camera {cam_idx} visibility results: {np.sum(current_visibility_results)}/{len(current_visibility_results)} points visible")
        
        # 첫 번째 선택된 카메라의 결과를 기본 visibility_results로 설정 (기존 코드 호환성)
        if selected_indices:
            visibility_results = current_visibility_results_dict[selected_indices[0]]
            




            
        
        # RGB-D 프레임에서 point cloud 생성 (카메라 좌표계)
        point_cloud_camera = rgbd_to_point_cloud(rgb_frame, depth_frame_wo_fake_depth, K_frame)
        
        # Point cloud를 월드 좌표계로 변환
        if len(point_cloud_camera.points) > 0:
            # 카메라 좌표계의 포인트들을 월드 좌표계로 변환
            points_camera = np.asarray(point_cloud_camera.points)
            colors = np.asarray(point_cloud_camera.colors)
            
            # 카메라 포즈를 사용하여 월드 좌표계로 변환
            points_camera_homo = np.hstack([points_camera, np.ones((len(points_camera), 1))])  # [N, 4]
            points_world_homo = (T_W_C @ points_camera_homo.T).T  # [N, 4]
            points_world = points_world_homo[:, :3]  # [N, 3]
            
            # 월드 좌표계의 point cloud 생성
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points_world)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            
            
        else:
            point_cloud = point_cloud_camera
        
        # active_camera 모드인 경우 active_camera_pose_list 생성
        active_camera_pose_list = None
        if active_camera:
            # 현재 카메라 pose를 기준으로 active_camera_pose_list 생성
            active_camera_pose_list = create_active_camera_pose_list(current_camera_pose_np, num_cameras=10, translation_step=-0.05)
            
            
            # 지정된 인덱스의 active camera들에 대한 visibility 계산 (기존 카메라와 동일한 인덱스 사용)
            if len(active_camera_pose_list) > 0:
                
                
                # 각 선택된 active camera에 대한 visibility 계산
                active_visibility_results_dict = {}
                
                for cam_idx in selected_indices:
                    if cam_idx < len(active_camera_pose_list):
                        active_camera_pose = active_camera_pose_list[cam_idx]
                        active_viewpoint = active_camera_pose[:3, 3]
                        active_viewdirection = R.from_matrix(active_camera_pose[:3, :3]).as_euler('xyz', degrees=True)
                        
                        
                        
                        # 현재 active camera에 대한 visibility 계산
                        current_visibility_results = np.zeros((tracking_3d.shape[0]), dtype=bool)
                        current_hit_distances = np.zeros((tracking_3d.shape[0]), dtype=np.float64)
                        
                        for query_idx in range(tracking_3d.shape[0]):
                            query_point = query_point_in_world_coordinate[query_idx]
                            
                            # 1. LOS 체크 (active camera에서)
                            direction = query_point - active_viewpoint
                            distance = np.linalg.norm(direction)
                            direction_normalized = direction / distance
                            
                            # 단일 ray에 대한 raycasting
                            origins_single = active_viewpoint.reshape(1, 3)
                            directions_single = direction_normalized.reshape(1, 3)
                            distances_single = np.array([distance - OFFSET_DISTANCE])
                            
                            los_visible, los_hit_distances = batch_raycasting_with_scene(
                                current_scene, origins_single, directions_single, distances_single
                            )
                            los_visible = los_visible[0]
                            los_hit_distance = los_hit_distances[0]
                            
                            # 2. 카메라 frustum 체크 (active camera에서)
                            frustum_visible = is_point_in_camera_frustum(
                                query_point, active_viewpoint, active_viewdirection, K_frame, image_size
                            )
                            
                            # 3. 최종 visibility: LOS 체크와 frustum 체크를 모두 통과해야 함
                            final_visible = los_visible and frustum_visible
                            
                            # 결과 저장
                            current_visibility_results[query_idx] = final_visible
                            current_hit_distances[query_idx] = los_hit_distance
                            
                            # 상세 로그 출력
                            los_status = "LOS_OK" if los_visible else "LOS_BLOCKED"
                            frustum_status = "FRUSTUM_OK" if frustum_visible else "FRUSTUM_OUT"
                            final_status = "VISIBLE" if final_visible else "OCCLUDED"
                            
                            vprint(f"      Active Camera {cam_idx} Query {query_idx + 1}: {final_status} (LOS: {los_status}, Frustum: {frustum_status}, dist(without offset): {distance:.3f}m, hit: {los_hit_distance:.3f}m) query_point: {query_point}")
                        
                        # 결과를 딕셔너리에 저장
                        active_visibility_results_dict[cam_idx] = current_visibility_results
                        vprint(f"Active camera {cam_idx} visibility results: {np.sum(current_visibility_results)}/{len(current_visibility_results)} points visible")
                
                
        
        
        # Reward tracking: 기존 카메라와 active 카메라의 visibility 결과를 reward로 저장
        # 기존 카메라 reward (첫 번째 선택된 카메라의 결과)
        reward_tracker.add_current_camera_reward(frame_idx, visibility_results)
        
        # Active 카메라 reward (첫 번째 선택된 active 카메라의 결과)
        if 'active_visibility_results_dict' in locals() and active_visibility_results_dict is not None:
            # 첫 번째 선택된 인덱스의 active 카메라 결과 사용
            first_selected_idx = selected_indices[0] if selected_indices else 0
            if first_selected_idx in active_visibility_results_dict:
                active_visibility_results = active_visibility_results_dict[first_selected_idx]
                reward_tracker.add_active_camera_reward(frame_idx, active_visibility_results)
            else:
                reward_tracker.add_active_camera_reward(frame_idx, None)
        else:
            reward_tracker.add_active_camera_reward(frame_idx, None)
        
        visualizer.visualize(
            color_mesh=scene_mesh, 
            point_cloud=point_cloud,
            camera_pose=pose_tensor,
            query_points=np.stack(query_point_in_world_coordinate), # [N, 3]
            visibility_results=visibility_results,  # 기존 카메라의 visibility 결과
            camera_intrinsics=K_frame,  # 카메라 내부 파라미터
            image_size=image_size,  # 이미지 크기
            current_camera_pose_list=current_camera_pose_list,  # 기존 카메라 pose 리스트
            current_visibility_results_dict=current_visibility_results_dict,  # 기존 카메라의 visibility 결과
            active_camera_pose_list=active_camera_pose_list,  # active camera pose 리스트
            active_visibility_results_dict=active_visibility_results_dict if 'active_visibility_results_dict' in locals() else None,  # 모든 선택된 active camera의 visibility 결과
            selected_indices=selected_indices  # 선택된 인덱스들
        )
        
        
        print(f"  Process frame time: {time.time() - start:.4f} seconds")
        
        
        
    print(f'Saving mesh at {save_path}')
    mapper.update_color_mesh()
    if droid:
        mapper.get_color_mesh().save(save_path+'/3d_visualization_animate_multiframe_nvblox_droid.ply')
    else:
        mapper.get_color_mesh().save(save_path+'/3d_visualization_animate_multiframe_nvblox.ply')
    
    
    # 비디오 저장
    if visualizer.video_save and (len(visualizer.captured_frames_mesh) > 0 or len(visualizer.captured_frames_point_cloud) > 0):
        visualizer.save_video()
        print(f"Mesh frames: {len(visualizer.captured_frames_mesh)}, Point cloud frames: {len(visualizer.captured_frames_point_cloud)}")
    
    # Reward 분석 및 plot 생성
    print("\n=== Generating Reward Analysis Plot ===")
    reward_tracker.plot_reward_changes(save_path)
    
    
            


def depth_processing(rgb, depth_raw, fake_depth_mask, model, camera, resize=False, resize_size=(256, 256), use_monodepth=False, use_fake_depth=False):
    # import cv2  # 함수 시작 부분에서 cv2 import
    
    if use_monodepth:
        vprint("Using UniDepth for depth estimation...")
        
        # UniDepth 모델 로드
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vprint(f"Using device: {device}")
        
        
        
        # Depth 추정 (원본 크기 RGB 사용)
        vprint(f"Running UniDepth on original RGB: {rgb.shape}")
        depth_pred_original = get_depth_with_unidepth(rgb, K, model, camera)
        
        # Raw depth로 스케일링 (원본 크기)
        depth_m_original = scale_depth_with_raw(depth_pred_original, depth_raw, K)


        if use_fake_depth:
            vprint("Applying fake depth (0) to masked regions...")
            depth_m_original = apply_fake_depth_to_mask(depth_m_original[None], fake_depth_mask[None], FAKE_DEPTH_VALUE)[0]
            
        
        # 리사이즈가 필요한 경우 depth만 리사이즈
        if resize:
            vprint(f"Resizing depth from {depth_m_original.shape} to {resize_size}")
            
            depth_m = cv2.resize(depth_m_original, resize_size, interpolation=cv2.INTER_LINEAR)
        else:
            depth_m = depth_m_original
        
        vprint(f"Final depth shape: {depth_m.shape}")
    else:
        additional_depth_preprocess = True # False
        vprint("Using raw depth data...")
        depth_raw = fill_depth_zeros(depth_raw)
        if use_fake_depth:
            vprint("Applying fake depth (0) to masked regions...")
            
            if additional_depth_preprocess:
                # 1. 마스크 여유띠(dilate)
                fake_depth_mask = cv2.dilate(fake_depth_mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=2).astype(bool)
                
            depth_raw = apply_fake_depth_to_mask(depth_raw[None], fake_depth_mask[None], FAKE_DEPTH_VALUE)[0]

            if additional_depth_preprocess:
                # 2. 작은 유효깊이 섬 제거(connected components)
                valid = (depth_raw > 0)  # np.nan 대신 > 0 체크 사용
                cnt, labels, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8), 8)
                min_area = 20  # 해상도에 맞춰 조절
                for i in range(1, cnt):
                    if stats[i, cv2.CC_STAT_AREA] < min_area:
                        depth_raw[labels == i] = FAKE_DEPTH_VALUE

        if resize:            
            depth_m = cv2.resize(depth_raw, resize_size, interpolation=cv2.INTER_LINEAR)
        else:
            depth_m = depth_raw
    
    

    return depth_m

# ========= 메인 =========
def main():
    vprint("=== URDF-based Robot Visibility Test with nvblox ===")
    vprint("Using URDF robot meshes with Inverse Kinematics for realistic robot positioning")
    
    # 전체 실행 시간 측정
    total_start_time = time.time()
    
    # 1) 입력 로드
    vprint("\n=== 1. Loading RGB-D Data ===")
    load_start_time = time.time()
    rgb_frames, depth_frames, relative_poses, combined_mask, T_mc_transformation, action, tracking_3d = load_data_from_zarr(BUFFER_PATH, EPISODE_IDX, USE_DROID)
    rgb, depth_raw, fake_depth_mask, relative_pose, T_mc, act, track_3d = get_data(rgb_frames, depth_frames, relative_poses, combined_mask, T_mc_transformation, action, tracking_3d, FRAME_IDX, DEPTH_SCALE)
    H, W = depth_raw.shape
    load_end_time = time.time()
    vprint(f"Loaded RGB: {rgb.shape}, Depth: {depth_raw.shape}")
    vprint(f"Data loading time: {load_end_time - load_start_time:.4f} seconds")

    # 1.5) 이미지 리사이즈 (옵션) - UniDepth 사용시에는 RGB만 리사이즈
    if RESIZE:
        vprint(f"\n=== 1.5. Resizing RGB to {RESIZE_SIZE} ===")
        resize_start_time = time.time()
        rgb_resized, _, scale_factor_x, scale_factor_y, _ = resize_image_and_depth(rgb, depth_raw, RESIZE_SIZE)
        K_adjusted = adjust_camera_intrinsics(K, scale_factor_x, scale_factor_y)
        resize_end_time = time.time()
        vprint(f"Resized RGB: {rgb_resized.shape}, Original depth: {depth_raw.shape}")
        vprint(f"Scale factors: x={scale_factor_x:.4f}, y={scale_factor_y:.4f}")
        vprint(f"Adjusted camera intrinsics: fx={K_adjusted[0,0]:.2f}, fy={K_adjusted[1,1]:.2f}, cx={K_adjusted[0,2]:.2f}, cy={K_adjusted[1,2]:.2f}")
        vprint(f"RGB resize time: {resize_end_time - resize_start_time:.4f} seconds")
    else:
        K_adjusted = K
        rgb_resized = rgb

    if K is None:
        raise ValueError("K (intrinsics)가 None 입니다. 코드 상단의 K를 사용자의 카메라 파라미터로 채워주세요.")
    if USE_MONODEPTH:
        name = f"unidepth-v2-vit{MODEL_TYPE}14"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
        model.interpolation_mode = "bilinear"
        model = model.to(device).eval()
        
        # 카메라 설정 (원본 크기로 UniDepth 실행)
        intrinsics_torch = torch.from_numpy(K).float()
        camera = Pinhole(K=intrinsics_torch.unsqueeze(0))
    else:
        model = None
        camera = None

    if isinstance(model, (UniDepthV2old, UniDepthV1)):
        camera = camera.K.squeeze(0)
    
    if RESIZE:
        image_size = RESIZE_SIZE
    else:
        image_size = IMAGE_SIZE

    # Z1 로봇 시각화 객체 생성
    urdf_path = "/home/dscho1234/Workspace/unitree_ros/robots/z1_description/xacro/z1.urdf"
    mesh_base_path = "/home/dscho1234/Workspace/unitree_ros/robots/z1_description/"
    robot_viz = Z1RobotVisualizer(urdf_path, mesh_base_path)


    # Multi-frame scene mesh 애니메이션을 위한 RGB-D 데이터 로드
    vprint("\nLoading multiple frames for multiframe visualization...")
    num_multiframe_frames = 340  # 디버깅을 위해 매우 적게 설정
    rgb_frames_list = []
    depth_frames_list = []
    depth_frames_wo_fake_depth_list = []
    relative_poses_list = []
    K_adjusted_list = []
    T_mc_list = []
    action_list = []
    tracking_3d_list = []
    
    for frame_offset in range(num_multiframe_frames):
        current_frame_idx = FRAME_IDX + frame_offset
        
        # RGB-D 데이터 로드
        rgb, depth_raw, fake_depth_mask, relative_pose, T_mc, act, track_3d = get_data(rgb_frames, depth_frames, relative_poses, combined_mask, T_mc_transformation, action, tracking_3d, current_frame_idx, DEPTH_SCALE)
        # 1.5) 이미지 리사이즈 (옵션) - UniDepth 사용시에는 RGB만 리사이즈
        if RESIZE:
            rgb_resized, _, scale_factor_x, scale_factor_y, _ = resize_image_and_depth(rgb, depth_raw, RESIZE_SIZE)
        else:
            rgb_resized = rgb
        
        depth_m = depth_processing(rgb, depth_raw, fake_depth_mask, model, camera, resize=RESIZE, resize_size=RESIZE_SIZE, use_monodepth=USE_MONODEPTH, use_fake_depth=USE_FAKE_DEPTH)
        depth_m_wo_fake_depth = depth_processing(rgb, depth_raw, fake_depth_mask, model, camera, resize=RESIZE, resize_size=RESIZE_SIZE, use_monodepth=USE_MONODEPTH, use_fake_depth=False)
        
        
        rgb_frames_list.append(rgb_resized)
        depth_frames_list.append(depth_m)
        depth_frames_wo_fake_depth_list.append(depth_m_wo_fake_depth)
        relative_poses_list.append(relative_pose)
        K_adjusted_list.append(K_adjusted)
        T_mc_list.append(T_mc)
        action_list.append(act)
        tracking_3d_list.append(track_3d)
        vprint(f"  Loaded frame {frame_offset + 1}/{num_multiframe_frames} (index: {current_frame_idx})")
            
    vprint(f"Successfully loaded {len(rgb_frames_list)} frames for multiframe visualization")
    
    
    
    # NVBlox 방식의 Multi-frame scene mesh 애니메이션 생성 (sun3d.py 방식)
    if len(rgb_frames_list) > 0:
        create_multiframe_nvblox(
            "visibility_test_output",
            rgb_frames_list, depth_frames_list, depth_frames_wo_fake_depth_list, relative_poses_list, K_adjusted_list, T_mc_list, action_list, tracking_3d_list,
            robot_viz=robot_viz,
            voxel_size=VOXEL_SIZE,
            image_size=image_size,
            follow_camera_view=True,
            active_camera=True,  # active_camera 모드 활성화
            vis_window=True,  # 윈도우 숨김 (비디오만 저장)
            droid=USE_DROID,
        )
        vprint("NVBlox multi-frame scene mesh animated visualization saved to: visibility_test_output/3d_visualization_animate_multiframe_nvblox.html")
    else:
        vprint("No frames loaded for multiframe visualization")
    
    
if __name__ == "__main__":
    main()
