#!/usr/bin/env python3
"""
Z1 로봇 URDF 메시 시각화 및 전진기구학 테스트
Open3D와 Plotly를 사용하여 3D 메시를 로드하고 전진기구학을 통해 월드 좌표로 변환하여 시각화
"""

import os
import sys
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import plotly.offline as pyo
from urdf_parser_py.urdf import URDF
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

import trimesh

# Unitree Z1 SDK 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "unitree_ros_to_real", "unitree_legged_sdk", "lib"))
sys.path.append("/home/dscho1234/Workspace/z1_sdk/lib")
import unitree_arm_interface

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
        
        # Unitree Z1 SDK 초기화
        try:
            self.arm_interface = unitree_arm_interface.ArmInterface(hasGripper=True)
            print("Unitree Z1 SDK 초기화 성공")
        except Exception as e:
            print(f"Unitree Z1 SDK 초기화 실패: {e}")
            self.arm_interface = None
        
        # URDF 파싱
        self.parse_urdf()
        
        # 메시 로딩
        self.load_meshes()
        
    def parse_urdf(self):
        """URDF 파일을 파싱하여 로봇 구조 정보 추출"""
        try:
            self.robot = URDF.from_xml_file(self.urdf_path)
            print(f"URDF 파싱 완료: {len(self.robot.joints)} 개 조인트, {len(self.robot.links)} 개 링크")
            
            # 조인트 정보 출력
            for i, joint in enumerate(self.robot.joints):
                print(f"Joint {i+1}: {joint.name}, Type: {joint.type}, Axis: {joint.axis}")
                
        except Exception as e:
            print(f"URDF 파싱 오류: {e}")
            
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
                            
                            try:
                                # 먼저 STL 파일 시도
                                stl_path = mesh_path.replace('.dae', '.STL').replace('visual/', 'collision/')
                                if os.path.exists(stl_path):
                                    # STL 파일 로드
                                    mesh_obj = o3d.io.read_triangle_mesh(stl_path)
                                    if len(mesh_obj.vertices) > 0:
                                        self.meshes[link_name] = mesh_obj
                                        print(f"STL 메시 로드 성공: {link_name} -> {stl_path} ({len(mesh_obj.vertices)} vertices)")
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
                                        print(f"DAE 메시 로드 성공: {link_name} -> {mesh_path} ({len(mesh_obj.vertices)} vertices)")
                                    else:
                                        print(f"빈 메시: {link_name}")
                                else:
                                    print(f"메시 형식 오류: {link_name}")
                            except Exception as e:
                                print(f"메시 로드 실패: {link_name} - {e}")
                                # 메시 로드 실패 시 간단한 기하학적 모양 생성
                                self.create_simple_geometry(link_name)
        
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
                        self.meshes[gripper_name] = mesh_obj
                        print(f"Gripper STL 메시 로드 성공: {gripper_name} -> {stl_path} ({len(mesh_obj.vertices)} vertices)")
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
                            print(f"Gripper DAE 메시 로드 성공: {gripper_name} -> {dae_path} ({len(mesh_obj.vertices)} vertices)")
                            continue
                            
            except Exception as e:
                print(f"Gripper 메시 로드 실패: {gripper_name} - {e}")
                                
    def create_simple_geometry(self, link_name):
        """메시 로드 실패 시 간단한 기하학적 모양 생성"""
        # 링크별로 적절한 크기의 기하학적 모양 생성
        if link_name == 'link00':
            # 베이스 링크 - 원통
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0325, height=0.051)
        elif link_name == 'link01':
            # 첫 번째 링크 - 원통
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=0.045)
        elif link_name == 'link02':
            # 두 번째 링크 - 박스
            mesh = o3d.geometry.TriangleMesh.create_box(width=0.35, height=0.102, depth=0.102)
        elif link_name == 'link03':
            # 세 번째 링크 - 박스
            mesh = o3d.geometry.TriangleMesh.create_box(width=0.116, height=0.059, depth=0.059)
        elif link_name == 'link04':
            # 네 번째 링크 - 원통
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0325, height=0.067)
        elif link_name == 'link05':
            # 다섯 번째 링크 - 원통
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=0.0492)
        elif link_name == 'link06':
            # 여섯 번째 링크 - 원통
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0325, height=0.051)
        else:
            # 기본 - 구
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            
        self.meshes[link_name] = mesh
        print(f"간단한 기하학적 모양 생성: {link_name}")
                                
    def set_joint_angles(self, angles):
        """조인트 각도 설정"""
        self.joint_angles = np.array(angles)
        print(f"조인트 각도 설정: {self.joint_angles}")
        
    def solve_inverse_kinematics(self, target_pose, gripper_angle=0.0):
        """Inverse Kinematics를 사용하여 목표 pose에서 조인트 각도 계산"""
        if self.arm_interface is None:
            print("Unitree Z1 SDK가 초기화되지 않았습니다.")
            return False
            
        try:
            # 현재 조인트 각도를 초기 추정값으로 사용
            current_q = self.joint_angles.copy()
            print(f"초기 조인트 각도: {current_q}")
            
            # Inverse Kinematics 계산
            success, q_forward = self.arm_interface._ctrlComp.armModel.inverseKinematics(
                target_pose, current_q, True  # checkInWorkSpace=True
            )
            
            print(f"계산 후 조인트 각도: {q_forward}")
            
            if success:
                # 계산된 조인트 각도로 업데이트
                self.joint_angles = q_forward
                print(f"Inverse Kinematics 성공: {self.joint_angles}")
                
                # Forward Kinematics로 검증
                fk_result = self.arm_interface._ctrlComp.armModel.forwardKinematics(q_forward, 6)
                print(f"Forward Kinematics 검증 결과:\n{fk_result}")
                
                return True
            else:
                print("Inverse Kinematics 실패: 목표 pose가 작업 공간 밖에 있습니다.")
                return False
                
        except Exception as e:
            print(f"Inverse Kinematics 오류: {e}")
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
        # 메시 복사 (Open3D 방식)
        transformed_mesh = o3d.geometry.TriangleMesh()
        transformed_mesh.vertices = mesh.vertices
        transformed_mesh.triangles = mesh.triangles
        
        # 정점들을 월드 좌표계로 변환
        vertices = np.asarray(transformed_mesh.vertices)
        vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        vertices_world = (transform @ vertices_homogeneous.T).T[:, :3]
        
        # 변환된 정점으로 메시 업데이트
        transformed_mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
        
        return transformed_mesh
        
    def create_plotly_visualization(self, output_file="z1_robot_visualization.html", gripper_angle=0.0):
        """Plotly를 사용하여 3D 시각화 생성"""
        # 전진기구학 계산
        self.compute_forward_kinematics(gripper_angle)
        
        # Plotly figure 생성
        fig = go.Figure()
        
        # 각 링크의 메시를 월드 좌표계로 변환하여 시각화
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        color_idx = 0
        
        for link_name, mesh in self.meshes.items():
            if link_name in self.link_transforms:
                # 메시를 월드 좌표계로 변환
                world_mesh = self.transform_mesh_to_world(mesh, self.link_transforms[link_name])
                
                # 정점과 면 추출
                vertices = np.asarray(world_mesh.vertices)
                triangles = np.asarray(world_mesh.triangles)
                
                if len(vertices) > 0 and len(triangles) > 0:
                    # Plotly Mesh3d로 시각화
                    fig.add_trace(go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1], 
                        z=vertices[:, 2],
                        i=triangles[:, 0],
                        j=triangles[:, 1],
                        k=triangles[:, 2],
                        name=link_name,
                        color=colors[color_idx % len(colors)],
                        opacity=0.7,
                        showscale=False
                    ))
                    
                    color_idx += 1
                    
        # World frame 축 표시
        world_origin = np.array([0, 0, 0])
        axis_length = 0.1  # 10cm 축 길이
        
        # X축 (빨간색)
        fig.add_trace(go.Scatter3d(
            x=[world_origin[0], world_origin[0] + axis_length],
            y=[world_origin[1], world_origin[1]],
            z=[world_origin[2], world_origin[2]],
            mode='lines+markers',
            line=dict(color='red', width=8),
            marker=dict(size=4, color='red'),
            name='X-axis',
            showlegend=False
        ))
        
        # Y축 (초록색)
        fig.add_trace(go.Scatter3d(
            x=[world_origin[0], world_origin[0]],
            y=[world_origin[1], world_origin[1] + axis_length],
            z=[world_origin[2], world_origin[2]],
            mode='lines+markers',
            line=dict(color='green', width=8),
            marker=dict(size=4, color='green'),
            name='Y-axis',
            showlegend=False
        ))
        
        # Z축 (파란색)
        fig.add_trace(go.Scatter3d(
            x=[world_origin[0], world_origin[0]],
            y=[world_origin[1], world_origin[1]],
            z=[world_origin[2], world_origin[2] + axis_length],
            mode='lines+markers',
            line=dict(color='blue', width=8),
            marker=dict(size=4, color='blue'),
            name='Z-axis',
            showlegend=False
        ))
        
        # 축 라벨 추가
        fig.add_trace(go.Scatter3d(
            x=[world_origin[0] + axis_length + 0.02],
            y=[world_origin[1]],
            z=[world_origin[2]],
            mode='text',
            text=['X'],
            textfont=dict(size=16, color='red'),
            name='X-label',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[world_origin[0]],
            y=[world_origin[1] + axis_length + 0.02],
            z=[world_origin[2]],
            mode='text',
            text=['Y'],
            textfont=dict(size=16, color='green'),
            name='Y-label',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[world_origin[0]],
            y=[world_origin[1]],
            z=[world_origin[2] + axis_length + 0.02],
            mode='text',
            text=['Z'],
            textfont=dict(size=16, color='blue'),
            name='Z-label',
            showlegend=False
        ))
        
        # 조인트 위치 표시
        joint_positions = []
        joint_names = []
        for link_name, transform in self.link_transforms.items():
            if 'link' in link_name:
                position = transform[:3, 3]
                joint_positions.append(position)
                joint_names.append(link_name)
                
        if joint_positions:
            joint_positions = np.array(joint_positions)
            fig.add_trace(go.Scatter3d(
                x=joint_positions[:, 0],
                y=joint_positions[:, 1],
                z=joint_positions[:, 2],
                mode='markers',
                marker=dict(size=5, color='black'),
                name='Joint Positions',
                text=joint_names,
                textposition="top center"
            ))
            
        # 레이아웃 설정
        fig.update_layout(
            title=f'Z1 Robot Visualization (Joint Angles: {self.joint_angles})',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            width=1200,
            height=800
        )
        
        # HTML 파일로 저장
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"시각화가 {output_file}에 저장되었습니다.")
        
        return fig
        

def main():
    """메인 함수"""
    # 파일 경로 설정
    urdf_path = "/home/dscho1234/Workspace/unitree_ros/robots/z1_description/xacro/z1.urdf"
    mesh_base_path = "/home/dscho1234/Workspace/unitree_ros/robots/z1_description/"
    
    # Z1 로봇 시각화 객체 생성
    robot_viz = Z1RobotVisualizer(urdf_path, mesh_base_path)
    
    # Inverse Kinematics 기반 테스트 - end effector 6DoF pose 설정
    target_poses = [
        # 테스트 1: 홈 포지션 (원점에서 약간 위)
        np.array([
            [1, 0, 0, 0.0],
            [0, 1, 0, 0.0], 
            [0, 0, 1, 0.4],
            [0, 0, 0, 1]
        ]),
        
        # 테스트 2: 앞쪽으로 뻗은 포지션 (더 멀리)
        np.array([[ 0.50622026,  0.        ,  0.86240423,  0.30484571],
       [ 0.        ,  1.        ,  0.        ,  0.        ],
       [-0.86240423,  0.        ,  0.50622026,  0.20909168],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
        
    ]
    
    # 각 테스트별 gripper 각도 설정 (라디안)
    gripper_angles = [
        0.0,    # 테스트 1: 닫힌 상태
        -1.2,   # 테스트 2: 열린 상태 (약 -69도)
        
    ]
    
    # 테스트 설명
    test_descriptions = [
        "홈 포지션 (원점 위)",
        "앞쪽으로 뻗은 포지션", 
        "옆쪽으로 뻗은 포지션",
        "복잡한 회전 포지션"
    ]
    
    print("Z1 로봇 Inverse Kinematics 기반 시각화 테스트 시작...")
    
    for i, target_pose in enumerate(target_poses):
        gripper_angle = gripper_angles[i]
        gripper_status = "열린" if gripper_angle != 0.0 else "닫힌"
        description = test_descriptions[i]
        
        print(f"\n=== 테스트 {i+1}: {description}, Gripper: {gripper_status} ===")
        print(f"목표 pose:\n{target_pose}")
        
        # 홈 포지션으로 초기화
        robot_viz.set_joint_angles([0, 0, 0, 0, 0, 0])
        
        # Inverse Kinematics로 조인트 각도 계산
        ik_success = robot_viz.solve_inverse_kinematics(target_pose, gripper_angle)
        
        if ik_success:
            # Plotly 시각화 생성 및 저장
            save_path = 'test_urdf_mesh'
            output_file = f"{save_path}/z1_robot_ik_test_{i+1}.html"
            fig = robot_viz.create_plotly_visualization(output_file, gripper_angle)
            
            print(f"테스트 {i+1} 완료: {output_file} 생성됨 (Gripper: {gripper_status})")
        else:
            print(f"테스트 {i+1} 실패: Inverse Kinematics 해결 불가")
        
    print("\n모든 테스트 완료!")
    print("생성된 HTML 파일들을 브라우저에서 열어서 3D 시각화를 확인하세요.")

if __name__ == "__main__":
    main()
