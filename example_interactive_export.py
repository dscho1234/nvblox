#!/usr/bin/env python3
"""
Interactive 3D Export 사용 예제
Open3D mesh를 interactive HTML로 export하는 방법을 보여줍니다.
"""

import numpy as np
import open3d as o3d
from interactive_3d_exporter import Interactive3DExporter, create_custom_threejs_viewer


def create_sample_scene():
    """샘플 3D 씬 생성"""
    # 바닥 평면
    floor = o3d.geometry.TriangleMesh.create_box(width=10, height=0.1, depth=10)
    floor.translate([-5, -0.05, -5])
    floor.paint_uniform_color([0.5, 0.5, 0.5])
    
    # 벽들
    wall1 = o3d.geometry.TriangleMesh.create_box(width=0.1, height=3, depth=10)
    wall1.translate([-5, 0, -5])
    wall1.paint_uniform_color([0.8, 0.6, 0.4])
    
    wall2 = o3d.geometry.TriangleMesh.create_box(width=10, height=3, depth=0.1)
    wall2.translate([-5, 0, -5])
    wall2.paint_uniform_color([0.8, 0.6, 0.4])
    
    # 테이블
    table = o3d.geometry.TriangleMesh.create_box(width=2, height=0.1, depth=1)
    table.translate([-1, 0.7, -0.5])
    table.paint_uniform_color([0.4, 0.2, 0.1])
    
    # 테이블 다리들
    for i, pos in enumerate([[-1, 0, -0.5], [1, 0, -0.5], [-1, 0, 0.5], [1, 0, 0.5]]):
        leg = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=0.7)
        leg.translate([pos[0], 0.35, pos[1]])
        leg.paint_uniform_color([0.2, 0.1, 0.05])
        table += leg
    
    # 컵
    cup = o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=0.3)
    cup.translate([0, 0.95, 0])
    cup.paint_uniform_color([0.1, 0.1, 0.8])
    
    # 모든 메시 결합
    scene = floor + wall1 + wall2 + table + cup
    
    return scene


def create_robot_arm():
    """간단한 로봇 팔 생성"""
    # 베이스
    base = o3d.geometry.TriangleMesh.create_cylinder(radius=0.2, height=0.3)
    base.paint_uniform_color([0.3, 0.3, 0.3])
    
    # 첫 번째 링크
    link1 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=1.0)
    link1.translate([0, 0.5, 0])
    link1.paint_uniform_color([0.7, 0.1, 0.1])
    
    # 두 번째 링크
    link2 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.08, height=0.8)
    link2.translate([0, 1.2, 0])
    link2.paint_uniform_color([0.1, 0.7, 0.1])
    
    # 그리퍼
    gripper = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.1, depth=0.3)
    gripper.translate([-0.1, 1.6, -0.15])
    gripper.paint_uniform_color([0.1, 0.1, 0.7])
    
    robot = base + link1 + link2 + gripper
    return robot


def main():
    """메인 함수"""
    print("Creating sample 3D scene...")
    
    # 샘플 씬 생성
    scene = create_sample_scene()
    robot = create_robot_arm()
    
    # 로봇을 씬에 추가
    robot.translate([2, 0, 2])
    combined_scene = scene + robot
    
    print(f"Scene created with {len(combined_scene.vertices)} vertices and {len(combined_scene.triangles)} triangles")
    
    # Interactive HTML export
    print("\nExporting to interactive HTML...")
    
    # PyVista 방식으로 export
    exporter = Interactive3DExporter()
    exporter.export_mesh_to_html(
        combined_scene,
        "example_scene_pyvista.html",
        title="Sample 3D Scene - PyVista",
        show_axes=True
    )
    
    # Custom Three.js 방식으로 export
    create_custom_threejs_viewer(
        combined_scene,
        "example_scene_threejs.html",
        title="Sample 3D Scene - Three.js"
    )
    
    # 카메라 궤적 시뮬레이션
    print("\nCreating camera trajectory...")
    camera_poses = []
    for i in range(10):
        # 원형 궤적으로 카메라 포즈 생성
        angle = i * 2 * np.pi / 10
        x = 5 * np.cos(angle)
        z = 5 * np.sin(angle)
        y = 2
        
        # 4x4 변환 행렬 생성
        pose = np.eye(4)
        pose[:3, 3] = [x, y, z]
        camera_poses.append(pose)
    
    # 카메라 궤적과 함께 export
    exporter.export_with_camera_trajectory(
        combined_scene,
        camera_poses,
        "example_scene_with_camera_trajectory.html",
        title="Sample 3D Scene with Camera Trajectory"
    )
    
    print("\nExport completed!")
    print("Generated files:")
    print("  - example_scene_pyvista.html (PyVista version)")
    print("  - example_scene_threejs.html (Three.js version)")
    print("  - example_scene_with_camera_trajectory.html (with camera trajectory)")
    print("\nOpen these files in a web browser to view the interactive 3D visualization!")


if __name__ == "__main__":
    main()
