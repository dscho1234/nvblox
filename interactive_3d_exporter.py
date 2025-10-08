#!/usr/bin/env python3
"""
Interactive 3D Visualization Exporter for nvblox
Open3D visualizer를 interactive HTML로 저장하는 도구

사용법:
    from interactive_3d_exporter import Interactive3DExporter
    
    exporter = Interactive3DExporter()
    exporter.export_mesh_to_html(open3d_mesh, "output.html")
"""

import numpy as np
import open3d as o3d
import pyvista as pv
import tempfile
import os
from typing import Optional, List, Dict, Any
import json


class Interactive3DExporter:
    """
    Open3D mesh를 interactive HTML로 export하는 클래스
    PyVista + Three.js 기반으로 구현
    """
    
    def __init__(self):
        """초기화"""
        self.temp_dir = tempfile.mkdtemp()
        
    def __del__(self):
        """임시 파일 정리"""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def open3d_to_pyvista(self, open3d_mesh: o3d.geometry.TriangleMesh) -> pv.PolyData:
        """
        Open3D mesh를 PyVista mesh로 변환
        
        Args:
            open3d_mesh: Open3D TriangleMesh 객체
            
        Returns:
            PyVista PolyData 객체
        """
        # Open3D mesh에서 vertices와 faces 추출
        vertices = np.asarray(open3d_mesh.vertices)
        faces = np.asarray(open3d_mesh.triangles)
        
        # PyVista 형식으로 faces 변환 (각 face 앞에 vertex 개수 추가)
        faces_pv = np.column_stack([
            np.full(faces.shape[0], 3),  # 각 face는 3개 vertex
            faces
        ]).flatten()
        
        # PyVista PolyData 생성
        pv_mesh = pv.PolyData(vertices, faces_pv)
        
        # 색상 정보가 있다면 추가
        if open3d_mesh.has_vertex_colors():
            colors = np.asarray(open3d_mesh.vertex_colors)
            # Open3D는 0-1 범위, PyVista는 0-255 범위
            colors_uint8 = (colors * 255).astype(np.uint8)
            pv_mesh.point_data['colors'] = colors_uint8
            
        return pv_mesh
    
    def export_mesh_to_html(self, 
                           open3d_mesh: o3d.geometry.TriangleMesh,
                           output_path: str,
                           title: str = "Interactive 3D Visualization",
                           background_color: str = "black",
                           show_axes: bool = True,
                           camera_position: Optional[List[float]] = None) -> str:
        """
        Open3D mesh를 interactive HTML로 export
        
        Args:
            open3d_mesh: Open3D TriangleMesh 객체
            output_path: 출력 HTML 파일 경로
            title: HTML 제목
            background_color: 배경색
            show_axes: 축 표시 여부
            camera_position: 카메라 위치 [x, y, z]
            
        Returns:
            생성된 HTML 파일 경로
        """
        # Open3D mesh를 PyVista로 변환
        pv_mesh = self.open3d_to_pyvista(open3d_mesh)
        
        # PyVista plotter 생성
        plotter = pv.Plotter(off_screen=True)
        
        # 메시 추가
        if pv_mesh.point_data.get('colors') is not None:
            # 색상이 있는 경우
            plotter.add_mesh(pv_mesh, scalars='colors', rgb=True)
        else:
            # 색상이 없는 경우 기본 색상 사용
            plotter.add_mesh(pv_mesh, color='lightblue')
        
        # 축 표시
        if show_axes:
            plotter.add_axes()
        
        # 배경색 설정
        plotter.background_color = background_color
        
        # 카메라 위치 설정
        if camera_position:
            plotter.camera_position = camera_position
        else:
            # 자동으로 적절한 카메라 위치 설정
            plotter.camera_position = 'iso'
        
        # HTML로 export (title 파라미터 제거)
        plotter.export_html(output_path)
        
        print(f"Interactive 3D visualization saved to: {output_path}")
        return output_path
    
    def export_multiframe_animation(self,
                                   open3d_meshes: List[o3d.geometry.TriangleMesh],
                                   output_path: str,
                                   title: str = "Interactive 3D Animation",
                                   frame_duration: float = 1.0) -> str:
        """
        여러 프레임의 mesh를 애니메이션으로 export
        
        Args:
            open3d_meshes: Open3D mesh 리스트
            output_path: 출력 HTML 파일 경로
            title: HTML 제목
            frame_duration: 각 프레임 지속 시간 (초)
            
        Returns:
            생성된 HTML 파일 경로
        """
        if not open3d_meshes:
            raise ValueError("No meshes provided")
        
        # 첫 번째 mesh로 기본 설정
        pv_meshes = [self.open3d_to_pyvista(mesh) for mesh in open3d_meshes]
        
        # PyVista plotter 생성
        plotter = pv.Plotter(off_screen=True)
        
        # 첫 번째 mesh 추가
        if pv_meshes[0].point_data.get('colors') is not None:
            plotter.add_mesh(pv_meshes[0], scalars='colors', rgb=True)
        else:
            plotter.add_mesh(pv_meshes[0], color='lightblue')
        
        # 애니메이션 설정
        plotter.add_axes()
        plotter.background_color = 'black'
        plotter.camera_position = 'iso'
        
        # 애니메이션 HTML 생성 (간단한 버전)
        # 실제로는 더 복잡한 애니메이션 로직이 필요할 수 있음
        plotter.export_html(output_path)
        
        print(f"Interactive 3D animation saved to: {output_path}")
        return output_path
    
    def export_with_camera_trajectory(self,
                                    open3d_mesh: o3d.geometry.TriangleMesh,
                                    camera_poses: List[np.ndarray],
                                    output_path: str,
                                    title: str = "Interactive 3D with Camera Trajectory") -> str:
        """
        카메라 궤적과 함께 mesh를 export
        
        Args:
            open3d_mesh: Open3D TriangleMesh 객체
            camera_poses: 카메라 포즈 리스트 (4x4 변환 행렬)
            output_path: 출력 HTML 파일 경로
            title: HTML 제목
            
        Returns:
            생성된 HTML 파일 경로
        """
        # 메시 변환
        pv_mesh = self.open3d_to_pyvista(open3d_mesh)
        
        # PyVista plotter 생성
        plotter = pv.Plotter(off_screen=True)
        
        # 메시 추가
        if pv_mesh.point_data.get('colors') is not None:
            plotter.add_mesh(pv_mesh, scalars='colors', rgb=True)
        else:
            plotter.add_mesh(pv_mesh, color='lightblue')
        
        # 카메라 궤적 추가
        camera_positions = []
        for pose in camera_poses:
            # 4x4 변환 행렬에서 위치 추출
            position = pose[:3, 3]
            camera_positions.append(position)
        
        if camera_positions:
            # 카메라 위치를 점으로 표시
            camera_points = pv.PolyData(camera_positions)
            plotter.add_mesh(camera_points, color='red', point_size=10)
            
            # 카메라 궤적을 선으로 표시
            if len(camera_positions) > 1:
                camera_line = pv.PolyData()
                camera_line.points = np.array(camera_positions)
                lines = np.column_stack([
                    np.full(len(camera_positions)-1, 2),
                    np.arange(len(camera_positions)-1),
                    np.arange(1, len(camera_positions))
                ]).flatten()
                camera_line.lines = lines
                plotter.add_mesh(camera_line, color='yellow', line_width=3)
        
        plotter.add_axes()
        plotter.background_color = 'black'
        plotter.camera_position = 'iso'
        
        plotter.export_html(output_path)
        
        print(f"Interactive 3D visualization with camera trajectory saved to: {output_path}")
        return output_path


def create_custom_threejs_viewer(open3d_mesh: o3d.geometry.TriangleMesh,
                                output_path: str,
                                title: str = "Custom Three.js Viewer") -> str:
    """
    Open3D mesh를 직접 Three.js 형식으로 변환하여 HTML 생성
    더 가벼운 파일 크기와 커스터마이징 가능
    
    Args:
        open3d_mesh: Open3D TriangleMesh 객체
        output_path: 출력 HTML 파일 경로
        title: HTML 제목
        
    Returns:
        생성된 HTML 파일 경로
    """
    # 메시 데이터 추출 및 디버깅 정보 출력
    vertices = np.asarray(open3d_mesh.vertices)
    faces = np.asarray(open3d_mesh.triangles)
    colors = np.asarray(open3d_mesh.vertex_colors) if open3d_mesh.has_vertex_colors() else None
    
    print(f"Debug - Mesh info:")
    print(f"  Vertices: {vertices.shape}")
    print(f"  Faces: {faces.shape}")
    print(f"  Has colors: {colors is not None}")
    if vertices.shape[0] == 0:
        print("  WARNING: No vertices found!")
        return output_path
    if faces.shape[0] == 0:
        print("  WARNING: No faces found!")
        return output_path
    
    # Three.js JSON 형식으로 변환
    geometry_data = {
        "vertices": vertices.flatten().tolist(),
        "faces": faces.flatten().tolist(),
        "colors": (colors.flatten() * 255).astype(np.uint8).tolist() if colors is not None else None
    }
    
    # HTML 템플릿 생성
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ margin: 0; padding: 0; background: #000; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info {{ position: absolute; top: 10px; left: 10px; color: white; font-family: Arial; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h3>{title}</h3>
        <p>Mouse: Rotate | Wheel: Zoom | Right-click: Pan</p>
        <p id="debug-info">Loading...</p>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor(0x000000);
        document.getElementById('container').appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Geometry data
        const geometryData = {json.dumps(geometry_data)};
        
        // Debug info
        document.getElementById('debug-info').innerHTML = 
            `Vertices: ${{geometryData.vertices.length/3}} | Faces: ${{geometryData.faces.length/3}} | Colors: ${{geometryData.colors ? 'Yes' : 'No'}}`;
        
        // Create geometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(geometryData.vertices, 3));
        geometry.setIndex(geometryData.faces);
        
        // Add colors if available
        if (geometryData.colors) {{
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(geometryData.colors, 3));
        }}
        
        // Compute bounding box for debugging
        geometry.computeBoundingBox();
        const bbox = geometry.boundingBox;
        console.log('Bounding box:', bbox);
        
        // Create material
        const material = new THREE.MeshLambertMaterial({{ 
            vertexColors: geometryData.colors ? true : false,
            side: THREE.DoubleSide,
            color: geometryData.colors ? 0xffffff : 0x888888  // 색상이 없으면 회색으로 설정
        }});
        
        // Create mesh
        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);
        
        // Lighting (더 밝게 설정)
        const ambientLight = new THREE.AmbientLight(0x404040, 1.0);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);
        
        // 추가 조명
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight2.position.set(-1, -1, -1);
        scene.add(directionalLight2);
        
        // Camera position
        const box = new THREE.Box3().setFromObject(mesh);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        
        console.log('Mesh center:', center);
        console.log('Mesh size:', size);
        console.log('Max dimension:', maxDim);
        
        // Ensure we have a valid size
        if (maxDim === 0) {{
            console.warn('Mesh has zero size, using default camera position');
            camera.position.set(0, 0, 5);
            controls.target.set(0, 0, 0);
        }} else {{
            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            cameraZ *= 2.0; // Add more margin
            
            camera.position.set(center.x, center.y, center.z + cameraZ);
            controls.target.copy(center);
        }}
        controls.update();
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        animate();
    </script>
</body>
</html>
    """
    
    # HTML 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Custom Three.js viewer saved to: {output_path}")
    return output_path


# 사용 예제
if __name__ == "__main__":
    # 예제: 간단한 큐브 생성 및 export
    cube = o3d.geometry.TriangleMesh.create_box()
    cube.paint_uniform_color([0.7, 0.1, 0.1])
    
    # PyVista 방식으로 export
    exporter = Interactive3DExporter()
    exporter.export_mesh_to_html(cube, "test_cube_pyvista.html", "Test Cube - PyVista")
    
    # Custom Three.js 방식으로 export
    create_custom_threejs_viewer(cube, "test_cube_threejs.html", "Test Cube - Three.js")
    
    print("Example exports completed!")
