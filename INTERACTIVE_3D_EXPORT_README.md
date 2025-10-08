# Interactive 3D Export for nvblox

Open3D visualizer에서 생성된 3D 시각화를 interactive HTML로 저장하여 언제든지 다시 열어볼 수 있도록 하는 도구입니다.

## 🎯 주요 기능

- **Open3D mesh를 interactive HTML로 변환**
- **PyVista + Three.js 기반 고품질 시각화**
- **Custom Three.js viewer (더 가벼운 파일 크기)**
- **카메라 궤적 시각화**
- **멀티프레임 애니메이션 지원**

## 📦 설치 요구사항

**추가 설치 불필요!** 이 도구는 표준 Python 라이브러리만 사용합니다.

- `numpy`
- `open3d` (이미 설치되어 있음)
- 웹 브라우저 (HTML 파일을 열기 위해)

> **참고**: PyVista 방식은 복잡한 의존성 문제로 인해 현재 비활성화되어 있습니다. Custom Three.js 방식이 더 안정적이고 가볍습니다.

## 🚀 사용법

### 1. 기본 사용법

```python
from interactive_3d_exporter import create_custom_threejs_viewer
import open3d as o3d

# Open3D mesh 생성
mesh = o3d.geometry.TriangleMesh.create_box()
mesh.paint_uniform_color([0.7, 0.1, 0.1])

# Custom Three.js 방식으로 export (권장)
create_custom_threejs_viewer(mesh, "output_threejs.html", "My 3D Scene")
```

### 2. nvblox와 함께 사용

기존 `create_multiframe_nvblox` 함수에 `export_interactive_html=True` 파라미터를 추가하면 자동으로 interactive HTML이 생성됩니다:

```python
create_multiframe_nvblox(
    "output.ply",
    rgb_frames_list, depth_frames_list, relative_poses_list, K_adjusted_list,
    robot_meshes_list=robot_meshes_list,
    voxel_size=0.01,
    export_interactive_html=True  # 이 옵션 추가
)
```

### 3. 카메라 궤적과 함께 export

```python
# 카메라 포즈 리스트 생성
camera_poses = []
for i in range(10):
    pose = np.eye(4)
    pose[:3, 3] = [i, 0, 5]  # 카메라 위치
    camera_poses.append(pose)

# 카메라 궤적과 함께 export
exporter.export_with_camera_trajectory(
    mesh, camera_poses, "output_with_trajectory.html"
)
```

## 📁 생성되는 파일들

`create_multiframe_nvblox`를 실행하면 다음과 같은 파일들이 생성됩니다:

- `*_interactive_threejs.html` - Custom Three.js 기반 interactive viewer
- 카메라 궤적 정보는 콘솔에 출력됩니다

## 🎮 Interactive 기능

생성된 HTML 파일은 다음 기능들을 지원합니다:

- **마우스 드래그**: 3D 모델 회전
- **마우스 휠**: 줌 인/아웃
- **우클릭 드래그**: 팬 (이동)
- **자동 카메라 위치**: 최적의 시점으로 자동 설정
- **축 표시**: X, Y, Z 축 표시
- **조명**: 실시간 조명 효과

## 🔧 고급 사용법

### 커스텀 설정

```python
exporter = Interactive3DExporter()

exporter.export_mesh_to_html(
    mesh,
    "output.html",
    title="My Custom Title",
    background_color="white",
    show_axes=True,
    camera_position=[5, 5, 5]  # 커스텀 카메라 위치
)
```

### 멀티프레임 애니메이션

```python
# 여러 프레임의 mesh 리스트
meshes = [mesh1, mesh2, mesh3, ...]

exporter.export_multiframe_animation(
    meshes,
    "animation.html",
    title="My Animation",
    frame_duration=1.0  # 각 프레임 지속 시간 (초)
)
```

## 🎯 Custom Three.js의 장점

| 특징 | Custom Three.js |
|------|-----------------|
| 파일 크기 | 매우 작음 (~4KB) |
| 기능 | 핵심 기능 완비 |
| 커스터마이징 | 완전 자유 |
| 성능 | 매우 좋음 |
| 의존성 | 없음 (표준 라이브러리만) |
| 안정성 | 높음 |

## 🐛 문제 해결

### 메모리 부족
- 큰 mesh의 경우 mesh를 단순화하거나 샘플링하세요
- Open3D의 `simplify_vertex_clustering` 또는 `simplify_quadric_decimation` 사용

### 브라우저에서 표시되지 않음
- 최신 브라우저를 사용하세요 (Chrome, Firefox, Safari)
- JavaScript가 활성화되어 있는지 확인하세요
- CDN에서 Three.js를 로드하므로 인터넷 연결이 필요합니다

### 파일이 너무 큼
- Custom Three.js 방식은 이미 매우 가벼움 (~4KB)
- 필요시 mesh의 vertex/triangle 수를 줄이세요

## 📝 예제 실행

```bash
# 예제 실행
python example_interactive_export.py
```

이 예제는 샘플 3D 씬을 생성하고 다양한 방식으로 export합니다.

## 🤝 기여하기

버그 리포트나 기능 요청은 GitHub Issues를 통해 제출해주세요.

## 📄 라이선스

이 프로젝트는 nvblox와 동일한 라이선스를 따릅니다.
