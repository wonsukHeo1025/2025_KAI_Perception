# PRISM 성능 최적화 적용 계획서 (실코드 변경 전 계획)

본 문서는 PRISM의 보간/프로젝션 경로에서 “확실히 효과가 있는” 최적화만 단계적으로 적용하기 위한 구체 실행 계획입니다. 실제 코드 수정 전 참고용이며, 각 변경의 적용 위치, 세부 절차, 검증 항목과 롤백 방법을 포함합니다.

## 목표와 범위
- 대상: `prism_fusion_node`의 LiDAR 보간 경로(grid/FILC-style) 및 퍼블리시 경로
- 센서 전제: Ouster OS1-32 (32x1024), ring-first 사용
- 제외: TBB 파이프라인/범용 메모리풀(효과 대비 복잡도↑)은 이번 계획에서 제외

---

## 변경 항목 A: Zero‑copy 입력 (PointCloud2 → SoA)
- 현재: `pcl::fromROSMsg` → PCL → SoA 2단계 복사
- 변경: `sensor_msgs::PointCloud2ConstIterator<float/uint16_t>`로 `x,y,z,intensity,ring`를 직접 읽어 SoA에 채움
- 적용 파일/위치:
  - 입력 변환 시작: `prism/src/nodes/prism_fusion_node.cpp:520`

### 절차
1) `pcl::fromROSMsg`로 만드는 `pcl::PointCloud<pcl::PointXYZI>` 제거
2) 필드 존재 여부 검사 함수 유지(`ring`/기타 필드)
3) 다음 이터레이터를 준비 후 단일 루프로 SoA 채움
   - `sensor_msgs::PointCloud2ConstIterator<float> it_x(*lidar_msg, "x");` (y,z,intensity 동등)
   - `sensor_msgs::PointCloud2ConstIterator<uint16_t> it_ring(*lidar_msg, has_ring?"ring":"x");`
4) SoA에 `addPoint(*it_x, *it_y, *it_z, *it_i, 0,0,0, ring)` 추가 후 이터레이터 전진

### 주의사항
- `intensity` 타입이 `float`(일반적)인지 확인. 타입 상이 시 별도 이터레이터 준비
- 필드 미존재 시 기본값(0) 처리 유지

### 검증
- 동일 프레임 포인트 수/분포(ring 포함) 일치
- 평균 처리시간(ms) 개선(복사 1회 제거)

### 롤백
- `pcl::fromROSMsg` 복귀 및 이터레이터 블록 주석 처리

---

## 변경 항목 B: Zero‑copy 출력 (SoA → PointCloud2 직접 구성)
- 현재: SoA → PCL → `pcl::toROSMsg` → PointCloud2
- 변경: `sensor_msgs::PointCloud2Modifier`/Iterator로 직접 PointCloud2 채움
- 적용 파일/위치:
  - 보간 결과 퍼블리시: `prism/src/nodes/prism_fusion_node.cpp:638` 및 `prism/src/nodes/prism_fusion_node.cpp:1100`

### 절차
1) `sensor_msgs::msg::PointCloud2 out; out.header = lidar_msg->header;`
2) `sensor_msgs::PointCloud2Modifier mod(out);`
3) `mod.setPointCloud2FieldsByString(2, "xyz", "intensity");` 또는 수동 필드 지정
4) `out.height = output_height; out.width = input_width; out.is_dense=false;`
5) `sensor_msgs::PointCloud2Iterator<float> ox(out, "x"), oy(out, "y"), oz(out, "z"), oi(out, "intensity");`
6) SoA에서 순서대로 복사(보간 결과 `gx/gy/gz/gi` 또는 `interpolated` SoA 기준)

### 주의사항
- ring 필드를 출력에 포함할지 정책 결정(보간 후 ring 재할당 가능: 새 행 index)
- `point_step`,`row_step`는 modifier가 자동 관리

### 검증
- RViz 표시/다운스트림 노드 호환성 확인(xyz/intensity 존재)
- 변환 제거로 지연 감소

### 롤백
- 기존 PCL 변환 블록 복구

---

## 변경 항목 C: 보간 쓰기 단계 OpenMP 병렬화
- 대상 루프: 행(r) 고정, 열(c)로 보간/쓰기 (`set_cell`) 수행 구간
- 적용 파일/위치:
  - 보간 루프 시작 지점: `prism/src/nodes/prism_fusion_node.cpp:606`

### 절차
1) 내부 열 루프에 `#pragma omp parallel for schedule(static, 32)` 추가
2) `set_cell`은 (r,c) 유일 인덱스에만 쓰므로 데이터 경합 없음(스레드 안전)
3) `inserts`/`discontinuous` 계산은 각 (r,c) 로컬 상태이므로 안전

### 주의사항
- 그 이전의 폴백 binning 단계(그리드 인덱스 채우기)는 경쟁이 있어 병렬화 금지
- 너무 작은 `input_width`에서는 병렬화 이득이 미미할 수 있음

### 검증
- 프레임당 보간 단계 시간 감소
- 결과 값 동일성(디버그 모드에서 checksum 비교)

### 롤백
- `#pragma omp`만 제거하면 끝

---

## 변경 항목 D: 프레임 간 버퍼 재사용(할당/초기화 비용 절감)
- 대상 버퍼: `grid_idx`, `best_diff`(폴백), 보간 결과 `gx/gy/gz/gi`
- 적용 파일/위치:
  - 선언 위치: `PrismFusionNode` 멤버(파일 하단 멤버 섹션 `prism/src/nodes/prism_fusion_node.cpp:1294` 이후)
  - 사용 전 준비: 그리드/보간 시작 시 capacity 확인 후 `assign`/`std::fill`로 재초기화

### 절차
1) 멤버 추가(예)
   - `std::vector<size_t> grid_idx_flat_;` (2D 대신 1D: size=H*W, 초기값=(size_t)-1)
   - `std::vector<float> best_diff_;`
   - `std::vector<float> gx_, gy_, gz_, gi_;`
2) fast-path에서는 `grid_idx_flat_[r*W+c]=i`로 인덱싱(기존 2D 벡터 제거)
3) out_size 변동 시 `resize`만, 여유 용량은 유지(shrink 금지)

### 주의사항
- 1D로 바꾸면 인덱싱 오류 주의
- `assign`은 size를 바꾸므로, `std::fill`을 사용하면 capacity 유지에 유리

### 검증
- 할당/초기화 카운터(allocator 정책 또는 추적 로그) 감소
- GC/메모리 파편화 및 지터 감소

### 롤백
- 로컬 벡터(스택 스코프)로 되돌림

---

## 변경 항목 E: 폴백 중심각 사전계산
- 대상: `center = -pi + (c+0.5f) * (2*pi/width)` 반복 계산 제거
- 적용 파일/위치:
  - 멤버: `std::vector<float> center_azimuths_;`
  - 사용: 폴백 binning 단계(`update_cell` 내 center 사용)

### 절차
1) 첫 프레임 또는 width 변경 시 `center_azimuths_.resize(width)` 후 채움
2) `update_cell`에서 `center_azimuths_[c]` 재사용

### 검증
- 성능 소폭 개선(분기/연산 감소), 결과 동일성 보장

### 롤백
- 즉시 수식 재계산으로 복귀

---

## 빌드/설정
- CMake: 기본 빌드 타입 Release 설정(이미 적용됨)
- OpenMP: `find_package(OpenMP REQUIRED)` 및 타겟 링크 유지(현 상태 유지)
- 선택: `PRISM_ENABLE_NATIVE_OPT=ON` 시 `-march=native` 추가(기본 OFF 권장)

---

## 검증/측정 계획
- 지표
  - 프레임 처리시간 평균/백분위수(P50/P90/P99)
  - FPS, CPU 사용률, 메모리 사용량/할당 횟수
- 시나리오
  - rosbag(고정 데이터) vs 실센서(변동) 각각 2분 이상
  - organized+ring fast‑path, unorganized 폴백 모두 커버
- 품질 확인
  - 포인트 개수/분포 동일성(허용 오차 0)
  - 불연속 경계 유지(시각 확인 및 통계)

---

## 위험/주의사항
- 입력 Zero‑copy에서 필드 타입 상이 시 런타임 예외 가능 → 필드 타입 검사/가드 추가
- 출력 Zero‑copy에서 필드 레이아웃 오설정 시 RViz/다운스트림 깨짐 → `PointCloud2Modifier` 사용 권장
- 병렬화 시 공유 상태 접근 금지(폴백 binning 단계는 단일 스레드 유지)

---

## 롤백 전략
- 각 항목은 독립적으로 주석/되돌리기 가능하도록 패치함
- 문제가 생기면 최근 항목부터 역순( E → D → C → B → A )으로 롤백

---

## 적용 순서 제안(스프린트 단위)
1) A 입력 Zero‑copy + B 출력 Zero‑copy
2) C 보간 쓰기 OpenMP 병렬화
3) D 버퍼 재사용(1D 전환 포함)
4) E 중심각 사전계산

각 단계마다 측정/검증 완료 후 다음 단계 진행.

