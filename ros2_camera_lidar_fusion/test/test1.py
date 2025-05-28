import cupy as cp

try:
    print(f"CuPy를 성공적으로 임포트했습니다. 사용 가능한 GPU 개수: {cp.cuda.runtime.getDeviceCount()}")
    # 간단한 CuPy 연산 테스트
    x_gpu = cp.array([1, 2, 3])
    y_gpu = x_gpu * 2
    print(f"CuPy 연산 결과 (GPU): {y_gpu}")
    print(f"CuPy 연산 결과 (CPU로 가져오기): {cp.asnumpy(y_gpu)}")
except Exception as e:
    print(f"CuPy 임포트 또는 테스트 중 오류 발생: {e}")

import numba
print(f"Numba 버전: {numba.__version__}")
try:
    from numba import cuda
    if cuda.is_available():
        print("Numba가 CUDA를 사용할 수 있습니다.")
        print(f"CUDA 디바이스: {cuda.list_devices()}")
    else:
        print("Numba가 CUDA를 사용할 수 없거나, CUDA 지원 드라이버/런타임이 없습니다.")
except ImportError:
    print("Numba CUDA 지원이 설치되지 않았거나 문제가 있습니다.")
except Exception as e:
    print(f"Numba CUDA 테스트 중 오류: {e}")