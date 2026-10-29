"""整帧坐标解算；GPU 失败时保留输入并在 CPU 上重算。"""
import importlib
import os
from pathlib import Path
import numpy as np


class PointCloudBackend:
    def __init__(self, requested='auto'):
        if requested not in ('auto', 'gpu', 'cpu'):
            raise ValueError('解析后端必须为 auto、gpu 或 cpu')
        self.active_backend = 'cpu'
        self.fallback_reason = ''
        self.cp = None
        self._kernel = None
        self.device_name = ''
        if requested != 'cpu':
            try:
                # 使用程序目录缓存内核，避免依赖用户主目录的写入权限。
                os.environ.setdefault('CUPY_CACHE_DIR', str(Path(__file__).resolve().parent / '.gpu_cache'))
                self.cp = importlib.import_module('cupy')
                if self.cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError('没有可用的 CUDA 设备')
                # 实际执行一次运算，提前发现运行库、内核编译或驱动问题。
                probe = self.cp.arange(4, dtype=self.cp.float32)
                self.cp.asnumpy(self.cp.sin(probe))
                properties = self.cp.cuda.runtime.getDeviceProperties(self.cp.cuda.Device().id)
                name = properties['name']
                self.device_name = name.decode() if isinstance(name, bytes) else str(name)
                self.active_backend = 'gpu'
            except InterruptedError:
                raise
            except Exception as error:
                self._fallback(error)

    def _fallback(self, error):
        self.active_backend = 'cpu'
        self.fallback_reason = f'{type(error).__name__}: {error}'

    @property
    def status(self):
        if self.active_backend == 'gpu':
            return f'GPU（{self.device_name}）'
        if self.fallback_reason:
            return f'CPU（GPU 已回退：{self.fallback_reason}）'
        return 'CPU'

    @staticmethod
    def _compute(packed, matrix, xp):
        # 输入列：距离、插值方位角、垂直角余弦/正弦、强度、线号。
        # 距离和输出保留 float64，与原解析器保持相同精度；角度按原算法用 float32。
        az_rad = xp.radians(packed[:, 1].astype(xp.float32))
        radial = packed[:, 0] * packed[:, 2]
        radar = xp.column_stack((radial * xp.cos(az_rad),
                                 -radial * xp.sin(az_rad),
                                 packed[:, 0] * packed[:, 3],
                                 packed[:, 4:]))
        homogeneous = xp.column_stack((radar[:, :3], xp.ones(len(radar), dtype=xp.float32)))
        site_xyz = (matrix @ homogeneous.T).T[:, :3]
        site = xp.column_stack((site_xyz, radar[:, 3:]))
        return radar, site

    def _compute_gpu(self, packed, matrix):
        cp = self.cp
        if self._kernel is None:
            # 三角函数、坐标解算和外参变换合成一个内核，避免每帧启动几十次小运算。
            self._kernel = cp.ElementwiseKernel(
                'raw float64 data, raw float64 matrix',
                'raw float64 radar, raw float64 site',
                r'''
                const long long k = (long long)i * 6;
                const long long o = (long long)i * 5;
                const float a = (float)data[k + 1] * 0.017453292519943295f;
                const double radial = data[k] * data[k + 2];
                const double x = radial * (double)cosf(a);
                const double y = -radial * (double)sinf(a);
                const double z = data[k] * data[k + 3];
                radar[o] = x;
                radar[o + 1] = y;
                radar[o + 2] = z;
                radar[o + 3] = data[k + 4];
                radar[o + 4] = data[k + 5];
                for (int row = 0; row < 3; ++row) {
                    const int m = row * 4;
                    site[o + row] = matrix[m] * x + matrix[m + 1] * y
                                  + matrix[m + 2] * z + matrix[m + 3];
                }
                site[o + 3] = data[k + 4];
                site[o + 4] = data[k + 5];
                ''', 'tm16_coordinates_and_extrinsics', options=('--fmad=false',))
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] != 4:
            raise ValueError('外参矩阵需要至少三行、四列')
        data = cp.asarray(np.ascontiguousarray(packed))
        transform = cp.asarray(np.ascontiguousarray(matrix))
        radar = cp.empty((len(packed), 5), dtype=cp.float64)
        site = cp.empty_like(radar)
        self._kernel(data, transform, radar, site, size=len(packed))
        # 拷回也在异常边界内，确保异步 CUDA 错误会触发当前帧重算。
        return cp.asnumpy(radar), cp.asnumpy(site)

    def compute(self, packed, matrix):
        if len(packed) == 0:
            empty = np.empty((0, 5), dtype=np.float32)
            return empty, empty.copy()
        if self.active_backend == 'gpu':
            try:
                return self._compute_gpu(packed, matrix)
            except InterruptedError:
                raise
            except Exception as error:
                self._fallback(error)
        return self._compute(packed, matrix, np)
