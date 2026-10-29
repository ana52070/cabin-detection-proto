"""批量解码、标定、切帧及 GPU 故障回退的回归测试。"""
import importlib
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import config
from tm16_decoder import TM16Decoder
from verification_gpu.baseline_decoder import TM16Decoder as ReferenceDecoder


def msop(start, empty=False, corrupt=False):
    data = bytearray(1206)
    for block in range(12):
        offset = block * 100
        struct.pack_into('<HH', data, offset, 0xEEFF, (start + block * 1000) % 36000)
        for channel in range(32):
            distance = 0 if empty else [0, 79, 80, 1000, 30000, 30001][channel % 6]
            struct.pack_into('<HB', data, offset + 4 + channel * 3, distance, channel * 7)
    if corrupt:
        data[500:502] = b'\x00\x00'
    return bytes(data)


def difop():
    data = bytearray(1206)
    data[:2] = b'\xa5\xff'
    struct.pack_into('<16f', data, 834, *[-14.75 + 2 * i for i in range(16)])
    return bytes(data)


def write_pcap(path, packets):
    with open(path, 'wb') as stream:
        stream.write(struct.pack('<IHHIIII', 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for index, (port, payload) in enumerate(packets):
            ethernet = bytes(12) + b'\x08\x00'
            ip = bytearray(20)
            ip[0], ip[9] = 0x45, 17
            udp = struct.pack('!HHHH', 1234, port, 8 + len(payload), 0)
            packet = ethernet + ip + udp + payload
            stream.write(struct.pack('<IIII', 100, index * 1000, len(packet), len(packet)))
            stream.write(packet)


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = str(Path(self.temp.name) / 'test.pcap')
        self.packets = [(2368, msop(i * 12000)) for i in range(13)]
        self.packets.insert(5, (8603, difop()))  # 在帧中途更改标定
        write_pcap(self.path, self.packets)

    def assert_frames_equal(self, actual, expected, tolerance=1e-10):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertEqual((a.frame_id, a.timestamp, a.packet_count, a.calibration_state),
                             (b.frame_id, b.timestamp, b.packet_count, b.calibration_state))
            np.testing.assert_array_equal(a.vertical_angles, b.vertical_angles)
            np.testing.assert_array_equal(a.points_radar[:, 3:], b.points_radar[:, 3:])
            np.testing.assert_allclose(a.points_radar[:, :3], b.points_radar[:, :3], atol=tolerance, rtol=0)
            np.testing.assert_allclose(a.points_site, b.points_site, atol=tolerance, rtol=0)

    def test_cpu_batch_preserves_calibration_and_frame_boundaries(self):
        decoder = TM16Decoder(backend='cpu')
        self.assert_frames_equal(decoder.decode_pcap(self.path), ReferenceDecoder().decode_pcap(self.path))
        self.assertEqual(decoder.active_backend, 'cpu')

    def test_empty_frames_corrupt_blocks_and_max_frames_match_reference(self):
        for packets in [self.packets[:4] + [(2368, msop(12000, corrupt=True))] + self.packets[4:],
                        [(2368, msop(i * 12000, empty=True)) for i in range(10)],
                        [(2368, msop(35000))]]:
            write_pcap(self.path, packets)
            for limit in [None, 1, 2]:
                with self.subTest(limit=limit, packets=len(packets)):
                    self.assert_frames_equal(TM16Decoder(backend='cpu').decode_pcap(self.path, max_frames=limit),
                                             ReferenceDecoder().decode_pcap(self.path, max_frames=limit))

    def test_missing_gpu_falls_back_without_dropping_points(self):
        with patch('pointcloud_backend.importlib.import_module', side_effect=ImportError('GPU dependency missing')):
            decoder = TM16Decoder(backend='auto')
            actual = decoder.decode_pcap(self.path)
        self.assertEqual(decoder.active_backend, 'cpu')
        self.assertIn('GPU dependency missing', decoder.fallback_reason)
        self.assert_frames_equal(actual, ReferenceDecoder().decode_pcap(self.path))

    def test_gpu_failure_recomputes_current_frame_then_stays_on_cpu(self):
        from pointcloud_backend import PointCloudBackend
        backend = PointCloudBackend('cpu')
        backend.active_backend = 'gpu'
        with patch.object(backend, '_compute_gpu', side_effect=RuntimeError('simulated CUDA out of memory')) as failing:
            with patch('tm16_decoder.PointCloudBackend', return_value=backend):
                decoder = TM16Decoder()
                actual = decoder.decode_pcap(self.path)
        self.assertEqual(failing.call_count, 1)
        self.assertEqual(decoder.active_backend, 'cpu')
        self.assertIn('out of memory', decoder.fallback_reason)
        self.assert_frames_equal(actual, ReferenceDecoder().decode_pcap(self.path))

    def test_failure_after_completed_gpu_frame_keeps_all_frames(self):
        from pointcloud_backend import PointCloudBackend
        backend = PointCloudBackend('cpu')
        backend.active_backend = 'gpu'
        calls = []
        def fail_second_batch(packed, matrix):
            calls.append(len(packed))
            if len(calls) == 2:
                raise RuntimeError('device lost during download')
            return backend._compute(packed, matrix, np)
        with patch.object(backend, '_compute_gpu', side_effect=fail_second_batch):
            with patch('tm16_decoder.PointCloudBackend', return_value=backend):
                decoder = TM16Decoder()
                actual = decoder.decode_pcap(self.path)
        self.assertEqual(len(calls), 2)
        self.assert_frames_equal(actual, ReferenceDecoder().decode_pcap(self.path))

    def test_custom_cut_extrinsic_and_disabled_calibration(self):
        matrix = np.array([[.8, .6, 0, 3], [-.6, .8, 0, -2], [0, 0, 1, 9], [0, 0, 0, 1]], dtype=np.float32)
        args = dict(cut_deg=90., extrinsic_matrix=matrix, use_calibration=False)
        self.assert_frames_equal(TM16Decoder(backend='cpu', **args).decode_pcap(self.path),
                                 ReferenceDecoder(**args).decode_pcap(self.path))

    def test_cpu_mode_does_not_import_gpu_dependency(self):
        with patch('pointcloud_backend.importlib.import_module', side_effect=AssertionError('must not import')):
            decoder = TM16Decoder(backend='cpu')
            self.assertGreater(len(decoder.decode_pcap(self.path)), 0)

    def test_gui_worker_reports_backend_and_elapsed_time(self):
        from visualizer_gui import PcapLoadWorker
        worker = PcapLoadWorker(self.path)
        received, failures = [], []
        worker.finished.connect(lambda frames, path: received.append(frames))
        worker.failed.connect(failures.append)
        with patch.dict('os.environ', {'TM16_DECODE_BACKEND': 'cpu'}):
            worker.run()
        self.assertFalse(failures)
        self.assertEqual(len(received), 1)
        self.assert_frames_equal(received[0], ReferenceDecoder().decode_pcap(self.path))
        self.assertIn('CPU', worker.decode_status)
        self.assertIn('耗时', worker.decode_status)

    def test_cancel_is_not_swallowed_as_gpu_failure(self):
        def cancel(frac, message):
            raise InterruptedError('cancelled')
        with self.assertRaises(InterruptedError):
            TM16Decoder(backend='cpu').decode_pcap(self.path, progress_callback=cancel)

    def test_real_gpu_matches_reference(self):
        if importlib.util.find_spec('cupy') is None:
            self.skipTest('未安装 CuPy')
        decoder = TM16Decoder(backend='gpu')
        self.assert_frames_equal(decoder.decode_pcap(self.path), ReferenceDecoder().decode_pcap(self.path), tolerance=5e-5)
        self.assertEqual(decoder.active_backend, 'gpu', decoder.fallback_reason)

    def test_real_gpu_preserves_double_precision_custom_extrinsics(self):
        if importlib.util.find_spec('cupy') is None:
            self.skipTest('未安装 CuPy')
        matrix = np.array([[.8123456789, .6123456789, .01, 3.123456789],
                           [-.6123456789, .8123456789, -.02, -2.987654321],
                           [.03, -.04, 1.123456789, 9.87654321], [0, 0, 0, 1]])
        options = dict(extrinsic_matrix=matrix, cut_deg=90.)
        decoder = TM16Decoder(backend='gpu', **options)
        self.assert_frames_equal(decoder.decode_pcap(self.path),
                                 ReferenceDecoder(**options).decode_pcap(self.path), tolerance=5e-5)
        self.assertEqual(decoder.active_backend, 'gpu', decoder.fallback_reason)


if __name__ == '__main__':
    unittest.main()
