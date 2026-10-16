"""Peak-memory unit tests; POSIX paths are simulated on non-POSIX hosts."""
import gc
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from runtime_resources import peak_rss_bytes


class PeakMemoryTests(unittest.TestCase):
    def test_windows_uses_peak_not_current(self):
        process = Mock()
        process.memory_info.return_value = SimpleNamespace(peak_wset=8192, rss=1024)
        with patch.object(sys, 'platform', 'win32'), patch('psutil.Process', return_value=process):
            self.assertEqual(peak_rss_bytes(), 8192)

    def test_windows_missing_peak_fails_explicitly(self):
        process = Mock()
        process.memory_info.return_value = SimpleNamespace(rss=1024)
        with patch.object(sys, 'platform', 'win32'), patch('psutil.Process', return_value=process):
            with self.assertRaises(AttributeError):
                peak_rss_bytes()

    def test_linux_kib_to_bytes(self):
        resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=Mock(return_value=SimpleNamespace(ru_maxrss=123)))
        with patch.object(sys, 'platform', 'linux'), patch.dict(sys.modules, {'resource': resource}):
            self.assertEqual(peak_rss_bytes(), 123 * 1024)
        resource.getrusage.assert_called_once_with(0)

    def test_macos_bytes(self):
        resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=Mock(return_value=SimpleNamespace(ru_maxrss=123)))
        with patch.object(sys, 'platform', 'darwin'), patch.dict(sys.modules, {'resource': resource}):
            self.assertEqual(peak_rss_bytes(), 123)

    def test_unknown_platform_rejected(self):
        with patch.object(sys, 'platform', 'unknown'):
            with self.assertRaisesRegex(RuntimeError, 'unsupported'):
                peak_rss_bytes()

    @unittest.skipUnless(sys.platform == 'win32', 'Windows-only host check')
    def test_native_windows_peak_survives_release(self):
        before = peak_rss_bytes()
        allocation = bytearray(32 * 1024 * 1024)
        high = peak_rss_bytes()
        del allocation
        gc.collect()
        after = peak_rss_bytes()
        self.assertGreaterEqual(high, before)
        self.assertGreaterEqual(after, high)
        self.assertGreater(after, 0)


if __name__ == '__main__':
    unittest.main()
