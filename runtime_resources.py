"""Current-process historical peak resident memory, in bytes."""
import sys


def peak_rss_bytes():
    """Return the OS high-water mark; never substitute sampled current RSS."""
    if sys.platform == 'win32':
        import psutil
        return int(psutil.Process().memory_info().peak_wset)
    if sys.platform == 'darwin' or sys.platform.startswith('linux'):
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        return int(peak) * (1 if sys.platform == 'darwin' else 1024)
    raise RuntimeError('Historical peak RSS is unsupported on ' + sys.platform)
