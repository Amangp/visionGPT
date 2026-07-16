"""
Thread-safe progress tracking module for managing concurrent download progress bars.
"""

import threading
from typing import Dict
from tqdm import tqdm


class DownloadProgressManager:
    """
    Manages multiple progress bars for concurrent file downloads.
    Ensures thread-safe console output and correct positioning of bars.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bars: Dict[str, tqdm] = {}
        self._positions: Dict[str, int] = {}
        self._next_position = 0

    def start_task(self, filename: str, total_bytes: int, initial_bytes: int = 0) -> None:
        """
        Starts tracking progress for a file, creating a new progress bar.
        """
        with self._lock:
            if filename in self._bars:
                # If a bar already exists, close it first
                try:
                    self._bars[filename].close()
                except Exception:
                    pass

            position = self._next_position
            self._next_position += 1
            self._positions[filename] = position

            self._bars[filename] = tqdm(
                desc=filename[:25].ljust(25),  # Limit description width for alignment
                total=total_bytes,
                initial=initial_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                position=position,
                leave=True,
                dynamic_ncols=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )

    def update_progress(self, filename: str, bytes_written: int) -> None:
        """
        Updates the progress of a specific file.
        """
        with self._lock:
            bar = self._bars.get(filename)
            if bar:
                bar.update(bytes_written)

    def finish_task(self, filename: str, success: bool = True) -> None:
        """
        Closes and cleans up the progress bar for a finished task.
        """
        with self._lock:
            bar = self._bars.get(filename)
            if bar:
                if not success:
                    bar.set_description(f"FAILED: {filename[:17]}")
                else:
                    bar.set_description(f"DONE: {filename[:19]}")
                bar.close()
                # Do not delete from dict to preserve final status display
                # but allow the position to be recycled optionally if we want

    def reset(self) -> None:
        """
        Resets the progress manager state.
        """
        with self._lock:
            for bar in self._bars.values():
                try:
                    bar.close()
                except Exception:
                    pass
            self._bars.clear()
            self._positions.clear()
            self._next_position = 0


# Global progress manager instance
progress_manager = DownloadProgressManager()
