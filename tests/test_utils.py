from pathlib import Path
from unittest.mock import patch

from ml_translate.utils import asMinutes, timeSince, get_project_root


class TestAsMinutes:
    def test_as_minutes_zero(self):
        """Test that 0 seconds returns '0 0'."""
        result = asMinutes(0)
        assert result == "0 0"

    def test_as_minutes_full_minutes(self):
        """Test exact minute values."""
        assert asMinutes(60) == "1 0"
        assert asMinutes(120) == "2 0"

    def test_as_minutes_with_remainder(self):
        """Test values with seconds remainder."""
        assert asMinutes(90) == "1 30"
        assert asMinutes(125) == "2 5"


class TestTimeSince:
    def test_time_since_elapsed_remaining(self):
        """Test elapsed and remaining time calculation."""
        with patch("ml_translate.utils.time.time") as mock_time:
            # Start time was 0, current time is 50, we're 50% done
            mock_time.return_value = 50
            result = timeSince(0, 0.5)
            # Elapsed: 50 seconds, estimated total: 100, remaining: 50
            assert "0 50" in result  # elapsed
            assert "(- 0 50" in result  # remaining (may have decimal)


class TestGetProjectRoot:
    def test_get_project_root_returns_valid_path(self):
        """Test that get_project_root returns a path containing pyproject.toml."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert (root / "pyproject.toml").exists()

    def test_get_project_root_returns_consistent_path(self):
        """Test that get_project_root returns the same path on multiple calls."""
        root1 = get_project_root()
        root2 = get_project_root()
        assert root1 == root2
