import tempfile
import unittest
from pathlib import Path

from scripts.validate_release_tag import read_project_version, validate_release_tag

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LINUX_ONEFILE_WORKFLOW = PROJECT_ROOT / ".github/workflows/build-linux-onefile.yml"


class ReleaseBuildTests(unittest.TestCase):
    def test_validate_release_tag_accepts_matching_semver_tag(self):
        self.assertEqual(validate_release_tag("v0.1.0", "0.1.0"), "v0.1.0")

    def test_validate_release_tag_rejects_non_matching_version(self):
        with self.assertRaisesRegex(
            ValueError,
            "must match project version",
        ):
            validate_release_tag("v0.1", "0.1.0")

    def test_read_project_version_reads_project_version_from_pyproject(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pyproject_path = Path(temp_dir) / "pyproject.toml"
            pyproject_path.write_text(
                '[project]\nname = "accio"\nversion = "1.2.3"\n',
                encoding="utf-8",
            )

            self.assertEqual(read_project_version(pyproject_path), "1.2.3")

    def test_linux_onefile_workflow_does_not_use_unbound_nuitka_version(self):
        workflow = LINUX_ONEFILE_WORKFLOW.read_text(encoding="utf-8")
        uses_version_placeholder = "{VERSION}" in workflow
        defines_nuitka_version = (
            "product-version:" in workflow or "file-version:" in workflow
        )

        self.assertFalse(
            uses_version_placeholder and not defines_nuitka_version,
            "Nuitka fails when {VERSION} is used without product-version or file-version",
        )

    def test_linux_onefile_workflow_does_not_pass_unknown_ccache_input(self):
        workflow = LINUX_ONEFILE_WORKFLOW.read_text(encoding="utf-8")

        self.assertNotIn(
            "enable-ccache:",
            workflow,
            "Nuitka/Nuitka-Action@v1.3 has no enable-ccache input",
        )


if __name__ == "__main__":
    unittest.main()
