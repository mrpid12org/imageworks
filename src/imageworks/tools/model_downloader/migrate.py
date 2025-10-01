"""
Registry migration utilities to fix legacy path issues.

Handles updating registry entries from old flat directory structure
to new consistent nested structure based on repository IDs.
"""

from pathlib import Path
from typing import List, Dict, Any
import shutil

from .registry import get_registry
from .config import get_config


def analyze_registry_inconsistencies() -> List[Dict[str, Any]]:
    """Analyze registry for inconsistencies with current directory logic."""
    registry = get_registry()
    config = get_config()

    inconsistencies = []

    for model in registry.get_all_models():
        expected_dir = config.get_target_directory(model.format_type, model.model_name)
        registry_dir = model.path_obj

        if str(expected_dir) != str(registry_dir):
            inconsistencies.append(
                {
                    "model_name": model.model_name,
                    "format_type": model.format_type,
                    "location": model.location,
                    "registry_path": str(registry_dir),
                    "expected_path": str(expected_dir),
                    "registry_exists": registry_dir.exists(),
                    "expected_exists": expected_dir.exists(),
                    "can_migrate": registry_dir.exists() and not expected_dir.exists(),
                }
            )

    return inconsistencies


def migrate_model_directory(
    old_path: Path, new_path: Path, dry_run: bool = False
) -> bool:
    """Migrate model files from old path to new path."""
    if not old_path.exists():
        print(f"❌ Source path does not exist: {old_path}")
        return False

    if new_path.exists():
        print(f"❌ Target path already exists: {new_path}")
        return False

    try:
        if dry_run:
            print(f"🔄 [DRY RUN] Would move: {old_path} → {new_path}")
            return True
        else:
            # Create parent directory if needed
            new_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the directory
            shutil.move(str(old_path), str(new_path))
            print(f"✅ Migrated: {old_path} → {new_path}")
            return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False


def fix_registry_inconsistencies(dry_run: bool = False) -> Dict[str, int]:
    """Fix all registry inconsistencies by migrating directories and updating paths."""
    registry = get_registry()

    inconsistencies = analyze_registry_inconsistencies()

    results = {
        "total": len(inconsistencies),
        "migrated": 0,
        "updated_registry": 0,
        "skipped": 0,
        "errors": 0,
    }

    print(f"Found {results['total']} registry inconsistencies")

    for issue in inconsistencies:
        model_name = issue["model_name"]
        format_type = issue["format_type"]
        location = issue["location"]
        old_path = Path(issue["registry_path"])
        new_path = Path(issue["expected_path"])

        print(f"\n🔧 Processing: {model_name}")
        print(f"   Old: {old_path}")
        print(f"   New: {new_path}")

        if not issue["can_migrate"]:
            reason = (
                "source missing" if not issue["registry_exists"] else "target exists"
            )
            print(f"   ⚠️  Skipped: {reason}")
            results["skipped"] += 1

            # If source is missing but target exists, update registry to point to target
            if not issue["registry_exists"] and issue["expected_exists"]:
                if dry_run:
                    print(
                        "   🔄 [DRY RUN] Would update registry to point to existing target"
                    )
                else:
                    try:
                        # Remove old registry entry
                        registry.remove_model(model_name, format_type, location)

                        # Find the model entry and update it
                        models = registry.get_all_models()
                        for model in models:
                            if (
                                model.model_name == model_name
                                and model.format_type == format_type
                                and model.location == location
                            ):

                                # Create new entry with correct path
                                registry.add_model(
                                    model_name=model.model_name,
                                    format_type=model.format_type,
                                    location=model.location,
                                    path=str(new_path),
                                    size_bytes=model.size_bytes,
                                    files=model.files,
                                    metadata=model.metadata,
                                )
                                results["updated_registry"] += 1
                                print(
                                    "   ✅ Updated registry to point to existing directory"
                                )
                                break
                    except Exception as e:
                        print(f"   ❌ Failed to update registry: {e}")
                        results["errors"] += 1
            continue

        # Migrate the directory
        if migrate_model_directory(old_path, new_path, dry_run):
            results["migrated"] += 1

            # Update registry entry
            if not dry_run:
                try:
                    # Remove old entry
                    registry.remove_model(model_name, format_type, location)

                    # Add new entry with correct path
                    # Get the original model entry for metadata
                    models = registry.get_all_models()
                    original_model = None
                    for model in models:
                        if (
                            model.model_name == model_name
                            and model.format_type == format_type
                            and model.location == location
                        ):
                            original_model = model
                            break

                    if original_model:
                        registry.add_model(
                            model_name=model_name,
                            format_type=format_type,
                            location=location,
                            path=str(new_path),
                            size_bytes=original_model.size_bytes,
                            files=original_model.files,
                            metadata=original_model.metadata,
                        )
                        results["updated_registry"] += 1
                        print("   ✅ Updated registry entry")
                    else:
                        print(
                            "   ⚠️  Could not find original model entry for registry update"
                        )

                except Exception as e:
                    print(f"   ❌ Failed to update registry: {e}")
                    results["errors"] += 1
        else:
            results["errors"] += 1

    return results


def clean_empty_directories(base_path: Path, dry_run: bool = False) -> int:
    """Remove empty directories left after migration."""
    removed = 0

    for item in base_path.rglob("*"):
        if item.is_dir() and not any(item.iterdir()):
            if dry_run:
                print(f"🔄 [DRY RUN] Would remove empty directory: {item}")
            else:
                try:
                    item.rmdir()
                    print(f"🗑️  Removed empty directory: {item}")
                    removed += 1
                except Exception as e:
                    print(f"⚠️  Could not remove {item}: {e}")

    return removed


def main():
    """CLI entry point for registry migration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix ImageWorks model registry inconsistencies"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--analyze-only", action="store_true", help="Only analyze and report issues"
    )
    parser.add_argument(
        "--clean-empty", action="store_true", help="Also remove empty directories"
    )

    args = parser.parse_args()

    if args.analyze_only:
        print("🔍 Analyzing registry inconsistencies...")
        inconsistencies = analyze_registry_inconsistencies()

        if not inconsistencies:
            print("✅ No registry inconsistencies found!")
            return

        print(f"\n📊 Found {len(inconsistencies)} inconsistencies:")
        for issue in inconsistencies:
            status = "✅ can migrate" if issue["can_migrate"] else "⚠️  needs attention"
            print(f"   {issue['model_name']}: {status}")
            print(
                f"      Registry: {issue['registry_path']} [exists: {issue['registry_exists']}]"
            )
            print(
                f"      Expected: {issue['expected_path']} [exists: {issue['expected_exists']}]"
            )
        return

    print("🔧 Fixing registry inconsistencies...")
    results = fix_registry_inconsistencies(dry_run=args.dry_run)

    print("\n📊 Migration Results:")
    print(f"   Total issues: {results['total']}")
    print(f"   Migrated: {results['migrated']}")
    print(f"   Registry updates: {results['updated_registry']}")
    print(f"   Skipped: {results['skipped']}")
    print(f"   Errors: {results['errors']}")

    if args.clean_empty:
        config = get_config()
        weights_dir = config.linux_wsl.root / "weights"
        removed = clean_empty_directories(weights_dir, dry_run=args.dry_run)
        print(f"   Empty directories removed: {removed}")

    if results["errors"] == 0:
        print("✅ Migration completed successfully!")
    else:
        print("⚠️  Migration completed with errors.")


if __name__ == "__main__":
    main()
