#!/usr/bin/env python3
"""
Check if all required packages are installed for FI-2POP generator.
Run this in your conda environment to verify setup.
"""

import sys

def check_package(package_name, import_name=None):
    """Try to import a package and report status."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name:20s} - installed")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} - MISSING")
        print(f"   Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Checking Dependencies for FI-2POP Generator")
    print("=" * 60)
    print()
    
    required_packages = [
        # Core Python packages (usually built-in)
        ("json", "json"),
        ("typing", "typing"),
        ("dataclasses", "dataclasses"),
        ("pathlib", "pathlib"),
        ("signal", "signal"),
        ("argparse", "argparse"),
        
        # External packages that need to be installed
        ("numpy", "numpy"),
        ("networkx", "networkx"),
        ("tqdm", "tqdm"),
    ]
    
    print("Core Python Packages:")
    print("-" * 60)
    core_results = []
    for pkg, imp in required_packages[:6]:
        core_results.append(check_package(pkg, imp))
    
    print()
    print("External Packages (need conda/pip install):")
    print("-" * 60)
    external_results = []
    for pkg, imp in required_packages[6:]:
        external_results.append(check_package(pkg, imp))
    
    # Check Python version
    print()
    print("Python Version:")
    print("-" * 60)
    py_version = sys.version_info
    print(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major >= 3 and py_version.minor >= 7:
        print("✅ Python version is compatible (>=3.7)")
        version_ok = True
    else:
        print("❌ Python version too old (need >=3.7)")
        version_ok = False
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_ok = all(core_results) and all(external_results) and version_ok
    
    if all_ok:
        print("✅ All dependencies are installed!")
        print("   You can run: python3 run_fi2pop.py")
    else:
        print("❌ Some dependencies are missing!")
        print()
        print("To install missing packages in conda environment:")
        print("   conda install numpy networkx tqdm")
        print()
        print("Or using pip:")
        print("   pip install numpy networkx tqdm")
    
    print()
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
