#!/usr/bin/env python
"""
Command-line interface for LVM stellar library processing.

Usage:
    lvmstars-process --input <spaxel_file> --output <output_file> [options]
    lvmstars-config --create [output_file]
"""

import argparse


def create_config_command(args):
    """Create a default configuration file."""
    from lvmstars.config import Config

    output_file = args.output or "lvmstars_config.json"
    config = Config()
    config.save(output_file)
    print(f"Configuration file created: {output_file}")


def process_command(args):
    """Process a spaxel file."""
    print("Processing spaxel data...")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")

    if args.config:
        print(f"  Config: {args.config}")

    # Note: This is a placeholder for actual processing
    # In a real implementation, you would:
    # 1. Load the spaxel data from the input file
    # 2. Apply the processing pipeline
    # 3. Save results to the output file

    print("\nNote: This is a demonstration CLI.")
    print("To implement full processing, load your LVM data format and use the lvmstars modules.")
    print("\nExample code:")
    print("  from lvmstars import StellarSpectrum, GaiaCrossmatcher, SkySubtractor, DustCorrector")
    print("  # Load your data")
    print("  # spectrum = StellarSpectrum.from_lvm_spaxel(data)")
    print("  # Process with the pipeline modules")


def info_command(args):
    """Display package information."""
    from lvmstars import __version__

    print(f"LVMstars version {__version__}")
    print("\nModules:")
    print("  - spectrum: Core spectral data structures")
    print("  - gaia: GAIA catalog cross-matching")
    print("  - subtraction: Sky/ISM residual subtraction")
    print("  - dust: Dust correction")
    print("  - parameters: Stellar parameter management")
    print("  - config: Configuration management")
    print("  - utils: Utility functions")
    print("\nFor more information, see the documentation and examples.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LVM Stellar Library Processing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument(
        "--create", action="store_true", help="Create default configuration file"
    )
    config_parser.add_argument("--output", "-o", help="Output configuration file")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process spaxel data")
    process_parser.add_argument("--input", "-i", required=True, help="Input spaxel file")
    process_parser.add_argument("--output", "-o", required=True, help="Output file")
    process_parser.add_argument("--config", "-c", help="Configuration file")
    process_parser.add_argument("--skip-gaia", action="store_true", help="Skip GAIA cross-matching")
    process_parser.add_argument("--skip-dust", action="store_true", help="Skip dust correction")

    # Info command
    subparsers.add_parser("info", help="Display package information")

    args = parser.parse_args()

    if args.command == "config":
        if args.create:
            create_config_command(args)
        else:
            parser.print_help()
    elif args.command == "process":
        process_command(args)
    elif args.command == "info":
        info_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
