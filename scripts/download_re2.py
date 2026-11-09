"""Compatibility entry point for the manifest-based downloader.

No arguments now download the committed sample, NOT the entire RE2 dataset.
For an explicit broad download use --pattern 're2*'.
"""
if __package__:
    from .download_data import main
else:
    from download_data import main


if __name__ == "__main__":
    main()
