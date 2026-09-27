from pathlib import Path
from collections import defaultdict
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================================================================
# DATA LOCATIONS
# ================================================================

LOCATIONS = {
    "INSAT": [
        PROJECT_ROOT / "data" / "raw" / "insat",
        PROJECT_ROOT / "data" / "processed" / "insat",
    ],

    "IMERG": [
        PROJECT_ROOT / "restdataset" / "DATA,RAW,GPM_IMERG 2020-05-01_03",
        PROJECT_ROOT / "data" / "processed" / "imerg",
    ],

    "ERA5": [
        PROJECT_ROOT / "restdataset" / "DATA,RAW,ERA5 2020-05-01_03",
        PROJECT_ROOT / "data" / "processed" / "era5",
    ],

    "LIS": [
        PROJECT_ROOT / "restdataset" / "DATA,RAW,LIGHTNING NASA_ISS_LIS 2020-05-01_03",
        PROJECT_ROOT / "data" / "processed" / "lis",
    ],

    "TARGETS": [
        PROJECT_ROOT / "data" / "processed" / "targets",
    ],

    "MULTIMODAL": [
        PROJECT_ROOT / "data" / "processed" / "multimodal",
    ],
}


# ================================================================
# DATE EXTRACTION
# ================================================================

DATE_PATTERNS = [
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)"),
]


def extract_date(path):
    """
    Try to find YYYYMMDD or YYYY-MM-DD / YYYY_MM_DD
    anywhere in a file/folder path.
    """

    text = str(path)

    for pattern in DATE_PATTERNS:

        match = pattern.search(text)

        if match:

            year = match.group(1)
            month = match.group(2)
            day = match.group(3)

            return f"{year}-{month}-{day}"

    return None


# ================================================================
# FILE SCANNING
# ================================================================

def scan_location(paths):

    dates = defaultdict(
        lambda: {
            "files": 0,
            "bytes": 0,
            "examples": [],
        }
    )

    for root in paths:

        if not root.exists():

            continue

        if root.is_file():

            files = [root]

        else:

            files = [
                p
                for p in root.rglob("*")
                if p.is_file()
            ]

        for path in files:

            date = extract_date(path)

            if date is None:

                continue

            try:
                size = path.stat().st_size
            except OSError:
                size = 0

            dates[date]["files"] += 1
            dates[date]["bytes"] += size

            if len(dates[date]["examples"]) < 3:

                dates[date]["examples"].append(
                    path.name
                )

    return dates


# ================================================================
# FORMATTERS
# ================================================================

def human_size(value):

    value = float(value)

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    for unit in units:

        if value < 1024:

            return f"{value:.1f} {unit}"

        value /= 1024

    return f"{value:.1f} PB"


def status_symbol(present):

    return "YES" if present else "---"


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 90)
    print("REAL-DATA EVENT INVENTORY")
    print("=" * 90)

    print()
    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print()

    inventories = {}

    all_dates = set()

    # ------------------------------------------------------------
    # Scan every modality
    # ------------------------------------------------------------

    for modality, paths in LOCATIONS.items():

        print(
            f"Scanning {modality}..."
        )

        inventory = scan_location(
            paths
        )

        inventories[modality] = inventory

        all_dates.update(
            inventory.keys()
        )

        print(
            f"  Dates found: {len(inventory)}"
        )

    # ------------------------------------------------------------
    # Date table
    # ------------------------------------------------------------

    print()
    print("=" * 90)
    print("DATE-BY-DATE AVAILABILITY")
    print("=" * 90)

    if not all_dates:

        print()
        print(
            "No dated files were discovered."
        )

        return

    header = (
        f"{'DATE':<12}"
        f"{'INSAT':>8}"
        f"{'IMERG':>8}"
        f"{'ERA5':>8}"
        f"{'LIS':>8}"
        f"{'TARGET':>9}"
        f"{'MULTI':>8}"
    )

    print()
    print(header)
    print("-" * len(header))

    for date in sorted(all_dates):

        values = []

        for modality in [
            "INSAT",
            "IMERG",
            "ERA5",
            "LIS",
            "TARGETS",
            "MULTIMODAL",
        ]:

            present = (
                date in inventories[modality]
            )

            values.append(
                status_symbol(present)
            )

        print(
            f"{date:<12}"
            f"{values[0]:>8}"
            f"{values[1]:>8}"
            f"{values[2]:>8}"
            f"{values[3]:>8}"
            f"{values[4]:>9}"
            f"{values[5]:>8}"
        )

    # ------------------------------------------------------------
    # Detailed inventory
    # ------------------------------------------------------------

    print()
    print("=" * 90)
    print("DETAILED INVENTORY")
    print("=" * 90)

    for modality in [
        "INSAT",
        "IMERG",
        "ERA5",
        "LIS",
        "TARGETS",
        "MULTIMODAL",
    ]:

        print()
        print(
            f"[{modality}]"
        )

        inventory = inventories[
            modality
        ]

        if not inventory:

            print(
                "  No dated files found."
            )

            continue

        for date in sorted(inventory):

            info = inventory[date]

            print(
                f"  {date} | "
                f"{info['files']} files | "
                f"{human_size(info['bytes'])}"
            )

            for example in info["examples"]:

                print(
                    f"      {example}"
                )

    # ------------------------------------------------------------
    # Candidate classification
    # ------------------------------------------------------------

    print()
    print("=" * 90)
    print("EVENT CANDIDATE ANALYSIS")
    print("=" * 90)

    complete_core = []
    lis_candidates = []
    multimodal_candidates = []

    for date in sorted(all_dates):

        insat = date in inventories["INSAT"]
        imerg = date in inventories["IMERG"]
        era5 = date in inventories["ERA5"]
        lis = date in inventories["LIS"]
        target = date in inventories["TARGETS"]
        multimodal = date in inventories["MULTIMODAL"]

        if multimodal:

            multimodal_candidates.append(
                date
            )

        if (
            insat
            and imerg
            and era5
        ):

            complete_core.append(
                date
            )

        if (
            insat
            and imerg
            and era5
            and lis
        ):

            lis_candidates.append(
                date
            )

    print()
    print(
        "Core meteorological coverage "
        "(INSAT + IMERG + ERA5):"
    )

    if complete_core:

        for date in complete_core:
            print(
                f"  {date}"
            )

    else:

        print(
            "  None"
        )

    print()
    print(
        "Potential multimodal events "
        "(INSAT + IMERG + ERA5 + LIS):"
    )

    if lis_candidates:

        for date in lis_candidates:
            print(
                f"  {date}"
            )

    else:

        print(
            "  None"
        )

    print()
    print(
        "Already processed multimodal events:"
    )

    if multimodal_candidates:

        for date in multimodal_candidates:
            print(
                f"  {date}"
            )

    else:

        print(
            "  None"
        )

    # ------------------------------------------------------------
    # Recommended next acquisition
    # ------------------------------------------------------------

    print()
    print("=" * 90)
    print("NEXT-DATA ACQUISITION CANDIDATES")
    print("=" * 90)

    missing_lis = []

    for date in complete_core:

        if date not in inventories["LIS"]:

            missing_lis.append(
                date
            )

    if missing_lis:

        print()
        print(
            "Dates with INSAT + IMERG + ERA5 "
            "but without detected LIS:"
        )

        for date in missing_lis:

            print(
                f"  {date}"
            )

    else:

        print()
        print(
            "No additional core dates detected."
        )

    print()
    print("=" * 90)
    print("INVENTORY COMPLETE")
    print("=" * 90)
    print()


if __name__ == "__main__":
    main()
