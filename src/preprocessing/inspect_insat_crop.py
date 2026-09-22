import h5py
import numpy as np

FILE = r"data\raw\insat\2026\0314\3SIMG_14MAR2026_2330_L1B_STD_V01R00.h5"

ROW_START = 885
ROW_END = 912

COL_START = 1153
COL_END = 1180


CHANNELS = {
    "MIR": ("IMG_MIR", "IMG_MIR_TEMP"),
    "TIR1": ("IMG_TIR1", "IMG_TIR1_TEMP"),
    "TIR2": ("IMG_TIR2", "IMG_TIR2_TEMP"),
    "WV": ("IMG_WV", "IMG_WV_TEMP"),
}


with h5py.File(FILE, "r") as h:

    lat = h["Latitude"][:].astype(np.float32) * 0.01
    lon = h["Longitude"][:].astype(np.float32) * 0.01

    print("=" * 60)
    print("INSAT MUMBAI CROP")
    print("=" * 60)

    print("Crop rows :", ROW_START, "to", ROW_END - 1)
    print("Crop cols :", COL_START, "to", COL_END - 1)
    print(
        "Crop size :",
        ROW_END - ROW_START,
        "x",
        COL_END - COL_START,
    )

    crop_lat = lat[ROW_START:ROW_END, COL_START:COL_END]
    crop_lon = lon[ROW_START:ROW_END, COL_START:COL_END]

    print()
    print("Latitude range :", crop_lat.min(), "to", crop_lat.max())
    print("Longitude range:", crop_lon.min(), "to", crop_lon.max())

    print()
    print("=" * 60)
    print("BRIGHTNESS TEMPERATURE CHANNELS")
    print("=" * 60)

    for name, (image_name, lookup_name) in CHANNELS.items():

        image = h[image_name][0]

        lookup = h[lookup_name][:]

        counts = image[
            ROW_START:ROW_END,
            COL_START:COL_END
        ]

        # Convert 10-bit pixel counts to brightness temperature
        temperature = lookup[counts]

        # Handle invalid/fill pixels
        fill_value = h[image_name].attrs.get("_FillValue", [1023])[0]

        valid = counts != fill_value

        valid_temperature = temperature[valid]

        print()
        print(name)

        print("  Raw image shape :", image.shape)
        print("  Crop shape      :", counts.shape)
        print("  Raw count range :", counts.min(), "to", counts.max())

        print("  Fill value      :", fill_value)
        print("  Valid pixels    :", valid.sum(), "/", counts.size)

        if valid_temperature.size > 0:
            print(
                "  Temperature     :",
                float(valid_temperature.min()),
                "to",
                float(valid_temperature.max()),
                "K"
            )

            print(
                "  Mean temperature:",
                float(valid_temperature.mean()),
                "K"
            )

    print()
    print("=" * 60)
    print("INSAT CROP TEST COMPLETE")
    print("=" * 60)