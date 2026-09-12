"""年単位の walk-forward 分割。"""


def walk_forward_windows(
    years,
    base_years=6,
    meta_years=3,
    calib_years=1,
    test_years=1,
):
    years = sorted(int(y) for y in years)
    windows = []
    start = 0
    while True:
        base_end = start + base_years
        meta_end = base_end + meta_years
        calib_end = meta_end + calib_years
        test_end = calib_end + test_years
        if test_end > len(years):
            break
        windows.append(
            {
                "name": f"test_{years[calib_end]}",
                "base": years[start:base_end],
                "meta": years[base_end:meta_end],
                "calib": years[meta_end:calib_end],
                "test": years[calib_end:test_end],
            }
        )
        start += 1
    return windows


def mask_years(df, years, col="year"):
    return df[col].isin(set(years))
