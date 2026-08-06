# -*- coding: utf-8 -*-
"""
This script extracts the start and end times from FITS files.
"""

#!/usr/bin/env python3
import os
import glob
import numpy as np
from astropy.io import fits
from decimal import Decimal, getcontext, ROUND_DOWN

import sys
data_dir = sys.argv[1]
# data_dir = '/media/cvke/Duchen/PT2025_0030/FRB20201124A/20251008'

# Set the decimal precision
getcontext().prec = 30

def get_fits_info(fits_file):
    """
    Read information such as the start time, end time, and duration of a PSRFITS file.

    Returns:
        stt_imjd
        stt_smjd
        stt_offs
        mjd_start_dec
        tbin
        nsblk
        nsubint
        file_duration_dec
        mjd_end_dec
    """
    with fits.open(fits_file, memmap=True) as hdul:
        hdr0 = hdul[0].header
        hdr1 = hdul['SUBINT'].header

        stt_imjd = hdr0.get('STT_IMJD')
        stt_smjd = hdr0.get('STT_SMJD')
        stt_offs = hdr0.get('STT_OFFS', 0.0)

        if stt_imjd is None or stt_smjd is None:
            raise ValueError(f"{fits_file} is missing STT_IMJD or STT_SMJD.")

        mjd_start_dec = Decimal(str(stt_imjd)) + (
            Decimal(str(stt_smjd)) + Decimal(str(stt_offs))
        ) / Decimal('86400')


        tbin = hdr1.get('TBIN')

        nsblk = hdr1.get('NSBLK')

        nsubint = hdr1.get('NAXIS2')

        if tbin is None or nsblk is None or nsubint is None:
            raise ValueError(f"{fits_file} is missing one or more required keywords: TBIN, NSBLK, or NAXIS2.")

        tbin_dec = Decimal(str(tbin))
        nsblk_dec = Decimal(str(nsblk))
        nsubint_dec = Decimal(str(nsubint))

        file_duration_dec = tbin_dec * nsblk_dec * nsubint_dec
        mjd_end_dec = mjd_start_dec + file_duration_dec / Decimal('86400')

        return (
            stt_imjd,
            stt_smjd,
            stt_offs,
            mjd_start_dec,
            tbin,
            nsblk,
            nsubint,
            file_duration_dec,
            mjd_end_dec
        )


fits_files = sorted(glob.glob(os.path.join(data_dir, '*.fits')))

if not fits_files:
    print("No FITS files found.")
    raise SystemExit

records = []
quant = Decimal('0.00000000000000000001')  
for f in fits_files:
    try:
        (
            stt_imjd,
            stt_smjd,
            stt_offs,
            mjd_start_dec,
            tbin,
            nsblk,
            nsubint,
            file_duration_dec,
            mjd_end_dec
        ) = get_fits_info(f)
        mjd_start_trunc = mjd_start_dec.quantize(quant, rounding=ROUND_DOWN)
        mjd_end_trunc   = mjd_end_dec.quantize(quant, rounding=ROUND_DOWN)
        records.append((
            os.path.basename(f),
            stt_imjd,
            stt_smjd,
            stt_offs,
            str(mjd_start_trunc),
            str(Decimal(str(tbin))),
            nsblk,
            nsubint,
            str(file_duration_dec),
            str(mjd_end_trunc)
        ))

        print(f"{os.path.basename(f)}")
        print(f"  MJD_START = {mjd_start_trunc}")
        print(f"  DURATION  = {file_duration_dec} s")
        print(f"  MJD_END   = {mjd_end_trunc}")

    except Exception as e:
        print(f"Failed to read: {f} ({e})")

print("\n===== Writing to a txt file =====")

outtxt = 'fits_start_end.txt'
with open(outtxt, 'w', encoding='utf-8') as fout:
    fout.write("# filename mjd_start mjd_end\n")
    for rec in records:
        filename, stt_imjd, stt_smjd, stt_offs, mjd_start, tbin_str, nsblk, nsubint, duration_str, mjd_end = rec
        fout.write(
            f"{filename} "
            # f"{stt_imjd} "
            # f"{stt_smjd} "
            # f"{stt_offs:.16f} "
            f"{mjd_start} "
            # f"{tbin_str} "
            # f"{nsblk} "
            # f"{nsubint} "
            # f"{duration_str} "
            f"{mjd_end}\n"
        )

print(f"Saved to {outtxt}")