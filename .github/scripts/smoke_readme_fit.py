import argparse

import pytensor

from ratingcurve import data
from ratingcurve.ratings import PowerLawRating

parser = argparse.ArgumentParser()
parser.add_argument('--expect-no-cxx', action='store_true')
args = parser.parse_args()

if args.expect_no_cxx:
    assert pytensor.config.cxx == '', pytensor.config.cxx

df = data.load('green channel')
rating = PowerLawRating(segments=2)
rating.fit(
    q=df['q'],
    h=df['stage'],
    q_sigma=df['q_sigma'],
    n=10,
    draws=10,
    progressbar=False,
    random_seed=0,
)
