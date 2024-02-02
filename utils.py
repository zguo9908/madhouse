from scipy.stats import expon
import decimal
import numpy as np

def pursuit_init(TimeArray, scale, prob):
    PSRewardTimePDF = expon.pdf(TimeArray, 0, scale)*prob
    PSRewardTimeCDF = expon.cdf(TimeArray, 0, scale)*prob
    return PSRewardTimePDF, PSRewardTimeCDF

def PrecisionOf(dt):
    d = decimal.Decimal(str(dt))
    DecimalDigit = -d.as_tuple().exponent
    return DecimalDigit


def FindIndexOfTime(TimeArray, TargetTimestamp):
    FindIndex = np.where(TimeArray == TargetTimestamp)
    IndexArray = FindIndex[0]

    index = IndexArray[0]
    return index