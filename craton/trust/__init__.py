"""The locks: what keeps a forecaster from being confidently wrong (P7).

predictive    a forecast as a full distribution: members, dispersion, abstention
calibration   reliability, ECE, randomized PIT, CRPS, and the post-hoc dispersion fit
meter         the predictability meter: how much space-time you watch to catch 90%
physics       hard sanity checks: rates, Gutenberg-Richter, Omori decay, the slip budget
attacks       fake precursors injected into the inputs, by name
guard         input hygiene: duplicates and magnitude glitches caught before a forecast
ood           "I have never seen this": nearest-neighbour distance in the trunk's space
lock          forecasts written and hashed before the day they forecast, graded after
referee       every discovery claim against fake clocks, shuffles and a locked test
"""
