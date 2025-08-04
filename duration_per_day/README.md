
I tried the following methods

  - review_medians = the current implementation
  -  review_means = this pr
  -  *_multiplier = the above but then take the average error and then multiply the durations by that in case it was a consistent factor higher or lower than the actual value
  -  review_trend = the median values then multiplied by a factor determined by how many reviews were on that day (fatigue kind of stuff)
  -  true_retention_trend = the median values multiplied by a factor determined by that days true retention (Trying a quick way to imitate your retention idea for CMRR)
  -  *_seigel the above 2 but using a different function to find the fit for the multiplier

