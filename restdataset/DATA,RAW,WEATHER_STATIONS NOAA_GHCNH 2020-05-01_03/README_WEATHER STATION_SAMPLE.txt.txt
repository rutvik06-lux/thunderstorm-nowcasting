DATASET: NOAA GHCNh Weather Station Data

SOURCE:
NOAA National Centers for Environmental Information (NCEI)
Global Historical Climatology Network hourly (GHCNh)

DATA TYPE:
Ground-based weather station observations

PROTOTYPE PERIOD:
2020-05-01 00:00:00 to 2020-05-03 23:59:59

REGION:
India

STATIONS:
10 geographically distributed Indian weather stations were selected
for the prototype after checking station availability for May 2020.

STATION VARIABLES AVAILABLE:
- Air temperature
- Dew point temperature
- Relative humidity
- Wind direction
- Wind speed
- Visibility
- Sea-level pressure where available
- Present weather codes where available
- Sky/cloud information where available
- Ceiling where available
- Other station observations where available

IMPORTANT:
Variable availability is not identical across all stations.
Missing values are preserved and are not automatically treated as zero.

PURPOSE:
This dataset provides ground observations for validating and
cross-referencing the thunderstorm prototype with:
1. GPM IMERG precipitation
2. ERA5 atmospheric/reanalysis data
3. NASA ISS LIS lightning observations

TEMPORAL CHARACTERISTICS:
Station observations may be hourly or sub-hourly depending on the
station and reporting period. The data must not be assumed to have
uniform hourly sampling.

PROTOTYPE STATUS:
Downloaded: YES
Initial validation: YES
Integrated with other datasets: NOT YET

NEXT PROCESSING STEP:
Extract the observations within the common prototype period and
spatially/temporally align them with GPM, ERA5, and NASA ISS LIS.

NOTE:
The station files are retained in their original form. No source
observations should be overwritten or fabricated during processing.