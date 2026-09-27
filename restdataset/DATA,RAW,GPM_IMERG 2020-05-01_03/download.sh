#!/bin/bash

GREP_OPTIONS=''

cookiejar=$(mktemp cookies.XXXXXXXXXX)
netrc=$(mktemp netrc.XXXXXXXXXX)
chmod 0600 "$cookiejar" "$netrc"
function finish {
  rm -rf "$cookiejar" "$netrc"
}

trap finish EXIT
WGETRC="$wgetrc"

prompt_credentials() {
    echo "Enter your Earthdata Login or other provider supplied credentials"
    read -p "Username (daiku): " username
    username=${username:-daiku}
    read -s -p "Password: " password
    echo "machine urs.earthdata.nasa.gov login $username password $password" >> $netrc
    echo
}

exit_with_error() {
    echo
    echo "Unable to Retrieve Data"
    echo
    echo $1
    echo
    echo "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S233000-E235959.1410.V07B.HDF5"
    echo
    exit 1
}

prompt_credentials
  detect_app_approval() {
    approved=`curl -s -b "$cookiejar" -c "$cookiejar" -L --max-redirs 5 --netrc-file "$netrc" https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S233000-E235959.1410.V07B.HDF5 -w '\n%{http_code}' | tail  -1`
    if [ "$approved" -ne "200" ] && [ "$approved" -ne "301" ] && [ "$approved" -ne "302" ]; then
        # User didn't approve the app. Direct users to approve the app in URS
        exit_with_error "Please ensure that you have authorized the remote application by visiting the link below "
    fi
}

setup_auth_curl() {
    # Firstly, check if it require URS authentication
    status=$(curl -s -z "$(date)" -w '\n%{http_code}' https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S233000-E235959.1410.V07B.HDF5 | tail -1)
    if [[ "$status" -ne "200" && "$status" -ne "304" ]]; then
        # URS authentication is required. Now further check if the application/remote service is approved.
        detect_app_approval
    fi
}

setup_auth_wget() {
    # The safest way to auth via curl is netrc. Note: there's no checking or feedback
    # if login is unsuccessful
    touch ~/.netrc
    chmod 0600 ~/.netrc
    credentials=$(grep 'machine urs.earthdata.nasa.gov' ~/.netrc)
    if [ -z "$credentials" ]; then
        cat "$netrc" >> ~/.netrc
    fi
}

fetch_urls() {
  if command -v curl >/dev/null 2>&1; then
      setup_auth_curl
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        curl -f -b "$cookiejar" -c "$cookiejar" -L --netrc-file "$netrc" -g -o $stripped_query_params -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  elif command -v wget >/dev/null 2>&1; then
      # We can't use wget to poke provider server to get info whether or not URS was integrated without download at least one of the files.
      echo
      echo "WARNING: Can't find curl, use wget instead."
      echo "WARNING: Script may not correctly identify Earthdata Login integrations."
      echo
      setup_auth_wget
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        wget --load-cookies "$cookiejar" --save-cookies "$cookiejar" --output-document $stripped_query_params --keep-session-cookies -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  else
      exit_with_error "Error: Could not find a command-line downloader.  Please install curl or wget"
  fi
}

fetch_urls <<'EDSCEOF'
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S233000-E235959.1410.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S230000-E232959.1380.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S223000-E225959.1350.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S220000-E222959.1320.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S213000-E215959.1290.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S210000-E212959.1260.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S203000-E205959.1230.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S200000-E202959.1200.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S193000-E195959.1170.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S190000-E192959.1140.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S183000-E185959.1110.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S180000-E182959.1080.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S173000-E175959.1050.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S170000-E172959.1020.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S163000-E165959.0990.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S160000-E162959.0960.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S153000-E155959.0930.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S150000-E152959.0900.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S143000-E145959.0870.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S140000-E142959.0840.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S133000-E135959.0810.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S130000-E132959.0780.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S123000-E125959.0750.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S120000-E122959.0720.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S113000-E115959.0690.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S110000-E112959.0660.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S103000-E105959.0630.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S100000-E102959.0600.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S093000-E095959.0570.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S090000-E092959.0540.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S083000-E085959.0510.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S080000-E082959.0480.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S073000-E075959.0450.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S070000-E072959.0420.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S063000-E065959.0390.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S060000-E062959.0360.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S053000-E055959.0330.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S050000-E052959.0300.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S043000-E045959.0270.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S040000-E042959.0240.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S033000-E035959.0210.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S030000-E032959.0180.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S023000-E025959.0150.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S020000-E022959.0120.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S013000-E015959.0090.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S010000-E012959.0060.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S003000-E005959.0030.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/124/3B-HHR.MS.MRG.3IMERG.20200503-S000000-E002959.0000.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S233000-E235959.1410.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S230000-E232959.1380.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S223000-E225959.1350.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S220000-E222959.1320.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S213000-E215959.1290.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S210000-E212959.1260.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S203000-E205959.1230.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S200000-E202959.1200.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S193000-E195959.1170.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S190000-E192959.1140.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S183000-E185959.1110.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S180000-E182959.1080.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S173000-E175959.1050.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S170000-E172959.1020.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S163000-E165959.0990.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S160000-E162959.0960.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S153000-E155959.0930.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S150000-E152959.0900.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S143000-E145959.0870.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S140000-E142959.0840.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S133000-E135959.0810.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S130000-E132959.0780.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S123000-E125959.0750.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S120000-E122959.0720.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S113000-E115959.0690.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S110000-E112959.0660.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S103000-E105959.0630.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S100000-E102959.0600.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S093000-E095959.0570.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S090000-E092959.0540.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S083000-E085959.0510.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S080000-E082959.0480.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S073000-E075959.0450.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S070000-E072959.0420.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S063000-E065959.0390.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S060000-E062959.0360.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S053000-E055959.0330.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S050000-E052959.0300.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S043000-E045959.0270.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S040000-E042959.0240.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S033000-E035959.0210.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S030000-E032959.0180.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S023000-E025959.0150.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S020000-E022959.0120.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S013000-E015959.0090.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S010000-E012959.0060.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S003000-E005959.0030.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/123/3B-HHR.MS.MRG.3IMERG.20200502-S000000-E002959.0000.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S233000-E235959.1410.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S230000-E232959.1380.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S223000-E225959.1350.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S220000-E222959.1320.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S213000-E215959.1290.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S210000-E212959.1260.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S203000-E205959.1230.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S200000-E202959.1200.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S193000-E195959.1170.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S190000-E192959.1140.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S183000-E185959.1110.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S180000-E182959.1080.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S173000-E175959.1050.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S170000-E172959.1020.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S163000-E165959.0990.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S160000-E162959.0960.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S153000-E155959.0930.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S150000-E152959.0900.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S143000-E145959.0870.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S140000-E142959.0840.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S133000-E135959.0810.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S130000-E132959.0780.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S123000-E125959.0750.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S120000-E122959.0720.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S113000-E115959.0690.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S110000-E112959.0660.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S103000-E105959.0630.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S100000-E102959.0600.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S093000-E095959.0570.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S090000-E092959.0540.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S083000-E085959.0510.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S080000-E082959.0480.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S073000-E075959.0450.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S070000-E072959.0420.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S063000-E065959.0390.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S060000-E062959.0360.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S053000-E055959.0330.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S050000-E052959.0300.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S043000-E045959.0270.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S040000-E042959.0240.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S033000-E035959.0210.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S030000-E032959.0180.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S023000-E025959.0150.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S020000-E022959.0120.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S013000-E015959.0090.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S010000-E012959.0060.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S003000-E005959.0030.V07B.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2020/122/3B-HHR.MS.MRG.3IMERG.20200501-S000000-E002959.0000.V07B.HDF5
EDSCEOF