# Deploy File Server

## Prepare directory

Prepare the directory where the server will be hosting files from:

`sudo mkdir /datasets`

`cd /datasets`

All data must be placed under the datasets directory to be hosted.

## Download data

From the datasets directory execute the load-\<dataset\>.sh script

`chmod +x ./<fedless_directory>/scripts/datasets_processing_scripts/load-<dataset>.sh`

`./<fedless_directory>/scripts/datasets_processing_scripts/load-<dataset>.sh`


## Host the dataset files

`kubectl apply -f <fedless_directory>/scripts/deployment_scripts/kubernetes/data-file-server`

