#!/bin/bash

POD=$1

gcho "Logs of pod $POD"

kubectl -n eidf230ns logs $POD
