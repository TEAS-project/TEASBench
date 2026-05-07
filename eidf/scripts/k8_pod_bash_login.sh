#!/bin/bash

POD=$1

echo "logging into $POD"

kubectl -n eidf230ns exec --stdin --tty $POD -- /bin/bash



