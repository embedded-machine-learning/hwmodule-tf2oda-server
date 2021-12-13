#!/bin/bash

echo "WARNING: Start script with . /[PATH_TO_SCRIPT], not with ./"
echo "WARNING: Set variable ENVROOT to your root for environment and TF2ODA models."


echo "Activate TF2 Object Detection API Python Environment"

PROJECTROOT=`pwd`
ENVROOT=../..

source $ENVROOT/venv/tf24_py36/bin/activate
cd $ENVROOT/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo New python path $PYTHONPATH

cd $PROJECTROOT

echo "Activation complete"
