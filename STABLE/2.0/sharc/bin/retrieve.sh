#!/bin/bash

if [ ! -f host_info ];
then
  echo 'No host_info file...'
  exit 1
fi

host=$(awk 'NR==1{print $NF}' host_info)
cwd=$(awk 'NR==2{print $NF}' host_info)

echo $host
echo $cwd

if [ "$host" == "" ] || [ "$cwd" == "" ];
then
  echo 'Not sufficient host or path info...'
  exit 1
fi

echo "Copying..."
if [ "$1" == "-lis" ];
then
  scp $USER@$host:$cwd/output.lis .
  exit 0
fi

scp  $USER@$host:$cwd/output.* .

if [ "$1" == "-res" ];
then
  scp  $USER@$host:$cwd/restart.* .
  scp  $USER@$host:$cwd/restart/* ./restart/
fi