#!/usr/bin/sh

#PBS -N cifar_test
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:30:00
#PBS -M afeeney@trinity.edu
#PBS -m abe

python ~/PoseNetwork/saiyan.py
