#!/bin/bash

echo "Iniciando prueba de CPU..."
echo "##############################"
echo $(date)
echo "##############################"
sysbench cpu --cpu-max-prime=140000 run

echo "Iniciando prueba 1 de Memoria..."
echo "##############################"
echo $(date)
echo "##############################"
sysbench memory --memory-total-size=4G run

echo "Iniciando prueba 2 de Memoria..."
echo "##############################"
echo $(date)
echo "##############################"
sysbench memory --memory-total-size=4G run

echo "Preparando archivos para prueba de Operaciones de Archivo..."
echo "##############################"
echo $(date)
echo "##############################"
sysbench fileio --file-total-size=6G prepare

echo "Iniciando prueba de Operaciones de Archivo..."
echo "##############################"
echo $(date)
echo "##############################"
sysbench fileio --file-total-size=6G --file-test-mode=seqwr run

echo "Iniciando prueba de Hilos..."
echo "##############################"
echo $(date)
echo "##############################"
sysbench threads --threads=10 --thread-yields=100 --thread-locks=4 run

echo "Iniciando prueba de Mutex..."
echo "##############################"
echo $(date)
echo "##############################"
sysbench mutex --mutex-num=1M --mutex-locks=1M --mutex-loops=1K run

echo "Limpiando despues de prueba de Operaciones de Archivo..."
echo "##############################"
echo $(date)
echo "##############################"
sysbench fileio cleanup

echo "Pruebas completadas. Contenedor en espera..."
echo "##############################"
echo $(date)
echo "##############################"
tail -f /dev/null
