#!/bin/bash

# Ejecuta la prueba de CPU
sysbench cpu --cpu-max-prime=20000 --threads=4 run

# Ejecuta la prueba de memoria
sysbench memory run

#sysbench memory --memory-total-size=1G run

# Espera indefinidamente
tail -f /dev/null
