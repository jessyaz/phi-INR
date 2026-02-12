#!/bin/bash

# Usage:
# ./create_env.sh cpu
# ./create_env.sh gpu

ENV_PATH="./mnt/nz-env"
ENV_FILE="environment.yml"

if [ -z "$1" ]; then
    echo "Usage: ./create_env.sh [cpu|gpu]"
    exit 1
fi

MODE=$1

echo "Création de l'environnement dans $ENV_PATH"

# Vérifie si l'environnement existe déjà
if [ -d "$ENV_PATH" ]; then
    echo "L'environnement existe déjà, mise à jour..."
    conda env update --prefix $ENV_PATH -f $ENV_FILE --prune
else
    conda env create --prefix $ENV_PATH -f $ENV_FILE
fi

if [ $? -ne 0 ]; then
    echo "Erreur création/mise à jour de l'environnement"
    exit 1
fi

echo "Activation de l'environnement"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -d "$ENV_PATH" ]; then
    conda activate $ENV_PATH
else
    echo "Erreur : l'environnement $ENV_PATH n'existe pas."
    exit 1
fi

echo "Installation PyTorch version $MODE"

if [ "$MODE" = "gpu" ]; then
    conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
elif [ "$MODE" = "cpu" ]; then
    conda install -y pytorch cpuonly -c pytorch
else
    echo "Argument invalide. Usage: ./create_env.sh [cpu|gpu]"
    exit 1
fi

echo "Environnement monté"
echo "> conda activate $ENV_PATH"
