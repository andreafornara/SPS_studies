if [ -z "$SIMPATH" ]; then
    SIMPATH=$(pwd)
fi

python simulation.py --yaml_path "$SIMPATH/config.yaml"