#!/bin/bash

if test $# -ne 1
then
    echo "Usage: $0 [install|clean]"
fi

if test "$1" == "install"
then
    echo "System ready to be installed"
    if ! grep -q avx /proc/cpuinfo
    then
        echo "AVX is not supported in this CPU."
        exit 1
    fi

    echo "AVX support: Yes"

    echo "Compiling PRACTICA2 User interface: test, eval_seq and eval_opt"
    g++ -mavx src/main_test.cpp src/practica2.cpp -o practica2_test
    g++ -mavx src/main_seq.cpp src/practica2.cpp -o practica2_eval_seq
    g++ -mavx src/main_opt.cpp src/practica2.cpp -o practica2_eval_opt
    if test $? -eq 1
    then
        echo "Please fix the code!"
	exit 1
    fi
    CFLAGS="-mavx" python setup.py install
    exit 0

elif test "$1" == "clean"
then
   rm -rf build/ dist/ torch_practica2.egg-info/ practica2_test practica2_eval_seq practica2_eval_opt
fi


