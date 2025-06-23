#!/bin/bash
#                                                              s Ω   λ_min max N  n  L  β    warm upd bin chk  file            re fq do                          
~/.julia/bin/mpiexecjl -n 32 --project=./ julia Holstein_PT.jl 1 0.5 0.325 0.5 8 0.7 4 5.0 1500 2000 10 1000.0  "./sim_temper" 25 25 true
~/.julia/bin/mpiexecjl -n 32 --project=./ julia Holstein_PT.jl 1 0.5 0.325 0.5 8 0.7 4 5.0 1500 2000 10 1000.0  "./sim_non" 0 25 false