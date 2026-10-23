ps -eo pid,etime,%cpu,%mem,args | grep -E 'run_registered_fold_ensemble|worker_cli|jax' | grep -v grep
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader