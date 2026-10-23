ps -eo pid,etime,%cpu,%mem,args | grep -E 'routeb_prior_gradient_smoke|python' | grep -v grep
