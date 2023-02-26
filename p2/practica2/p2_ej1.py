#perf stat -e fp_ret_sse_avx_ops.all,cycles,instructions ./practica2_eval_seq -LINEAR -N=5000 -M=5000 -K=5000
import os

FILE = 'resultados/times1.csv'

def main():
    with open(FILE, 'w') as f:
        pass
    os.system(f'echo -n "" > {FILE}')
    
    
    
    mn = [100,128,200,256,500,512,1000,1024,1024]
    k = [100,128,200,256,500,512,500,512,1024]
    
if __name__ == "__main__":
    main()