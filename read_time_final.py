import sys
import os
from statistics import mean, stdev

prec = 2

# read time or prec?
def read_out_analysis(time, prefix, os_path):
    # prefix = "After CtS    :"
    # prefix = "After Sine   :"
    # prefix = "After StC    : "
    # prefix = "Eval: Relu Done in"

    # prefix = "Total done in "
    # prefix = "Done in "
    # prefix = "Conv (with BN) Done in" 
    # prefix = "(until CtoS):"
    # prefix = "Eval: Eval: ReLU Done in"
    # prefix = "Boot (StoC) Done in "
    # prefix = "AVG Prec : ("

    list_results = []
    if os.path.exists(os_path):
        with open(os_path, 'r') as read_obj:
            for line in read_obj:
                if "iter" in line:
                    new_iter = True
                    count = 0           ## count the number of appearence in each iter
                if prefix in line:
                    count += 1
                    if prefix == "(until CtoS):":
                        time_str = next(read_obj,'').strip("Done in")
                    else:
                        # time_str = line.strip("Boot out: ").strip(prefix)
                        time_str = line.strip(prefix)
                    
                    if new_iter:
                        if time:
                            list_results.append(get_seconds(time_str))
                        else:
                            prec_str, _ = time_str.split(',')
                            list_results.append(float(prec_str))
                        new_iter = False
                    else:
                        if time:
                            list_results[-1] += get_seconds(time_str)
                        else:
                            prec_str, _ = time_str.split(',')
                            list_results[-1] += float(prec_str)
    else:
        print("No file exists")
        exit(1)
    
    return(count, list_results)


def get_seconds(time_str):
    try:
        ms, _ = time_str.split('ms')
        try:
            s, ms = ms.split('s')
            try: 
                m, s = s.split('m')
            except:
                m = 0
        except:
            s = 0
            m = 0
    except:
        ms = 0
        try:
            m, s = time_str.split('m')
            try:
                s, _ = s.split('s')
            except:
                s = 0
        except:
            m = 0
            s, _ = time_str.split('s')
            
        
    return float(m)*60 + float(s) + float(ms)*0.001


## read output timing or prec ##


crop = True
ker = int(sys.argv[1])
depth = int(sys.argv[2])
wide = int(sys.argv[3])
suffix = sys.argv[4]
measure = "time"

if measure == "time":
    prefix_choices = ["Total done in "]
#"Conv (with BN) Done in",
#"(until CtoS):",
#"Eval: Eval: ReLU Done in",
#"Boot (StoC) Done in ",
#"Plaintext (kernel) preparation, Done in"]
# prefix = "Final (reduce_mean & FC):", "for odd stride, offset time"
## for conv test
#"Evaluation total done in "
#"Conv (with BN) Done in "
elif measure == "prec":
    prefix_choices = ["AVG Prec : ("]
else:
    print("wrong measure type: time or prec")
    exit(1)

os_path = 'out_kdw_'+str(ker)+str(depth)+str(wide)+'_'+suffix+'.txt'
print(os_path)

for prefix in prefix_choices:
    result_count, result_list = read_out_analysis(False, prefix, os_path)

    if (len(result_list)==1):
        print(prefix, result_count, "each", " total iters: ", len(result_list), "\n")
    else:
        print(prefix, result_count, "each", " total iters: ", len(result_list), "mean: ", round(mean(result_list), prec), "std: ", round(stdev(result_list), prec), "min/max: ", min(result_list), "/", max(result_list), "\n")
    #print(result_list)
    #for res in result_list:
        #print(round(res, prec), end=', ')
    print("\n")
    
