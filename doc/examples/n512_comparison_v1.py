import cf
import time
import ppfive

EXAMPLE_FILE = "/Volumes/Lawrence4TB/xjanpa.pa19910301"

def timecf(path):
    t0 = time.perf_counter()
    r = cf.read(path)
    return time.perf_counter() - t0, r

def timep5(path):
    t0 = time.perf_counter()
    with ppfive.File(path) as f:
        r = cf.read(f)
    return time.perf_counter() - t0, r

def compare(path):
    cf_time, cf_result = timecf(path)
    p5_time, p5_result = timep5(path)   
    cfdict = {f.identity(): f for f in cf_result}
    p5dict = {f.identity(): f for f in p5_result}
    print(cfdict.keys())
    print(p5dict.keys())
    if set(cfdict) != set(p5dict):
        print('CF Variable identities:', set(cfdict))
        print('P5 Variable identities:', set(p5dict))

    results = [('meta', cf_time, p5_time)]
    keys = sorted(cfdict.keys() & p5dict.keys())
    for k in keys:
        p0 = time.perf_counter()
        cfm = cfdict[k].array.max()
        p1 = time.perf_counter()
        p5m = p5dict[k].array.max()
        p2 = time.perf_counter()
        if cfm != p5m:
            raise ValueError(f"Variable {k} has different max values: CF={cfm} P5={p5m}")
        results.append((k, p1-p0,p2-p1))
    for r in results:
        print(f"{r[0]:<20} CF time: {r[1]:.3f} sec, P5 time: {r[2]:.3f} sec")

if __name__ == "__main__":
    N_TRIALS = 2
    for i in range(N_TRIALS):
        compare(EXAMPLE_FILE)
        
     

