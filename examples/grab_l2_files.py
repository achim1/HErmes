import utils.files as f

all = f.harvest_files("/data/exp/IceCube/2015/filtered/level2/1127",prefix=None,ending=".bz2")
gcd = f.harvest_files("/data/exp/IceCube/2015/filtered/level2/",prefix=None,ending=".gz")
all_w_gcd = all + gcd

runs = f.group_names_by_regex(all_w_gcd,firstpattern=f.GCD,estimate_first= lambda x : sorted(x)[1])

#remove runs which have only gcd
runs_on_1127 = dict()
for k in runs.keys():
    if len(runs[k] > 1):
        runs_on_1127[k] = runs[k]

print runs_on_1127

