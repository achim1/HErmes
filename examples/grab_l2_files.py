from __future__ import print_function
import pyevsel.utils.files as f

allfiles = f.harvest_files("/data/exp/IceCube/2015/filtered/level2/1127",prefix="",ending=".bz2")
gcd = f.harvest_files("/data/exp/IceCube/2015/filtered/level2/VerifiedGCD",prefix="",ending=".gz")
all_w_gcd = allfiles + gcd

runs = f.group_names_by_regex(all_w_gcd,firstpattern=f.GCD,estimate_first= lambda x : sorted(x)[1])

#remove runs which have only gcd
runs_on_1127 = dict()
for k in list(runs.keys()):
    if len(runs[k]) > 1:
        runs_on_1127[k] = runs[k]

print(runs_on_1127)

