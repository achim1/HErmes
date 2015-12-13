"""
Generate expressions from the patterns in PATTERNS.cfg
"""

from ConfigParser import ConfigParser
from logger import Logger
from glob import glob

import re
import os.path
import subprocess as sub
import os

PATTERNFILE = "PATTERNS.cfg"
config = ConfigParser()
config.read(PATTERNFILE)

example = "PFFilt_PhysicsFiltering_Run00127080_Subrun00000000_00000244.tar.bz2"
gcd = "Level2_IC86.2015_data_Run00127178_1129_7_213_GCD.i3.gz"


def _regex_compiler(cfgsection,cfgname,transform = lambda x : x
):
    
    def safe_return(filename):
        res = []
        cmp = re.compile(config.get(cfgsection,cfgname))
        try:
            grps = cmp.search(filename).groups()
        except AttributeError,ValueError:
            return None
        for i in grps:
            res.append(transform(i))
        if len(res) == 1:
            return res[0]
        return res
    return safe_return

# getting the stuff!
ENDING     = lambda filename : _regex_compiler("files","ENDING")(filename)
DS_ID      = lambda filename : _regex_compiler("datasets","DS_ID",transform=int)(filename)
EXP_RUN_ID = lambda filename : _regex_compiler("dataruns","EXP_RUN_ID",transform=int)(filename)
SIM_RUN_ID = lambda filename : _regex_compiler("simruns","SIM_RUN_ID",transform=int)(filename)
GCD = lambda filename : _regex_compiler("metainfo","GCD",transform = lambda x: x == "GCD")(filename)

def strip_all_endings(filename):
    """
    Split a filename at the first dot and declare
    everything which comes after it and consists of 3 or 4 
    characters (including the dot) as "ending"
    """
    ending = ENDING(filename)
    print ending
    print ending[0]
    while ENDING(ending[0]) is not None:
        ending = ENDING(ending[0]) + ending[1:]
        print ending
    return [ending[0],"".join(ending[1:])]

#############################################

def harvest_files(path,ending=".bz2",sanitizer=lambda x : x,use_ls=False,prefix="dcap://"):
    """
    Get all the files with a specific ending
    from a certain path
    :sanitizer: clean the filelist with a filter
    :use_ls: try to use unix ls to compile the filelist (if the usual approach
             failes)
    """

    if (not os.path.exists(path)) or (not os.path.isdir(path)):
        raise SystemError('Path does not exist or it might not be a directory! %s' %path)
    
    path = os.path.abspath(path)
    if use_ls:
        files = []
        ls = sub.Popen(["ls","-a",path],stdout=sub.PIPE,stdin=sub.PIPE).communicate()[0].split()
        # remove by-products
        ls = filter(lambda x : (x != ".") or (x != ".."),ls)
        for subpath in ls:
            if os.path.isdir(os.path.join(path,subpath)):
                sub_ls = sub.Popen(["ls","-a",os.path.join(path,subpath)],stdout=sub.PIPE,stdin=sub.PIPE).communicate()[0].split()
                sub_ls = filter(lambda x : (x != ".") and (x != ".."),sub_ls)
                files += [os.path.join(path,os.path.join(subpath,subsubpath)) for subsubpath in sub_ls]
            elif os.path.isfile(os.path.join(path,subpath)):
                files += [os.path.join(path,subpath)]
                 
            if "*" in ending:
                ending = ending.replace("*","")
            files = filter(lambda x: x.endswith(ending),files)
    else:
        if not ending.startswith("*"):
            ending = "*" + ending

        tmpindirs = [item[0] for item in os.walk(path,followlinks=True)]
        files = reduce(lambda x,y : x+y,map(glob,[os.path.join(direc,ending) for direc in tmpindirs]))
    files = filter(sanitizer,files)
    files = map(lambda x : prefix + x,files)
    return files

##############################################################

def group_names_by_regex(names,regex=EXP_RUN_ID,firstpattern=GCD,estimate_first=lambda x : x[0]):
    """
    Generate sublists with files which all have the same 
    name patterns
    Group by regex,
    :firstpattern: the leading element of each group
    :estimate_first: if there are several elements which match
    firstpattern, estimate which is the first with a given fct
    """
    identifiers        = map(regex,names)
    unique_identifiers = set(identifiers)
    meta_names         = zip(identifiers,names)
    meta_names         = sorted(meta_names)
    groupdict          = dict()
    for i in unique_identifiers:
        groupdict[i] = [j[1] for j in meta_names if j[0] == i]

    if firstpattern is not None:
        for k in groupdict.keys():
            first = filter(firstpattern,groupdict[k])
            if len(first) > 1: 
                Logger.info("First entry is not unique! %s" %first.__repr__())  
                for j in first:
                    groupdict[k].remove(j)
                first = estimate_first(first)
                Logger.info("Picked %s by given estimate_first fct!" %first[0])
        
            else:
                first = first[0]
                groupdict[k].remove(first)
            groupdict[k] = [first] + groupdict[k]
    
    return groupdict

##############################################################

print ENDING(example)
print EXP_RUN_ID(example)
print strip_all_endings(example)
print harvest_files(".",ending=".py",use_ls=False)
print harvest_files(".",ending=".py",use_ls=True)
print GCD(gcd)

