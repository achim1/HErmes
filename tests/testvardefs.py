import HErmes.selection.variables as v
import numpy as np

# define some test variables
# run and event id
V = v.Variable
CV = v.CompoundVariable
run   = V("RUN",definitions=[("header","runid")])
event = V("EVENT",definitions=[("header","eventid")])

runstart       = V("START", definitions=[("header", "start")])
runsend        = V("STOP",  definitions=[("header", "stop")])


# mc primary
# the variable names are magic!
mc_p_en     = V("mc_p_en", definitions=[("MCPrimary","energy"),("mostEnergeticPrimary","energy")])
mc_p_ty     = V("mc_p_ty", definitions=[("MCPrimary","type"),("mostEnergeticPrimary","type"    )])
mc_p_ze     = V("mc_p_ze", definitions=[("MCPrimary","zenith"),("mostEnergeticPrimary","zenith")], transform=np.cos)
mc_p_we     = V("mc_p_we", definitions=[("I3MCWeightDict","TotalInteractionProbabilityWeight"),("CorsikaWeightMap","DiplopiaWeight")])
mc_p_gw     = V('mc_p_gw', definitions=[('CorsikaWeightMap','Weight')])
mc_p_ts     = V('mc_p_ts', definitions=[('CorsikaWeightMap','TimeScale')])

generation_weights = CV("generation_weights", variables=[mc_p_en],operation= lambda w: np.ones(len(w)), role=CV.ROLES.GENERATORWEIGHT)

# more MC related things
mc_nevents  = V("mc_nevents", definitions=[("I3MCWeightDict","NEvents")])

energy = v.Variable("energy",bins=np.linspace(0,10,20),\
                     label=r"$\log(E_{rec}/$GeV$)$",\
                     definitions=[("readout","energy")])

