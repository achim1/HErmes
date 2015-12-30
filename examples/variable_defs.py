"""
Define all variables here
"""

import numpy as n


from pyevsel.variables.variables import Variable as V
from pyevsel.variables.variables import CompoundVariable as CV
from pyevsel.variables.categories import RUN_START,RUN_STOP

from pyevsel.icecube_goodies import conversions as conv
# abbreviations
nbins = 70
ls = n.linspace
ar = lambda mi,ma,delta : n.arange(mi,ma+delta, delta)

# define all the bins
coszenbins    = ls(-1.025,1.025,nbins)
azimuthbins   = ls(0,n.pi*2,nbins)
vertexbins    = ls(-750,750,nbins)
timingbins    = ls(7.5,12.5,nbins)
r_timingbins  = ls(6,15,nbins)
dt_timebins   = ls(-500,300,nbins)
energybins    = ls(0.,9,nbins)
scalebins     = ls(0,1.5,nbins)
nchanbins     = ls(0,500,nbins)
totchargebins = ls(0,6.,nbins)
qmaxdombins   = ls(-2,6,nbins)
poletoibins   = ls(0,.35,nbins)
fillratiobins = ls(0.0,1.1,nbins)
magnetbins    = ls(-0.1,1.0,nbins)
dr_bins       = ls(0,250,nbins)
dz_bins       = ls(-250,250,nbins)
dt_bins       = ls(-1500,1500,nbins)
ratiobins     = ls(0,4,nbins)

nstringbins   = n.arange(1,90)
# totchargebins = ls(0,6.,61)
# qmaxdombins   = n.linspace(-2,6,101)
# poletoibins   = ls(0,.35,101)
# fillratiobins = n.linspace(0.45,1.1,101)
# magnetbins    = ls(-0.1,1.0,nbins)
containedbins = n.linspace(0,1.5,10)
# dr_bins       = ls(0,300,nbins)
# dz_bins       = ls(-250,250,nbins)
# dt_bins       = ls(-1000,2000,nbins)


# transformations
identity     = lambda i : i
micsec       = lambda ns : ns/1000.
pdist        = lambda p1, p2 : ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2).map(n.sqrt)


# mc primary
mc_p_en = V("mc_p_en",definitions=[("MCPrimary","energy"),("mostEnergeticPrimary","energy")])
mc_p_ty = V("mc_p_ty",definitions=[("MCPrimary","type"),("mostEnergeticPrimary","type")],transform=conv.ConvertPrimaryToPDG)
mc_p_ze = V("mc_p_zen",definitions=[("MCPrimary","zenith"),("mostEnergeticPrimary","zenith")],transform=conv.ConvertPrimaryToPDG)


# MC refvis cascade
mc_refvis_logE = V("mc_refvis_logE",bins=energybins,definitions=[("RefCscdVisEn_new","value")],label=r"$\log_{10}(E_{ref,vis}$ / GeV)",transform=n.log10)
mc_refvis_x    = V("mc_refvis_x",bins=vertexbins,definitions=[("mostEnergeticCascade","x")],label=r"$x_{ref}$ [m]")
mc_refvis_y    = V("mc_refvis_y",bins=vertexbins,definitions=[("mostEnergeticCascade","y")],label=r"$y_{ref}$ [m]")
mc_refvis_z    = V("mc_refvis_z",bins=vertexbins,definitions=[("mostEnergeticCascade","z")],label=r"$z_{ref} [m]$")
mc_refvis_t    = V("mc_refvis_t",bins=timingbins,definitions=[("mostEnergeticCascade","time")],label=r"$t_{ref}$ [ns]",transform=lambda x : x/1000.)


# recovars
l3credo_energy = V("l3credo_energy",bins=energybins,transform=n.log10,label=r"$\log(E_{rec}/$GeV$)$",definitions=[("CredoFit","energy")])
monopod_energy = V("monopod_energy",bins=energybins,transform=n.log10,label=r"$\log(E_{rec}/$GeV$)$",definitions=[("Monopod4","energy")])
monopod_x      = V("monopod_x",bins=vertexbins,label=r"$x$ [m]",definitions=[("Monopod4","x")])
monopod_y      = V("monopod_y",bins=vertexbins,label=r"$y$ [m]",definitions=[("Monopod4","y")])
monopod_z      = V("monopod_z",bins=vertexbins,label=r"$z$ [m]",definitions=[("Monopod4","z")])
monopod_zenith = V("monopod_zenith",bins=coszenbins,transform=n.cos,label=r"$\cos(\Theta)$",definitions=[("Monopod4","zenith")])
monopod_azimuth= V("monopod_azimuth",bins=azimuthbins,label=r"$\Phi$",definitions=[("Monopod4","azimuth")])

monopod_en_res = CV("monopod_en_res",variables=[monopod_energy,mc_refvis_logE],bins=ls(-.4,.4,nbins),label=r"$\log(E_{rec}/E_{ref}$",operation=lambda x,y : x/y)

scale_xy       = V("scale_xy",bins=scalebins,definitions=[("I3XYScale2","value")],label="scalefactor (x surface area)")
dt_nearly_ice  = V("dt_nearly_ice",bins = dt_timebins,definitions=[("DT_Nearly_ice2","value")],label="ns")
nchan          = V("nchan",bins = nchanbins,definitions=[("EventInfofromRecoPulses","nchan")],label=r"N_{DOM}")
qtot           = V("tot_charge",bins=totchargebins,definitions=[("EventInfofromRecoPulses","tot_charge")], label= r"$\log_{10}$(NPE)",transform=n.log10)
toposplit_cnt  = V("toposplit_count",bins = ls(-0.5,4.5,6),definitions=[("TopologicalSplitSplitCount","value")],label=r"counts")
toi_evalr      = V("PoleToI_evalratio",bins = poletoibins,definitions=[("PoleToIParams","evalratio"),("CascadeFilt_ToiVal","value")],label =r"$E_{min}/(E_1+E_2+E_3)$")
spe_zen        = V("spefit4",bins = coszenbins,definitions=[("SPEFit4","zenith"),("SPEFit2","zenith")],label=r"$\cos(\theta)$",transform=n.cos)
fillratio      = V("frM4_fillratio_from_rms",bins= fillratiobins,definitions=[("FillRatioMonopod4","fillratio_from_rms")], label=r"$DOM_{hit}/DOM_{nohit}$")
magnet         = V("magnet",bins=magnetbins,definitions=[("TWSRTDipoleFitParams","magnet"),("CascadeDipoleFitParams","magnet")],label=r"magnet x")
timesplit_dr   = V("timesplit_dr",bins=dr_bins,definitions=[("timesplit_dr2","value")],label= r"$abs(\vec{x}_{cscdllh}^{1} - \vec{x}_{cscdllh}^{2})$ [m]")
timesplti_dt   = V("timesplit_dt",bins = dt_bins,definitions=[("timesplit_dt2","value")],label=r"$t_{cscdllh}^{1} - t_{cscdllh}^{2}$ [ns]")
coronasplit_dt = V("coronasplit_dt",bins = dt_bins,definitions=[("coronasplit_dt2","value")],label=r"$t_{cscdllh}^{corona} - t_{cscdllh}^{core}$ [ns]")
acer_energy    = V("acer_logE",bins=energybins,definitions=[("AtmCscdEnergyReco","energy")],label=r"$\log_{10}(E_{acer}/$GeV$)$",transform=n.log10)

# timing information from header
endtime_day    = V("endtime_day",definitions=[("I3EventHeader","time_end_mjd_day")],transform= lambda x: 24*3600.*x)
endtime_sec    = V("endtime_sec",definitions=[("I3EventHeader","time_end_mjd_sec")]) 
endtime_ns     = V("endtime_ns",definitions=[("I3EventHeader","time_end_mjd_ns")],transform=lambda x : x*1e-9)
starttime_day  = V("starttime_day",definitions=[("I3EventHeader","time_start_mjd_day")],transform= lambda x: 24*3600.*x)
starttime_sec  = V("starttime_sec",definitions=[("I3EventHeader","time_start_mjd_sec")]) 
starttime_ns   = V("starttime_ns",definitions=[("I3EventHeader","time_start_mjd_ns")],transform=lambda x : x*1e-9)
runstart       = CV(RUN_START,variables=[starttime_day,starttime_sec,starttime_ns],operation=lambda x,y : x + y  )
runsend        = CV(RUN_STOP,variables=[endtime_day,endtime_sec,endtime_ns],operation=lambda x,y : x + y  )


