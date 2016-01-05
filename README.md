# pyevsel
Event selection tools targeted for HEP analysis

Requirements
---------------------------
`dashi`
`pandas`
`root_numpy` (if root support is needed)
`yaml`

Rationale
----------------------------

This software was developed while doing an eventselection
for the IceCube Experiment. Doing so, I found that if one is 
not eager to use root, plenty of code needed to be written
to set up a basic work environment, which allowed to
try out cuts on different variables quickly and study the effects.

One of the main issues was bookkeeping, as many files of
different categories were involved, a sort of dictionary
was needed to keep track of which variable was read out from
which file. In principle, `dashi.datasets.hub` proved to be suitable for the task and was used for the analysis in the first place.


[//]: #However, while trying to improve the analysis, especially if new machine learning methods should be tested (which can be interfaced with python), the `bundle` architecture came to its limits.





