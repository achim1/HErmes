
.. _contents:

HErmes - *highly efficient rapid multipurpose event selection* toolset
========================================================================

**What is an event selection?**

In the context of high energy physics, event selection means the enhancement of the signal-to-noise rate by implementing filter criteria on the data.
Since the signal consists of individual "events" (like a collision of particles in a collider) selecting only events which appear to be "signal-like" as defined by certain criteria is one of the basic tasks for a typical analysis in high energy physics. Typically the number of these kinds of events is very small compared to the number of background events (which are not "interesting" to the respective analyzer).

**How can this package help with the task?**

Selecting events is easy. However what is more complicated is the bookkeeping. To illustrate this, we have to go a bit more into the details: 

First, let`s start with some definitions:

- A **variable** describes a quantity which can describe signalness, e.g. energy.

- A **cut** describes a quality criterion, which is a condition imposed on a variable, e.g. "All events with energies larger then 100TeV"

- A data **category** is given by the fact that in many cases there is more than one type of data of interest which have to be studied simultaniously. For example this can be:

 - Real data, and a simulation of the signal and background

 - Different types of signal and background simulations for different kinds of hypothesis

 - Different types of data, e.g. different years of experimental data which need to be compared.

 and so on...

* A **dataset** means in this context a compilation of categories.

With these definitions, it is now possible to talk about **bookkeeping**: it is simply the necessity to ensure that every cut which is done the same way on each category of a dataset. This software intends to perform this task as painless as possible.

**Another problem: fragmented datasources..**

Often times, the data does not reach the analyzer in a consistent way: There might be several data files for a category, or different names for a variable. This software fixes some of these issues.

**Why not just use root?**

.. _Root: https://root.cern.ch/
Root_ is certainly the most popular framework used in particle physics. The here described package does not intend to reimplement all the statistical and physics oriented features of root. The HErmes toolset allows for a quick inspection of a dataset and pre-analysis with the focus of questions like: "How well does my simulation agree with data?" or "What signal rate can I expect from a certain dataset?". If questions like that need to be accessed quickly, then this package might be helpful. For elaborated analysis tools, other software (like Root_) might be a better choice.

The *HErmes* package is especially optimized to make the step from a bunch of files to a distribution after applications of some cuts as painless as possible. 


HErmes documentation contents
=============================

.. toctree::

   HErmes


Indices and tables
==================

.. only:: builder_html

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
   * :ref:`glossary`

.. only:: not builder_html

   * :ref:`modindex`
   * :ref:`glossary`

