.. _estimation:

**********
Estimation
**********

The directory *src.estimation* contains all files that are used for estimating the Model.


The Likelihoodfunction
============================================

.. automodule:: src.estimation.likelihoodfct
    :members:

Parameters
----------

* beta: present bias parameter
* betahat: perceived present bias parameter
* delta: usual time-discounting parameter
* gamma, phi: parameters controlling the cost of effort function
* alpha: projection bias parameter
* sigma: standard deviation of the normal error term ϵ

Arguments
---------

* netdistance: Difference between the payment date T and the work time t
* wage: Amount paid per task in a certain session
* today: dummy variable equal to one if the decision involves the choice of work today
* prediction: dummy variable equal to one if the decision involves the choice of work in the future
* pb: dummy equal to one if the subject completed 10 mandatory tasks on subject-day
* effort: number of tasks completed by a subject in a session. It can range from a minimum of 10 to a maximum of 110
* ind_effort10: dummy equal to one if the subject's effort was equal to 10
* ind_effort110: dummy equal to one if the subject's effort was equal to 110

Estimation of table 1 and 2
===========================

.. automodule:: src.estimation.estimation
    :members:

Estimating the individual likelihood
====================================

.. automodule:: src.estimation.task_estimateindividual
    :members:

Functions that prepare arguments for table 1 and 2
==================================================

.. automodule:: src.estimation.auxiliaryfct
    :members:


