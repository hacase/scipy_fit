# python fit with scipy.optimize
`exefit` with function model to fit with curve_fit.
`exefit_gauss` has gauss function with/without offset, calculates initial values automatically.
'exefit' fits with kafe2 if error in x Axis is given.
# example
`fit.saveplot(fig_width=5.5)` gives nice plot from matplotlib as output with latex formatting.
`fit.revert_params()` is neccesary at end of subplot to flush internal parameters.
Parameters can be set to 'True':  
Set `flag` to see plot.  
Set `copy` to get matplotlib code for fit to copy.  
Set `retoure` to return results from kafe2.  
Set `offs` to fit with a gaussian with offset.
