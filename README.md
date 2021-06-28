# python fit with scipy
`exefit` with function model to fit with curve_fit.
`exefit_gauss` has gauss function with/out offset, calculates initial values automatically.
`exefit` fits with scipy.odr if error in x axis is given. Set `init` with number of variables of function model or initial guesses.  
# example
`fit.saveplot(fig_width=5.5)` gives nice plot from matplotlib as output with latex formatting.
`fit.revert_params()` is neccesary at end of subplot to flush internal parameters.
Parameters can be set to `True`:  
Set `flag` to see plot, set `fill` to disable sigma enviromnment.  
Set `copy` to get matplotlib code for fit to copy.  
Set `retoure` to return results from kafe2 or scipy.  
Set `offs` to fit with a gaussian with offset.  
Set `plot` with `ax` from plt.subplot to plot in a subplot environment, set `l` with label if needed.
