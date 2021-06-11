import matplotlib
import numpy as np

# the code currently (May 2020) gives a lot of matplotlib deprecation warnings
# which are probably cleaned up with matplotlibs version 3.3.
# until then we just turn off these warnings:
import warnings
warnings.filterwarnings("ignore")

# store original plot parameters so that we can revert:
ORIG_MATPLOTLIB_CONF = dict(matplotlib.rcParams)

def saveplot(fig_width=None, fig_height=None, columns=1, fontsize=8):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this function before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}; 1 is default
    fontsize : 8 pt default
    """

    assert(columns in [1,2])

    if fig_width is None:
        # figure widths for one-column and two-column printing
        # (taken from https://www.elsevier.com/authors/author-schemas/artwork-and-media-instructions/artwork-sizing)
        if columns == 2:
            fig_width = 3.5
        else:
            fig_width = 7.0

    if fig_height is None:
        golden_mean = (np.sqrt(5.0) - 1.0) / 2.0    # Aesthetic ratio
        fig_height = fig_width * golden_mean # height in inches

    params = {'backend': 'ps',
              'text.latex.preamble':
              [ r'\usepackage{siunitx}',
                r'\usepackage[utf8]{inputenc}',
                r'\usepackage[T1]{fontenc}',
                r'\DeclareSIUnit \jansky {Jy}' ],
              'axes.labelsize' : fontsize,
              'axes.titlesize' : fontsize,
              'font.size': fontsize,
              'legend.fontsize' : fontsize,
              'xtick.labelsize' : fontsize,
              'ytick.labelsize' : fontsize,
              'axes.linewidth' : 1,
              'lines.linewidth' : 1,
              'text.usetex' : True,
              'figure.figsize' : [fig_width, fig_height],
              'font.family' : 'serif',
              'savefig.bbox' : 'tight',
              'savefig.dpi' : 300  # set to 600 for poster printing or PR
                                  # figures
    }

    matplotlib.rcParams.update(params)

def revert_params():
    """
    reverts any changes done to matplotlib parameters and restores
    the state before homogenize_plot was called
    """

    matplotlib.rcParams.update(ORIG_MATPLOTLIB_CONF)
    
    
# execute fit with kafe2
# set retoure to 2 to return results
from kafe2 import XYContainer, Fit, Plot, ContoursProfiler

def exefit(model_function, x_data, y_data, xerr=False, yerr=False, retoure=1):
    
    xy_data = XYContainer(x_data=x_data, y_data=y_data)
    if xerr is not False:
        xy_data.add_error(axis='x', err_val= xerr)
    if yerr is not False:
        xy_data.add_error(axis='y', err_val= yerr)
        
    xy_fit = Fit(data=xy_data, model_function=model_function) 
    results = xy_fit.do_fit()
    if results['did_fit']:
        xy_plot = Plot([xy_fit])
        xy_plot.plot()
        if retoure == 1:
            print('#===== Results of Fit =====#')
    
            if results['did_fit']:

                length = len(results['parameter_values'])

                print('\n=== Parameter Values ===')
                val = np.zeros(length)
                err = np.zeros(length)
                i=0
                for param in results['parameter_values']:
                    val[i] = results['parameter_values'][param]
                    err[i] = results['parameter_errors'][param]

                    # print every parameter with error
                    print(f'{param} = {val[i]:.3e} +/- {err[i]:.3e}')
                    i+=1
                
                return xy_fit
        elif retoure == 2:
            return results
    else:
        print('Fit failed')
        
        
# set flag to 1 to print certain information
# set cor, cov to 2, if given variables are longer than one symbole
def giveres(results, cor=None, cov=None, cost=None):
    print('#===== Results of Fit =====#')
    
    if results['did_fit']:
        
        length = len(results['parameter_values'])
        
        print('\n=== Parameter Values ===')
        val = np.zeros(length)
        err = np.zeros(length)
        i=0
        for param in results['parameter_values']:
            val[i] = results['parameter_values'][param]
            err[i] = results['parameter_errors'][param]
            
            # print every parameter with error
            print(f'{param} = {val[i]:.3e} +/- {err[i]:.3e}')
            i+=1
            
        if length == 2:
            # only if only two variables x and y are given
            # does only makes sense for y=f(x)
            print('\n\n=== Parameter Correlation Coefficient ===')
            x = [val[0]+err[0],val[0]-err[0]]
            y = [val[1]+err[1],val[1]-err[1]]
            
            # covariance
            cov = np.mean([x[0]*y[0], x[1]*y[1]])-np.mean(x)*np.mean(y)
            # sigma_x times sigma_y
            sigmaxy = np.sqrt(np.mean([x[0]**2, x[1]**2])-np.mean(x)**2) * np.sqrt(np.mean([y[0]**2, y[1]**2])-np.mean(y)**2)
            
            print(f'rho = {cov/sigmaxy:.3f}')
            
        
        if cor == 1:
            print('\n\n=== Parameter Correlation Matrix ===')
            # formatting dynamical matrix dpending of variable number
            # first line
            string = ' '*2
            for param in results['parameter_values']:
                string = string + ' '*6 + param + ' '*6
            print(string)
            # second line
            string = ' +'
            for param in np.arange(length):
                string = string + '-'*12
            string = string + '-'
            print(string)
            
            i = 0
            for param in results['parameter_values']:
                print(param,'|',results['parameter_cor_mat'][i],sep='')
                i+=0
        elif cor == 2:
            print('\n\n=== Parameter Correlation Matrix ===')
            print(results['parameter_cor_mat'])
            
        
        if cov == 1:
            print('\n\n=== Parameter Covariance Matrix ===')
            # formatting dynamical matrix dpending of variable number
            # first line
            string = ' '*2
            for param in results['parameter_values']:
                string = string + ' '*6 + param + ' '*6
            print(string)
            # second line
            string = ' +'
            for param in np.arange(length):
                string = string + '-'*12
            string = string + '-'
            print(string)
            
            i = 0
            for param in results['parameter_values']:
                print(param,'|',results['parameter_cov_mat'][i],sep='')
                i+=0
        elif cov == 2:
            print('\n\n=== Parameter Covariance Matrix ===')
            print(results['parameter_cov_mat'])
        

        if cost == 1:
            print('\n\n=== Cost Function ===')
            cost = results['cost']
            ndf = results['ndf']
            chi2_probability = results['chi2_probability']
            # print information
            print(f'chi^2/ndf = {cost:.3f}/{ndf:.3f} = {cost/ndf:.3f}')
            print(f'chi^2 probability = {chi2_probability:.3f}')
    else:
        print('Fit failed!')