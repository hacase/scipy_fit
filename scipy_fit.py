# fit program for python with scipy.optimize and kafe 2
import numpy as np
from scipy.optimize import curve_fit
from kafe2 import XYContainer, Fit, Plot, ContoursProfiler
import matplotlib.pyplot

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
    
    
import matplotlib.pyplot as plt
import inspect

# fitprogram
# if xerr, executed with kafe2
# else executed with scipy.optimize, curve_fit
# returns fit parameter
# set copy to True to return plot command
# set flag to True to return fit plot
# set retoure to True to return results from kafe2
def exefit(model_function, x_data, y_data, xerr=False, yerr=False, copy=False, flag=False, retoure=False, plot=False):
    if plot is not False:
        if xerr is not False:
            xy_data = XYContainer(x_data=x_data, y_data=y_data)
            xy_data.add_error(axis='x', err_val= xerr)
            if yerr is not False:
                xy_data.add_error(axis='y', err_val= yerr)

            xy_fit = Fit(data=xy_data, model_function=model_function) 
            results = xy_fit.do_fit()
            if results['did_fit']:
                xy_plot = Plot([xy_fit])
                xy_plot.plot()
                return xy_fit
        else:
            if yerr is not False:
                    popt, pcov = curve_fit(model_function, x_data, y_data, sigma=yerr)
            else:
                    popt, pcov = curve_fit(model_function, x_data, y_data)
            return plot.plot(x_data, model_function(x_data, *popt), linewidth=0.5, c='red')
        
    else:
        if xerr is not False:
            xy_data = XYContainer(x_data=x_data, y_data=y_data)
            xy_data.add_error(axis='x', err_val= xerr)
            if yerr is not False:
                xy_data.add_error(axis='y', err_val= yerr)

            xy_fit = Fit(data=xy_data, model_function=model_function) 
            results = xy_fit.do_fit()
            if results['did_fit']:
                xy_plot = Plot([xy_fit])
                xy_plot.plot()
                if retoure is False:
                    print('#===== Results of Fit =====#')
                    string = str(model_function)
                    word_2 = string.split()[1]
                    print('Fit model:', word_2)

                    length = len(results['parameter_values'])

                    print('=== Parameter Values ===')
                    val = np.zeros(length)
                    err = np.zeros(length)
                    i=0
                    for param in results['parameter_values']:
                        val[i] = results['parameter_values'][param]
                        err[i] = results['parameter_errors'][param]

                        # print every parameter with error
                        print(f'{param} = {val[i]:.3e} +/- {err[i]:.3e}')
                        i+=1

                    if copy is not False:
                        frame = inspect.currentframe()
                        frame = inspect.getouterframes(frame)[1]
                        string = inspect.getframeinfo(frame[0]).code_context[0].strip()
                        varbl = string[string.find('(') + 1:-1].split(',')

                        arry=[]
                        for param in results['parameter_values']:
                            arry.append("{:.3e}".format(results['parameter_values'][param]))
                        params = ', '.join(map(str, arry))

                        print(f'\nax.plot({varbl[1]},{varbl[0]}({varbl[1]},{params}),lw=0.5,label=\'Fit\')')
                    print('\n')

                    return xy_fit

                elif retoure == 2:
                    return results
        else:
            if yerr is not False:
                popt, pcov = curve_fit(model_function, x_data, y_data, sigma=yerr)
            else:
                popt, pcov = curve_fit(model_function, x_data, y_data)

            perr=np.sqrt(np.diag(pcov))

            print('#===== Results of Fit =====#')
            string = str(model_function)
            word_2 = string.split()[1]
            print('Fit model:', word_2)

            print('=== Parameter Values ===')
            for i in range(len(popt)):
                print(f'Par. {i+1}: {popt[i]:.2e} +/- {perr[i]:.2e}')

            print('=== Goodness of fit ===')
            residuals = y_data- model_function(x_data, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data-np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot)
            print('R^2:', r_squared)

            if copy is not False:
                frame = inspect.currentframe()
                frame = inspect.getouterframes(frame)[1]
                string = inspect.getframeinfo(frame[0]).code_context[0].strip()
                varbl = string[string.find('(') + 1:-1].split(',')

                arry=[]
                for i in range(len(popt)):
                    arry.append("{:.3e}".format(popt[i]))
                params = ','.join(map(str, arry))

                print(f'\nax.plot({varbl[1]},{varbl[0]}({varbl[1]},{params}),lw=0.5,label=\'Fit\')')

            if flag is not False:
                nstd = 2.
                popt_up = popt + nstd * perr
                popt_dw = popt - nstd * perr

                fit = model_function(x_data, *popt)
                fit_up = model_function(x_data, *popt_up)
                fit_dw = model_function(x_data, *popt_dw)

                fig, ax = plt.subplots()
                if yerr is not False:
                    ax.errorbar(x_data, y_data, yerr=yerr, ms=3, mew=0.5, marker="x", lw=0.5, capsize=2, label='data')
                else:
                    ax.plot(x_data, y_data, ms=1, marker='o', mew=0.5, label='data')
                ax.fill_between(x_data, fit_up, fit_dw, alpha=.25, label='2$\sigma$')
                ax.plot(x_data, model_function(x_data, *popt), linewidth=0.5, c='red', label='fit')
                plt.legend()
                ax.set_xlabel('x')
                ax.set_ylabel('y')

                revert_params()

            if retoure is not False:
                return popt, perr
            
            print('\n')
        
def exefit_gauss(x_data, y_data, model_function=False, yerr=False, offs=False, copy=False, flag=False, retoure=False, plot=False):
    def gauss(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2.*sigma**2))
    def gauss_offs(x,a,x0,sigma,b):
        return a*np.exp(-(x-x0)**2/(2.*sigma**2))+b
    
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    varbl = string[string.find('(') + 1:-1].split(',')
    
    if plot is not False:
        if model_function is False:
            if offs is False:
                model_function=gauss
            else:
                model_function=gauss_offs
        A = y_data.max()
        mu = x_data[y_data.argmax()]
        b = (y_data[0] + y_data[-1]) / 2
        FWHM = np.absolute(mu - np.where(y_data > (y_data.max() * 0.5))[0][0])
        if offs is not False:
            A = y_data.max()-b
            FWHM = np.absolute(mu - x_data[np.where(y_data > ((y_data.max() - b) * 0.5 + b))[0][0]])
        if yerr == True:
            if offs is not False:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, mu, FWHM, b], sigma = yerr)
            else:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, mu, FWHM], sigma = yerr)
        else:
            if offs is not False:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, mu, FWHM, b])
            else:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, mu, FWHM])
                
        return plot.plot(x_data, model_function(x_data, *popt), linewidth=0.5, c='red')
    
    else:    
        print('#===== Results of Fit =====#')    
        local = False
        if model_function is False:
            local = True
            if offs is False:
                model_function=gauss
                string = str(model_function)
                word_2 = string.split()[1]
                print('Fit model:', word_2)
                print('def gauss(x,a,x0,sigma):')
                print('    return a*np.exp(-(x-x0)**2/(2.*sigma**2))')
            else:
                model_function=gauss_offs
                string = str(model_function)
                word_2 = string.split()[1]
                print('Fit model:', word_2)
                print('def gauss_offs(x,a,x0,sigma,b):')
                print('    return a*np.exp(-(x-x0)**2/(2.*sigma**2))+b')
        else:
            string = varbl[2]
            function = string.replace('model_function=', '')
            print('Fit model:', function)

        A = y_data.max()
        mu = x_data[y_data.argmax()]
        b = (y_data[0] + y_data[-1]) / 2
        FWHM = np.absolute(mu - np.where(y_data > (y_data.max() * 0.5))[0][0])
        if offs is not False:
            A = y_data.max()-b
            FWHM = np.absolute(mu - x_data[np.where(y_data > ((y_data.max() - b) * 0.5 + b))[0][0]])
        if yerr == True:
            if offs is not False:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, mu, FWHM, b], sigma = yerr)
            else:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, mu, FWHM], sigma = yerr)
        else:
            if offs is not False:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, mu, FWHM, b])
            else:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, mu, FWHM])

        perr=np.sqrt(np.diag(pcov))

        print('=== Parameter Values ===')
        if local is True:
            if offs is False:
                print(f'A: {popt[0]:.2e} +/- {perr[0]:.2e}')
                print(f'mu: {popt[1]:.2e} +/- {perr[1]:.2e}')
                print(f'sigma: {popt[2]:.2e} +/- {perr[2]:.2e}')
            else:
                print(f'A: {popt[0]:.2e} +/- {perr[0]:.2e}')
                print(f'mu: {popt[1]:.2e} +/- {perr[1]:.2e}')
                print(f'sigma: {popt[2]:.2e} +/- {perr[2]:.2e}')
                print(f'b: {popt[3]:.2e} +/- {perr[3]:.2e}')
        else:
            for i in range(len(popt)):
                print(f'Par. {i+1}: {popt[i]:.2e} +/- {perr[i]:.2e}')

        residuals = y_data- model_function(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data-np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)

        print('=== Goodness of fit ===')
        print('R^2:', r_squared)

        if copy is not False:
            arry=[]
            for i in range(len(popt)):
                arry.append("{:.3e}".format(popt[i]))
            params = ','.join(map(str, arry))

            if local is True:
                if offs is not False:
                    print(f'\nax.plot({varbl[0]},gauss_offs({varbl[0]},{params}),lw=0.5,label=\'Fit\')')
                else:
                    print(f'\nax.plot({varbl[0]},gauss({varbl[0]},{params}),lw=0.5,label=\'Fit\')')
            else:
                print(f'\nax.plot({varbl[0]},{function}({varbl[0]},{params}),lw=0.5,label=\'Fit\')')

        if flag is not False:
            nstd = 5.
            popt_up = popt + nstd * perr
            popt_dw = popt - nstd * perr

            fit = model_function(x_data, *popt)
            fit_up = model_function(x_data, *popt_up)
            fit_dw = model_function(x_data, *popt_dw)

            fig, ax = plt.subplots()
            if yerr is not False:
                ax.errorbar(x_data, y_data, yerr=yerr, ms=3, mew=0.5, marker="x", lw=0.5, capsize=2, label='data')
            else:
                ax.plot(x_data, y_data, ms=1, marker='o', mew=0.5, label='data')
            ax.fill_between(x_data, fit_up, fit_dw, alpha=.25, label='5$\sigma$')
            ax.plot(x_data, model_function(x_data, *popt), linewidth=0.5, c='red', label='fit')
            plt.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            revert_params()

        if retoure is not False:
            return popt, perr

        print('\n')

        
def exefit_kafe(model_function, x_data, y_data, xerr=False, yerr=False, retoure=1):
    
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
