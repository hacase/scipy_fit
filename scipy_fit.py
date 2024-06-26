# fit program for python with scipy.optimize and scipy.odr
import numpy as np
from scipy.optimize import curve_fit
import scipy.odr as sodr
import matplotlib.pyplot
import inspect

import warnings
warnings.filterwarnings("ignore")

# store original plot parameters so that we can revert:
ORIG_MATPLOTLIB_CONF = dict(matplotlib.rcParams)

def saveplot(fig_width=None, fig_height=None, columns=1, fontsize=10):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this function before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}; 1 is default
    fontsize : 10 pt default
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
              'savefig.dpi' : 300  # set to 600 for poster printing or PR figures
    }

    matplotlib.rcParams.update(params)
    

def revert_params():
    """
    reverts any changes done to matplotlib parameters and restores
    the state before homogenize_plot was called
    """

    matplotlib.rcParams.update(ORIG_MATPLOTLIB_CONF)
    
    
import matplotlib.pyplot as plt
# fitprogram
# if xerr, executed with scipy.odr, but with init set to variable number
# or initial values
# else executed with scipy.optimize, curve_fit
# returns fit parameter
# set copy to return plot command
# set flag to return fit plot, set fill to disable sigma environment
# set retoure to return internal results
# set plot to ax to plot in subplot environment, set l for label
# fit program for python with scipy.optimize and scipy.odr
def exefit(model_function, x_data, y_data, xerr=False, yerr=False, copy=False, flag=False, retoure=False, plot=False, l=False, init=False, fill=False, sur=False):
    dist = np.abs(x_data[-1]-x_data[0])
    step = np.arange(x_data[0], x_data[-1], dist / 1000.)
    if not len(step):
        step = np.arange(x_data[0], x_data[-1], -dist / 1000.)
    if init is not False:
        if isinstance(init, int):
            init = np.ones(init)
    if plot is not False:
        if xerr is not False:
            lin_model = sodr.Model(model_function)
            fit_data = sodr.RealData(x_data, y_data, sx=xerr)
            if yerr is not False:
                fit_data = sodr.RealData(x_data, y_data, sx=xerr, sy=yerr)
            odr = sodr.ODR(fit_data, lin_model, beta0 = init)
            out = odr.run()
            if l is False:
                return plot.plot(step, model_function(out.beta, step), linewidth=0.5, c='red')
            else:
                return plot.plot(step, model_function(out.beta, step), linewidth=0.5, c='red', label=l)
        else:
            if yerr is not False:
                    popt, pcov = curve_fit(model_function, x_data, y_data, sigma=yerr)
            else:
                    popt, pcov = curve_fit(model_function, x_data, y_data)
            if l is False:
                return plot.plot(step, model_function(step, *popt), linewidth=0.5, c='red')
            else:
                return plot.plot(step, model_function(step, *popt), linewidth=0.5, c='red', label=l)
        
    else:
        if xerr is not False:
            lin_model = sodr.Model(model_function)
            fit_data = sodr.RealData(x_data, y_data, sx=xerr)
            if yerr is not False:
                fit_data = sodr.RealData(x_data, y_data, sx=xerr, sy=yerr)
            odr = sodr.ODR(fit_data, lin_model, beta0 = init)
            out = odr.run()

            if sur is False:
                print('#===== Results of Fit =====#')
                string = str(model_function)
                word_2 = string.split()[1]
                print('Fit model:', word_2)

                print('=== Parameter Values ===')
                for i in range(len(init)):
                    print(f'Par. {i+1}: {out.beta[i]:.2e} +/- {out.sd_beta[i]:.2e}')

                print('=== Goodness of fit ===')
                print('Chi^2:', out.res_var)
                
            if copy is not False:
                frame = inspect.currentframe()
                frame = inspect.getouterframes(frame)[1]
                string = inspect.getframeinfo(frame[0]).code_context[0].strip()
                varbl = string[string.find('(') + 1:-1].split(',')

                arry=[]
                for param in out.beta:
                    arry.append("{:.3e}".format(param))
                params = ','.join(map(str, arry))
                
                print(f'\nparams=np.array([{params}])')
                print(f'ax.plot({varbl[1]},{varbl[0]}(params,{varbl[1]}),lw=0.5,label=\'Fit\')')
                print('\n')

            if flag is not False:                
                fig, ax = plt.subplots()
                ax.errorbar(x_data, y_data, yerr=yerr, xerr=xerr, ms=3, mew=0.5, marker="x", lw=0.5, capsize=2, label='data')
                if fill is False:
                    nstd = 2.
                    popt_up = out.beta + nstd * out.sd_beta
                    popt_dw = out.beta - nstd * out.sd_beta
                                       
                    fit_up = model_function(popt_up, step)
                    fit_dw = model_function(popt_dw, step)
                    for i in range(len(fit_up)):
                        if fit_up[i] < fit_dw[i]:
                            temp = fit_dw[i]
                            fit_dw[i] = fit_up[i]
                            fit_up[i] = temp
                            
                    ax.fill_between(step, fit_up, fit_dw, alpha=.25, label='2$\sigma$')
                ax.plot(step, model_function(out.beta, step), linewidth=0.5, c='red', label='fit')
                plt.legend()
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                
            if retoure is not False:
                return out.beta, out.sd_beta
        else:
            if yerr is not False:
                popt, pcov = curve_fit(model_function, x_data, y_data, sigma=yerr)
            else:
                popt, pcov = curve_fit(model_function, x_data, y_data)

            perr=np.sqrt(np.diag(pcov))

            if sur is False:
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
                fig, ax = plt.subplots()
                if yerr is not False:
                    ax.errorbar(x_data, y_data, yerr=yerr, ms=3, mew=0.5, marker="x", lw=0.5, capsize=2, label='data')
                else:
                    ax.plot(x_data, y_data, ms=1, marker='o', mew=0.5, label='data')
                if fill is False:
                    nstd = 2.
                    popt_up = popt + nstd * perr
                    popt_dw = popt - nstd * perr
                    
                    fit_up = model_function(step, *popt_up)
                    fit_dw = model_function(step, *popt_dw)                    
                    for i in range(len(fit_up)):
                        if fit_up[i] < fit_dw[i]:
                            temp = fit_dw[i]
                            fit_dw[i] = fit_up[i]
                            fit_up[i] = temp
                            
                    ax.fill_between(step, fit_up, fit_dw, alpha=.25, label='2$\sigma$')
                ax.plot(step, model_function(step, *popt), linewidth=0.5, c='red', label='fit')
                plt.legend()
                ax.set_xlabel('x')
                ax.set_ylabel('y')

            if retoure is not False:
                if xerr is not False:
                    return popt, perr, out.res_var
                else:
                    residuals = y_data- model_function(x_data, *popt)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y_data-np.mean(y_data))**2)
                    r_squared = 1 - (ss_res / ss_tot)
                    return popt, perr, r_squared
            
            print('\n')
        
# gaussian fit only with scipy.optimize
# normal gaussian function with/without offset build in
# calculates initial guesses
def exefit_gauss(x_data, y_data, model_function=False, yerr=False, offs=False, copy=False, flag=False, retoure=False, plot=False, l=False, fill=False, sur=False):
    dist = np.abs(x_data[-1]-x_data[0])
    step = np.arange(x_data[0], x_data[-1], dist / 1000.)
    if not len(step):
        step = np.arange(x_data[0], x_data[-1], -dist / 1000.)
    def gauss(x,A,x0,sigma):
        return A*np.exp(-(x-x0)**2/(2.*sigma**2))
    def gauss_offs(x,A,x0,sigma,b):
        return A*np.exp(-(x-x0)**2/(2.*sigma**2))+b
    
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
        x0 = x_data[y_data.argmax()]
        b = (y_data[0] + y_data[-1]) / 2
        FWHM = np.absolute(x0 - np.where(y_data > (y_data.max() * 0.5))[0][0])
        if offs is not False:
            A = y_data.max()-b
            FWHM = np.absolute(x0 - x_data[np.where(y_data > ((y_data.max() - b) * 0.5 + b))[0][0]])
        if yerr is not False:
            if offs is not False:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, x0, FWHM, b], sigma = yerr)
            else:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, x0, FWHM], sigma = yerr)
        else:
            if offs is not False:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, x0, FWHM, b])
            else:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, x0, FWHM])
                
        if l is not False:
            return plot.plot(step, model_function(step, *popt), linewidth=0.5, c='red')
        else:
            return plot.plot(step, model_function(step, *popt), linewidth=0.5, c='red', label=l)
    
    else:    
        if sur is False:
            print('#===== Results of Fit =====#')    
            local = False
            if model_function is False:
                local = True
                if offs is False:
                    model_function=gauss
                    string = str(model_function)
                    word_2 = string.split()[1]
                    print('Fit model:', word_2)
                    print('def gauss(x,A,x0,sigma):')
                    print('    return A*np.exp(-(x-x0)**2/(2.*sigma**2))')
                else:
                    model_function=gauss_offs
                    string = str(model_function)
                    word_2 = string.split()[1]
                    print('Fit model:', word_2)
                    print('def gauss_offs(x,A,x0,sigma,b):')
                    print('    return A*np.exp(-(x-x0)**2/(2.*sigma**2))+b')
            else:
                string = varbl[2]
                function = string.replace('model_function=', '')
                print('Fit model:', function)

        A = y_data.max()
        x0 = x_data[y_data.argmax()]
        b = (y_data[0] + y_data[-1]) / 2
        FWHM = np.absolute(x0 - np.where(y_data > (y_data.max() * 0.5))[0][0])
        if offs is not False:
            A = y_data.max()-b
            FWHM = np.absolute(x0 - x_data[np.where(y_data > ((y_data.max() - b) * 0.5 + b))[0][0]])
        if yerr is not False:
            if offs is not False:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, x0, FWHM, b], sigma = yerr)
            else:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, x0, FWHM], sigma = yerr)
        else:
            if offs is not False:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, x0, FWHM, b])
            else:
                popt, pcov = curve_fit(model_function, x_data, y_data, p0=[A, x0, FWHM])

        perr=np.sqrt(np.diag(pcov))

        if sur is False:
            print('=== Parameter Values ===')
            if local is True:
                if offs is False:
                    print(f'A:     {popt[0]:.2e} +/- {perr[0]:.2e}')
                    print(f'x0:    {popt[1]:.2e} +/- {perr[1]:.2e}')
                    print(f'sigma: {popt[2]:.2e} +/- {perr[2]:.2e}')
                else:
                    print(f'A:     {popt[0]:.2e} +/- {perr[0]:.2e}')
                    print(f'x0:    {popt[1]:.2e} +/- {perr[1]:.2e}')
                    print(f'sigma: {popt[2]:.2e} +/- {perr[2]:.2e}')
                    print(f'b:     {popt[3]:.2e} +/- {perr[3]:.2e}')
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
            fig, ax = plt.subplots()
            if yerr is not False:
                ax.errorbar(x_data, y_data, yerr=yerr, ms=3, mew=0.5, marker="x", lw=0.5, capsize=2, label='data')
            else:
                ax.plot(x_data, y_data, ms=1, marker='o', mew=0.5, label='data')
            if fill is False:                
                nstd = 5.
                popt_up = popt + nstd * perr
                popt_dw = popt - nstd * perr
                
                fit_up = model_function(step, *popt_up)
                fit_dw = model_function(step, *popt_dw)                    
                for i in range(len(fit_up)):
                    if fit_up[i] < fit_dw[i]:
                        temp = fit_dw[i]
                        fit_dw[i] = fit_up[i]
                        fit_up[i] = temp

                ax.fill_between(step, fit_up, fit_dw, alpha=.25, label='5$\sigma$')
            ax.plot(step, model_function(step, *popt), linewidth=0.5, c='red', label='fit')
            plt.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            revert_params()

        if retoure is not False:
                if xerr is not False:
                    return popt, perr, out.res_var
                else:
                    residuals = y_data- model_function(x_data, *popt)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y_data-np.mean(y_data))**2)
                    r_squared = 1 - (ss_res / ss_tot)
                    return popt, perr, r_squared

        print('\n')

# returns value and error in scientific notation      
def sciexpo(val, err, rnd=2, tbl=False, ltx=False, cdot=False):
    exp=np.zeros(len(val),dtype=int)
    rst=np.zeros(len(val),dtype=int)

    for i in range(len(val)):
        for j in np.arange(-100,100,1):
            if int(val[i]/10.**j) == 0:
                exp[i]=j-1
                break

    err=err/10.**(exp)

    for i in range(len(err)):
        for j in np.arange(-100,100,1):
            if (int(err[i]/10.**j) == 0):
                if j>-rnd:
                    err[i]/10.**j
                else:
                    rst[i]=j-1
                break



    new_val=np.round(val/10.**exp,rnd)

    new_err=np.round(err/10.**rst,rnd)
    
    if cdot is not False:
        if ltx == 1:
            for i in range(len(val)):
                if exp[i]==0:
                    if rst[i]==0:
                        print('$\\sciexpo{',f'{new_val[i]:.{rnd}f}',
                              '}{',f'{new_err[i]:.{rnd}f}',
                              '}$\\\\',sep='')
                    else:
                        print('$\\sciexpo{',f'{new_val[i]:.{rnd}f}',
                              '}{',f'{new_err[i]:.{rnd}f}',
                              '\\cdot\\num{e',f'{rst[i]}',
                              '}}$\\\\',sep='')
                else:
                    if rst[i]==0:
                        print('$\\sciexp{',f'{new_val[i]:.{rnd}f}',
                              '}{',f'{new_err[i]:.{rnd}f}',
                              '}{e',f'{exp[i]}','}$\\\\',sep='')
                    else:
                        print('$\\sciexp{',f'{new_val[i]:.{rnd}f}',
                              '}{',f'{new_err[i]:.{rnd}f}',
                              '\\cdot\\num{e',f'{rst[i]}',
                              '}}{e',f'{exp[i]}','}$\\\\',sep='')

        if ltx == 2:
            for i in range(len(val)):
                if exp[i]==0:
                    if rst[i]==0:
                        print('$\\sciexpo{',f'{new_val[i]:.{rnd}f}',
                              '}{',f'{new_err[i]:.{rnd}f}',
                              '}$',sep='')
                    else:
                        print('$\\sciexpo{',f'{new_val[i]:.{rnd}f}',
                              '}{',f'{new_err[i]:.{rnd}f}',
                              '\\cdot\\num{e',f'{rst[i]}',
                              '}}$',sep='')
                else:
                    if rst[i]==0:
                        print('$\\sciexp{',f'{new_val[i]:.{rnd}f}',
                              '}{',f'{new_err[i]:.{rnd}f}',
                              '}{e',f'{exp[i]}','}$',sep='')
                    else:
                        print('$\\sciexp{',f'{new_val[i]:.{rnd}f}',
                              '}{',f'{new_err[i]:.{rnd}f}',
                              '\\cdot\\num{e',f'{rst[i]}',
                              '}}{e',f'{exp[i]}','}$',sep='')
    else:
        if ltx == 1:
            for i in range(len(val)):
                if exp[i]==0:
                    if rst[i]==0:
                        print('$\\sciexpo{',f'{new_val[i]:.{rnd}f}',
                              '}{',f'{new_err[i]:.{rnd}f}',
                              '}$\\\\',sep='')
                    else:
                        print('$\\sciexpo{',f'{new_val[i]:.{rnd}f}',
                              '}{\\num{',f'{new_err[i]:.{rnd}f}',
                              'e',f'{rst[i]}',
                              '}}$\\\\',sep='')
                else:
                    if rst[i]==0:
                        print('$\\sciexp{',f'{new_val[i]:.{rnd}f}',
                              '}{',f'{new_err[i]:.{rnd}f}',
                              '}{e',f'{exp[i]}','}$\\\\',sep='')
                    else:
                        print('$\\sciexp{',f'{new_val[i]:.{rnd}f}',
                              '}{\\num{',f'{new_err[i]:.{rnd}f}',
                              'e',f'{rst[i]}',
                              '}}{e',f'{exp[i]}','}$\\\\',sep='')

        if ltx == 2:
            for i in range(len(val)):
                if exp[i]==0:
                    if rst[i]==0:
                        print('$\\sciexpo{',f'{new_val[i]:.{rnd}f}',
                              '}{',f'{new_err[i]:.{rnd}f}',
                              '}$',sep='')
                    else:
                        print('$\\sciexpo{',f'{new_val[i]:.{rnd}f}',
                              '}{\\num{',f'{new_err[i]:.{rnd}f}',
                              'e',f'{rst[i]}',
                              '}}$',sep='')
                else:
                    if rst[i]==0:
                        print('$\\sciexp{',f'{new_val[i]:.{rnd}f}',
                              '}{',f'{new_err[i]:.{rnd}f}',
                              '}{e',f'{exp[i]}','}$',sep='')
                    else:
                        print('$\\sciexp{',f'{new_val[i]:.{rnd}f}',
                              '}{\\num{',f'{new_err[i]:.{rnd}f}',
                              'e',f'{rst[i]}',
                              '}}{e',f'{exp[i]}','}$',sep='')
    
    if tbl is not False:
        if cdot is not False:
            print('for i in range(len(val)):')
            print('    if exp[i]==0:')
            print('        if rst[i]==0:')
            print("            text.append(' $\\\\sciexpo{')")
            print("            text.append(f' {new_val[i]:.",f'{rnd}',"f}')",sep='')
            print("            text.append(' }{')")
            print("            text.append(f' {new_err[i]:.",f'{rnd}',"f}')",sep='')
            print(r"            text.append(' }$\\\\\n')")
            print('        else:')
            print("            text.append(' $\\\\sciexpo{')")
            print("            text.append(f' {new_val[i]:.",f'{rnd}',"f}')",sep='')
            print(r"            text.append(' }{')")
            print("            text.append(f' {new_err[i]:.",f'{rnd}',"f}')",sep='')
            print(r"            text.append(' \\cdot\\num{e')")
            print("            text.append(f' ","{rst[i]}","$')",sep='')
            print(r"            text.append('\\\\\n')")
            print('    else:')
            print('        if rst[i]==0:')
            print("            text.append(' $\\\\sciexp{')")
            print("            text.append(f' {new_val[i]:.",f'{rnd}',"f}')",sep='')
            print("            text.append(' }{')")
            print("            text.append(f' {new_err[i]:.",f'{rnd}',"f}')",sep='')
            print("            text.append(' }{e')")
            print("            text.append(f' {exp[i]}')")
            print(r"            text.append(' }$\\\\\n')")
            print('        else:')
            print("            text.append(' $\\\\sciexp{')")
            print("            text.append(f' {new_val[i]:.",f'{rnd}',"f}')",sep='')
            print(r"            text.append(' }{')")
            print("            text.append(f' {new_err[i]:.",f'{rnd}',"f}')",sep='')
            print(r"            text.append(' \\cdot\\num{e')")
            print("            text.append(f' ","{rst[i]}","')",sep='')
            print("            text.append(' }}{e')")
            print("            text.append(f' {exp[i]}')")
            print(r"            text.append(' }$\\\\\n')")
        else:
            print('for i in range(len(val)):')
            print('    if exp[i]==0:')
            print('        if rst[i]==0:')
            print("            text.append(' $\\\\sciexpo{')")
            print("            text.append(f' {new_val[i]:.",f'{rnd}',"f}')",sep='')
            print("            text.append(' }{')")
            print("            text.append(f' {new_err[i]:.",f'{rnd}',"f}')",sep='')
            print(r"            text.append(' }$\\\\\n')")
            print('        else:')
            print("            text.append(' $\\\\sciexpo{')")
            print("            text.append(f' {new_val[i]:.",f'{rnd}',"f}')",sep='')
            print(r"            text.append(' }{\\num{')")
            print("            text.append(f' {new_err[i]:.",f'{rnd}',"f}e')",sep='')
            print("            text.append(f' ","{rst[i]}","$')",sep='')
            print(r"            text.append('\\\\\n')")
            print('    else:')
            print('        if rst[i]==0:')
            print("            text.append(' $\\\\sciexp{')")
            print("            text.append(f' {new_val[i]:.",f'{rnd}',"f}')",sep='')
            print("            text.append(' }{')")
            print("            text.append(f' {new_err[i]:.",f'{rnd}',"f}')",sep='')
            print("            text.append(' }{e')")
            print("            text.append(f' {exp[i]}')")
            print(r"            text.append(' }$\\\\\n')")
            print('        else:')
            print("            text.append(' $\\\\sciexp{')")
            print("            text.append(f' {new_val[i]:.",f'{rnd}',"f}')",sep='')
            print(r"            text.append(' }{\\num{')")
            print("            text.append(f' {new_err[i]:.",f'{rnd}',"f}e')",sep='')
            print("            text.append(f' ","{rst[i]}","')",sep='')
            print("            text.append(' }}{e')")
            print("            text.append(f' {exp[i]}')")
            print(r"            text.append(' }$\\\\\n')")
        
    return new_val, new_err, exp, rst