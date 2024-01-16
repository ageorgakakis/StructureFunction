import sys
import os
import stan
import numpy as np
import httpstan.models
import httpstan.cache
from astropy.table import Table


model_code = """


data {

  int<lower=1> Nobjects; // number of objects
  int<lower=0> counts0[Nobjects]; // observed counts for object 0
  int<lower=0> counts1[Nobjects]; // observed counts for object 1

  vector[Nobjects] exptime0; // exposure time (including ECF and EEF) 
  vector[Nobjects] exptime1; // exposure time (including ECF and EEF) 

  vector[Nobjects] background0; // background counts for object 0
  vector[Nobjects] background1; // background counts for object 1
 }

parameters {

   vector[Nobjects] logf0; // log10 flux of the first epoch observation
   vector[Nobjects] logSF; // logarithmic flux difference between the two epochs SF = log10(f1)-log10(f0)
   real <lower=1e-5, upper=2.0> sigma;  // stucture function
}

model{

  // these are the Poisson expectations for first and second epoch
  vector[Nobjects] theta0, theta1;

  // Priors
  logf0 ~ normal(-14.0, 10.0); // wide prior on logf0
  logSF ~ normal(0.0, sigma); // logSF is normal 

  // Likelihoods
  theta0 = pow(10.0,logf0) .* exptime0 + background0;
  theta1 = pow(10.0,logf0+logSF) .* exptime1 + background1;
  counts0 ~ poisson(theta0);
  counts1 ~ poisson(theta1);

}
"""

# read input table with two-epoch X-ray count products
new  = Table.read('SF_DRQ16.fits')


# define the range of black-hole masses in the catalogue to be used     
LGMBH_ARRAY = [6.5, 11.25]

# the define the range of Eddington ratios in the catalogue to be used
LGLEDD_ARRAY = [-3.0, 2.0]

# define the rest-frame time-scale intervals in *years* for the structure function calculation 
edges = np.arange(-2,2.0,0.4)
DTloyr =  10**edges[0:-1]
DThiyr =  10**edges[1:]

for ILGMBH in range(len(LGMBH_ARRAY)-1):
    for ILGLEDD in range(len(LGLEDD_ARRAY)-1):

        # define masks to select sources with black hole mass and Eddington ratios
        # within LGMBH_ARRAY and LGLEDD_ARRAY.
        # It expects to find column names 'LOGMBH' and 'LOGLEDD_RATIO' in the
        # input file. 
        m1 = new['LOGMBH'] < LGMBH_ARRAY[ILGMBH+1]
        m1 = np.logical_and(m1, new['LOGMBH'] >= LGMBH_ARRAY[ILGMBH])
        m1 = np.logical_and(m1, new['LOGLEDD_RATIO'] >= LGLEDD_ARRAY[ILGLEDD])
        m1 = np.logical_and(m1, new['LOGLEDD_RATIO'] < LGLEDD_ARRAY[ILGLEDD+1])         

        # loop through the rest-frame time scales
        # it expects to find the column 'DTREST', i.e. rest-frame time
        # difference in *days* between the two epochs. 
        for i, (DT1, DT2) in enumerate(zip(DTloyr, DThiyr)):
   
            mt = np.logical_and(new['DTREST']>DT1 * 365, new['DTREST']< DT2 * 365)
            mt = np.logical_and(m1, mt)
 
            tmp = new[mt]


            if(len(tmp)>0):

                # input parmeters to STAN
                # 'EP1_CTS': column in input catalogue with the 1st epoch X-ray counts
                # 'EP1_CTS': column in input catalogue with the 2nd epoch X-ray counts
                # 'EP1_EXP': column in input catalogue with the 1st epoch "exposure time".
                #            This is the product of the exposure time (in sec), the
                #            counts-to-flux conversion factor and PSF EEF (Encircled
                #            Energy Fraction). Under this definition the number of source
                #            X-ray counts (photons) on the detector is FLUX *  EP1_EXP
                #            FLUX is the assumed flux of the source. 
                # 'EP2_EXP': column in input catalogue with the 2nd epoch "exposure time".
                #            This is the product of the exposure time (in sec), the
                #            counts-to-flux conversion factor and PSF EEF (Encircled
                #            Energy Fraction). Under this definition the number of source
                #            X-ray counts (photons) on the detector is FLUX *  EP1_EXP
                #            FLUX is the assumed flux of the source. 
                # 'EP1_BKG': column in input catalogue with the 1st epoch background level
                # 'EP2_BKG': column in input catalogue with the 2nd epoch background level
                 
                data_dict = {'Nobjects': len(tmp), 
                             'counts0' : tmp['EP1_CTS'].data, 
                             'counts1' : tmp['EP2_CTS'].data,
                             'exptime0': tmp['EP1_EXP'].data,
                             'exptime1': tmp['EP2_EXP'].data,
                             'background0': tmp['EP1_BKG'].data,
                            'background1': tmp['EP2_BKG'].data,
                             }
                chainfile = "results.fits"
                posterior = stan.build(model_code, data=data_dict)
                fit_nonsmooth = posterior.sample(num_chains=4, num_samples=1500, num_warmup=1500)
    
                N, NCHAINS = fit_nonsmooth['logf0'].shape

                # write the output best-fit parameters (median and 1-sigma errors)
                # for individual QSOs in the input catalogue to a fits table/file.
                # the output parameters of STAN inlcude:
                # 'logf0': the 1st epoch  flux in the 0.2-2keV
                # 'logSF': the logarithmic ratio log10(F1/F0), where
                #          F0, F1 represents the 1st and 2nd epoch flux
                #          of the source, respectively
                # 'sigma': the structure function of the full sample
                # Uses astropy Tables to define the fits table
                out1 = Table()
                for k in tmp.keys():
                    out1[k] = tmp[k]
                out1['F0_median'] = np.quantile(fit_nonsmooth['logf0'], q=0.5, axis=1)
                out1['F0_lo'] = np.quantile(fit_nonsmooth['logf0'], q=0.16, axis=1)
                out1['F0_hi'] = np.quantile(fit_nonsmooth['logf0'], q=0.84, axis=1)
                out1['SF_median'] = np.quantile(fit_nonsmooth['logSF'], q=0.5, axis=1)
                out1['SF_lo'] = np.quantile(fit_nonsmooth['logSF'], q=0.16, axis=1)
                out1['SF_hi'] = np.quantile(fit_nonsmooth['logSF'], q=0.84, axis=1)

                out1.write(chainfile,overwrite=True)
                 
                sigma = np.quantile(fit_nonsmooth['sigma'], q=[0.5,0.16,0.84]),
                 
                print(np.mean(tmp['DTREST'].data)/365, DT1, DT2, LGMBH_ARRAY[ILGMBH], LGMBH_ARRAY[ILGMBH+1], LGLEDD_ARRAY[ILGLEDD], LGLEDD_ARRAY[ILGLEDD+1], sigma[0][0], sigma[0][0]-sigma[0][1], sigma[0][2]-sigma[0][0])


