from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset


class S2SDataset(Dataset):
    
    """
    Class instantiation for file lists of cesm/era5 train, validate, and test data.
    
    Args:
        week (int): lead time week (1, 2, 3, 4, 5, or 6).
        variable (str): variable (tas2m, prsfc, tas2m_anom, or prsfc_anom).
        norm (str): normalization. Defaults to zscore. Also use None, minmax, or negone.
        norm_pixel (boolean): normalize each grid cell. Defaults to False.
        dual_norm (boolean): normalize the input and labels separately. Defaults to False.
        region (str): region method used. 'fixed', 'random', 'quasi', 'global'.
        minv (float): minimum value for normalization. Defaults to None.
        maxv (float): maximum value for normalization. Defaults to None.
        mini (float): minimum value for normalization (input, if dual_norm). Defaults to None.
        maxi (float): maximum value for normalization (input, if dual_norm). Defaults to None.
        mnv (float): mean value for normalization. Defaults to None.
        stdv (float): standard deviation value for normalization. Defaults to None.
        mni (float): mean value for normalization (input, if dual_norm). Defaults to None.
        stdi (float): standard deviation value for normalization (input, if dual_norm). Defaults to None.
        lon0 (float): bottom left corner of 'fixed' region (0 to 360). Defaults to None.
        lat0 (float): bottom left corner of 'fixed' region (-90 to 90). Defaults to None.
        dxdy (float): number of grid cells for 'fixed' or 'random' region. Defaults to 32.
        feat_topo (boolean): use terrian heights (era5) as feature. Defaults True.
        feat_lats (boolean): use latitudes as feature. Defaults True.
        feat_lons (boolean): use longitudes as feature. Defaults True.
        startdt (str): datetime start. Defaults to 1999-02-01.
        enddt (str): datetime end. Defaults to 2021-12-31.
        homedir (str): home directory. Defaults to /glade/scratch/molina/.
        
    Returns:
        list of cesm and era5 files for train/val/test sets.
    
    """
    
    def __init__(self, week, variable, 
                 norm='zscore', norm_pixel=False, dual_norm=False, region='fixed', 
                 minv=None, maxv=None, mini=None, maxi=None, 
                 mnv=None, stdv=None, mni=None, stdi=None,
                 lon0=None, lat0=None, dxdy=32, 
                 feat_topo=True, feat_lats=True, feat_lons=True, 
                 startdt='1999-02-01', enddt='2021-12-31', 
                 homedir='/glade/scratch/molina/'):
        
        self.week = week
        self.day_init, self.day_end = self.leadtime_help()
        
        self.variable_ = variable
        self.startdt = startdt
        self.enddt = enddt
        
        self.homedir = homedir
        self.cesm_dir = f'{self.homedir}cesm_{self.variable_}_week{self.week}/'
        self.era5_dir = f'{self.homedir}era5_{self.variable_}_week{self.week}/'
            
        self.region_ = region
        
        if self.region_ == 'fixed':
            self.lon0=lon0
            self.lat0=lat0
            self.dxdy=dxdy
            
        if self.region_ == 'random':
            self.dxdy=dxdy
            self.lon0, self.lat0=self.rand_coords()
            
        if self.region_ == 'quasi':
            self.dxdy=dxdy
            self.lon0, self.lat0=self.quasi_coords()
            
        if self.region_ == 'quasi_global':
            self.dxdy=180
            self.lon0, self.lat0=self.quasi_global()
            
        if self.region_ == 'global':
            self.dxdy=360
            self.lon0, self.lat0=0.0, -90.0
            
        if self.region_ == 'global1':
            self.dxdy=180
            self.lon0, self.lat0=0.0, -90.0
            
        if self.region_ == 'global2':
            self.dxdy=180
            self.lon0, self.lat0=179.0, -90.0
            
        self.feat_topo=feat_topo
        self.feat_lats=feat_lats
        self.feat_lons=feat_lons
            
        self.filelists()
        
        self.norm = norm
        self.norm_pixel = norm_pixel
        self.dual_norm = dual_norm
        
        if not self.dual_norm:
            if self.norm == 'zscore':
                self.zscore_values(mnv, stdv)
            if self.norm == 'minmax':
                self.minmax_values(minv, maxv)
            if self.norm == 'negone':
                self.minmax_values(minv, maxv)
            
        if self.dual_norm:
            if self.norm == 'zscore':
                self.zscore_values(mnv, stdv, mni, stdi)
            if self.norm == 'minmax':
                self.minmax_values(minv, maxv, mini, maxi)
            if self.norm == 'negone':
                self.minmax_values(minv, maxv, mini, maxi)
            
        
    def __len__(self):
        
        return len(self.list_of_cesm)

    
    def __getitem__(self, idx):
        """
        assembles input and label data
        """
        # create files using random indices
        self.create_files(idx)
        
        image = self.img_train
        label = self.img_label
        
        # need to convert precip cesm file
        if self.variable_ == 'prsfc':
            image = image * 84600 # convert kg/m2/s to mm/day
        
        # normalization options applied here
        if self.norm == 'zscore':
            image, label = self.zscore_compute(image, inpvar=True), self.zscore_compute(label)
        if self.norm == 'minmax':
            image, label = self.minmax_compute(image, inpvar=True), self.minmax_compute(label)
        if self.norm == 'negone':
            image, label = self.negone_compute(image, inpvar=True), self.negone_compute(label)
        
        # add the spatial variable to coordinate data
        self.coord_data["cesm"]=(['sample','x','y'], 
                                 image.transpose('sample','lon','lat').values)
        
        self.coord_data["era5"]=(['sample','x','y'], 
                                 label.transpose('sample','x','y').values)
        
        # features including terrain, lats, and lons
        if self.feat_topo and self.feat_lats and self.feat_lons:
            
            # input features
            img = xr.concat([self.coord_data['top'],
                             self.coord_data['lat'],
                             self.coord_data['lon'],
                             self.coord_data['cesm']],dim='feature')
            
        # features including terrain
        if self.feat_topo and not self.feat_lats and not self.feat_lons:
            
            # input features
            img = xr.concat([self.coord_data['top'],
                             self.coord_data['cesm']],dim='feature')
            
        # features including lats and lons
        if not self.feat_topo and self.feat_lats and self.feat_lons:
            
            # input features
            img = xr.concat([self.coord_data['lat'],
                             self.coord_data['lon'],
                             self.coord_data['cesm']],dim='feature')
            
        # features including terrain and lat
        if self.feat_topo and self.feat_lats and not self.feat_lons:
            
            # input features
            img = xr.concat([self.coord_data['top'],
                             self.coord_data['lat'],
                             self.coord_data['cesm']],dim='feature')
            
        # features including terrain and lon
        if self.feat_topo and not self.feat_lats and self.feat_lons:
            
            # input features
            img = xr.concat([self.coord_data['top'],
                             self.coord_data['lon'],
                             self.coord_data['cesm']],dim='feature')
            
        # no extra features
        if not self.feat_topo and not self.feat_lats and not self.feat_lons:
            
            # input features
            img = xr.concat([self.coord_data['cesm']],dim='feature')
        
        lbl = xr.concat([self.coord_data['era5']],dim='feature') # label
        lsm = self.lmask # lsm
        
        if self.region_ == 'global':
            img = img.pad(pad_width={'x':[12,12],'y':[6,5]}, constant_values=0.0)
            lbl = lbl.pad(pad_width={'x':[12,12],'y':[6,5]}, constant_values=0.0)
            lsm = lsm.pad(pad_width={'x':[12,12],'y':[6,5]}, constant_values=0.0)
            
        if self.region_ == 'global1' or self.region_ == 'global2' or self.region_ == 'quasi_global':
            img = img.pad(pad_width={'x':[6,5],'y':[6,5]}, constant_values=0.0)
            lbl = lbl.pad(pad_width={'x':[6,5],'y':[6,5]}, constant_values=0.0)
            lsm = lsm.pad(pad_width={'x':[6,5],'y':[6,5]}, constant_values=0.0)
            
        return {'input': img.transpose('feature','sample','x','y').values, 
                'label': lbl.transpose('feature','sample','x','y').values,
                'lmask': lsm.transpose('sample','x','y').values}
    
    
    def leadtime_help(self):
        """
        helps with lead time start and end period
        """
        # start dict
        weekdict_init = {
            1: 1,
            2: 8,
            3: 15,
            4: 22,
            5: 29,
            6: 36,
        }
        
        # end dict
        weekdict_end = {
            1: 7,
            2: 14,
            3: 21,
            4: 28,
            5: 35,
            6: 42,
        }
        
        return weekdict_init[self.week], weekdict_end[self.week]
    
    
    def rand_coords(self):
        """
        Get random coords.
        Returns lon0, lat0.
        """
        range_x = np.arange(0., 359. + 1 - self.dxdy, 1)
        range_y = np.arange(-90., 90. + 1 - self.dxdy, 1)

        ax = np.random.choice(range_x, replace=False)
        by = np.random.choice(range_y, replace=False)

        return ax, by
    
    
    def quasi_coords(self):
        """
        Get quasi-random coords.
        Returns lon0, lat0.
        """
        lon_quasi = [190, 230, 262, 280, 280, 112, 110, 125,  0, 60, 0,  10]
        lat_quasi = [ 45,  25,  24, -23, -55,  22, -12, -44, 35,  5, 0, -35]

        rand_indx = np.random.choice(len(lon_quasi), replace=False)

        ax = lon_quasi[rand_indx]
        by = lat_quasi[rand_indx]

        return ax, by
    
    
    def quasi_global(self):
        """
        Get quasi-random global coords.
        Returns lon0, lat0.
        """
        lon_quasi = [  0.0, 179.0]
        lat_quasi = [-90.0, -90.0]

        rand_indx = np.random.choice(len(lon_quasi), replace=False)

        ax = lon_quasi[rand_indx]
        by = lat_quasi[rand_indx]

        return ax, by
    
    
    def zscore_compute(self, data, inpvar=False):
        """
        Function for normalizing data prior to training using z-score.
        """
        if not inpvar:
            
            return (data - self.mean_val) / self.std_val
        
        if inpvar:
            
            return (data - self.mean_inp) / self.std_inp


    def minmax_compute(self, data, inpvar=False):
        """
        Min max computation.
        """
        if not inpvar:
            
            return (data - self.min_val) / (self.max_val - self.min_val)
        
        if inpvar:
            
            return (data - self.min_inp) / (self.max_inp - self.min_inp)
    
    
    def negone_compute(self, data, inpvar=False):
        """
        Scale between negative 1 and positive 1.
        """
        if not inpvar:
            
            return (2 * (data - self.min_val) / (self.max_val - self.min_val)) - 1
        
        if inpvar:
            
            return (2 * (data - self.min_inp) / (self.max_inp - self.min_inp)) - 1
    
    
    def datetime_range(self):
        """
        helps with creating datetime range for data assembly
        """
        dt_cesm = pd.date_range(start=self.startdt, end=self.enddt, freq='W-MON')
        
        # remove missing july date/file
        dt_cesm = dt_cesm[~((dt_cesm.month==7)&(dt_cesm.day==26)&(dt_cesm.year==2021))]
        
        # anomalies have a lead time issue, this resolves it
        if self.variable_ == 'prsfc_anom' or self.variable_ == 'tas2m_anom':
            dt_cesm = dt_cesm[~((dt_cesm.month==2)&(dt_cesm.day==29)&(dt_cesm.year==2016))]
        
        # list containing datetime array
        matches = []

        # loop through datetimes
        for num, (yr, mo, dy) in enumerate(zip(
            dt_cesm.strftime("%Y"), dt_cesm.strftime("%m"), dt_cesm.strftime("%d"))):

            # time adjustment for leap year
            if mo == '02' and dy == '29':
                matches.append(dt_cesm[num] - timedelta(days=1))
            else:
                matches.append(dt_cesm[num])
        
        self.dt_cesm = pd.to_datetime(matches)
        
        
    def filelists(self):
        """
        creates list of files from the datetime range
        """
        # run related method
        self.datetime_range()

        # lists to be populated with filenames
        self.list_of_cesm = []
        self.list_of_era5 = []

        # loop through datetimes
        for i in self.dt_cesm:
            
            # convert datetime to string
            dt_string = str(i.strftime("%Y")+i.strftime("%m")+i.strftime("%d"))
            
            self.list_of_cesm.append(
                f'{self.cesm_dir}cm_{self.variable_}_'+dt_string+'.nc') # cesm2 list
            self.list_of_era5.append(
                f'{self.era5_dir}e5_{self.variable_}_'+dt_string+'.nc') # era5 list
            
            
    def zscore_values(self, mnv, stdv, mni=False, stdi=False):
        """
        compute zscore values
        """
        # help with variable names inside files
        if self.variable_ == 'tas2m':
            var = 'tas_2m'
        if self.variable_ == 'prsfc':
            var = 'pr_sfc'
        if self.variable_ == 'prsfc_anom' or self.variable_ == 'tas2m_anom':
            var = 'anom'
        
        # if not dual_norm
        if not self.dual_norm:
            
            # if mean and standard deviation are NOT provided do this (only era5)
            if np.any(mnv) == None or np.any(stdv) == None:

                # open file
                tmp = xr.open_mfdataset(self.list_of_era5, concat_dim='sample', combine='nested')[var]
                tmp = self.box_cutter(tmp)

                if not self.norm_pixel:

                    self.mean_val = tmp.mean(skipna=True).values # era5
                    self.std_val = tmp.std(skipna=True).values   # era5
                    
                    self.mean_inp = self.mean_val # cesm
                    self.std_inp = self.std_val   # cesm
                    
                    return

                if self.norm_pixel:

                    self.mean_val = tmp.mean('sample', skipna=True).values # era5
                    self.std_val = tmp.std('sample', skipna=True).values   # era5
                    
                    self.mean_inp = self.mean_val # cesm
                    self.std_inp = self.std_val   # cesm
                    
                    return

            # if mean and standard deviation ARE provided do this
            if np.any(mnv) != None and np.any(stdv) != None:

                self.mean_val = mnv # era5
                self.std_val = stdv # era5
                
                self.mean_inp = self.mean_val # cesm
                self.std_inp = self.std_val   # cesm
                
                return
                
        # if dual_norm
        if self.dual_norm:
            
            # if mean and standard deviation are NOT provided do this
            if np.any(mnv) == None or np.any(stdv) == None:

                # open file
                tmp0 = xr.open_mfdataset(self.list_of_era5, concat_dim='sample', combine='nested')[var]
                tmp1 = xr.open_mfdataset(self.list_of_cesm, concat_dim='sample', combine='nested')[var]

                # need to convert precip cesm file
                if self.variable_ == 'prsfc':
                    tmp1 = tmp1 * 84600 # convert kg/m2/s to mm/day

                tmp0 = self.box_cutter(tmp0) # era5
                tmp1 = self.box_cutter(tmp1, cesm_help=True) # cesm

                if not self.norm_pixel:

                    self.mean_val = tmp0.mean(skipna=True).values # era5
                    self.std_val = tmp0.std(skipna=True).values   # era5
                    
                    self.mean_inp = tmp1.mean(skipna=True).values # cesm
                    self.std_inp = tmp1.std(skipna=True).values   # cesm
                    
                    return

                if self.norm_pixel:

                    self.mean_val = tmp0.mean('sample', skipna=True).values # era5
                    self.std_val = tmp0.std('sample', skipna=True).values   # era5
                    
                    self.mean_inp = tmp1.mean('sample', skipna=True).values # cesm
                    self.std_inp = tmp1.std('sample', skipna=True).values   # cesm
                    
                    return

            # if mean and standard deviation ARE provided do this
            if np.any(mnv) != None and np.any(stdi) != None:

                self.mean_val = mnv # era5
                self.std_val = stdv # era5
                
                self.mean_inp = mni # cesm
                self.std_inp = stdi # cesm
                
                return
        

    def minmax_values(self, minv, maxv, mini=False, maxi=False):
        """
        compute minmax values
        """
        # help with variable names inside files
        if self.variable_ == 'tas2m':
            var = 'tas_2m'
        if self.variable_ == 'prsfc':
            var = 'pr_sfc'
        if self.variable_ == 'prsfc_anom' or self.variable_ == 'tas2m_anom':
            var = 'anom'
        
        # if not dual_norm
        if not self.dual_norm:

            # if min and max are NOT provided do this (only era5)
            if np.any(minv) == None or np.any(maxv) == None:

                # open file
                tmp = xr.open_mfdataset(self.list_of_era5, concat_dim='sample', combine='nested')[var]
                tmp = self.box_cutter(tmp)

                if not self.norm_pixel:

                    self.max_val = tmp.max(skipna=True).values # era5
                    self.min_val = tmp.min(skipna=True).values # era5
                    
                    self.max_inp = self.max_val # cesm
                    self.min_inp = self.min_val # cesm
                    
                    return

                if self.norm_pixel:

                    self.max_val = tmp.max('sample', skipna=True).values # era5
                    self.min_val = tmp.min('sample', skipna=True).values # era5
                    
                    self.max_inp = self.max_val # cesm
                    self.min_inp = self.min_val # cesm
                    
                    return

            # if min and max ARE provided do this
            if np.any(minv) != None and np.any(maxv) != None:

                self.max_val = maxv # era5
                self.min_val = minv # era5
                
                self.max_inp = self.max_val # cesm
                self.min_inp = self.min_val # cesm
                
                return
                
        # if dual_norm
        if self.dual_norm:
            
            # if min and max are NOT provided do this
            if np.any(minv) == None or np.any(maxv) == None:

                # open file
                tmp0 = xr.open_mfdataset(self.list_of_era5, concat_dim='sample', combine='nested')[var]
                tmp1 = xr.open_mfdataset(self.list_of_cesm, concat_dim='sample', combine='nested')[var]

                # need to convert precip cesm file
                if self.variable_ == 'prsfc':
                    tmp1 = tmp1 * 84600 # convert kg/m2/s to mm/day

                tmp0 = self.box_cutter(tmp0) # era5
                tmp1 = self.box_cutter(tmp1, cesm_help=True) # cesm

                if not self.norm_pixel:

                    self.max_val = tmp0.max(skipna=True).values # era5
                    self.min_val = tmp0.min(skipna=True).values # era5
                    
                    self.max_inp = tmp1.max(skipna=True).values # cesm
                    self.min_inp = tmp1.min(skipna=True).values # cesm
                    
                    return

                if self.norm_pixel:

                    self.max_val = tmp0.max('sample', skipna=True).values # era5
                    self.min_val = tmp0.min('sample', skipna=True).values # era5
                    
                    self.max_inp = tmp1.max('sample', skipna=True).values # cesm
                    self.min_inp = tmp1.min('sample', skipna=True).values # cesm
                    
                    return

            # if min and max ARE provided do this
            if np.any(minv) != None and np.any(maxi) != None:

                self.max_val = maxv # era5
                self.min_val = minv # era5
                
                self.max_inp = maxi # cesm
                self.min_inp = mini # cesm
                
                return
        
        
    def create_files(self, indx):
        """
        create input and label data using file list and indx from sampling
        """
        # help with variable names inside files
        if self.variable_ == 'tas2m':
            var = 'tas_2m'
        if self.variable_ == 'prsfc':
            var = 'pr_sfc'
        if self.variable_ == 'prsfc_anom' or self.variable_ == 'tas2m_anom':
            var = 'anom'
            
        # coordinates (terrain and lat/lon features)
        tmpcd = xr.open_dataset(self.homedir+'/ml_coordsv2.nc').expand_dims('sample')
        self.coord_data = self.box_cutter(tmpcd)
        
        # coordinates (terrain and lat/lon features)
        lmask = xr.open_dataset(self.homedir+'/era5_lsmask.nc').expand_dims('sample')
        self.lmask = self.box_cutter(lmask)['lsm']
        
        # open files using lists and indices
        imgtr = xr.open_mfdataset(self.list_of_cesm[indx], concat_dim='sample', combine='nested')[var]
        self.img_train = self.box_cutter(imgtr, cesm_help=True)
        
        imglb = xr.open_mfdataset(self.list_of_era5[indx], concat_dim='sample', combine='nested')[var]
        self.img_label = self.box_cutter(imglb)
        
        
    def box_cutter(self, ds1, ds2=None, cesm_help=False):
        """
        help slicing region
        if statements due to different coord names
        """ 
        if not cesm_help:
            
            if np.any(ds2):

                # slicing occurs here using data above
                ds1 = ds1.sel(y=slice(self.lat0, self.lat0 + self.dxdy), 
                              x=slice(self.lon0, self.lon0 + self.dxdy))
                ds2 = ds2.sel(y=slice(self.lat0, self.lat0 + self.dxdy), 
                              x=slice(self.lon0, self.lon0 + self.dxdy))

                return ds1, ds2

            if not np.any(ds2):

                # slicing occurs here using data above
                ds1 = ds1.sel(y=slice(self.lat0, self.lat0 + self.dxdy), 
                              x=slice(self.lon0, self.lon0 + self.dxdy))

                return ds1
            
        if cesm_help:
            
            # slicing occurs here using data above
            ds1 = ds1.sel(lat=slice(self.lat0, self.lat0 + self.dxdy), 
                          lon=slice(self.lon0, self.lon0 + self.dxdy))

            return ds1