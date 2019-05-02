import os

import time
import random
import pickle
import numpy as np

from math import ceil
from sklearn import metrics

import build_features

class preprocess:
    """
    Preprocessing pipeline to streamline data transformation processes.

    Parameters
    ----------
    data : dict
        data_dict : dict, default {}
            Data dictionary of children's EHR.
        data_dict_moms : dict, default {}
            Data dictionary of of mother's EHR at time of child's birth.
        data_dic_hist_moms : dict, default {}
            Data dictionary of mother's EHR that is within the same hospital system.
        lat_lon_dict : dict, default {}
            Data dictionary of the child's or mother's geocoded address information.
        env_dict : dict, default {}
            Data dictionary of census features.
        
        x1 : np.ndarray, default None
            Data array.
        x2 : np.ndarray, default None
            Data array that has been or will be transformed.
        y1 : np.ndarray, default None, shape (x1.shape[0], )
            Target data array.
        y2 : np.ndarray, default None, shape (x2.shape[0], )
            Target data array that has been or will be transformed.
        y1_label : np.ndarray, default None, shape (x1.shape[0], )
            Target labels corresponding to y1. Can either be 1 column or 6.
        y2_label : np.ndarray, default None, shape (x2.shape[0], )
            Target labels corresponding to y2. Can either be 1 column or 6.
        feature_headers1 : np.ndarray, default None, shape (x1.shape[1], )
            Column names corresponding to x1.
        feature_headers2 : np.ndarray, default None, shape (x2.shape[1], )
            Column names corresponding to x2.
        mrns1 : np.ndarray, default None, shape (x1.shape[0], )
            MRNs corresponding to x1.
        mrns2 : np.ndarray, default None, shape (x2.shape[0], )
            MRNs corresponding to x2.
    params : dict
        build_from_scratch : bool
            Indicator to build data from scratch or work with provided arrays.
        pred_age_low : int or float
            Lower bound for a child's age for the prediction window.
        pred_age_high : int or float
            Upper bound for a child's age for the prediction window.
        months_from : int
            Lower bound on data validity window.
        months_to int
            Upper bound on data validity window.
        percentile : bool, default False
            Filter to ensure certain types of features exist for each data point.
        label_build : str, default 'multi'
            Obesity threshold for bmi/age percentile for outcome class.
            Source: https://www.cdc.gov/obesity/childhood/defining.html
            'underweight': 0.0 <= bmi percentile < 0.05
            'normal': 0.05 <= bmi percentile < 0.85
            'overweight': 0.85 <= bmi percentile < 0.95
            'obese': 0.95 <= bmi percentile <= 1.0
            'class I severe obesity': class I severe obesity; 120% of the 95th percentile
            'class II severe obesity': class II severe obesity; 140% of the 95th percentile
            'multi': multiclass label for columns ['underweight','normal','overweight','obese','class I severe obesity','class II severe obesity']
                NOTE: will return redundant labels for obese and severe obese classes as they are a subset
        prediction : str, default 'obese'
            Label to use for prediction. Only required if label_build == 'multi'
            Source: https://www.cdc.gov/obesity/childhood/defining.html
            'underweight': 0.0 <= bmi percentile < 0.05
            'normal': 0.05 <= bmi percentile < 0.85
            'overweight': 0.85 <= bmi percentile < 0.95
            'obese': 0.95 <= bmi percentile <= 1.0
            'class I severe obesity': class I severe obesity; 120% of the 95th percentile
            'class II severe obesity': class II severe obesity; 140% of the 95th percentile
            'multi': multiclass label for columns ['underweight','normal','overweight','obese','class I severe obesity','class II severe obesity']
                NOTE: will return redundant labels for obese and severe obese classes as they are a subset
        mrns_for_filter : list, default []
            Filter for valid MRNs. All other data will be skipped.

        do_normalize : bool, default False
            Indicator for data normalization.
        do_impute : bool, default False
            Indicator to perform imputation of data.
        min_occur : int, default 0
            Minimum number of non-empty rows for a column to be considered in the final data set.
        lasso_selection : bool, default False
            Indicator to use LASSO feature selection. Performs 10 bootstrapped analyses to use only the non-zero features across all runs.
        binarize_diagnosis : bool, default True
            Indicator to binarize the diagnosis features.
        require_maternal : bool, default False
            Indicator to require that maternal data exists for a child to be considered in the final data set.
        require_vital : bool, default False
            Indicator to require that vital data exists for a child to be considered in the final data set.

        filter_str : str or list, default []
                List of feature names to filter with respect to the filter_str_thresh.
        filter_str_thresh : int or float or list, default [0.5], shape len(filter_str)
            List of filter thresholds for features in filter_str_th.
        feature_subset : list
            List of features to subset the data.

        add_time : bool, default False
            Indicator to add timeseries data.
        num_clusters : int, default 16
            Number of clusters to use.
        num_iters : int, default 100)
            Number of iterations to use.
        dist_type : str, default 'euclidean'
            Distance type.
        subset : np.ndarray, default np.array([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
            Used to determine timeseries subset.
        
        corr_vars_exclude : list, default ['Vital']
            Feature names to exclude in correlation results.
        filter_percentile_more_than_percent : int, default 5
            Filter normalized values greater than this to be 0.
        mu : list, default []
            Column means for normalization.
        std : list, default []
            Column standard deviations for normalization.

        bin_ix : array-like, default []
            List of indices for binary features or truth array for binary features.

    Returns
    -------

    """
    def __init__(self, data, params):
        # initialize the data
        self.data_dict = data.get('data_dict')
        self.data_dict_moms = data.get('data_dict_moms', {})
        self.data_dict_hist_moms = data.get('data_dict_hist_moms', {})
        self.lat_lon_dict = data.get('lat_lon_dict', {})
        self.env_dict = data.get('env_dict', {})

        # initialize the data arrays for later use
        self.x1 = data.get('x1', None)
        self.x2 = data.get('x2', None)
        self.y1 = data.get('y1', None)
        self.y2 = data.get('y2', None)
        self.y1_label = data.get('y1_label', None)
        self.y2_label = data.get('y2_label', None)
        self.feature_headers1 = data.get('feature_headers1', None)
        self.feature_headers2 = data.get('feature_headers2', None)
        self.mrns1 = data.get('mrns1', None)
        self.mrns2 = data.get('mrns2', None)
        self.ix_filter = data.get('ix_filter', None)
        self.ix_feature_filter = None
        self.corr_headers_filtered = None
        self.corr_matrix_filtered = None
        self.ix_corr_headers = None
        
        # initialize the parameters for data creation
        self.build_from_scratch = params.get('build_from_scratch', True)
        self.label_build = params.get('label_build', 'multi')
        self.prediction = params.get('prediction', 'obese')
        self.percentile = params.get('percentile', False)
        self.mrns_for_filter = params.get('mrns_for_filter', [])
        self.pred_age_low = params.get('pred_age_low', 4.5)
        self.pred_age_high = params.get('pred_age_high', 5.5)
        self.months_from = params.get('months_from', 0)
        self.months_to = params.get('months_to', 24)

        # initialize the data prep arguments
        self.do_normalize = params.get('do_normalize', False)
        self.do_impute = params.get('do_impute', False)
        self.min_occur = params.get('min_occur', 0)
        self.lasso_selection = params.get('lasso_selection', False)
        self.binarize_diagnosis = params.get('binarize_diagnosis', True)
        self.require_maternal = params.get('require_maternal', False)
        self.require_vital = params.get('require_vital', False)

        self.filter_str = params.get('filter_str', [])
        self.filter_str_thresh = params.get('filter_str_thresh', [0.5])
        self.feature_subset = params.get('variable_subset', ['Vital'])

        self.add_time = params.get('add_time', False)
        self.num_clusters = params.get('num_clusters', 16)
        self.num_iters = params.get('num_iters', 100)
        self.dist_type = params.get('dist_type', 'euclidean')
        self.corr_vars_exclude = params.get('corr_vars_exclude', ['Vital'])

        self.filter_percentile_more_than_percent = params.get('filter_percentile_more_than_percent', 5)
        self.mu = params.get('mu', [])
        self.std = params.get('std', [])

        self.bin_ix = params.get('bin_ix', [])
        self.feature_info = params.get('feature_info', False)
        self.subset = params.get('subset', np.array([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]))

        self.label_ix = {'underweight':0,'normal':1,'overweight':2,'obese':3,'class I severe obesity':4,'class II severe obesity':5, 'none':6}

    def build_data(self):
        """
        Calls build_features.call_build_function to create the data arrays.

        Parameters
        ----------
        self : preprocess() data object
            data_dict : dict, default {}
                Data dictionary of children's EHR.
            data_dict_moms : dict, default {}
                Data dictionary of of mother's EHR at time of child's birth.
            data_dic_hist_moms : dict, default {}
                Data dictionary of mother's EHR that is within the same hospital system.
            lat_lon_dict : dict, default {}
                Data dictionary of the child's or mother's geocoded address information.
            env_dict : dict, default {}
                Data dictionary of census features.
            pred_age_low : int or float
                Lower bound for a child's age for the prediction window.
            pred_age_high : int or float
                Upper bound for a child's age for the prediction window.
            months_from : int
                Lower bound on data validity window.
            months_to int
                Upper bound on data validity window.
            percentile : bool, default False
                Filter to ensure certain types of features exist for each data point.
            prediction : str, default 'obese'
                Obesity threshold for bmi/age percentile for outcome class.
                Source: https://www.cdc.gov/obesity/childhood/defining.html
                'underweight': 0.0 <= bmi percentile < 0.05
                'normal': 0.05 <= bmi percentile < 0.85
                'overweight': 0.85 <= bmi percentile < 0.95
                'obese': 0.95 <= bmi percentile <= 1.0
                'class I severe obesity': class I severe obesity; 120% of the 95th percentile
                'class II severe obesity': class II severe obesity; 140% of the 95th percentile
                'multi': multiclass label for columns ['underweight','normal','overweight','obese','class I severe obesity','class II severe obesity']
                    NOTE: will return redundant labels for obese and severe obese classes as they are a subset
            mrns_for_filter : list, default []
                Filter for valid MRNs. All other data will be skipped.

        Returns
        -------
        x1 : np.ndarray, shape (len(data_dict), len(feature_headers))
            Data array.
        y1 : np.ndarray, shape (len(data_dict), )
            Target data.
        y1_label : np.ndarray, shape (len(data_dict), 1 or 6)
            Classification labels corresponding to y1. 6 columns if prediction == 'multi', otherwise 1 column.
        feature_headers : np.ndarray (x1.shape[1], )
            Names of features corresponding to x1.
        mrns1 : np.ndarray, shape (len(data_dict), )
            MRNs corresponding to patients in the data.
        NOTE: if mrns_for_filter is not empty, then there will be len(mrns_for_filter) less rows in x1, y1, y1_label, and mrns1.
        """
        args = (
            self.data_dict, self.data_dict_moms, self.data_dict_hist_moms, self.lat_lon_dict, self.env_dict,
            self.pred_age_low, self.pred_age_high, self.months_from, self.months_to, self.percentile
            )
        kwargs = {'prediction': self.label_build, 'mrnsForFilter': self.mrns_for_filter}
        self.x1, self.y1, self.y1_label, feature_headers1, self.mrns1 = build_features.call_build_function(*args, **kwargs)
        self.feature_headers1 = np.array(feature_headers1)
        return self.x1, self.y1, self.y1_label, self.feature_headers1, self.mrns1
        
    def binarize(self, **kwargs):
        """
        Binarize all diagnosis codes in the data

        Parameters
        ----------
        self : preprocess() data object
            x2 : np.ndarray
                Data array.
            feature_headers2 : np.ndarray, shape (x2.shape[1], )
                Feature names for x2.

        Returns
        -------
        x2 : np.ndarray
            Copy of preprocess.x2 with binarized diagnosis columns.
        bin_ix : np.ndarray, shape (x2.shape[1], )
            Truth array to indicate which columns contain diagnosis data.
        """
        self.x2 = kwargs.get('x2', self.x2)
        self.feature_headers2 = kwargs.get('feature_headers2', self.feature_headers2)

        bin_ix = np.array(['Diagnosis' in h for h in self.feature_headers2])
        print('{:,.0f} features are binary'.format(bin_ix.sum()))
        x2 = np.copy(self.x2)
        x2[:, bin_ix] = (x2[:, bin_ix] > 0) * 1.
        self.x2 = x2
        return self.x2, bin_ix

    def filter_train(self, **kwargs):
        """
        Filter the training set based off filter_str, filter_str_thresh, require_maternal, require_vital, and valid bmi readings for a prediction.

        Parameters
        ----------
        self : preprocess() data object
            filter_str : str or list, default []
                List of feature names to filter with respect to the filter_str_thresh.
            filter_str_thresh : int or float or list, default [0.5], shape len(filter_str)
                List of filter thresholds for features in filter_str_th.
            require_maternal : bool, default False
                Require corresponding maternal data for a child to be considered in the final data set.
            require_vital : bool, default False
                Require vital readings data for a child to be considered in the final data set.
            x2 : np.ndarray
                Data array.
            y2 : np.ndarray, shape (x2.shape[0],)
                Target data array.
            feature_headers2 : np.ndarray, shape (x2.shape[1],)
                Feature names.

        Returns
        -------
        self
        """
        self.x2 = kwargs.get('x2', self.x2)
        self.y2 = kwargs.get('y2', self.y2)
        self.feature_headers2 = kwargs.get('feature_headers2', self.feature_headers2)
        self.filter_str = kwargs.get('filter_str', self.filter_str)
        self.filter_str_thresh = kwargs.get('filter_str_thresh', self.filter_str_thresh)
        self.require_maternal = kwargs.get('require_maternal', self.require_maternal)
        self.require_maternal = kwargs.get('require_maternal', self.require_maternal)

        if self.filter_str.__class__ == list:
            pass
        else:
            self.filter_str = [self.filter_str]

        if self.feature_headers2.__class__ != np.ndarray:
            self.feature_headers2 = np.array(self.feature_headers2)

        if len(self.filter_str_thresh) != len(self.filter_str):
            self.filter_str_thresh = []
        if len(self.filter_str_thresh) == 0 :
            self.filter_str_thresh = [0.5]*len(self.filter_str) #make it binary

        print('Original cohort size is: {0:,d}, number of features: {1:,d}'.format(self.x2.shape[0], len(self.feature_headers2)))

        index_finder_filterstr = np.zeros(len(self.feature_headers2))
        index_finder_maternal = [f.startswith('Maternal') for f in self.feature_headers2]

        for i, fstr in enumerate(self.filter_str):
            index_finder_filterstr_tmp = np.array([h.startswith(fstr) for h in self.feature_headers2])
            if index_finder_filterstr_tmp.sum() > 1:
                print('ALERT: filter returned more than one feature: {0:s}'.format(fstr))
                index_finder_filterstr_tmp = np.array([h == fstr for h in self.feature_headers2])
                print('using all instances of features starting with: {0:s}, set filter to one of the following if incorrect: {0:s}'.format(fstr, str(headers[index_finder_filterstr_tmp])))
                    
            index_finder_filterstr += index_finder_filterstr_tmp
            print('total number of people who have: {0:s} is: {1:,.0f}'.format(str(headers[index_finder_filterstr_tmp]), (x[:,index_finder_filterstr_tmp].ravel() > filterSTRThresh[i]).sum()))

        index_finder_filterstr = (index_finder_filterstr > 0)

        non_dem_feats = ['Diagnosis', 'Lab', 'Maternal Diagnosis', 'Maternal Diagnosis', 'Maternal Lab History', 'Maternal Procedure History', 'Maternal Vital', 'Newborn Diagnosis', 'Vital']
        non_dem_filt = [any(f.startswith(ndf) for ndf in non_dem_feats) for f in self.feature_headers2]
        
        ix_vital = ((self.x2[:, non_dem_filt] != 0.).sum(axis=1) > 0)
        ix_maternal = ((self.x2[:,index_finder_maternal] != 0).sum(axis=1) >= 1)
        ix_valid_bmi = (self.y2 > 10) & (self.y2 < 40)
        ix_user_filter = (((self.x2[:,index_finder_filterstr] > np.array(self.filter_str_thresh)).sum(axis=1) >= index_finder_filterstr.sum()).ravel())

        self.ix_filter = ix_valid_bmi
        if len(self.filter_str) != 0 and self.percentile == False:
            self.ix_filter = self.ix_filter & ix_user_filter        
        if self.require_maternal:
            self.ix_filter = self.ix_filter & ix_maternal
        if self.require_vital:
            self.ix_filter = self.ix_filter & ix_vital

        self.x2 = self.x2[self.ix_filter, :]
        self.y2 = self.y2[self.ix_filter]
        self.y2_label = self.y2_label[self.ix_filter]
        self.mrns2 = self.mrns2[self.ix_filter]
        print('total number of children who have a valid BMI measured (10 > BMI < 40): {0:,.0f}'.format(ix_valid_bmi.sum()))
        print('total number of children who have all filtered variables: {0:,.0f}'.format(ix_user_filter.sum()))
        print('total number of children who have maternal data available: {0:,.0f}'.format(ix_maternal.sum()))
        print('total number of children who have vital data available: {0:,.0f}'.format(ix_vital.sum()))
        print('final number of children to be considered: {0:,.0f}'.format(self.ix_filter.sum()))
        return self.x2, self.y2, self.y2_label, self.mrns2, self.ix_filter

    def normalize(self, **kwargs):
        """
        Function to normalize the data.
        
        Parameters
        ----------
        self : preprocess() data object
            x2 : np.ndarray
                Data array.
            filter_percentile_more_than_percent : int or float, default 5
                Default inputs greater this percent to be 0.
            mu : list, default []
                List of means for normalizing data. Will determine if not provided.
            std : list, default []
                List of standard deviations for normalizing data. Will determine if not provided.
            bin_ix : list, default []
                List of column indices for binary variables. Will determine if not provided.

        Returns
        -------
        self.x2 : np.ndarray
            Data aray.
        self.mu : np.ndarray
            Array of means.
        self.std : np.ndarray
            Array of standard deviations.
        """
        self.x2 = kwargs.get('x2', self.x2)
        self.bin_ix = kwargs.get('bin_ix', self.bin_ix)
        self.mu = kwargs.get('mu', self.mu)
        self.std = kwargs.get('std', self.std)
        self.filter_percentile_more_than_percent = kwargs.get('filter_percentile_more_than_percent', self.filter_percentile_more_than_percent)

        unobserved = (self.x2 == 0.) * 1.0
        if len(self.bin_ix) == 0:
            self.bin_ix = (self.x2.min(axis=0) == 0) & (self.x2.max(axis=0) == 1)
        xcop = self.x2 * 1.0
        xcop[xcop == 0] = np.nan
        if len(self.mu) == 0:
            self.mu = np.nanmean(xcop, axis=0)
            self.mu[self.bin_ix] = 0.0
            self.mu[np.isnan(self.mu)] = 0.0
        if len(self.std) == 0:
            self.std = np.nanstd(xcop, axis=0)
            self.std[self.std == 0] = 1.0
            self.std[self.bin_ix] = 1.0
            self.std[np.isnan(self.std)] = 1.0
        normed_x = (self.x2 != 0) * ((self.x2 - self.mu) / self.std * 1.0)
        normed_x[abs(normed_x) > self.filter_percentile_more_than_percent] = 0
        self.x2 = normed_x
        return self.x2, self.mu, self.std

    def filter_min_occurrences(self, **kwargs):
        """
        Filter columns that have less than min_occur ocurrences.

        Parameters
        ----------
        self : preprocess data object
            x2 : array
                Data array.
            feature_headers2 : np.ndarray, shape (x2.shape[1], )
                Feature names corresponding to x2.
            min_occur : int, default 0
                Minimum number of occurrences for a feature to be considered in the data.

        Returns
        -------
        self.x2 : np.ndarray
            Filtered data array.
        feature_filter : np.ndarray
            Truth array of features with more than min_occur features.
        """
        self.x2 = kwargs.get('x2', self.x2)
        self.feature_headers2 = kwargs.get('feature_headers2', self.feature_headers2)
        self.min_occur = kwargs.get('min_occur', self.min_occur)

        feature_filter = (np.count_nonzero(x2, axis=0) >= min_occur)
        self.feature_headers2 = self.feature_headers2[feature_filter]
        self.x2 = self.x2[:, feature_filter]
        print('{0:,d} features filtered with number of occurrences less than {1:,d}'.format(feature_filter.sum(), min_occur))
        return self.x2, feature_filter

    def variable_subset(self, **kwargs):
        """
        Function to remove specific variables from the data.

        Parameters
        ----------
        self : preprocess() data object
            x2 : np.ndarray
                Data array.
            feature_subset : list, default ['Vital']
                List of features to subset the data
            feature_headers2 : np.ndarray, shape (x2.shape[1])
                Feature names corresponding to the columns of x2.

        Returns
        -------
        self.x2 : np.ndarray
            Data array.
        self.feature_headers2 : np.ndarray
            Array of feature names.
        ix_subset : np.ndarray
            Truth array of features to be included and excluded.
        """
        self.x2 = kwargs.get('x2', self.x2)
        self.feature_subset = kwargs.get('feature_subset', self.feature_subset)
        self.feature_headers2 = kwargs.get('feature_headers2', self.feature_headers2)

        ix_subset = np.array([any(x in self.feature_subset for x in (f, f.split(':')[0].strip())) for f in self.feature_headers2])
        ix_subset = np.zeros(len(self.feature_headers2), dtype=bool)
        for i, ft in enumerate(self.feature_headers2):
            for subset in self.feature_subset:
                if ft == subset or ft.startswith(subset):
                    ix_subset[i] = True
                    break
        print('Filtered feature size from {0:,d} variables to {1:,.0f}\n'.format(self.x2.shape[1], ix_subset.sum()))
        self.x2 = self.x2[:, ix_subset]
        self.feature_headers2 = self.feature_headers2[ix_subset]
        return self.x2, self.feature_headers2, ix_subset

    def lasso_filter(self):
        """
        Filter any columns that have zeroed out feature weights

        Parameters
        ----------
        self : preprocess() data object
            x2 : np.ndarray
                Data array.
            y2 : np.ndarray
                Target data.
            y2_label : np.ndarray
                Target labels.
            feature_headers2 : np.ndarray
                Column names associated with x. Not actually used in this process

        Returns
        -------
        self.x2 : np.ndarray
            Filtered data array.
        self.feature_headers2 : np.ndarray
            Column feature names.
        ix_lasso_filter : np.ndarray
            Truth array of features to be included and excluded.
        """
        N = self.x2.shape[0]
        ix_list = np.arange(0,N).tolist()
        random.shuffle(ix_list)
        ix_list = ix_list[0: int(N * 0.8)]
        iters = 10
        hyperparamlist = [0.001, 0.005, 0.01, 0.1] #[alpha]
        arguments = []
        for it in range(iters):
            ix_filter = ix_list.copy()
            random.shuffle(ix_filter)
            ix_filter = ix_filter[0: int(N * 0.9)]
            ix_train = ix_filter[0:int(len(ix_filter) * 0.7)]
            ix_test = ix_filter[int(len(ix_filter) * 0.7):]
            arguments.append([it, ix_train, ix_test, hyperparamlist])

        node_count = ceil(cpu_count() * 0.8)
        coefs = np.zeros((iters, self.feature_headers2.shape[0]))
        print(' | '.join(('Iter', 'N non-zero', 'N zero', 'AUC Train', 'AUC Test', 'Explained Var Train', 'Explained Var Test')))
        with ProcessPoolExecutor(max_workers=node_count) as p:
            outputs = p.map(self._run_lasso_single, arguments)
            for i, coef in outputs:
                coefs[i] = coef
        ix_lasso_filter = (coefs.abs().mean(axis=0) > 0)
        self.x2 = self.x2[:, ix_lasso_filter]
        self.feature_headers2 = self.feature_headers2[ix_lasso_filter]
        print('Only considering the below features after LASSO filtering:')
        print(self.feature_headers2.tolist())
        return self.x2, self.feature_headers2, ix_lasso_filter

    def _run_lasso_single(self, args):
        from sklearn.linear_model import Lasso
        run, ix_train, ix_test, hyperparamlist = args
        xtrain = self.x2[ix_train]
        xtest = self.x2[ix_test]
        ytrain = self.y2[ix_train]
        ytest = self.y2[ix_test]
        ytrainlabel = self.y2_label[ix_train]
        ytestlabel = self.y2_label[ix_test]

        best_score = -1
        best_alpha = -1
        for alpha_i in hyperparamlist:
            clf = Lasso(alpha=alpha_i, max_iter=1000)
            clf.fit(xtrain, ytrain)
            exp_var_test = metrics.explained_variance_score(ytest, clf.predict(xtest))
            if exp_var_test > best_score:
                best_score = exp_var_test
                best_alpha = alpha_i
        clf = Lasso(alpha=best_alpha)
        clf.fit(xtrain,ytrain)


        fpr, tpr, thresholds = metrics.roc_curve(ytrainlabel, clf.predict(xtrain))
        auc_train = metrics.auc(fpr, tpr)
        exp_var_train = metrics.explained_variance_score(ytrain, clf.predict(xtrain))
        fpr, tpr, thresholds = metrics.roc_curve(ytrainlabel, clf.predict(xtrain))
        auc_test = metrics.auc(fpr, tpr)
        exp_var_test = metrics.explained_variance_score(ytrain, clf.predict(xtrain))
        non_zero = int((clf.coef_ != 0).sum())
        zero = int((clf.coef_ == 0).sum())
        string = ' | '.join((
            '{:,d}'.format(run).rjust(4), '{:,d}'.format(non_zero).rjust(10), '{:,d}'.format(zero).rjust(6), '{:4.3f}'.format().rjust(9), 
            '{:4.3f}'.format(x).rjust(8), '{:4.3f}'.format(x).rjust(19), '{:4.3f}'.format(x).rjust(18)
            ))
        print(string)
        return run, clf.coef_

    def filter_correlations_via(corr_headers, corr_matrix, corr_vars_exclude):
        """
        Parameters
        ----------
        corr_headers : np.ndarray
            Feature names of correlation matrix.
        corr_matrix : np.ndarray
            np.corcoef matrix.
        corr_vars_exclude : array-like
            List/array/tuple of features to exclude from the correlation matrix.

        Returns
        -------
        corr_matrix : np.ndarray
            np.corcoef matrix.
        corr_headers : np.ndarray
            Feature names of correlation matrix.
        """
        ix_header = np.ones((len(corr_headers)), dtype=bool)
        if len(corr_headers) == 1:
            corr_matrix = np.array([[corr_matrix]])
        for ind, item in enumerate(corr_headers):
            if (item in corr_vars_exclude) or sum([item.startswith(ii) for ii in corr_vars_exclude]) > 0 :
                ix_header[ind] = False
        print('filtered correlated features to: {0:,d}'.format(ix_header.sum()))
        return corr_matrix[:, ix_header], corr_headers[ix_header]

    def autoencoder_impute(self, hidden_nodes=100, latent_dim=10, **kwargs):
        """
        Parameters
        ----------
        self : preprocess() data object
            x2 : np.ndarray
                Data array.
            bin_ix : np.ndarray
                Truth array indicating binary columns.

        Returns
        -------
        self.x2 : np.ndarray
            Imputed data array.
        """
        try:
            import auto_encoder
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.autograd import Variable
        except:
            print('imputation requires pytorch. please install and make sure you can import it')
            raise

        self.x2 = kwargs.get('x2', self.x2)
        self.bin_ix = kwargs.get('bin_ix', self.bin_ix)
        
        cuda = torch.cuda.is_available()
        print('Using GPU: {}'.format(cuda))    
        
        cont_ix = (self.bin_ix == False)
        non_zero_ix = (self.x2.sum(axis=0) != 0)
        old_shape = self.x2.shape
        bin_ix = np.array([b & nz for b, nz in zip(self.bin_ix, non_zero_ix)])
        cont_ix = np.array([~b & nz for b, nz in zip(self.bin_ix, non_zero_ix)])
        # x = self.x2[:, non_zero_ix]
        x_cont = self.x2[:, cont_ix]
        x_bin = self.x2[:, bin_ix]
        print(sum(bin_ix), sum(cont_ix), hidden_nodes)

        model = auto_encoder.VariationalAutoencoder(x_bin.shape[1], x_cont.shape[1])
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1, 2):
            model.train()
            train_loss = 0
            for ix in range(len(self.x2)):
                databoth = Variable(torch.from_numpy(np.hstack([x_bin[ix], x_cont[ix]])).float()).cuda() if cuda else Variable(torch.from_numpy(np.hstack([x_bin[ix], x_cont[ix]])).float())
                optimizer.zero_grad()
                impute_x, mu, logvar = model(databoth)
                loss = auto_encoder.vae_loss(impute_x, databoth, mu, logvar, x_bin.shape[1], x_cont.shape[1])
                loss.backward()
                train_loss += loss.data[0]
                optimizer.step()
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.x2)))
        
        model.eval()
        xout = np.zeros(self.x2.shape)
        for ix in range(len(self.x2)):
            Variable(torch.from_numpy(np.hstack([x_bin[ix], x_cont[ix]])).float()).cuda() if cuda else Variable(torch.from_numpy(np.hstack([x_bin[ix], x_cont[ix]])).float())
            impute_x, mu, logvar = model(databoth)
            impute_x = impute_x.cpu().detach().numpy()
            xout[ix, bin_ix] = impute_x[:, :x_bin.shape[1]]
            xout[ix, cont_ix] = impute_x[:, x_bin.shape[1]:]

        self.x2[:,non_zero_ix] = xout[:,non_zero_ix]

        # autoencoder = auto_encoder.AutoencoderConinBinar(x_bin.shape[1], x_cont.shape[1], hidden_nodes)
        # optimizer = optim.SGD(autoencoder.parameters(), lr=0.5)
        # np.random.seed(0)
        # lossfuncBin = nn.BCELoss()
        # lossfunccont = nn.MSELoss()
        # loss_list = []
        # for epoch in range(1, 200):
        #     autoencoder.train()
        #     for ix in range(len(self.x2)):
        #         databin = Variable(torch.from_numpy(x_bin[ix]).float())
        #         datacont = Variable(torch.from_numpy(x_cont[ix]).float())
        #         databoth = Variable(torch.from_numpy(np.hstack([x_bin[ix], x_cont[ix]])).float())
        #         optimizer.zero_grad()
        #         xoutBin, xoutCont = autoencoder(databoth)
        #         loss = lossfuncBin(xoutBin, databin) + lossfunccont(xoutCont, datacont)
        #         loss_list.append(loss)
        #         loss.backward()
        #         optimizer.step()

        # autoencoder.eval()
        # xout = np.zeros(self.x2.shape)
        # for ix in range(len(self.x2)):
        #     databoth = Variable(torch.from_numpy(np.hstack([x_bin[ix], x_cont[ix]]))).float()
        #     outbin, outcont = autoencoder(databoth)
        #     xout[ix, bin_ix] = outbin.data.numpy()
        #     xout[ix, cont_ix] = outcont.data.numpy()

        # # xfinal = np.zeros(old_shape)
        # # xfinal[:,non_zero_ix] = xout
        # self.x2[:,non_zero_ix] = xout[:,non_zero_ix]
        return self.x2

    def preprocess(self, test_ix=[], test_size=0.2):
        # build the data
        if self.build_from_scratch:
            self.x1, self.y1, self.y1_label, self.feature_headers1, self.mrns1 = self.build_data()

        # create a copy of the data for manipulation
        self.x2 = np.copy(self.x1)
        self.y2 = np.copy(self.y1)
        self.y2_label = np.copy(self.y1_label) if self.label_build != 'multi' else np.copy(self.y1_label[:, self.label_ix[self.prediction]])
        self.feature_headers2 = np.copy(self.feature_headers1)
        self.mrns2 = np.copy(self.mrns1)
        
        # binarize
        if self.binarize_diagnosis:
            self.x2, self.bin_ix = self.binarize()
        
        # filter the data with respect to the provided filtering criteria and data requirements
        self.x2, self.y2, self.y2_label, self.mrns2, self.ix_filter = self.filter_train()

        #normalize the data
        if self.do_impute or self.do_normalize or self.add_time:
            self.normalize()

        # perform imputation
        if self.do_impute:
            x2 = self.autoencoder_impute()

        # if add_time:
        #     x2, feature_headers, centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew = add_temporal_features(x2, feature_headers, num_clusters, num_iters, y2, y2label, dist_type, True, mux, stdx, do_impute, subset)
        # else:
        #     centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew = ['NaN']*7

        if self.min_occur > 0:
            self.x2, feature_filter = self.filter_min_occurrences()

        if len(self.feature_subset) != 0:
            self.x2, self.feature_headers2, ix_subset = self.variable_subset()

        if self.lasso_selection:
            self.x2, self.feature_headers2, ix_lasso_filter = self.lasso_filter()

        self.ix_feature_filter = np.array([f in self.feature_headers2 for f in self.feature_headers1])
        print('output is: average: {0:4.3f}, min: {1:4.3f}, max: {2:4.3f}'.format(self.y2.mean(), self.y2.min(), self.y2.max()))
        print('total patients: {0:,d}, positive: {1:,.2f}, negative: {2:,.2f}'.format(self.y2.shape[0], self.y2_label.sum(), self.y2.shape[0]-self.y2_label.sum()))
        # return self.x2, self.y2, self.y2_label, self.mrns2, self.feature_headers2, self.ix_filter

        # corr_headers = np.array(self.feature_headers2)
        # corr_matrix = np.corrcoef(self.x2.transpose())
        # self.corr_headers_filtered, self.corr_matrix_filtered, self.ix_corr_headers = filter_correlations_via(corr_headers, corr_matrix, corr_vars_exclude, print_out=not delay_print)
        # print('corr matrix is filtered to size: '+ str(corr_matrix_filtered.shape))

        # if delay_print:
        #     reporting += 'output is: average: {0:4.3f}, min: {1:4.3f}, max: {2:4.3f}\n'.format(y2.mean(), y2.min(), y2.max())
        #     reporting += 'total patients: {0:,d}, positive: {1:,d}, negative: {2:,d}\n'.format(y2.shape[0], y2label.sum(), y2.shape[0]-y2label.sum())
        #     reporting += 'normalizing output...\n'
        # else:
        #     print('output is: average: {0:4.3f}, min: {1:4.3f}, max: {2:4.3f}'.format(y2.mean(), y2.min(), y2.max()))
        #     print('total patients: {0:,d}, positive: {1:,.2f}, negative: {2:,.2f}'.format(y2.shape[0], y2label.sum(), y2.shape[0]-y2label.sum()))
        #     print('normalizing output...')
        # y2 = (y2-y2.mean())/y2.std()

        # reporting += 'Predicting BMI at age: '+str(agex_low)+ ' to '+str(agex_high)+ ' years, from data in ages: '+ str(months_from)+' - '+str(months_to) + ' months\n'
        # if filterSTR != '':
        #     reporting += 'filtering patients with: '+str(filterSTR)+'\n'

        # reporting += 'total size: {0:,d} x {1:,d}'.format(x2.shape[0], x2.shape[1])
        # print(reporting)
        # if (ix_filter.sum() < 50):
        #     print('Not enough subjects. Next.')
        #     return (filterSTR, [])
        # return x2, y2, y2label, mrns, ix_filter, feature_headers, corr_headers_filtered, corr_matrix_filtered, ix_corr_headers





# def add_temporal_features(x2, feature_headers, num_clusters, num_iters, y2, y2label, dist_type='eucledian', cross_valid=True, mux=None, stdx=None, do_impute=False, subset=[]):
#     if isinstance(feature_headers, list):
#         feature_headers = np.array(feature_headers)
#     header_vital_ix = np.array([h.startswith('Vital') for h in feature_headers])
#     headers_vital = feature_headers[header_vital_ix]
#     x2_vitals = x2[:, header_vital_ix]
#     mu_vital = mux[header_vital_ix]
#     std_vital = stdx[header_vital_ix]
#     import timeseries
#     xnew, hnew, muxnew, stdxnew = timeseries.construct_temporal_data(x2_vitals, headers_vital, y2, y2label, mu_vital, std_vital, subset)
#     centroids, assignments, trendArray, standardDevCentroids, cnt_clusters, distances = timeseries.k_means_clust(xnew, num_clusters, num_iters, hnew, distType=dist_type, cross_valid=cross_valid)
#     trendArray[trendArray!=0] = 1
#     trend_headers = ['Trend:'+str(i)+' -occ:'+str(cnt_clusters[i]) for i in range(0, len(centroids))]
#     return np.hstack([x2, trendArray]), np.hstack([feature_headers , np.array(trend_headers)]), centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew
