# -*- coding: utf-8 -*-
"""
Created on Fri May 04 10:26:49 2018

@author: Mostafa Meliani <melimostafa@gmail.com>
Multi-Fidelity co-Kriging: recursive formulation with autoregressive model of 
order 1 (AR1)
"""

from __future__ import division
import numpy as np
from smt.surrogate_models import KrgBased, KRG, KPLS
from types import FunctionType
from smt.utils.kriging_utils import l1_cross_distances, componentwise_distance
from scipy.linalg import solve_triangular
from scipy import linalg
from sklearn.metrics.pairwise import manhattan_distances
import copy 
from functools import partial
from smt.utils.kriging_utils import standardization
"""
The MFK class.
"""



class MFKPLS(KrgBased):

    """
    - MFK
    """
    def _initialize(self):
        super(MFKPLS, self)._initialize()
        declare = self.options.declare
        declare('model', 'KRG',\
                values=('KRG', 'KPLS'), desc='model designation')
        declare('rho_regr', 'constant',types=FunctionType,\
                values=('constant', 'linear', 'quadratic'), desc='regr. term')
        declare('theta0', None, types=(list, np.ndarray), \
                desc='Initial hyperparameters')
        declare('optim_var', True, types = bool, \
                values = (True, False), \
                desc ='Turning this option to True, forces variance to zero at HF samples ')
        declare('n_comp', None, types = int, desc='nb of components')
        
        
        self.name = 'MFKPLS'
        if self.options['rho_regr'] != 'constant':
            raise 'regression rho supported for constant'
    
    
    def create_trained_model(self, xt, yt, LF_Train = True, eval_noise = True):
        flag = self.options['noise0'] and eval_noise
            
        
        if self.options['model']== 'KRG':
            print flag, self.options['model']
            mypoly = self.options['poly'] if LF_Train else self.poly_mfk
            
            model = KRG(theta0 = self.options['theta0'], eval_noise = flag,
                        noise0 = self.options['noise0'] , poly = mypoly,
                        corr = self.options['corr'], normalize = False, print_global = False)
                        #, data_dir = self.options['data_dir']    
            model.set_training_values(xt, yt)
            model.train()
            
            return model
        elif self.options['model']== 'KPLS':
            mypoly = self.options['poly'] if LF_Train else self.poly_mfk
            nugget_mutiplier = 50. if LF_Train else 10.
            model = KPLS(n_comp =self.options['n_comp'], theta0 = self.options['theta0'], eval_noise = flag,
                        noise0 = self.options['noise0'], poly = mypoly,
                        corr = self.options['corr'], normalize = False, 
                        print_global = False, nugget = nugget_mutiplier)
                        #, data_dir = self.options['data_dir']    
            model.set_training_values(xt, yt)
            model.train()
            
            return model
        else :
            raise 'the model chosen is not supported as of now'
    
   

    def poly_mfk(self, X_HF):
        
        F_rho = self.options['rho_regr'](X_HF)
        q = F_rho.shape[1]
        F = self.options['poly'](X_HF)
        poly = np.hstack((F_rho*np.dot(self.LF_model.predict_values(X_HF),
                                              np.ones((1,q))), F))
        return poly
    
    
    def _new_train(self):
        self._check_param()
        
#         print self.training_points
        i=0
        xt_LF = self.training_points[i][0][0]
        yt_LF = self.training_points[i][0][1]
        xt_HF = self.training_points[None][0][0]
        yt_HF = self.training_points[None][0][1]
        print 'shapes :', xt_LF.shape, xt_HF.shape, yt_LF.shape, yt_HF.shape
        
        _, _, self.X_mean, self.y_mean, self.X_std, \
            self.y_std = standardization(np.concatenate([xt_LF, xt_HF],axis=0), \
                                         np.concatenate([yt_LF, yt_HF],axis=0))
        
#         self.X_mean, self.y_mean, self.X_std, \
#             self.y_std = np.zeros((xt_LF.shape[1],)), 0.,np.ones((xt_LF.shape[1],)),1.

        xt_LF = (xt_LF-self.X_mean)/self.X_std
        xt_HF = (xt_HF-self.X_mean)/self.X_std
        yt_LF = (yt_LF-self.y_mean)/self.y_std
        yt_HF = (yt_HF-self.y_mean)/self.y_std
        
        self.LF_model = self.create_trained_model(xt_LF,yt_LF)
        
        if self.options['eval_noise'] and self.options['optim_var']:
             
            self.LF_model = self.create_trained_model(xt_LF,\
                                                self.LF_model.predict_values(xt_LF), 
                                                eval_noise = False)
            
        
        self.HF_model = self.create_trained_model(xt_HF, yt_HF, LF_Train=False)
        if self.options['eval_noise'] and self.options['optim_var']:
            
            self.HF_model = self.create_trained_model(xt_HF,\
                                                self.HF_model.predict_values(xt_HF), \
                                                LF_Train=False, 
                                                eval_noise = False)
        
        rho = self.HF_model.optimal_par['beta'][0]
        print rho
        
        
    def _predict_values(self, x, descale = True):
        x = (x-self.X_mean)/self.X_std
        return self.HF_model.predict_values(x)*self.y_std+self.y_mean
    
    
    def _predict_variances(self, x):
        x = (x-self.X_mean)/self.X_std
        rho = self.HF_model.optimal_par['beta'][0]
        return (rho**2*self.LF_model.predict_variances(x)+ \
            self.HF_model.predict_variances(x))*self.y_std**2
    
    def predict_variances_all_levels(self, x):
        x = (x-self.X_mean)/self.X_std
        rho = self.HF_model.optimal_par['beta'][0]
        MSE = np.concatenate((self.LF_model.predict_variances(x)*self.y_std**2,
                             self._predict_variances(x)),
                              axis = 1)
        return MSE, [np.array([rho**2])]
        
            
    def _predict_derivatives(self, x, kx):
        x = (x-self.X_mean)/self.X_std
        n_eval = x.shape[0]
        rho = self.HF_model.optimal_par['beta'][0]
        p1 = np.ravel(rho*self.LF_model.predict_derivatives(x, kx))*self.y_std/self.X_std
        
        
        F = self.HF_model.F
        C = self.HF_model.optimal_par['C']
        dx = manhattan_distances(x, Y=self.HF_model.X_norma, sum_over_features=False)
        d = self.HF_model._componentwise_distance(dx)
        r_ = self.options['corr'](self.HF_model.optimal_theta, d).reshape(n_eval,\
                                                                          self.HF_model.nt)
        

        beta = self.HF_model.optimal_par['beta']
        gamma = self.HF_model.optimal_par['gamma']
        

        d_dx=x[:,kx].reshape((n_eval,1))-self.HF_model.X_norma[:,kx].reshape((1,self.HF_model.nt))
        theta = self.HF_model.optimal_theta
            # scaled predictor
        if self.options['model']=='KPLS':
            theta = np.sum(self.HF_model.optimal_theta * self.HF_model.coeff_pls**2,axis=1)
        
        p2 = np.ravel(-2*theta[kx]*np.dot(d_dx*r_,gamma))*self.y_std/self.X_std
        
        
        return p1 + p2
    
    def predict_jacobian(self, x):
#         print 'jacobian : ', x
        jac = np.zeros(x.shape)
        for dim in range(x.shape[1]):
            jac[:,dim] = self._predict_derivatives(x, dim)
        return jac
    
    def get_noise(self):
        return [self.LF_model.noise, self.HF_model.noise]
    
    
    
