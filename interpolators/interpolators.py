from __future__ import division
import numpy as np
import numbers

import matplotlib.pyplot as pl
import piplot as pipl

import itertools as itt

from toolz import reduce



def mysum(l):
    return reduce( lambda x,y: x+y, l)

### Splines
def CR_spline_coeffs(t0, t1, t2, t3, t, dif = False):
    """
    Returns the coefficients of the control points p0, p1, p2, p3 
    in the Catmull-Rom spline.
    t is between t1, t2
    """
    p0, p1, p2, p3 = np.eye(4)
    
    ## the tangent vectors
    v1 = (p2 - p0) / (t2 - t0)
    v2 = (p3 - p1) / (t3 - t1)
    
    ## we normalize the time in the interval [ t1, t2]
    v1 = (t2 - t1) * v1
    v2 = (t2 - t1) * v2
    t = (t - t1) / (t2 - t1)
    
    if not dif:
        return (2* t**3 - 3 * t**2 + 1) * p1 + ( t**3 - 2* t**2 + t) * v1 + ( - 2* t**3 + 3* t**2) * p2 + ( t**3 - t**2) * v2
    else:
        return 1/(t2 - t1) * ( (6* t**2 - 6 * t) * p1 + ( 3*t**2 - 4* t + 1) * v1 + 
                                ( - 6* t**2 + 6* t) * p2 + ( 3*t**2 - 2*t) * v2 )

               
        
def linear_spline_coeffs( t0, t1, t2, t3, t, dif = False):
    if not dif:
        return np.array([ 0, t2 - t, t - t1, 0]) / (t2 - t1)
    else:
        return np.array([0, -1, 1, 0]) / (t2 - t1)
    
### 
def tensor_product(*args):
    """
    Tensor product of np.arrays.
    E.g. tensor_product( a, b, c) is an array whose shape is the concatenation 
    of shapes of a, b, c.
    """
    shape = []
    res = np.array(1)
    
    for t in args:
        res = np.outer(res, t)
        shape += t.shape        
    
    return res.reshape(shape)

    
def get_spline_coeffs_1d_simple(controls, t, method = 'Catmull-Rom', dif = False):
    """
    controls is a list (or 1d array) of numbers
    t is number
    dif is a boolean indicating whether we compute the derivative or the value.
    
    
    This function finds the indices of the four controls (stored in the list i) 
    around t and the corresponding spline-coefficients (stored in the list ci) 
    of these points. 
    It returns the two lists i, ci of length 4.
    """
    ## force t inside
    t = max( controls[0], min(t, controls[-1]))
    
    ## the cell at which our t is
    cell = max([i for i, p in enumerate(controls[:-1]) if p <= t ])

    ## the indices of the 4 points around our cell
    i = [max( cell - 1, 0),   cell,   cell + 1,  min( cell + 2, len(controls) -1 ) ]
    ## values of the active parameter at these points
    ti = [ controls[k] for k in i ]
    
    #print cell, i, ti

    ## coefficietns of these four points in the spline will be denoted ci
    if method == 'linear':
        ci = linear_spline_coeffs(*ti, t = t, dif = dif)
    elif method == 'Catmull-Rom':                
        ci = CR_spline_coeffs(*ti, t = t, dif = dif)
    else:
        raise ValueError("Unsupported method.")
     
    return i, ci
    
    
def get_spline_coeffs_simple( params, x, method = 'Catmull-Rom', derivative = None, ravel_multi_index = False):
    """
    multidimensional analogue of get_spline_coeffs_1d_simple.
    
    'params' is a list of len n of lists of numbers. 
    (I.e our spline is n-dimensional and params contains for each coordinate the list of
    corresponding controls of our grid.)
    'x' is an array of len n --- it is the point where we want to evalueate our spline.
    'derivative' is a natural number --- the index of the coordinate by which we want to differentiate.
    if it is None, we don't differentiate.
    
    We return index_list, coeff_list.
    index_list is a list of lists of len n of natural numbers.
    coef_list is a list (of the same len as index_list) of real numbers.
    It means: to calculate value at x, take the grid-point at position index_list[k] 
    with coefficient coef_list[k].
    """
    index_list = [[]]
    coeff_list = [1]
    for k, (controls, t) in enumerate(zip(params, x)):
        ind, coef = get_spline_coeffs_1d_simple(controls, t, method, dif = (derivative == k))
        
        index_list = [ il + [i] for il in index_list for i in ind]
        coeff_list = [ cl *  c  for cl in coeff_list for c in coef]
        
        
    if ravel_multi_index:
        index_list = [np.ravel_multi_index(jj, dims = [len(c) for c in params]) for jj in index_list]
        
    return index_list, coeff_list
    
def  get_spline_coeffs_multi(params, points, probas, method = 'Catmull-Rom', derivative = None, ravel_multi_index = False):
    """
    Evaluate in multiple points and then compute the average with coeffs in 'probas'.
    """
    coeffs = dict()
    for x, p in zip(points, probas):
        new_indices, new_coefficients = get_spline_coeffs_simple(params, x, method, derivative, ravel_multi_index)
        new_indices = [ tuple(i) for i in new_indices]
        for i, c in zip(new_indices, new_coefficients):
            if i in coeffs:
                coeffs[i] += p * c
            else:
                coeffs[i]  = p * c
                
    return coeffs.keys(), coeffs.values()

    
## the Interpolator class

class Interpolator(object):
    def __init__(self, params, values, method = 'Catmull-Rom'):
        """ 
        If we have say 3d interpolation dependent on params a,b,c then parameters 
        are expected to form an orthogonal 3d grid where a varies in the
        set params[0] and b varies in the set params[1] and c in params[3].
        That is, params is a list of 1d numpy arrays.
        It's expected, that a,b,c are increasing.        
        
        Values form a numpy array of dimension
        len(params[1]) * len(params[2]) *... * dimension of the values of the approximated function.
        
        The type of values can be arbitrary, but it must be possible to multiply them by real numbers
        and to add them.
        
        Method is a string and can be either 'linear', or 'Catmull-Rom'.
        """
        
        self.params = params
        self.values = values
        self.method = method
        
        for i in range(len(params)):
            if len( params[i]) != self.values.shape[i]:
                ValueError("Interpolator initialization dimension error.")

    @staticmethod
    def identity(params, method = 'Catmull-Rom'):
        shape = [len(c) for c in params]
        vals = np.empty(shape, np.object).flatten()
        vals[:] = list( itt.product(*params))
        vals = vals.reshape(shape)

        return Interpolator(params, vals, method)

    @staticmethod
    def from_fun(params, fun, method = 'Catmull-Rom'):
        return Interpolator.identity(params, method).apply(fun)
                
    def __str__(self):
        return "Interpolator with params:\n" + str( self.params) + "\n and values:\n" + str(self.values)
            
                
    
    ### we define binary operations
    def __add__(self, other):
        """
        We assume implicitly, that self.params = other.params, self.method = other.method.
        """
        
        if isinstance(other, Interpolator):
            return Interpolator( self.params, self.values + other.values, self.method)
        else:
            ## change other to a constant interpolator with the same domain as self
            vals = np.empty_like(self.values)
            vals.fill( other )
            oi = Interpolator( params = self.params, values = vals, method = self.method )            
            return self + oi
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return (-1) * self     
        
    def __sub__(self, other):
        """
        We assume implicitly, that self.params = other.params, self.method = other.method.
        """
        return self + (-other)
    
    def __mul__(self, other):
        if isinstance( other, numbers.Number):
            return Interpolator( self.params, other * self.values, self.method )
        else:
            raise ValueError("You can multiply an Interpolator object only by numbers. "\
                             "You tried to myltiply it by an " + other.__class__.__name__  + ' object.')
                             
    def __rmul__(self, other):
        return self * other
    
    def __div__(self, other):
        return self * (1/ other)
        
    def __truediv__(self, other):
        return self * (1/ other)
        
        
    def plot(self, *args, **kwargs):
        f = self
        if len(f.params) == 1:
            x0, x1 = min(f.params[0]), max(f.params[0])
            
            xxx = np.linspace( x0, x1, 100)
            yyy = [ f(x) for x in xxx]
            pl.plot(xxx, yyy, *args, **kwargs)
        else:
            raise NotImplementedError("This methods works only for one-dimensional interpolators.")
            
    def heatmap(self):
        f = self
        assert len(f.params) == 2, "Heat map can be drawn only for 2-parametric meshes."
        xxx, yyy = [np.linspace(par[0], par[-1], 50) for par in f.params] 
        pipl.plot_fun_heatmap(f, xxx, yyy)
        
    
    def get_range_of_params(self):
        """
        Returns a list containg a list of minimal values and a list of maximal values  
        """
        return [ [min(par) for par in self.params], [max(par) for par in self.params] ]
                
        

    def get_coeffs_simple(self, x, derivative = None):
        if isinstance(x, numbers.Number):
            x = [x]
        return get_spline_coeffs_simple( self.params, x, self.method, derivative )

    def ev(self, x, derivative = None):
        """
        evaluation or calculation of derivative in the direction 'derivative'
        """
        index_list, coeff_list = self.get_coeffs_simple(x, derivative)
        return mysum([ c * self.values[ tuple(i) ]  for i, c in zip(index_list, coeff_list)])

    def multi_eval(self, points, probas, derivative = None):
        """
        Evaluate in multiple points and then compute the average with coeffs in 'probas'.
        """
        coeffs = dict()
        for x, p in zip(points, probas):
            new_indices, new_coefficients = self.get_coeffs_simple(x, derivative)
            new_indices = [ tuple(i) for i in new_indices]
            for i, c in zip(new_indices, new_coefficients):
                if i in coeffs:
                    coeffs[i] += p * c
                else:
                    coeffs[i]  = p * c             
              
        #print [ [i, c] for i, c in coeffs.iteritems()]
        return mysum([ c * self.values[tuple(i)] for i, c in coeffs.items()])
        
        
    def ev_partial(self, x):
        """
        x is a vector that can contain None elements.
        The function returns an interpolator on unspecified variables.
        
        I.e. if x contains k times None, then the returned interpolator
        will have k-dimensional domain.
        """
        def unshuffle( mask):
            """
            mask is a list of 0,1 e.g. [1, 0, 1, 0, 1, 1]
            function returns a pair of lists 
            [ index of 1st 0, ind of 2nd 0, ...], [ind of 1ft 1, ...]
            In our example: [1, 3], [0, 2, 4, 5]
    
            """
            le = [ (v, i) for i, v in enumerate(mask) ]
            sl =  sorted(le) 
            return [ [ b for (a, b) in sl if a == i] for i in [0, 1] ]
        
        
        i1, i2 = unshuffle( [ xe is None for xe in x])
        index_list, coeff_list = get_spline_coeffs_simple( 
                                                x = [ x[i] for i in i1], 
                                               params = [ self.params[i] for i in i1],
                                               method = self.method)
        vals = self.values.transpose(i1 + i2)                                                
        new_vals = mysum([ c * vals[ tuple(i) ]  for i, c in zip(index_list, coeff_list)])
        return Interpolator(params = [ self.params[i] for i in i2], values = new_vals)
        
   
    def __call__(self, x):
        return self.ev(x)
        
        
    def grad(self, x):
        """
        Returns the gradient at x as a list.
        """
        return [ self.ev(x, derivative = i) for i in range(len(self.params))]

    def apply( self, fun):
        def apply_at_level( a, level, fun):
            if level == 0:
                return fun(a)
            return [apply_at_level( aa, level - 1, fun) for aa in a ]
            
        new_vals = np.array(
                        apply_at_level(a = self.values, 
                                  level =  len(self.params),
                                  fun = fun )
                    )
        return Interpolator( self.params, new_vals)
        
###
def concat( l):
    """ 
    Joins interpolators from the list l. All interpoladors must have 1D domain.
    """
    if l is None or l == []:
        return None
    
    params = [ l[0].params[0][0] ]
    values = [ l[0].values[0]]
    #print params, values
    for f in l:
        xxx,  yyy = f.params[0],  f.values
        #print yyy, values[-1], list( np.array(yyy) - yyy[0] + params[-1])
        params = params[:-1] + list( np.array(xxx) - xxx[0] + params[-1])
        values = values[:-1] + list( np.array(yyy) - yyy[0] + values[-1])
        
    return Interpolator([params], values, l[0].method)        
        
