import numpy as np
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import vega
from scipy import stats
from numpy import log, exp, sqrt
from functools import wraps
from time import time
import yfinance as yf

class OptionPricing():
    
    def __init__(self):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.N = N
        self.u = u
        self.d = 1/self.u
        self.m_price = m_price
        self.sigma = sigma
        
        
    ''' Create the Black Scholes formula call option
    '''
    
    def call_option_price(self):
        #calculate d1 and d2 parameters
        d1 = (log(self.S0/self.K) + (self.r+(self.sigma)**2/2)*self.T)/(self.sigma*sqrt(self.T))
        
        d2 = d1 - self.sigma * sqrt(self.T)
        # use the d1 and d2 to calculate the price of the call option
        return self.S0*stats.norm.cdf(d1) - self.K*exp(-self.r*self.T)*stats.norm.cdf(d2)
    
    
    ''' Implement two methods (slow vs fast) to the binomial model
    '''
    
    # The slow method: use for loops to iterate through nodes j at each time step i
    def binomial_tree_slow(self):
        # precompute constants
        dt = self.T/self.N
        q = (np.exp(self.r*dt) - self.d) / (self.u-self.d)
        disc = np.exp(-self.r*dt)
    
        # initialize asset prices at maturity - Time step N
        S = np.zeros(self.N+1)
        S[0] = self.S0*self.d**self.N
        for j in range(1, self.N+1):
            S[j] = S[j-1]*self.u/self.d
    
        # initialize option values at maturity
        C = np.zeros(self.N+1)
        for j in range(0, self.N+1):
            C[j] = max(0, S[j] - self.K)
        
        # step backwards through tree
        for i in np.arange(self.N, 0, -1):
            for j in range(0, i):
                C[j] = disc * ( q*C[j+1] + (1-q)*C[j] )
        
        
        return C[0]
    
    # The fast method: uses numpy arrays instead of for loops through j nodes
    def binomial_tree_fast(self):
        #precompute constants
        dt = self.T/self.N
        q = (np.exp(self.r*dt) - self.d) / (self.u-self.d)
        disc = np.exp(-self.r*dt)
    
        # initialize asset prices at maturity - Time step N
        C = self.S0 * self.d ** (np.arange(self.N, -1, -1)) * self.u ** (np.arange(0, self.N+1, 1))
    
        # initialize option values at maturity
        C = np.maximum( C - self.K, np.zeros(self.N+1) )
    
        # step backwards through tree
        for i in np.arange(self.N, 0, -1):
            C = disc * ( q * C[1:i+1] + (1-q) * C[0:i] )
            
        return C[0]
    
    ''' Calculate the implied volatility of the call option using the Newton-Raphson method
    '''

    def implied_vol(self):
        self.max_iter = max_iter = 100  # max number of iterations
        self.vol_old = vol_old = 0.3    # initial guess of implied volatility
        self.tol = tol = 0.00001        # tolerance limit
        self.opttype = 'c'    # option type: c = call, p = put
    
        for k in range(max_iter):
            bs_price = bs(self.opttype, self.S0, self.K, self.T, self.r, vol_old)
            Cprime = vega(self.opttype, self.S0, self.K, self.T, self.r, vol_old) * 100
            C = bs_price - self.m_price
        
            vol_new = vol_old - C/Cprime
            new_bs_price = bs(self.opttype, self.S0, self.K, self.T, self.r, vol_new)
            if (abs(vol_old-vol_new) < tol or abs(new_bs_price-m_price) < tol):
                break
        
            vol_old = vol_new
        
        implied_vol = vol_new
        return implied_vol

if __name__ == '__main__':
    
    S0 = 108.16      # Stock Price
    K = 97.5         # Strike price
    T = (53/365)     # time to maturity in years
    r = 0.06         # annual risk-free rate
    N = 8            # number of time steps
    u = 1.1          # up-factor in binomial models
    d = 1/u          # recombining binomial tree
    m_price = 13.30  # Market price of the option
    sigma = 0        # volatility
    
    options_pricing = OptionPricing()
    
    
    black_scholes = options_pricing.call_option_price()
    binomial_slow = options_pricing.binomial_tree_slow()
    binomial_fast = options_pricing.binomial_tree_fast()
    imp_vol = options_pricing.implied_vol()
    
    print("Black-Scholes call option price: %.2f " % (black_scholes))
    print('Market Price of the option: %.2f ' % (m_price))
    print('Slow method of Binomial Option Price: %.2f ' % (binomial_slow))
    print('Fast method of Binomial Option Price: %.2f ' % (binomial_fast))
    print('Implied Vol: %.2f%%' % (imp_vol * 100))
    
    def timing(f):
        @wraps(f)
        def wrap(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()
            print('func: %r args:[%r, %r] took: %2.4f sec' % \
                  (f.__name__, args, kw, te-ts))
            return result
        return wrap

    
    @timing
    def slow_metrics(K, T, S0, r, N, u, d, opttype='C'):
        #precompute constants
        dt = T/N
        q = (np.exp(r*dt) - d) / (u-d)
        disc = np.exp(-r*dt)
    
        # initialize asset prices at maturity - Time step N
        S = np.zeros(N+1)
        S[0] = S0*d**N
        for j in range(1, N+1):
            S[j] = S[j-1]*u/d
    
        # initialize option values at maturity
        C = np.zeros(N+1)
        for j in range(0, N+1):
            C[j] = max(0, S[j] - K)
            
            # step backwards through tree
            for i in np.arange(N, 0, -1):
                for j in range(0, i):
                    C[j] = disc * ( q*C[j+1] + (1-q)*C[j] )
                
        return C[0]
    
    @timing
    def fast_metrics(K, T, S0, r, N, u, d, opttype='C'):
        #precompute constants
        dt = T/N
        q = (np.exp(r*dt) - d) / (u-d)
        disc = np.exp(-r*dt)
        
        # initialize asset prices at maturity - Time step N
        C = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N+1, 1))
        
        # initialize option values at maturity
        C = np.maximum( C - K, np.zeros(N+1) )
        
        # step backwards through tree
        for i in np.arange(N, 0, -1):
            C = disc * ( q * C[1:i+1] + (1-q) * C[0:i] )
                
        return C[0]
    
    
    for N in [500]:
        slow_metrics(K, T, S0, r, N, u, d, opttype='C')
        fast_metrics(K, T, S0, r, N, u, d, opttype='C')
        

        