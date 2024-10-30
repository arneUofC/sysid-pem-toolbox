import control as ct
import numpy as np
import scipy as sp  
import matplotlib.pyplot as plt


def theta_2_BCDF(theta,n):
    nb = n[0]
    nc = n[1]
    nd = n[2]
    nf = n[3]
    nk = n[4]

    theta_b = theta[0:n[0]]
    theta_c = np.concatenate(([1],theta[n[0]:n[0]+n[1]]))
    theta_d = np.concatenate(([1],theta[n[0]+n[1]:n[0]+n[1]+n[2]]))
    theta_f = np.concatenate(([1],theta[n[0]+n[1]+n[2]:n[0]+n[1]+n[2]+n[3]]))

    if nf+1 > nb:
        B = np.concatenate((theta_b,np.zeros(nf+1-nb)))
    elif nf+1==nb:
        B = theta_b
    else:
        print('Must choose proper transfer function for plant model.')
    
    if nk > 0:
        F = np.concatenate((theta_f,np.zeros(nk)))
    else:
        F = theta_f 
        
    if nd > nc:
        C = np.concatenate((theta_c, np.zeros(nd-nc)))
    elif nc==nd:
        C = theta_c
    else:
        print('Must choose proper transfer function for noise model.')

    D = theta_d
    return B, C, D, F


def theta_2_tf_box_jenkins(theta,n,Ts):

    B,C,D,F = theta_2_BCDF(theta,n)
    G_theta = ct.tf(B, F, Ts)
    H_theta = ct.tf(C, D, Ts)

    return G_theta, H_theta


def jac_V_bj(theta, n, y, u):
    N = y.shape[0]
    nb = n[0]
    nc = n[1]
    nd = n[2]
    nf = n[3]
    nk = n[4]

    B,C,D,F = theta_2_BCDF(theta,n)

    G_theta = ct.tf(B, F, True)
    H_theta = ct.tf(C, D, True)
    tt, y_hat_1 = ct.forced_response(G_theta/H_theta, U=u) 
    tt, y_hat_2 = ct.forced_response(1 - 1/H_theta, U=y)
    y_hat = y_hat_1 + y_hat_2
    epsilon = y - y_hat

    tt, y_hat_3 = ct.forced_response(G_theta, U=u) 
    e = y - y_hat_3

    depsilondB = np.empty((N,nb))
    for ii in range(nb):
        d = ct.tf(1,np.concatenate(([1],np.zeros(ii))), True)
        P = ct.tf(np.concatenate(([1],np.zeros(nf))),F, True)
        #print(-d*P/H_theta)
        tt, depsilon = ct.forced_response(-d*P/H_theta,U=u)
        depsilondB[:,ii] = depsilon
        #dVdB[ii] = 2*(np.sum(epsilon * depsilon))

    depsilondC = np.empty((N,nc))
    for ii in range(nc):
        d = ct.tf(1,np.concatenate(([1],np.zeros(ii+1))), True)
        P = ct.tf(np.concatenate(([1],np.zeros(nc))),C, True)
        tt, depsilon = ct.forced_response(-d*P/H_theta,U=e)
        depsilondC[:,ii] = depsilon
        #dVdC[ii] = 2*(np.sum(epsilon * depsilon))
   
    depsilondD = np.empty((N,nd))
    for ii in range(nd):
        d = ct.tf(1,np.concatenate(([1],np.zeros(ii+1))), True)
        P = ct.tf(np.concatenate(([1],np.zeros(nc))),C, True)
        tt, depsilon = ct.forced_response(d*P,U=e)
        depsilondD[:,ii] = depsilon
        #dVdD[ii] = 2*(np.sum(epsilon * depsilon))
    
    depsilondF = np.empty((N,nf))
    for ii in range(nf):
        d = ct.tf(1,np.concatenate(([1],np.zeros(ii+1))), True)
        P = ct.tf(np.concatenate(([1],np.zeros(nf+nk))),F, True)
        tt, depsilon = ct.forced_response(d*P*G_theta/H_theta,U=u)
        depsilondF[:,ii] = depsilon
        #dVdF[ii] = 2*(np.sum(epsilon * depsilon))
    depsilonTot = np.concatenate((depsilondB, depsilondC, depsilondD, depsilondF),axis=1)
    return depsilonTot


def V_box_jenkins(theta, n, y, u):
    N = y.shape[0]
    y_hat = y_hat_box_jenkins(theta,n,y,u)
    epsilon = y - y_hat
    
    #return np.sum(epsilon**2)/N
    return epsilon


def y_hat_box_jenkins(theta, n, y, u):
    B,C,D,F = theta_2_BCDF(theta,n)
    G_theta = ct.tf(B, F, True)
    H_theta = ct.tf(C, D, True)    
    tt, y_hat_1 = ct.forced_response(G_theta/H_theta, U=u) 
    tt, y_hat_2 = ct.forced_response(1 - 1/H_theta, U=y)
    y_hat = y_hat_1 + y_hat_2
    
    return y_hat


def V_oe(theta, n, y, u):
    theta_b = theta[0:n[0]]
    theta_f = np.concatenate(([1],theta[n[0]:n[0]+n[1]]))

    G_theta = ct.tf(theta_b, theta_f, True)
    tt, y_hat = ct.forced_response(G_theta, U=u) 
   
    epsilon = y - y_hat
    return np.sum(epsilon**2)


def theta_2_tf_oe(theta,n, Ts):
    theta_b = theta[0:n[0]]
    theta_f = np.concatenate(([1],theta[n[0]:n[0]+n[1]]))

    G_theta = ct.tf(theta_b, theta_f, Ts)
    H_theta = ct.tf(1,1,Ts)

    return G_theta, H_theta


def y_hat_oe(theta, n, y, u):
    theta_b = theta[0:n[0]]
    theta_f = np.concatenate(([1],theta[n[0]:n[0]+n[1]]))

    G_theta = ct.tf(theta_b, theta_f, True)
    tt, y_hat = ct.forced_response(G_theta, U=u) 
   
    epsilon = y - y_hat

    return epsilon


def V_arx_lin_reg(n, y, u):
    
    na = n[0]
    nb = n[1]
    nk = n[2]

    t0 = np.maximum(na-1, nb+nk-1)
    N = y.shape[0]
    phi = np.zeros((N-t0,na+nb))
    for ii in range(N-t0):
        for jj in range(na):
            phi[ii,jj] = -y[ii+t0-jj-1]
    
    for ii in range(N-t0):
        for jj in range(nb):
            phi[ii,jj+na] = u[ii+t0-jj-nk]

    theta = np.linalg.inv( phi.T @ phi ) @ (phi.T @ y[t0:N])
    return theta

def theta_2_tf_arx(theta,n,Ts):
    
    na = n[0]
    nb = n[1]
    nk = n[2]

    theta_a = np.concatenate(([1],theta[0:n[0]]))
    theta_b = theta[n[0]:n[0]+n[1]]

    if na+1 > nb:
        B = np.concatenate((theta_b,np.zeros(na+1-nb)))
    elif na+1==nb:
        B = theta_b
    else:
        print('Must choose proper transfer function for plant model.')
    
    if nk > 0:
        A = np.concatenate((theta_a,np.zeros(nk)))
    else:
        A = theta_a 
        
    C = np.zeros(na+1)
    C[0] = 1

    G_theta = ct.tf(B, A, Ts)
    H_theta = ct.tf(C, theta_a, Ts)
    return G_theta, H_theta



def cross_correlation_test(epsilon,u,tau = 50):
    N = u.shape[0]
    Reu = np.correlate(epsilon,u,'full')
    
    Reu = Reu[N-tau:N+tau]

    Re = np.correlate(epsilon,epsilon,'full')
    Ru = np.correlate(u,u,'full')
    P = np.sum(Re*Ru)
    bound = np.sqrt(P/N)*1.95 

    fig,ax = plt.subplots(1)
    ax.plot(np.arange(-tau,tau),Reu)
    ax.plot(np.arange(-tau,tau),np.ones(2*tau)*bound,'k:')
    ax.plot(np.arange(-tau,tau),-np.ones(2*tau)*bound,'k:')
    ax.set_title('Cross Correlation of Prediction Error')
    ax.set_xlabel('Lag (samples)')


    Re = np.correlate(epsilon,epsilon,'full')


def auto_correlation_test(epsilon,tau = 50):
    N = epsilon.shape[0]
    
    Re = np.correlate(epsilon,epsilon,'full')
    Re_pos = Re[N:N+tau]

    bound_e = 1.95/np.sqrt(N)

    fig,ax = plt.subplots(1)
    ax.plot(np.arange(1,tau+1),Re_pos/Re[N-1])
    ax.plot(np.arange(1,tau+1),np.ones(tau)*bound_e,'k:')
    ax.plot(np.arange(1,tau+1),-np.ones(tau)*bound_e,'k:')
    ax.set_title('Auto Correlation of Prediction Error')
    ax.set_xlabel('Lag (samples)')


def FIR_estimates_GH(n, y, u):

    na = n[0]
    nb = n[1]
    nk = n[2]

    ng = nb
    nh = na

    theta = sid.V_arx_lin_reg(n,y,r)

    A = -theta[0:na]
    B = theta[na:nb+na]

    rB = np.concatenate(([B[0]], np.zeros(na-1)))
    cB = B
    
    rA = np.concatenate(([1], np.zeros(na-1)))
    cA = np.concatenate(([1], -A[0:na-1]))

    CB = sp.linalg.toeplitz(cB,r=rB)
    CA = sp.linalg.toeplitz(cA,r=rA)
    
    M = np.block([[np.zeros((na,nb)), CA], [np.eye(nb), -CB]])
    
    theta_gh = np.linalg.inv( M.T @ M ) @ (M.T @ np.concatenate((A, B)))

    g = np.concatenate((np.zeros(nk), theta_gh[0:ng]))
    h = np.concatenate(([1], theta_gh[ng:ng+nh]))

    return g, h


def tf_realization_GH(g,h,n):

    na = n[0]
    nb = n[1]
    nc = n[2]
    nd = n[3]
    nk = n[4]

    nh = h.shape[0]-1
    ng = g.shape[0]-nk

    Cg = np.array(sp.linalg.toeplitz(np.concatenate(([0],g[nk:nk+ng-1])),r=np.zeros(na)))
    Meye = np.concatenate((np.eye(nb), np.zeros((ng-nb,nb))),axis=0)
    M = np.concatenate((Meye,-Cg),axis=1)
    thetaBA = np.linalg.inv( M.T @ M ) @ (M.T @ g[nk:ng+nk] )

    Ch = np.array(sp.linalg.toeplitz(h[0:nh],r=np.concatenate(([1],np.zeros(nd-1)))))
    Meye = np.concatenate((np.eye(nc), np.zeros((nh-nc,nc))),axis=0)
    M = np.concatenate((Meye,-Ch),axis=1)
    thetaCD = np.linalg.inv( M.T @ M ) @ (M.T @ h[1:nh+1] )

    return np.concatenate((thetaBA,thetaCD))

    

def get_regression_matrix(w,t0,i1,i2):
    
    N = w.shape[0]
    phi = np.zeros((N-t0+i1,i2-i1))

    for ii in range(N-t0+i1):
        for jj in range(i1,i2):
            phi[ii,jj] = w[ii+t0-jj]   
    return phi