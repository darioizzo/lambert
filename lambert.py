###############################################################################
def a2tof(aam,lam,N=0):
	"""
	Computes the non-dimensional time of flight T := 1/2 (t_2-t_1) * sqrt(mu/am^3)
	as a function of a/am > 1 where am is the semi-major axis of the minimum
	energy ellipse. The function returns one or two values accordingly (hyperbolas, ellipses)
	"""
	from numpy import sin,cos,sqrt,arccos,arcsin,pi,sinh,cosh,arccosh, arcsinh, nan
	if (aam>0): #ellipses
		beta = 2.0 * arcsin(sqrt(lam*lam/aam))
		if lam<0:
			beta = -beta
		alfa1 = 2.0 * arcsin(sqrt(1.0/aam))
		alfa2 = 2.0 * pi-alfa1
		t1 = aam*sqrt(aam)*((alfa1-sin(alfa1)) - (beta-sin(beta)) + 2*pi*N)
		t2 = aam*sqrt(aam)*((alfa2-sin(alfa2)) - (beta-sin(beta)) + 2*pi*N)
	elif (aam<0): #hyperbolas
		beta = 2 * arcsinh(sqrt(-lam*lam/aam))
		if lam<0:
			beta = -beta
		alfa = 2.0 * arcsinh(sqrt(-1.0/aam))
		t1 = -aam*sqrt(-aam)*((beta-sinh(beta)) - (alfa-sinh(alfa)))
		t2 = nan
	return t1 / 2,t2 / 2

def plota(lam,N=0):
	"""
	Plots the time of flight curve for a/am varying from 1 up (ellipses)
	and from -1.0 to 0.0 (hyperbolas)
	"""
	from matplotlib import pyplot as pl
	from numpy import linspace, array
	axes = linspace(1.0,2.0,100)
	T1 = list()
	T2 = list()
	for a in axes:
		t1,t2 = a2tof(a,lam,N)
		T1.append(min(t1,t2))
		T2.append(max(t1,t2))
	T3 = list()
	axes2 = linspace(-1.0,-0.0001,100)
	for a in axes2:
		t1,t2 = a2tof(a,lam,N)
		T3.append(t1)
	pl.plot(axes,T1,'k')
	pl.plot(axes,T2,'k')
	if N==0:
		pl.plot(axes2,T3,'k')
	pl.xlim((-1,2))
	pl.ylim((0,10))
	pl.show()

def plot_a_curves(lam,N=0):
	"""
	Plots the time of flight curves for different values of lambda. here lam is a list of values
	"""
	from numpy import linspace
	if ((type(lam) == float) or (type(lam) == int)):
		lam = (lam)
	for l in lam:
		plota(l,N)
			
def make_a_plot():
	"""
	Makes the (a,T) plot appearing in the paper
	"""
	from numpy import linspace
	from matplotlib import pyplot as pl
	pl.rc('text', usetex=True)
	pl.rc('font', family='serif')
	lam=-0.9
	plota(lam,0)
	plota(lam,1)
	plota(lam,2)
	pl.xlabel(r'$$a / a_m$$',fontsize=16)
	pl.ylabel(r'$$T$$',fontsize=16)
	pl.text(0.7, 2, r'$$M=0$$', bbox=dict(facecolor='grey', alpha=0.5))
	pl.text(-0.6, 1, r'$$M=0$$', bbox=dict(facecolor='grey', alpha=0.5))
	pl.text(0.7, 5.3, r'$$M=1$$', bbox=dict(facecolor='grey', alpha=0.5))
	pl.text(0.7, 8.5, r'$$M=2$$', bbox=dict(facecolor='grey', alpha=0.5))
	pl.text(-0.5, 4, r'hyperbolic')
	pl.text(1.5, 4, r'elliptic')

###############################################################################

def x2tof(x,lam,N=0):
	"""
	Computes the non-dimensional time of flight T T := 1/2 (t_2-t_1) * sqrt(mu/am^3) 
	as a function of x where x is the Lancaster-Blanchard variable. This version
	uses Lagrange tof expression
	"""
	from numpy import sin,cos,sqrt,arccos,arcsin,pi,sinh,cosh,arccosh, arcsinh, nan
	aam = 1.0/(1.0-x*x)
	if (aam>0): #ellipses
		beta = 2.0 * arcsin(sqrt(lam*lam/aam))
		if lam<0:
			beta = -beta
		alfa = 2.0 * arccos(x)
		t = aam*sqrt(aam)*((alfa-sin(alfa)) - (beta-sin(beta)) + 2*pi*N)
	elif (aam<0): #hyperbolas
		beta = 2 * arcsinh(sqrt(-lam*lam/aam))
		if lam<0:
			beta = -beta
		alfa = 2.0 * arccosh(x)
		t = -aam*sqrt(-aam)*((beta-sinh(beta)) - (alfa-sinh(alfa)))
	return t / 2.0

def x2tof2(x,lam,N=0):
	"""
	Computes the non-dimensional time of flight T as a function of x where
	x is the Lancaster-Blanchard variable. It uses different expressions (Lagrange, 
	Battin or Lancaster-Blanchard)) according
	to the distance to x=1, as to avoid numerical problems of different formulations
	"""
	from numpy import sqrt,arccos,pi,log
	#from math import atan2
	series = 0.01
	lagrange = 0.2
	dist = abs(x-1)
	if dist < lagrange and dist > series:
		return x2tof(x,lam,N)
	K = lam*lam
	E = x*x-1
	rho = abs(E)
	z = sqrt(1+K*E)	
	if dist < series:
		eta = z-lam*x
		S1 = 0.5*(1-lam-x*eta)
		Q,_ = F(3.0,1.0,5.0/2.0,S1,1e-11)
		Q = 4.0/3.0*Q
		return (eta**3*Q+4*lam*eta)/2 + N*pi / (rho**(3.0/2.0))
	else:
		y=sqrt(rho)
		g = x*z - lam*E
		if E<0:
			#l = atan2(f,g) #strange behavior when lambda = +-1
			l = arccos(g)
			d=N*pi+l
		if E>0:
			f = y*(z-lam*x)
			d=log(f+g)
		return (x-lam*z-d/y)/E

def plotx(lam,N=0,logplot=False,linestyle='k',minval=-0.99,maxval=3.0):
	"""
	Plots the time of flight curve in the x-T plane or xi-tau plane
	
	* N number of revolutions
	* logplot if True plots in the xi-tau plane
	* linestyle plot linestyle
	* minval minimum x value to use
	* maximum maximum x value to use
	"""
	from matplotlib import pyplot as pl
	from numpy import linspace, array,log
	if N>0:
		axes = linspace(minval,-minval,200)
	elif N==0:
		axes = linspace(minval,maxval,200)
		
	if (lam==1) and (N==0):
		axes = linspace(minval,-1e-5)
	T = list()
	for x in axes:
		t = x2tof2(x,lam,N)
		T.append(t)
	if logplot:
		if N==0:
			axes = [a+1 for a in axes]
		elif N>0:
			axes = [(a+1)/(1-a) for a in axes]
		pl.plot(log(axes),log(T),linestyle)
		pl.xlim( log(minval+1),log(maxval+1))
		#pl.ylim((-2,log(10)))
	elif not logplot:
		pl.plot(axes,T,linestyle)
		pl.xlim((-1,2.0))
		pl.ylim((0,20))
	pl.show()

def plot_x_curves(lam,N=0,logplot=False,minval=-0.99,maxval=3.0):
	"""
	Plots the time of flight curves for different values of lambda. Here lam is a list of values

	* N number of revolutions
	* logplot if True plots in the xi-tau plane
	* linestyle plot linestyle
	* minval minimum x value to use
	* maximum maximum x value to use
	"""
	from numpy import linspace
	if ((type(lam) == float) or (type(lam) == int)):
		lam = (lam)
	for l in lam:
		plotx(l,N,logplot,minval=minval,maxval=maxval)

def make_x_plot():
	from numpy import linspace
	from matplotlib import pyplot as pl
	pl.rc('text', usetex=True)
	pl.rc('font', family='serif')
	lamdas = [-1,-0.9,-0.7,0,0.7,0.9,1]
	for lam in lamdas:
		plotx(lam,0)
		plotx(lam,1,linestyle='k--')
		plotx(lam,2)
		plotx(lam,3,linestyle='k--')
	pl.axvline(x=1,color='k')
	pl.xlabel(r'$$x$$',fontsize=16)
	pl.ylabel(r'$$T$$',fontsize=16)
	pl.text(0.0, 1.5, r'$$M=0$$', bbox=dict(facecolor='white', alpha=1))
	pl.text(0.0, 4.7, r'$$M=1$$', bbox=dict(facecolor='white', alpha=1))
	pl.text(0.0, 8.0, r'$$M=2$$', bbox=dict(facecolor='white', alpha=1))
	pl.text(0.5, 4, r'hyperbolic')
	pl.text(1.2, 4, r'elliptic')
	pl.annotate(r'$$\lambda = 1$$', xy=(-0.25, 1.1), xytext=(-0.8, 0.2),
            arrowprops=dict(facecolor='black', shrink=0.15, width=1,headwidth=5),
            )
	pl.annotate(r'$$\lambda = -1$$', xy=(0.5, 2.0), xytext=(0.7, 3.0),
            arrowprops=dict(facecolor='black', shrink=0.15, width=1,headwidth=5),
            )

def make_xi_plot():
	from numpy import linspace
	from matplotlib import pyplot as pl
	pl.rc('text', usetex=True)
	pl.rc('font', family='serif')
	lamdas = [-1,-0.9,-0.7,0,0.7,0.9,1]
	for lam in lamdas:
		plotx(lam,0,linestyle='k',minval=-0.9,maxval=10.0,logplot=True)
		plotx(lam,1,linestyle='k--',logplot=True)
		plotx(lam,2,logplot=True)
		plotx(lam,3,linestyle='k--',logplot=True)
	pl.vlines(x=1,ymin=-2,ymax=-0.076,color='k')
	pl.xlabel(r'$$\xi$$',fontsize=16)
	pl.ylabel(r'$$\tau$$',fontsize=16)
	pl.text(0.0, 0, r'$$M=0$$', bbox=dict(facecolor='white', alpha=1))
	pl.text(0.0, 1.4, r'$$M=1$$', bbox=dict(facecolor='white', alpha=1))
	pl.text(0, 2.48, r'$$M=2$$', bbox=dict(facecolor='white', alpha=1))
	pl.text(1.3, -1.5, r'hyperbolic')
	pl.text(0.3, -1.5, r'elliptic')
	pl.annotate(r'$$\lambda = 1$$', xy=(-0.29, -0.19), xytext=(-1, -1),
            arrowprops=dict(facecolor='black', shrink=0.15, width=1,headwidth=5),
            )
	pl.annotate(r'$$\lambda = -1$$', xy=(0.7, 0.4), xytext=(0.8,0.8),
            arrowprops=dict(facecolor='black', shrink=0.15, width=1,headwidth=5),
            )
	pl.xlim((-2,2))
	pl.ylim((-2,3))

def make_xi_plot_M0():
	from numpy import linspace
	from matplotlib import pyplot as pl
	pl.rc('text', usetex=True)
	pl.rc('font', family='serif')
	lambdas = linspace(-.9,.9,20)
	plot_x_curves(lam=lambdas,N=0,logplot=True, maxval=30)
	pl.axvline(x=1,color='k')
	pl.xlabel(r'$$\xi$$',fontsize=16)
	pl.ylabel(r'$$\tau$$',fontsize=16)
	pl.text(1.5, 4, r'hyperbolic')
	pl.text(-0.2, 4, r'elliptic')
	pl.text(-3, 6, r'$$M=0$$', bbox=dict(facecolor='grey', alpha=0.5))
	pl.annotate(r'$$\lambda = 0.9$$', xy=(0.1, -0.25), xytext=(-1, -2),
            arrowprops=dict(facecolor='black', shrink=0.15, width=1,headwidth=5),
            )
	pl.annotate(r'$$\lambda = -0.9$$', xy=(2.12, -1.11), xytext=(2.2, 0),
            arrowprops=dict(facecolor='black', shrink=0.15, width=1,headwidth=5),
            )

def make_xi_plot_M1():
	from numpy import linspace
	from matplotlib import pyplot as pl
	pl.rc('text', usetex=True)
	pl.rc('font', family='serif')
	lambdas = linspace(-.9,.9,20)
	plot_x_curves(lam=lambdas,N=1,logplot=True, minval=-0.8)
	pl.xlabel(r'$$\xi$$',fontsize=16)
	pl.ylabel(r'$$\tau$$',fontsize=16)
	pl.text(-1, 2.7, r'$$M=1$$', bbox=dict(facecolor='grey', alpha=0.5))
	pl.annotate(r'$$\lambda = 0.9$$', xy=(-0.7, 1.7), xytext=(-1.25, 1.2),
            arrowprops=dict(facecolor='black', shrink=0.15, width=1,headwidth=5),
            )
	pl.annotate(r'$$\lambda = -0.9$$', xy=(0.3, 1.7), xytext=(0.5, 2),
            arrowprops=dict(facecolor='black', shrink=0.15, width=1,headwidth=5),
            )
	pl.ylim([1,3])

def make_starter_plot():
	from numpy import linspace
	from math import pi
	from matplotlib import pyplot as pl
	pl.rc('text', usetex=True)
	pl.rc('font', family='serif')
	pl.axvline(x=1,color='k')
	pl.xlabel(r'$$x$$',fontsize=16)
	pl.ylabel(r'$$T$$',fontsize=16)
	pl.text(0.5, 4, r'hyperbolic')
	pl.text(1.2, 4, r'elliptic')
	plot_x_curves([-1,-0.85,0.85,1],N=0,logplot=False,maxval=10,minval=-0.9)
	T0s = [pi,pi/2,1.1]
	for T0 in T0s:
		T  =linspace(T0,10*pi)
		pl.plot((T0/T)**(2.0/3.0)-1.0,T,':k')
		T  =linspace(0.1,T0)
		pl.plot((T0/T)**(1.0)-1.0,T,':k')
	pl.plot([],'k-',label="tof curves [-1,-0.85,0.85,1]")
	pl.plot([],'k:',label="initial guesses")
	pl.legend()
	pl.annotate(r'$$\lambda \le -0.85$$', xy=(-0.4, 7.5), xytext=(-0.1, 8),
            arrowprops=dict(facecolor='black', shrink=0.15, width=1,headwidth=5),
            )
	pl.annotate(r'$$\lambda \ge 0.85$$', xy=(-0.33, 2.1), xytext=(-0.9, 1.09),
            arrowprops=dict(facecolor='black', shrink=0.15, width=1,headwidth=5),
            )

def make_accuracy_plot():
	from numpy import linspace, mean
	from math import pi
	from matplotlib import pyplot as pl
	pl.rc('text', usetex=True)
	pl.rc('font', family='serif')
	pl.axvline(x=1,color='k')
	pl.xlabel(r'$$\lambda$$',fontsize=16)
	pl.ylabel(r'$$\epsilon$$',fontsize=16)
	print "Testing M=0"
	it_list = list()
	#We test for zero revolutions
	err,it,la,x,T = test_standard(M=100000,gooding=False,
		house=True,tau=False,iter_max=15,eps=1e-5, rnd_seed=4562,N=0)
	it_list.append(mean(it))
	pl.semilogy(la,err,'k.')
	#We test for multiple revolutions
	for rev in range(50):
		print "Testing M=" + str(rev)
		err,it,la,x,T = test_standard(M=10000,gooding=False,
			house=True,tau=False,iter_max=15,eps=1e-8, rnd_seed=4562+rev+1,N=rev+1)
		pl.semilogy(la,err,'k.')
		it_list.append(mean(it))
	pl.figure()
	pl.plot(it_list)
	
	

def derivatives(T,x,lam):
	from numpy import sqrt
	l2 = lam*lam
	l3 = l2*lam
 	umx2 = 1.0-x*x
	y = sqrt(1.0-l2*umx2)
	y2 = y*y
	y3 = y2*y
	DT = 1.0/umx2 * (3.0*T*x-2.0+2.0*l3*x/y)
	DDT = 1.0 / umx2 * (3.0*T+5.0*x*DT+2.0*(1.0-l2)*l3/y3)
	DDDT = 1.0 / umx2 * (7.0*x*DDT+8.0*DT-6.0*(1.0-l2)*l2*l3*x/y3/y2)
	return DT,DDT,DDDT

def newton(T,x0,lam,N=0,eps=1e-13,iter_max = 3):
	it=0
	err = 1
	xold = x0
	while ((err>eps) and it < iter_max):
		tof = x2tof2(xold,lam,N)
		DT,_,_ = derivatives(tof,xold,lam)
		xnew = xold - (tof-T) / DT
		err=abs(xold-xnew)
		xold=xnew
		it=it+1
	return xnew,it,err
	
def halley(T,x0,lam,N=0,eps=1e-13, iter_max = 3):
	it=0
	err = 1
	xold = x0
	while ((err>eps) and it < iter_max):
		tof = x2tof2(xold,lam,N)
		DT,DDT,_ = derivatives(tof,xold,lam)
		dt = T-tof
		if (not dt == 0):
			xnew = xold + dt * DT / (DT * DT + dt * DDT / 2.0);
		err=abs(xold-xnew)
		xold=xnew
		it=it+1
	return xnew,it,err

def halley_xM(x0,T0,lam,N=1,eps=1e-13):
	it=0
	err = 1
	xold = x0
	tof = T0
	DT,DDT,DDDT = derivatives(tof,xold,lam)
	dt = 0.0-DT
	if (not dt == 0):
		xnew = xold + dt * DDT / (DDT * DDT + dt * DDDT / 2.0);
	err=abs(xold-xnew)
	xold=xnew
	while (err>eps):
		tof = x2tof2(xold,lam,N)
		DT,DDT,DDDT = derivatives(tof,xold,lam)
		dt = 0.0-DT
		if (not dt == 0):
			xnew = xold + dt * DDT / (DDT * DDT + dt * DDDT / 2.0);
		err=abs(xold-xnew)
		xold=xnew
		it=it+1
	return xnew,tof,it,err


def householder(T,x0,lam,N=0,eps=1e-13, iter_max = 3):
	it=0
	err = 1
	xold = x0
	while ((err>eps) and it < iter_max):
		tof = x2tof2(xold,lam,N)
		DT,DDT,DDDT = derivatives(tof,xold,lam)
		delta = tof-T
		DT2 = DT*DT
		xnew = xold - delta * (DT2-delta*DDT/2.0) / (DT*(DT2-delta*DDT) + DDDT*delta*delta/6.0)
		err=abs(xold-xnew)
		xold=xnew
		it=it+1
		
	return xnew,it, err

def derivatives_tau(tau,xi,lam,N=0):
	from numpy import sqrt,exp
	x = exp(xi)-1	
	T = exp(tau)	
	y = sqrt(1-lam**2*(1-x*x))
	DT,DDT,DDDT = derivatives(T,x,lam)
	if N==0:
		Dtau = DT*(1+x)/T
		DDtau = (1+x)**2/T*DDT+Dtau-Dtau*Dtau
		DDDtau = (1+x)**3/T*DDDT + (DDtau-Dtau+Dtau*Dtau)*(2-Dtau) + DDtau - 2* Dtau*DDtau
	else:
		Dtau = (1-x**2)/2/T*DT
		DDtau = (1-x**2)**2/4.0/T*DDT - x*Dtau-Dtau*Dtau
		DDDtau = (1-x**2)**3/8/T*DDDT - (DDtau+x*Dtau+Dtau*Dtau)*(2*x+Dtau) - (1-x**2)/2*Dtau-x*DDtau- 2* Dtau*DDtau
	return Dtau,DDtau,DDDtau

def newton_tau(tau,xi0,lam,N=0,eps=1e-13, iter_max = 3):
	it=0
	from numpy import log,exp
	err = 1
	xiold = xi0
	while ((err>eps) and it < iter_max):
		tof = log( x2tof2(exp(xiold)-1,lam,N) )
		Dtau,_,_ = derivatives_tau(tof,xiold,lam)
		xinew = xiold - (tof-tau) / Dtau
		err=abs(xiold-xinew)
		xiold=xinew
		it=it+1
	return xinew,it, err

def halley_tau(tau,xi0,lam,N=0,eps=1e-13, iter_max = 3):
	it=0
	from numpy import log,exp
	err = 1
	xiold = xi0
	while ((err>eps) and it < iter_max):
		tof = log( x2tof2(exp(xiold)-1,lam,N) )
		Dtau,DDtau,_ = derivatives_tau(tof,xiold,lam)
		dt = tau-tof
		if (not dt == 0):
			xinew = xiold + dt * Dtau / (Dtau * Dtau + dt * DDtau / 2.0);
		err=abs(xiold-xinew)
		xiold=xinew
		it=it+1
	return xinew,it,err


def householder_tau(tau,xi0,lam,N=0,eps=1e-13, iter_max = 3):
	from numpy import log,exp
	it=0
	err = 1
	xiold = xi0
	while ((err>eps) and it < iter_max):
		tof = log( x2tof2(exp(xiold)-1,lam,N) )
		Dtau,DDtau,DDDtau = derivatives_tau(tof,xiold,lam)
		xinew = xiold - (tof-tau) * (Dtau*Dtau-(tof-tau)*DDtau/2) / (Dtau*Dtau*Dtau-(tof-tau)*Dtau*DDtau + DDDtau*(tof-tau)*(tof-tau)/6)
		err=abs(xiold-xinew)
		xiold=xinew
		it=it+1
	return xinew,it,err

def make_table(eps=1e-14):
	from numpy import log,exp,linspace
	lam=linspace(-0.99,0.99,10)
	x_true = linspace(-0.9,3,10)
	resITER = list()
	resX = list()
	for i in xrange(len(lam)):
		lineresITER = list()
		lineresX = list()
		for j in xrange(len(x_true)):
			T_true = x2tof2(x_true[j],lam[i],N=0)
			x0=gooding_guess(lam[i],T_true,0)
			#x0=0.5
			#x0 = -0.5
			#x1,it1 = newton_tau(log(T[j]),log(x0+1),lam[i],eps=eps,iter_max=10)
			#x,it,_ = halley_tau(log(T[j]),log(x0+1),lam[i],eps=eps,iter_max=10)
			#xi,it,_ = householder_tau(log(T_true),log(x0+1),lam[i],eps=eps,iter_max=10)
			#x0=gooding_guess(lam[i],T[j],0)
			#x2,it2 = newton(T[j],x0,lam[i],eps)
			x,it,_ = halley(T_true,x0,lam[i],eps=eps,iter_max=10)
			#x2,it2 = householder(T[j],x0,lam[i],eps)
			lineresITER.append(it)
			lineresX.append(x-x_true)
			#lineresX.append(exp(xi)-1-x_true[j])
		resITER.append(lineresITER)
		resX.append(lineresX)
	return resX, resITER

def gooding_guess(lam,TT,N=0):
	from numpy import arcsin, pi,sqrt
	from math import atan2
	# our T is half as much as Lancaster
	T = TT *2
	if N==0:
		T0 = x2tof2(0.0,lam,N) * 2
		phi = atan2(1-lam**2,2*lam)
		if T<T0: # x > 0
			return (T0*(T0-T)/4.0/T)
		else: # x < 0
			x01 = -(T-T0)/(T-T0+4.0)
			x02 = -sqrt((T-T0)/(T+T0/2.0))
			W = x01 + 1.7*sqrt(2-phi/pi)
			if W>= 0:
				x03=x01
			else:
				x03 = x01 + (-W)**(1.0/16.0) * (x02-x01)
			c1=0.5
			c2 = 0.03
			l = 1+c1*x03*(1+x01)-c2*(x03*x03)*sqrt(1+x01)
			return (l * x03) 
	elif N>0:
		# We find XM
		phi = atan2(1-lam**2,2*lam)
		XMpi = 4.0 / (3*pi*(2*N+1))
		if phi < pi:
			XM0 = XMpi * (phi/pi)**(1.0/8.0)
		else:
			XM0 = XMpi * ( 2 - (2-phi/pi)**(1.0/8.0) )
		xM,tM,_,_ = halley_xM(XM0,T,lam,N,eps=1e-13)
		return my_guess(lam,T,N=N)
		
def my_guess(lam,T,N=0):
	from math import pi,log
	if N==0:
		T0 = x2tof2(0,lam,N=0)
		T1 = 2.0/3.0*(1-lam**3.0)
		log2 = log(2)
		if T>T0:
			#return (T0/T)**(2.0/3.0) - 1
			return -(T-T0)/(T-T0+4)
		elif T<T1:
			#return 2*(T1/T) - 1
			return T1*(T1-T) / ( 2.0/5.0*(1-lam**5) * T ) + 1
		else:	
			return (T/T0)**(log2/ log(T1/T0)) - 1
	elif N>0:
		Ar = (8*T/(N*pi))**(2.0/3.0)
		Al = ((N*pi+pi)/(8.0*T))**(2.0/3.0)
		xr = (Ar-1)/(Ar+1)
		xl = (Al-1)/(Al+1)
		return xl,xr

def make_table2(niter=3):
	from numpy import log,exp,linspace
	lam=linspace(-0.99,0.99,10)
	x_true = linspace(-0.99,3,10)
	resX = list()
	for i in xrange(len(lam)):
		lineresX = list()
		for j in xrange(len(x_true)):
			T = x2tof2(x_true[j],lam[i],N=0)
			x0=gooding_guess(lam[i],T,0)
			#x,it = newton_tau(log(T),log(x0+1),lam[i])
			#x,it,_ = halley_tau(log(T),log(x0+1),lam[i],iter_max=niter)
			#x,it,_ = householder_tau(log(T),log(x0+1),lam[i],iter_max=niter)
			#x,it = newton(T[j],x0,lam[i],eps)
			#x,it,_ = halley(T,x0,lam[i],iter_max=niter,eps=1e-16)
			#x,it,_ = householder(T,x0,lam[i])
			#TT = x2tof2(exp(x)-1,lam[i],N=0)
			TT = x2tof2(x,lam[i],N=0)
			lineresX.append(T-TT)
		resX.append(lineresX)
	return resX

def random_lambda():
	from math import sqrt,pi,cos
	from numpy.random import rand
	#r1 = 10.0*rand()
	#r2 = 10.0*rand()
	#theta = 2.0*pi*rand()
	#c = sqrt(r1**2+r2**2-2.0*r1*r2*cos(theta))
	#s = (r1+r2+c)/2.0
	#return sqrt(r1*r2) / s * cos(theta/2.0)
	return -0.999 + rand()*(0.999*2)

def test_standard(M=1000,gooding=True,house=False,tau=False,
		iter_max=3,eps=1e-13,rnd_seed=1234,N=0):
	from numpy.random import rand, seed
	from math import exp,log,pi
	from numpy import sign, array

	seed(rnd_seed)
	la = array([0.0]*M)
	x = array([0.0]*M)
	error = array([0.0]*M)
	it = array([0.0]*M)
	T = array([0.0]*M)

	for i in xrange(M):
		la[i] = random_lambda()

		if N==0:
			x_true = -0.99 + 3*rand()
			T[i] = x2tof2(x_true,la[i],N=N)
			if gooding:
				x0=gooding_guess(la[i],T[i],0)
			else:
				x0=my_guess(la[i],T[i],0)
			if tau:
				if (house):
					x[i],it[i],_ = householder_tau(log(T[i]),
						log(x0+1),la[i],iter_max=iter_max,eps=eps)
				else:
					x[i],it[i],_ = halley_tau(log(T[i]),log(x0+1),
						la[i],iter_max=iter_max,eps=eps)
				x[i] = exp(x[i]) + 1
			else:
				if (house):
					x[i],it[i],_ = householder(T[i],x0,
						la[i],iter_max=iter_max,eps=eps)
				else:
					x[i],it[i],_ = halley(T[i],x0,la[i],
						iter_max=iter_max,eps=eps)
			error[i] = abs(x_true - x[i])
		else:
			x_true = -0.99+(rand()*(0.99+0.99))
			T[i] = x2tof2(x_true,la[i],N=N)
			if gooding:
				x0l,x0r=gooding_guess(la[i],T[i],N)
			else:
				x0l,x0r=my_guess(la[i],T[i],N)
			if tau:
				print "Not implemented"
				return
			else:
				if (house):
					xr,itr,_ = householder(T[i],x0r,
						la[i],iter_max=iter_max,eps=eps,N=N)
				else:
					xr,itr,_ = halley(T[i],x0r,la[i],
						iter_max=iter_max,eps=eps,N=N)
				err_r = abs(x_true - xr)
				if (house):
					xl,itl,_ = householder(T[i],x0l,
						la[i],iter_max=iter_max,eps=eps,N=N)
				else:
					xl,itl,_ = halley(T[i],x0l,la[i],
						iter_max=iter_max,eps=eps,N=N)
				err_l = abs(x_true - xl)
				if err_l<err_r:
					x[i]=xl
					it[i]=itl
					error[i] = err_l
				else:
					x[i]=xr
					it[i]=itr
					error[i] = err_r

	return error,it,la,x,T

def findN(T,T0,lam):
	from math import pi
	M = int(T/pi)
	T0M = T0+M*pi
	if T > T0M:
		return M
	xM,TM,_,_ = halley_xM(0.1,T0M,lam,N=M,eps=1e-13)
	if TM < T:
		return M
	return M-1
		

def F(a,b,c,z,tol=1e-13):
	Sj=1
	Cj=1
	err=1.0
	j=0
	while err > tol:
		Cj1 = Cj*(a+j)*(b+j)/(c+j)*z/(j+1)
		Sj1 = Sj + Cj1
		err=abs(Cj1)
		Sj = Sj1
		Cj=Cj1
		j=j+1
	return Sj,j
	
def tlamb(m, q, qsqfm1, x, n):

	# Gooding lambert support function

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	from math import sqrt,atan2,pi,log

	lm1 = False;

	l1 = False;

	l2 = False;

	l3 = False;

	sw = 0.4;

	lm1 = n == -1;

	l1 = n >= 1;

	l2 = n >= 2;

	l3 = n == 3;

	qsq = q * q;

	xsq = x * x;

	u = (1.0 - x) * (1.0 + x);

	# needed if series, and otherwise useful when z = 0

	if (~lm1):
	    dt = 0.0;

	    d2t = 0.0;

	    d3t = 0.0;
	

	if (lm1 or m > 0 or x < 0.0 or abs(u) > sw):
	    # direct computation, not series
	    
	    y = sqrt(abs(u));

	    z = sqrt(qsqfm1 + qsq * xsq);

	    qx = q * x;

	    if (qx <= 0.0):
		a = z - qx;
		b = q * z - x;
	    

	    if (qx < 0 and lm1):
		aa = qsqfm1 / a;
		bb = qsqfm1 * (qsq * u - xsq) / b;
	    

	    if (qx == 0.0 and lm1 or qx > 0.0):
		aa = z + qx;
		bb = q * z + x;
	    

	    if (qx > 0):
		a = qsqfm1 / aa;
		b = qsqfm1 * (qsq * u - xsq) / bb;
	    

	    if (lm1):
		dt = b;
		d2t = bb;
		d3t = aa;
	    else:
		if (qx * u >= 0.0):
		    g = x * z + q * u;
		else:
		    g = (xsq - qsq * u) / (x * z - q * u);
		
		f = a * y;

		if (x <= 1):
		    t = m * pi + atan2(f, g);
		else:
		    if (f > sw):
		        t = log(f + g);
		    else:
		        fg1 = f / (g + 1.0);
		        term = 2.0 * fg1;
		        fg1sq = fg1 * fg1;
		        t = term;
		        twoi1 = 1.0;
		        told = 0.0;

		        while (t != told):
		            twoi1 = twoi1 + 2.0;
		            term = term * fg1sq;
		            told = t;
		            t = t + term / twoi1;	        

		t = 2.0 * (t / y + b) / u;

		if (l1 and z != 0.0):
		    qz = q / z;
		    qz2 = qz * qz;
		    qz = qz * qz2;
		    dt = (3.0 * x * t - 4.0 * (a + qx * qsqfm1) / z) / u;

		    if (l2):
		        d2t = (3.0 * t + 5.0 * x * dt + 4.0 * qz * qsqfm1) / u;
		    

		    if (l3):
		        d3t = (8.0 * dt + 7.0 * x * d2t - 12.0 * qz * qz2 * x * qsqfm1) / u;
		    	    
	else:
	    # compute by series
	    
	    u0i = 1.0;

	    if (l1):
		u1i = 1.0;
	    
	    if (l2):
		u2i = 1.0;
	    
	    if (l3):
		u3i = 1.0;
	    
	    term = 4.0;

	    tq = q * qsqfm1;

	    if (q < 0.5):
		tqsum = 1.0 - q * qsq;
	    

	    if (q >= 0.5):
		tqsum = (1.0 / (1.0 + q) + q) * qsqfm1;
	    

	    ttmold = term / 3.0;

	    t = ttmold * tqsum;

	    # start of loop
	    
	    icounter = 0;

	    told = 0;

	    while (icounter < n or t != told):
		icounter = icounter + 1;

		p = icounter;

		u0i = u0i * u;

		if (l1 and icounter > 1):
		    u1i = u1i * u;
		

		if (l2 and icounter > 2):
		    u2i = u2i * u;
		

		if (l3 and icounter > 3):
		    u3i = u3i * u;
		

		term = term * (p - 0.5) / p;

		tq = tq * qsq;

		tqsum = tqsum + tq;

		told = t;

		tterm = term / (2.0 * p + 3.0);

		tqterm = tterm * tqsum;

		t = t - u0i * ((1.5 * p + 0.25) * tqterm / (p * p - 0.25) - ttmold * tq);

		ttmold = tterm;

		tqterm = tqterm * p;

		if (l1):
		    dt = dt + tqterm * u1i;
		

		if (l2):
		    d2t = d2t + tqterm * u2i * (p - 1.0);
		

		if (l3):
		    d3t = d3t + tqterm * u3i * (p - 1.0) * (p - 2.0);
	    

	    if (l3):
		d3t = 8.0 * x * (1.5 * d2t - xsq * d3t);
	    

	    if(l2):
		d2t = 2.0 * (2.0 * xsq * d2t - dt);
	    

	    if (l1):
		dt = -2.0 * x * dt;
	    

	    t = t / xsq;
	return dt/2, d2t/2, d3t/2, t/2
	
