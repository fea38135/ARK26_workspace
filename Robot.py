import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
from scipy.spatial.transform import Rotation, RotationSpline

class Robot:
    def __init__(self, DH_a, DH_d, DH_alpha, axisType = None, nSols=None):
        self.nAxes = len(DH_a)
        if axisType == None:
            self.axisType = 'R'*self.nAxes
        self.B = [T_link(DH_a[i], DH_d[i], DH_alpha[i]) for i in range(self.nAxes)]
        self.cartLim = 1 # Cartesian max val for plotting (set in specific robot class)
        self.nSols = nSols

    def fkine(self, q, n=None):
        '''
        Compute forward kinematics of robot with arbitrary number of R/P joints. The output poses are given wrt. the KUKAbase coordinate system.

        input:  q:      axis configuration
                n:      compute partial kinematics of n joints

        output: T_EE:   pose of end effector as a matrix
                T:      list containing poses of each joint
        '''
        if n == None:
            n = self.nAxes
        T_EE = np.eye(4)
        T = []

        for i in range(n):
            if self.axisType[i] == 'R':
                T_EE = T_EE * rotz(q[i]) * self.B[i]
            else:
                T_EE = T_EE * transz(q[i])*self.B[i]
            T.append(T_EE)

        return [T_EE,T]
    
    def jac(self, q):
        q = np.array(q)
        if len(q.shape) > 1:
            return np.array([self.jac(qi) for qi in q])
        # as in Angeles
        J = np.eye(6)
        Ts = self.fkine(q)[1]
        Ts = [np.matrix(np.eye(4))] + Ts[:-1]
        p = [t[:3,3] for t in Ts]
        e = np.zeros([6,3])
        r = np.zeros([6,3])
        for i in range(6):
            e[i] = Ts[i][:3,2].T
            r[i] = (p[5] - p[i]).T
            J[:3,i] = e[i]
            J[3:,i] = np.cross(e[i], r[i])
        return J
    
    def jacdet(self, q):
        q = np.array(q)
        if len(q.shape) > 1:
            return np.array([self.jacdet(qi) for qi in q])
        return np.linalg.det(self.jac(q))
    
    def linPath(self, T0, T1, nSteps=100):
        '''
        Interpolate poses between T0 and T1 linearly
        '''
        if len(T0)==6:
            T0 = self.fkine(T0)[0]
        Ts = np.array([T0, T1])
        lmbda = np.linspace(0, 1, nSteps)
        Rs = Rotation.from_matrix(Ts[:,:3,:3])
        R_spline = RotationSpline([0,1],Rs)
        T_spline = np.zeros((nSteps,4,4))
        T_spline[:,3,3] = 1
        T_spline[:,:3,:3] = R_spline(lmbda).as_matrix()
        lmbda = lmbda.reshape(-1,1)
        T_spline[:,:3,3] = (1-lmbda) * Ts[0,:3,3] + lmbda * Ts[1,:3,3]
        return T_spline
    
    def plotPose(self, T, plot_axes=True, plot_a6=False, ar_length=None, ax=None, ax_eq=True, **kwargs):
        '''
        A simple visualisation of the robot. Works only if the robot has just rotational joints.
        input:      T:      Either a list of hom. transf. matrices giving the joint poses wrt. KUKAbase (i.e. as in the output of robot.fkine()/robot.forward()) or a list specifying joint angles.
                    plot_axes:    enables a visualisation of the rotational axes of each joint
                    ar_length:  sets the length of the joint axes
                    ax:     specify an axis for the plot
        output:     axis object
        '''

        if ax is None:
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection='3d')
        if not np.array(T).ndim == 3:
            [_,T] = self.fkine(T)
        T = np.concatenate([[np.eye(4)],T])
        T_pos = np.array(T)
        T_pos = T_pos[:,:3,-1]
        ax.plot(T_pos[:,0],T_pos[:,1],T_pos[:,2],'b', **kwargs)
        if ar_length is None:
            ar_length = 1/20*self.cartLim
        if plot_axes:
            for t in T[:-1]:
                dir = np.array(np.matrix(t[:-1,:-1])*np.matrix([[0,0,-ar_length/2],[0,0,ar_length/2]]).transpose())
                dir = np.array(dir+t[:-1,-1:])
                ax.plot(dir[0],dir[1],dir[2],'r', linewidth=2)
        if plot_a6:
            t = T[-1]
            dir = np.array(np.matrix(t[:-1,:-1])*np.matrix([[0,0,0],[0,0,ar_length/2]]).transpose())
            dir = np.array(dir+t[:-1,-1:])
            ax.plot(dir[0],dir[1],dir[2],'r', linewidth=2)
        if ax_eq:
            ax.axis('equal')
        return ax
    
    def plotIKnSols(self, steps=[201,501], lims=None, nSols=None, circ_res=500, fignum=11):
        if nSols is None:
            nSols = self.nSols
        rot_steps = steps[0]
        cart_steps = steps[1]
        if lims == None:
            lims = self.cartLim
        x_max = lims
        y_max = x_max
        z_max = x_max
        nSol_shape = nSols.shape
        sol_steps = [x_max / nSol_shape[0], z_max / nSol_shape[1], 2*pi / nSol_shape[2], pi/nSol_shape[3]]

        alphas = np.linspace(0, 2*pi, rot_steps)
        betas = np.linspace(0, pi, int(rot_steps/2))
        xs = np.linspace(0, x_max, cart_steps)
        ys = np.linspace(0, y_max, cart_steps)
        zs = np.linspace(0, z_max, cart_steps)

        cols = ['g','purple','c','yellow','cyan','magenta','b','grey']
        fig = plt.figure(fignum)
        fig.clf()
        ax = fig.add_subplot(121, projection='3d', computed_zorder=False)

        T = T_euler(0,0,0,0,0)

        circ = self.getCirclePoints(T, n = 25)
        cx, cy, cz = circ.T
        ax.plot(cx,cy,cz,'orange')

        sols = self.numIK(T_euler(0,0,0,0,0), n=circ_res)
        for i in range(np.min([len(sols),8])): # ignores more than 8 sols
            self.plotPose(sols[i][1], ax = ax, plot_axes=False, c=cols[i], linestyle='--', linewidth=0.8)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.set_xlim((0,self.cartLim))
        ax.set_ylim((0,self.cartLim))
        ax.set_zlim((0,self.cartLim))

        ax2 = fig.add_subplot(122)

        a,b,r,z = 0,0,0,0
        im = ax2.imshow(nSols[:,:,a,b].T, origin='lower', extent=[0,self.cartLim]*2, cmap='inferno', vmin=0, vmax=9)
        ax2.set_xlabel('rho')
        ax2.set_ylabel('z')
        # ax2.get_xaxis().set_ticks([])
        # ax2.get_yaxis().set_ticks([])


        ax2.scatter(0,nSol_shape[1])

        fig.subplots_adjust(left=0.3)

        ax_a = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
        ax_b = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
        ax_x = fig.add_axes([0.15, 0.25, 0.0225, 0.63])
        ax_y = fig.add_axes([0.2, 0.25, 0.0225, 0.63])
        ax_z = fig.add_axes([0.25, 0.25, 0.0225, 0.63])

        slider_a = Slider(
            ax=ax_a,
            label="a1",
            valmin = alphas[0],
            valmax = alphas[-1],
            valstep=alphas,
            valinit=0,
            orientation="vertical"
        )
        slider_b = Slider(
            ax=ax_b,
            label="b1",
            valmin = betas[0],
            valmax = betas[-1],
            valstep=betas,
            valinit=0,
            orientation="vertical"
        )
        slider_x = Slider(
            ax=ax_x,
            label="x",
            valmin = xs[0],
            valmax = xs[-1],
            valstep=xs,
            valinit=0,
            orientation="vertical"
        )
        slider_y = Slider(
            ax=ax_y,
            label="y",
            valmin = ys[0],
            valmax = ys[-1],
            valstep=ys,
            valinit=0,
            orientation="vertical"
        )
        slider_z = Slider(
            ax=ax_z,
            label="z",
            valmin = zs[0],
            valmax = zs[-1],
            valstep=zs,
            valinit=0,
            orientation="vertical"
        )

        def update(val):
            nonlocal a,b,r,z
            lines = ax.get_lines()
            for line in lines:
                line.remove()
                del line
            for scatter in ax2.collections:
                scatter.remove()
            
            T = T_euler(slider_a.val, slider_b.val, slider_x.val, slider_y.val, slider_z.val)
            if btn2state:
                circ = self.getCirclePoints(T, n = 25)
                cx, cy, cz = circ.T
                ax.plot(cx,cy,cz,'orange')

            sols = self.numIK(T, n=circ_res)
            for i in range(np.min([len(sols),8])):
                self.plotPose(sols[i][1], plot_axes=False, ax_eq=False, plot_a6=True, ax = ax, c=cols[i], linestyle='--', linewidth=0.8) 
            
            [a,b,r,z] = get_euler(T, True)
            # rho = np.sqrt(slider_x.val**2 + slider_y.val**2)
            # a1 = slider_a.val - np.arctan2(slider_y.val, slider_x.val)

            if btnstate:
                im.set_data(nSols[:,:,int(a/sol_steps[2]),int(b/sol_steps[3])].T)
                sc = ax2.scatter(r, z)
            else:
                im.set_data(nSols[int(r/sol_steps[0]),int(z/sol_steps[1]),:,:].T)
                sc = ax2.scatter(a, b)

            fig.canvas.draw_idle()
            
        slider_a.on_changed(update)
        slider_b.on_changed(update)
        slider_x.on_changed(update)
        slider_y.on_changed(update)
        slider_z.on_changed(update)

        ax_btn = fig.add_axes([0.05, 0.1, 0.4, 0.07])
        btn = Button(ax_btn, 'Change View')
        btnstate = True
        def btn_action(event):
            nonlocal btnstate
            nonlocal im, a, b, r, z
            btnstate = not btnstate
            ax2.clear()
            if btnstate:
                im = ax2.imshow(nSols[:,:,int(a/sol_steps[2]),int(b/sol_steps[3])].T, origin='lower', extent=[0,self.cartLim]*2, cmap='inferno', vmin=0, vmax=9)
                ax2.set_xlabel('rho')
                ax2.set_ylabel('z')
            else:
                im = ax2.imshow(nSols[int(r/sol_steps[0]),int(z/sol_steps[1]),:,:].T, origin='lower', extent=[0,2*pi,0,pi], cmap='inferno', vmin=0, vmax=9)
                ax2.set_xlabel('a')
                ax2.set_ylabel('b')
                
            fig.canvas.draw_idle()
        btn.on_clicked(btn_action)

        ax_btn2 = fig.add_axes([0.5, 0.1, 0.4, 0.07])
        btn2 = Button(ax_btn2, 'Plot circle around joint 6')
        btn2state = True
        def btn2_action(event):
            nonlocal btn2state
            btn2state = not btn2state
        btn2.on_clicked(btn2_action)

        plt.show()
    
    def plotnSols(self, steps=[201,501], lims=None, nSols=None, circ_res=500, fignum=11):
        # plt.rcParams['text.usetex'] = True
        if nSols is None:
            nSols = self.nSols
        rot_steps = steps[0]
        cart_steps = steps[1]
        if lims == None:
            lims = self.cartLim
        x_max = lims
        y_max = x_max
        z_max = x_max
        nSol_shape = nSols.shape
        sol_steps = [x_max / nSol_shape[0], z_max / nSol_shape[1], 2*pi / nSol_shape[2], pi/nSol_shape[3]]

        alphas = np.linspace(0, 2*pi, rot_steps)
        betas = np.linspace(0, pi, int(rot_steps/2))
        xs = np.linspace(0, x_max, cart_steps)
        ys = np.linspace(0, y_max, cart_steps)
        zs = np.linspace(0, z_max, cart_steps)

        fig = plt.figure(fignum)
        fig.clf()
        ax2 = fig.add_subplot(121)
        init_a = 0
        init_b = 0
        im = ax2.imshow(nSols[:,:,init_a,init_b].T, origin='lower', extent=[0,self.cartLim]*2, cmap='inferno', vmin=0, vmax=9)
        ax2.set_xlabel('r')
        ax2.set_ylabel('z')
        # ax2.get_xaxis().set_ticks([])
        # ax2.get_yaxis().set_ticks([])
        ax2.scatter(0,0)

        ax3 = fig.add_subplot(122)
        init_a = 0
        init_b = 0
        im3 = ax3.imshow(nSols[0,0,:,:].T, origin='lower', extent=[0,2*pi,0,pi], cmap='inferno', vmin=0, vmax=9)
        ax3.set_xlabel('a')
        ax3.set_ylabel('b')
        # ax3.get_xaxis().set_ticks([])
        # ax3.get_yaxis().set_ticks([])
        ax3.scatter(0,0)

        fig.subplots_adjust(left=0.25)
        ax_a = fig.add_axes([0.03, 0.25, 0.0225, 0.63])
        ax_b = fig.add_axes([0.06, 0.25, 0.0225, 0.63])
        ax_x = fig.add_axes([0.09, 0.25, 0.0225, 0.63])
        ax_y = fig.add_axes([0.12, 0.25, 0.0225, 0.63])
        ax_z = fig.add_axes([0.15, 0.25, 0.0225, 0.63])
        slider_a = Slider(
            ax=ax_a,
            label="a1",
            valmin = alphas[0],
            valmax = alphas[-1],
            valstep=alphas,
            valinit=0,
            orientation="vertical"
        )
        slider_b = Slider(
            ax=ax_b,
            label="b1",
            valmin = betas[0],
            valmax = betas[-1],
            valstep=betas,
            valinit=0,
            orientation="vertical"
        )
        slider_x = Slider(
            ax=ax_x,
            label="x",
            valmin = xs[0],
            valmax = xs[-1],
            valstep=xs,
            valinit=0,
            orientation="vertical"
        )
        slider_y = Slider(
            ax=ax_y,
            label="y",
            valmin = ys[0],
            valmax = ys[-1],
            valstep=ys,
            valinit=0,
            orientation="vertical"
        )
        slider_z = Slider(
            ax=ax_z,
            label="z",
            valmin = zs[0],
            valmax = zs[-1],
            valstep=zs,
            valinit=0,
            orientation="vertical"
        )
        ax_btn = fig.add_axes([0.05, 0.1, 0.7, 0.07])
        btn = CheckButtons(ax_btn, ["Use modified reference frame for a1, b1\n otherwise a1 and b1 are in world coordinate system, a and b in modified \n (will take effect after moving slider)"]) #, $(\phi, \psi) = (\alpha, \beta)$

        def update(val):
            for scatter in ax2.collections:
                scatter.remove()
            for scatter in ax3.collections:
                scatter.remove()
                
            if not btn.get_status()[0]:
                T = T_euler(slider_a.val, slider_b.val, slider_x.val, slider_y.val, slider_z.val)
                [a,b,r,z] = get_euler(T, True)
            else:
                [a,b,r,z] = [slider_a.val, slider_b.val, np.linalg.norm([slider_x.val, slider_y.val]), slider_z.val]
            
            im.set_data(nSols[:,:,int(a/sol_steps[2]),int(b/sol_steps[3])].T)
            ax2.scatter(r, z)
            # ax2.scatter(r/sol_steps[0], nSol_shape[1]-z/sol_steps[1])
            im3.set_data(nSols[int(r/sol_steps[0]), int(z/sol_steps[1]),:,:].T)
            ax3.scatter(a, b)
            # ax3.scatter(a/sol_steps[2], b/sol_steps[3])

            fig.canvas.draw_idle()

            
        slider_a.on_changed(update)
        slider_b.on_changed(update)
        slider_x.on_changed(update)
        slider_y.on_changed(update)
        slider_z.on_changed(update)

        plt.show()
    
class Transpressor(Robot):
    def __init__(self, a2, a5, d1, d4, nSols=None):
        self.a2, self.a5, self.d1, self.d4 = a2, a5, d1, d4
        DH_a = [0, a2, 0, 0, a5, 0]
        DH_d = [d1, 0, 0, d4, 0, 0]
        DH_alpha = [pi/2,0,-pi/2,pi/2,-pi/2,0]
        super().__init__(DH_a, DH_d, DH_alpha, nSols=nSols)
        self.cartLim = a2 + d4 + a5
        
    def wristIK(self, p):
        # Compute the four inverse kinematik of the first three axes to reach given point with the "wrist", takes positional vector p as input

        l1 = self.a2
        l2 = self.d4

        z = p[2]-self.d1
        rho2 = p[0]**2 + p[1]**2

        r2z2 = rho2 + z**2

        if (l1+l2)**2 < r2z2 or (l1-l2)**2 > r2z2:
                return np.array([None]*4)

        q0 = np.arctan2(p[1],p[0])

        c2 = (r2z2 - l1**2 - l2**2)/(2*l1*l2)
        s2 = np.sqrt(abs(1-c2**2)) # abs is just to avoid errors edge cases not considered above, maybe the abs causes errors?
        s2 = np.array([s2,-s2]) # ellbow up/down

        q2 = np.arctan2(s2,c2)
        q1 = np.arctan2(z, np.sqrt(rho2)) - np.arctan2(l2*s2,l1+l2*c2)

        sol01 = [[q0,q1[i],q2[i]-pi/2] for i in range(2)]
        sol23 = [[pi+q0,pi-q1[i],-q2[i]-pi/2] for i in range(2)]

        sol = sol01+sol23
        sol = [[(s)%(2*pi) for s in so] for so in sol]

        return np.array(sol)
        
    def circleAroundTarget(self, T_t, alpha):
        # returns point on circle around T_t with radius a5, parametrized by alpha
        p0 = np.matrix([self.a5,0,0,1]).transpose()
        return np.array(T_t*rotz(alpha)*p0).flatten()[:3]
    
    def getCirclePoints(self, T_t, n=100, l=None):
        # returns circle around T_t with radius a5, given by n evenly spaced points or for all angles defined in l (if l is given, n is ignored)
        if l is None:
            l = np.linspace(0,2*pi,n)
        return np.array([self.circleAroundTarget(T_t,al) for al in l])
    
    def dotProdCirc(self, T_t, n=100, l=None):
        # computes the wristIK on the circle points and dot product of axis 4 with the tangent to the circle (only consider ellbow up/down, for shoulder one case is enough)
        if l is None:
            l = np.linspace(0,2*pi,n)            
        p_t = np.array(T_t[:3,3]).flatten() # target point
        cps = self.getCirclePoints(T_t, l=l)
        cptangent = self.getCirclePoints(T_t, l=l+pi/2) - p_t # tangent directions of the circle
        
        dp = np.zeros((2,n))
        q = np.zeros((2,n,3))
        sols = []
        isol = [[],[]]

        for i in range(n):
            iks = self.wristIK(cps[i])

            # either none or all the solutions exist
            if iks[0] is None:
                    dp[:,i] = [None, None]
                    continue
            
            dk0 = self.fkine(iks[0], 3)[0] # compute wrist forward kinematic, iks no 0
            wz0 = np.array(dk0[:3,2]).flatten() # z-axis of joint 4 
            dp[0,i] = np.dot(cptangent[i],wz0)
            q[0,i] = iks[0]

            dk1 = self.fkine(iks[1], 3)[0] # compute wrist forward kinematic, iks no 1
            wz1 = np.array(dk1[:3,2]).flatten() # z-axis of joint 4 
            dp[1,i] = np.dot(cptangent[i],wz1)
            q[1,i] = iks[1]

            # save solutions (approximate a zero in the dot product by detecting a sign change)
            if dp[0,i-1] is not None:                    
                if (np.sign(dp[0,i-1]) + np.sign(dp[0,i]) == 0) or np.sign(dp[0,i]) == 0:
                    sols.append([0, iks[0], cptangent[i], dk0, i])
                    isol[0].append(i)
                    if i==n-1 and len(isol[0])>0:
                        if isol[0][0] < 2:
                            sols = sols[:-1]        #avoid counting edge cases double
                            isol[0] = isol[0][:-1]

                if np.sign(dp[1,i-1]) + np.sign(dp[1,i]) == 0 or np.sign(dp[1,i]) == 0:
                    sols.append([1, iks[1], cptangent[i], dk1, i])
                    isol[1].append(i)
                    if i==n-1 and len(isol[1])>0:
                        if isol[1][0] < 2:
                            sols = sols[:-1]        #avoid counting edge cases double
                            isol[1] = isol[1][:-1]
        
        return [l, dp, q, sols]
        
    def numIK(self, T_t, n=100, compute_all = False):
        _, _, _, wristSols = self.dotProdCirc(T_t, n)
        sols = []
        for ws in wristSols:
            q = np.append(ws[1],[0,0,0])

            # compute 4th angle as the angle between the third x-axis and the tangent to the circle
            x3 = np.array(ws[3][:3,0]).flatten()
            z3 = np.array(ws[3][:3,2]).flatten()
            tangent = ws[2]

            t4 = np.arctan2(np.dot(np.cross(x3,tangent),z3), np.dot(x3,tangent))
            t4 = t4 + pi/2 

            q = np.append(ws[1],[t4,0,0])


            # compute 5th angle as the angle between the fourth x-axis and the line from p4 to target
            T4 = self.fkine(q,4)[0] # actually one could just store T3 above so that not the entire fkine is necessary
            p4 = np.array(T4[:3,3]).flatten()
            x4 = np.array(T4[:3,0]).flatten()
            z4 = np.array(T4[:3,2]).flatten()
            p_t = np.array(T_t[:3,3]).flatten()
            tardir = p_t - p4

            t5 = np.arctan2(np.dot(np.cross(x4,tardir),z4), np.dot(x4,tardir))

            q = np.append(ws[1],[t4,t5,0])


            #check if orientation of final z-axis coincides with target z-axis
            T5 = self.fkine(q)[0] # actually one could just store T4 above so that not the entire fkine is necessary
            z5 = np.array(T5[:3,2]).flatten()
            z_t = np.array(T_t[:3,2]).flatten()
            diff1 = np.linalg.norm(z5-z_t)
            if diff1 > 0.1:
                t4a = t4 + pi
                t5a = pi - t5
                T5a = self.fkine(np.append(ws[1],[t4a,t5a,0]))[0]
                z5a = np.array(T5a[:3,2]).flatten()
                if np.linalg.norm(z5a-z_t) < diff1:
                    t4 = t4a
                    t5 = t5a
            q = np.append(ws[1],[t4,t5,0])


            # compute last angle
            T5 = self.fkine(q)[0] # actually one could just store T4 above so that not the entire fkine is necessary
            x5 = np.array(T5[:3,0]).flatten()
            z5 = np.array(T5[:3,2]).flatten()
            x_t = np.array(T_t[:3,0]).flatten()

            t6 = np.arctan2(np.dot(np.cross(x5,x_t),z5), np.dot(x5,x_t))

            q = np.append(ws[1],[t4%(2*pi),t5%(2*pi),t6%(2*pi)])

            sols.append([ws[0],q])

        if compute_all:
            ls = len(sols)
            for i in range(ls):
                sol = sols[i]
                q = sol[1].copy()
                q[0] = (q[0] + pi)%(2*pi)
                q[1] = (-q[1] + pi)%(2*pi)
                q[2] = (-q[2] + pi)%(2*pi)
                q[3] = (q[3] + pi)%(2*pi)
                sols.append([sol[0]+2,q])
        return sols


class FanucCR(Robot):
        def __init__(self, a2=540, d4=540, d5=150, nSols=None):
            self.a2, self.d4, self.d5 = a2, d4, d5
            DH_a = [0, a2, 0, 0, 0, 0]
            DH_d = [0, 0, 0, -d4, d5, 0]
            DH_alpha = [-pi/2,pi,-pi/2,pi/2,-pi/2,0]
            super().__init__(DH_a, DH_d, DH_alpha, nSols=nSols)
            self.cartLim = a2+d4+d5
        
        def wristIK(self, p):
            # Compute the four inverse kinematik of the first three axes to reach given point with the "wrist", takes positional vector p as input

            l1 = self.a2
            l2 = self.d4

            z = p[2]#-self.d1
            rho2 = p[0]**2 + p[1]**2

            r2z2 = rho2 + z**2

            if (l1+l2)**2 < r2z2 or (l1-l2)**2 > r2z2:
                 return np.array([None]*4)

            q0 = np.arctan2(p[1],p[0])

            c2 = (r2z2 - l1**2 - l2**2)/(2*l1*l2)
            s2 = np.sqrt(abs(1-c2**2)) # abs is just to avoid errors edge cases not considered above, maybe the abs causes errors?
            s2 = np.array([s2,-s2]) # ellbow up/down

            q2 = np.arctan2(s2,c2)
            q1 = np.arctan2(z, np.sqrt(rho2)) - np.arctan2(l2*s2,l1+l2*c2)

            sol01 = [[q0,-q1[i],q2[i]+pi/2] for i in range(2)]
            sol23 = [[pi+q0,pi+q1[i],-q2[i]+pi/2] for i in range(2)]

            sol = sol01+sol23
            sol = [[(s)%(2*pi) for s in so] for so in sol]

            return np.array(sol)
        
        def circleAroundTarget(self, T_t, alpha):
            # returns point on circle around T_t with radius a5, parametrized by alpha
            p0 = np.matrix([self.d5,0,0,1]).transpose()
            return np.array(T_t*rotz(alpha)*p0).flatten()[:3]
        
        def getCirclePoints(self, T_t, n=100, l=None):
            # returns circle around T_t with radius a5, given by n evenly spaced points or for all angles defined in l (if l is given, n is ignored)
            if l is None:
                l = np.linspace(0,2*pi,n)
            return np.array([self.circleAroundTarget(T_t,al) for al in l])
        
        def dotProdCirc(self, T_t, n=100, l=None):
            # computes the wristIK on the circle points and dot product of axis 4 with the tangent to the circle (only consider ellbow up/down, for shoulder one case is enough)
            if l is None:
                l = np.linspace(0,2*pi,n+1)[:-1]            
            p_t = np.array(T_t[:3,3]).flatten() # target point
            cps = self.getCirclePoints(T_t, l=l)
            cptangent = cps - p_t # axis 5 directions of the circle
            #q = np.zeros((2,n,3))
            q=None
            sols = []
            sol_indices = [[],[]]
            dp = None # not computed

            # compute 'i=-1'
            dp_sign = np.zeros(2)
            dp_sign_prev = np.empty(2)
            iks = self.wristIK(cps[-1])

            # either none or all the solutions exist
            if iks[0] is None:
                dp_sign_prev[:] = [None, None]
            else:
                dk0 = self.fkine(iks[0], 3)[0] # compute wrist forward kinematic, iks no 0
                wz0 = np.array(dk0[:3,2]).flatten() # z-axis of joint 4 
                dp_sign_prev[0] = np.sign(np.dot(cptangent[-1],wz0))
                #q[0,-1] = iks[0]

                dk1 = self.fkine(iks[1], 3)[0] # compute wrist forward kinematic, iks no 1
                wz1 = np.array(dk1[:3,2]).flatten() # z-axis of joint 4 
                dp_sign_prev[1] = np.sign(np.dot(cptangent[-1],wz1))
                #q[1,-1] = iks[1]

            for i in range(n):
                iks = self.wristIK(cps[i])

                # either none or all the solutions exist
                if iks[0] is None:
                    dp_sign_prev[:] = [None, None]
                    continue
                
                dk0 = self.fkine(iks[0], 3)[0] # compute wrist forward kinematic, iks no 0
                wz0 = np.array(dk0[:3,2]).flatten() # z-axis of joint 4 
                dp_sign[0] = np.sign(np.dot(cptangent[i],wz0))
                #q[0,i] = iks[0]

                dk1 = self.fkine(iks[1], 3)[0] # compute wrist forward kinematic, iks no 1
                wz1 = np.array(dk1[:3,2]).flatten() # z-axis of joint 4 
                dp_sign[1] = np.sign(np.dot(cptangent[i],wz1))
                #q[1,i] = iks[1]

                # save solutions (approximate a zero in the dot product by detecting a sign change)
                if not np.isnan(dp_sign_prev[0]) and dp_sign[0] - dp_sign_prev[0]:
                    if dp_sign_prev[0] != 0 or sol_indices[0][-1] != i-1:
                        if np.linalg.norm(cps[i][:2]) > 1:
                            sols.append([0, iks[0], cptangent[i], dk0, i])
                            sol_indices[0].append(i)
                if not np.isnan(dp_sign_prev[1]) and dp_sign[1] - dp_sign_prev[1]:
                    if dp_sign_prev[1] != 0 or sol_indices[1][-1] != i-1:
                        if np.linalg.norm(cps[i][:2]) > 1:
                            sols.append([1, iks[1], cptangent[i], dk1, i])
                            sol_indices[1].append(i)

                dp_sign_prev[:] = dp_sign
            
            return [l, dp, q, sols]
        
        def numIK(self, T_t, n=100, compute_all = False):
            _, _, _, wristSols = self.dotProdCirc(T_t, n)
            sols = []
            for ws in wristSols:
                q = np.append(ws[1],[0,0,0])

                # compute 4th angle as the angle between the third x-axis and the tangent to the circle
                x3 = np.array(ws[3][:3,0]).flatten()
                z3 = np.array(ws[3][:3,2]).flatten()
                tangent = ws[2]

                t4 = np.arctan2(np.dot(np.cross(x3,tangent),z3), np.dot(x3,tangent))
                t4 = t4 - pi/2 

                q = np.append(ws[1],[t4,0,0])


                # compute 5th angle as the angle between the fourth x-axis and the line from p4 to target
                T5 = self.fkine(q,5)[0] # actually one could just store T3 above so that not the entire fkine is necessary
                #p4 = np.array(T4[:3,3]).flatten()
                y5 = np.array(T5[:3,1]).flatten()
                z5 = np.array(T5[:3,2]).flatten()
                #p_t = np.array(T_t[:3,3]).flatten()
                tardir = np.array(T_t[:3,2]).flatten()

                t5 = np.arctan2(np.dot(np.cross(tardir,z5),y5), np.dot(z5,tardir))

                q = np.append(ws[1],[t4,t5,0])

                # compute last angle
                T5 = self.fkine(q)[0] # actually one could just store T4 above so that not the entire fkine is necessary
                x5 = np.array(T5[:3,0]).flatten()
                z5 = np.array(T5[:3,2]).flatten()
                x_t = np.array(T_t[:3,0]).flatten()

                t6 = np.arctan2(np.dot(np.cross(x5,x_t),z5), np.dot(x5,x_t))

                q = np.append(ws[1],[t4%(2*pi),t5%(2*pi),t6%(2*pi)])

                sols.append([ws[0],q])

            if compute_all:
                ls = len(sols)
                for i in range(ls):
                    sol = sols[i]
                    q = sol[1].copy()
                    q[0] = (q[0] + pi)%(2*pi)
                    q[1] = (-q[1] + pi)%(2*pi)
                    q[2] = (-q[2] + pi)%(2*pi)
                    q[3] = (q[3] + pi)%(2*pi)
                    sols.append([sol[0]+2,q])
            return sols
        
        def convertJoints(self, q, adj_range = True, return_string = True):
            q = np.array(q)
            q = q*180/pi
            q[1] = q[1] + 90
            q[2] = q[2] - q[1]
            q[4] = q[4] + 180
            q[5] = -(q[5] + 180) #not sure why, but this works at least for RoboDK
            q = q % 360
            if not adj_range:
                return q
            for i in [0,1,3,4,5]:
                if q[i] > 180:
                    q[i] = q[i] - 360
            if q[2] > 180 - q[1]:
                q[2] = q[2] - 360
            if return_string:
                s = '['
                for i in range(len(q)-1):
                    s = s + str(q[i]) + ', '
                s = s + str(q[-1]) + ']'
                return s
            return q
    

# --------------------------------------
#  helper functions

def rotx(phi):
    s = np.sin(phi)
    c = np.cos(phi)
    R = np.matrix([[1, 0, 0, 0],
                 [0, c, -s, 0],
                 [0, s, c, 0],
                 [0, 0, 0, 1]])
    return R

def roty(phi):
    s = np.sin(phi)
    c = np.cos(phi)
    R = np.matrix([[c, 0, s, 0],
                 [0, 1, 0, 0],
                 [-s, 0, c, 0],
                 [0, 0, 0, 1]])
    return R

def rotz(phi):
    s = np.sin(phi)
    c = np.cos(phi)
    R = np.matrix([[c, -s, 0 ,0],
                 [s, c, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
    return R

def transl(x,y = None, z= None):
    if y is None and z is None:
        P = x
        F = np.matrix([[1, 0, 0, P[0]],
                      [0, 1, 0, P[1]],
                      [0, 0, 1, P[2]],
                      [0, 0, 0, 1]])
    else:
        F = np.matrix([[1, 0, 0, x],
                      [0, 1, 0, y],
                      [0, 0, 1, z],
                      [0, 0, 0, 1]])
    return F

def transx(x): return transl(x,0,0)
def transy(y): return transl(0,y,0)
def transz(z): return transl(0,0,z)

def T_link(a,d,alpha):
    return transl(a,0,d) * rotx(alpha)

def T_euler(a,b,x,y,z):
    return transl([x,y,z])*rotz(a)*rotx(b)

def get_euler(T, rotated_ref_frame=False):
    # get a,b,x,y,z from hom. transf. matrix
    x = T[0,3]
    y = T[1,3]
    z = T[2,3]
    a = (np.arctan2(T[1,2],T[0,2]) + pi/2) % (2*pi)
    # a = np.arccos(-T[1,2] / np.sqrt(1 - T[2,2]**2))
    b = np.arccos(T[2,2])
    if rotated_ref_frame:
        return [(a - np.arctan2(y, x)) % (2*pi), b, np.linalg.norm([x, y]), z]
    return [a, b, x, y, z]
