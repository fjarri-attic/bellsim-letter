import pickle, json
import numpy
from scipy.misc import factorial

import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

n = 255.0
cm_custom = LinearSegmentedColormap.from_list(
    "riedel",
    [
        (0.0, (1.0, 1.0, 1.0)),
        (2.5/15, (68/n, 84/n, 158/n)),
        (5.0/15, (117/n, 192/n, 235/n)),
        (10.0/15, (234/n, 230/n, 76/n)),
        (12.5/15, (206/n, 59/n, 46/n)),
        (1.0, (142/n, 33/n, 39/n))
    ]
    )
cm_custom.set_under(color='white')
color_lblue = (0.5686274509803921, 0.6823529411764706, 0.8901960784313725)
color_dblue = (0.13725490196078433, 0.28627450980392155, 0.5568627450980392)
color_dred = (1.0, 0.20784313725490197, 0.16470588235294117)
color_dgreen = (0.10588235294117647, 0.6470588235294118, 0.17254901960784313)
color_dyellow = (0.8352941176470589, 0.592156862745098, 0.13725490196078433)


P_GLOBAL = {
    # backend parameters
    'backend': 'ps',
    'text.usetex': False,

    # main parameters
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica'],
    'font.size': 9,
    'lines.linewidth': 1,
    'lines.dash_capstyle': 'round',

    # axes
    'axes.labelsize': 8,
    'axes.linewidth': 0.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.major.pad': 3,
    'xtick.minor.pad': 3,
    'ytick.major.pad': 3,
    'ytick.minor.pad': 3,

    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
}

P_INSET = {
    'font.size': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'axes.labelsize': 6,
    'lines.linewidth': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.minor.size': 1,
    'ytick.minor.size': 1,
}


class plot_params:

    def __init__(self, **kwds):
        old_vals = {}
        for key in kwds:
            old_val = matplotlib.rcParams[key]
            if old_val != kwds[key]:
                old_vals[key] = old_val
        self.old_vals = old_vals
        self.new_vals = kwds

    def __enter__(self):
        matplotlib.rcParams.update(self.new_vals)

    def __exit__(self, *args):
        matplotlib.rcParams.update(self.old_vals)

class inset:

    def __init__(self, *args, **kwds):
        self.args = args
        self.kwds = kwds
        self.params = plot_params(**P_INSET)

    def __enter__(self):
        self.params.__enter__()
        return makeSubplot(*self.args, **self.kwds)

    def __exit__(self, *args):
        self.params.__exit__()


class figsize:

    def __init__(self, columns, aspect):
        column_width_inches = 2.3

        fig_width = column_width_inches * columns
        fig_height = fig_width * aspect # height in inches
        fig_size = [fig_width, fig_height]

        self.params = plot_params(**{'figure.figsize': fig_size})

    def __enter__(self):
        self.params.__enter__()
        return plt.figure()

    def __exit__(self, *args):
        self.params.__exit__()


def dashes(s):
    if s.endswith('--'):
        return (6, 3)
    elif s.endswith('-'):
        return []
    elif s.endswith('-.'):
        return (5,3,1,3)
    elif s.endswith(':'):
        return (0.5,2)
    raise Exception("Unknown dash style " + s)


def buildGHZDistributions():

    with open('data/ghz_binning_ardehali_2p_Q.pickle') as f:
        qdata = pickle.load(f)

    with open('data/ghz_binning_ardehali_2p_number.pickle') as f:
        pdata = pickle.load(f)

    q_s1x_s2x, q_s12x_s12y = qdata[1]
    p_s1x_s2x, p_s12x_s12y = pdata[1]

    q1, q1_edges = q_s1x_s2x
    q2, q2_edges = q_s12x_s12y
    p1, p1_edges = p_s1x_s2x
    p2, p2_edges = p_s12x_s12y

    # Adjust data to get rid of "seam" in the middle of the vertical wall
    for i in xrange(q1.shape[0]):
        for j in xrange(q1.shape[1]):
            if q1[i,j] > 0:
                q1[i,j] = 0
                break

    for j in xrange(q1.shape[1]):
        for i in xrange(q1.shape[0]):
            if q1[i,j] > 0:
                q1[i,j] = 0
                break

    corrs = [
        dict(data=p1, edges=q1_edges, zmax=5.5, single=True, Q=False),
        dict(data=p2, edges=q2_edges, zmax=3.5, single=False, Q=False),
        dict(data=q1, edges=p1_edges, zmax=5.5, single=True, Q=True),
        dict(data=q2, edges=p2_edges, zmax=3.5, single=False, Q=True)
    ]

    with figsize(2, 0.8) as fig:
        for i, corr in enumerate(corrs):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            ax.view_init(elev=40., azim=245.)

            data = corr['data']
            edges = corr['edges']

            X = (edges[0][1:] + edges[0][:-1]) / 2
            Y = (edges[1][1:] + edges[1][:-1]) / 2

            # normalize on 1
            data = data.astype(numpy.float64) / data.sum() / (X[1] - X[0]) / (Y[1] - Y[0]) * 100

            X, Y = numpy.meshgrid(X, Y)

            ax.plot_surface(X, Y, data.T, rstride=2, cstride=2, cmap=cm_custom,
               linewidth=0, antialiased=False)

            #ax.contour(X, Y, data.T, cmap=cm_custom,
            #    levels=numpy.linspace(0, corr['zmax'], 25))

            representation = 'Q' if corr['Q'] else 'pos-P'
            ax.set_zlabel('\n\nprobability ($\\times 10^{-2}$)')
            fig.text(0.2 + 0.5 * (i % 2), 0.93 - 0.48 * (i / 2),
                "SU(2)-Q" if corr['Q'] else 'positive-P', fontsize=P_GLOBAL['font.size']-1)

            if corr['single']:
                ax.set_xlabel('\n\n$\\mathrm{Re}\,\\sigma_1^x$')
                ax.set_ylabel('\n\n$\\mathrm{Re}\,\\sigma_2^x$')
                #ax.set_zlabel('\n\n$P_{\\mathrm{' + representation + '}}$, $\\times 10^{-2}$')

                ax.set_xlim3d(-3.5, 3.5)
                ax.xaxis.set_ticks(range(-3, 4))
                ax.yaxis.set_ticks(range(-3, 4))
            else:
                ax.set_xlabel('\n\n$\\mathrm{Re}\,\\sigma_1^x \\sigma_2^x$')
                ax.set_ylabel('\n\n$\\mathrm{Re}\,\\sigma_1^y \\sigma_2^y$')
                #ax.set_zlabel('\n\n$P_{\\mathrm{' + representation + '}}$, $\\times 10^{-2}$')

                ax.set_xlim3d(-8.5, 6.5)
                ax.set_ylim3d(-6.5, 8.5)
                ax.xaxis.set_ticks(range(-8, 8, 2))
                ax.yaxis.set_ticks(range(-6, 10, 2))
                ax.zaxis.set_ticks([0, 1, 2, 3])

            ax.set_zlim3d(0, corr['zmax'])

            # clear background panes
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

            #ax.w_xaxis.set_rotate_label(False)
            #ax.w_yaxis.set_rotate_label(False)

            fig.text(0.1 + 0.5 * (i % 2), 0.93 - 0.48 * (i / 2), ('A', 'B', 'C', 'D')[i],
                fontsize=P_GLOBAL['font.size'] + 1, fontweight='bold')


        fig.tight_layout(pad=1.3)
        fig.savefig('figures/ghz_distributions.eps')


def buildCooperative():

    def G_analytic(gamma, I, J, N):
        if gamma is None:
            s = 0
            for n in xrange(N - J + 1):
                s += factorial(N - n) / factorial(N - J - n)
            return factorial(N) / ((N + 1) * (factorial(N - I))) * s
        else:
            def binom(n, k):
                if k < 0 or k > n: return 0
                return factorial(n) / factorial(k) / factorial(n - k)
            s = 0
            for n in xrange(N - J + 1):
                for i in xrange(I + 1):
                    s += factorial(n) * factorial(N - n) ** 2 / factorial(N - I) / factorial(N - J - n) * \
                        binom(I, i) * binom(N - J, n - i) * (1 - gamma) ** i * gamma ** (I - i)
            return s / (N + 1)

    def g_analytic(theta, J, N):
        gamma = numpy.cos(theta) ** 2
        return G_analytic(gamma, J, J, N) / G_analytic(None, J, J, N)

    def deltas_analytic(thetas, J, N):
        gs = g_analytic(thetas, J, N)
        gs_3theta = g_analytic(thetas * 3, J, N)
        return (3 * gs - gs_3theta - 2)

    with open('data/cooperative-N1-J1-21.json') as f:
        n1 = json.load(f)
    with open('data/cooperative-N2-J2-25.json') as f:
        n2 = json.load(f)

    with figsize(1, 1.2) as fig:
        for i, n in enumerate([n1, n2]):
            ax = fig.add_subplot(2, 1, i + 1)

            thetas_scaled = numpy.array(n['thetas'])
            deltas = numpy.array(n['deltas_mean'])
            err = numpy.array(n['deltas_err'])

            #ax.errorbar(thetas, deltas, yerr=err)
            ax.fill_between(thetas_scaled, deltas-err, deltas+err,
                facecolor=color_lblue, interpolate=True,
                color=color_dblue,
                linewidth=0.3)

            #ax.plot(thetas_scaled, deltas, 'k-')
            ax.plot(thetas_scaled,
                deltas_analytic(thetas_scaled / numpy.sqrt(i + 1), i + 1, i + 1),
                    'k--', dashes=dashes('--'))

            ax.set_xlim((thetas_scaled[0], thetas_scaled[-1]))
            ax.set_ylim((-0.05, 0.45 if i == 0 else 0.55))

            ax.set_xlabel("$\\theta" + ("\\sqrt{2}" if i == 1 else "") + "$ (rad)")
            ax.set_ylabel("Violation")

            fig.text(0.01, 0.95 - 0.48 * i, 'A' if i == 0 else 'B',
                fontsize=P_GLOBAL['font.size'] + 1, fontweight='bold')

            ax.text(0.02, 0.25 + 0.1 * i,
                ("1 pair" if i == 0 else "2 pairs") +
                ",\n$2^{" + str(21 if i == 0 else 25) + "}$ samples",
                fontsize=P_GLOBAL['font.size']-1)

            xs = numpy.linspace(0.2, 0.25)
            ax.plot(xs, [0.1] * xs.size, "k--", dashes=dashes('--'))
            ax.fill_between(xs, [-0.02] * xs.size, [0.03] * xs.size,
                facecolor=color_lblue, interpolate=True, color=color_dblue, linewidth=0.3)
            ax.text(0.27, 0.08, "QM prediction", fontsize=P_GLOBAL['font.size']-1)
            ax.text(0.27, -0.01, "pos-P sampling", fontsize=P_GLOBAL['font.size']-1)


        fig.tight_layout(pad=0.3)
        fig.savefig('figures/cooperative.eps')


def buildGHZCorrelations():

    def getF_analytical(particles, quantity):
        """
        Returns 'classical' and 'quantum' predictions for the
        Mermin's/Ardehali's state and operator.
        """
        if quantity == 'F_mermin':
            return 2. ** (particles / 2), 2. ** (particles - 1)
        elif quantity == 'F_ardehali':
            return 2. ** ((particles + 1) / 2), 2. ** (particles - 0.5)
        else:
            raise NotImplementedError(quantity)

    def filter_data(data, **kwds):
        result = []
        for d in data:
            for key in kwds:
                if kwds[key] != d[key]:
                    break
            else:
                result.append(d)

        ns = []
        vals = []
        errs = []
        lhvs = []
        qms = []
        for r in sorted(result, key=lambda x: x['particles']):
            if r['quantity'] in ('F_ardehali', 'F_mermin'):
                cl, qm = getF_analytical(r['particles'], r['quantity'])
                if r['error'] / qm > 0.5:
                    continue
                lhvs.append(cl)
                qms.append(qm)

            ns.append(r['particles'])
            vals.append(r['mean'])
            errs.append(r['error'])

        return dict(ns=numpy.array(ns), mean=numpy.array(vals), error=numpy.array(errs),
            lhvs=numpy.array(lhvs), qms=numpy.array(qms))


    with open('data/ghz_sampling.json') as f:
        data = json.load(f)

    with figsize(1, 1.2) as fig:

        G = matplotlib.gridspec.GridSpec(2, 2)

        ax1 = fig.add_subplot(G[0,0])
        ax2 = fig.add_subplot(G[0,1])

        ax1.set_xlabel('particles', color='white') # need it to make matplotlib create proper spacing
        fig.text(0.5, 0.52, 'particles', fontsize=P_GLOBAL['axes.labelsize'])
        ax1.set_ylabel('$\\langle F \\rangle / \\langle F \\rangle_{\\mathrm{QM}}$')

        representation = 'Q'
        violations = filter_data(data['violations'], representation=representation, size=10**8)

        ns = violations['ns'][:50]
        qms = violations['qms'][:50]
        mean = violations['mean'][:50] / qms
        err = violations['error'][:50] / qms

        cl_ns = numpy.arange(1, 51)
        cl_qm = [getF_analytical(n, 'F_ardehali' if n % 2 == 0 else 'F_mermin') for n in cl_ns]
        cl_qm = numpy.array(zip(*cl_qm)[0]) / numpy.array(zip(*cl_qm)[1])

        ax1.set_xlim((0, 10.5))
        ax1.set_ylim((-0.05, 1.2))
        ax2.set_xlim((39.5, 51))
        ax2.set_ylim((-0.05, 1.2))

        for ax in (ax1, ax2):
            ax.plot(cl_ns, numpy.ones(50), color='grey', linewidth=0.75,
                linestyle='--', dashes=dashes('--'))
            ax.errorbar(ns, mean, yerr=err, color=color_dblue, linestyle='None')
            ax.plot(cl_ns, cl_qm, color=color_dred, linestyle='-.', dashes=dashes('-.'))

        # hide the spines between ax and ax2
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.yaxis.tick_right()
        ax2.tick_params(labelright='off') # don't put tick labels at the right side
        ax1.yaxis.tick_left()

        # add cut-out lines
        d = .015 # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False,
            linewidth=P_GLOBAL['axes.linewidth'])
        ax2.plot((-d,+d),(-d,+d), **kwargs)
        ax2.plot((-d,+d),(1-d,1+d), **kwargs)

        kwargs.update(transform=ax1.transAxes,
            linewidth=P_GLOBAL['axes.linewidth'])  # switch to the bottom axes
        ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)
        ax1.plot((1-d,1+d),(-d,+d), **kwargs)

        xs = numpy.linspace(8, 10)
        ax1.plot(xs, [0.25] * xs.size, color=color_dred, linestyle='-.', dashes=dashes('-.'))
        ax1.plot(xs, [0.45] * xs.size, color='grey', linewidth=0.75,
                linestyle='--', dashes=dashes('--'))
        ax1.errorbar([9], [0.65], yerr=[0.05], color=color_dblue, linestyle='None')

        ax2.text(40, 0.6, "SU(2)-Q\nsampling", fontsize=P_GLOBAL['font.size']-1)
        ax2.text(40, 0.4, "QM prediction", fontsize=P_GLOBAL['font.size']-1)
        ax2.text(40, 0.2, "LHV prediction", fontsize=P_GLOBAL['font.size']-1)



        ax = fig.add_subplot(G[1,:])
        ax.set_xlabel('particles')
        ax.set_ylabel('$\\log_{2}($relative error$)$')

        corr1 = filter_data(data['different_order_correlations'],
            representation='Q', quantity='N_total', size=10**8)
        corrm = filter_data(data['violations'],
            representation='Q', size=10**8)

        ax.plot(corr1['ns'], numpy.log2(corr1['error'] / corr1['ns'] * 2.),
            color=color_dgreen, linestyle='--', dashes=dashes('--'))
        ax.plot(corrm['ns'][:50], numpy.log2(corrm['error'] / corrm['qms'])[:50],
            color=color_dblue)

        ref_ns = numpy.arange(1, 36)
        ax.plot(ref_ns, ref_ns / 2. - 18, linestyle=':', dashes=dashes(':'), linewidth=0.75, color='grey')

        ax.set_xlim((0, 51))
        ax.set_ylim((-22, 0))
        ax.yaxis.set_ticks(range(-20, 1, 5))

        xs = numpy.linspace(18, 23)
        ax.plot(xs, [-15] * xs.size, color=color_dgreen, linestyle='--', dashes=dashes('--'))
        ax.plot(xs, [-17.5] * xs.size, color=color_dblue)
        ax.text(25, -15.5, "Single order", fontsize=P_GLOBAL['font.size']-1)
        ax.text(25, -18.5, "Max order ($F\\,$)", fontsize=P_GLOBAL['font.size']-1)

        xs = numpy.linspace(2, 7)
        ax.plot(xs, [-5] * xs.size, color='grey', linestyle=':', dashes=dashes(':'), linewidth=0.75)
        ax.text(7, -6.5, "Reference\n($\\propto 2^{M/2}$)", fontsize=P_GLOBAL['font.size']-1)


        for i in (0, 1):
            fig.text(0.01, 0.95 - 0.48 * i, 'A' if i == 0 else 'B',
                fontsize=P_GLOBAL['font.size'] + 1, fontweight='bold')

        fig.tight_layout(pad=0.3)

        fig.savefig('figures/ghz_correlations.eps')


def buildGHZDecoherence():

    with open('data/ghz_decoherence.json') as f:
        data = json.load(f)

    def find(**kwds):
        for dataset in data:
            for kwd in kwds:
                if kwds[kwd] != dataset[kwd]:
                    break
            else:
                return dataset
        return None

    ns = (2, 3, 4, 6)
    colors = {2: color_dblue, 3: color_dred, 4: color_dgreen, 6: color_dyellow}
    ndashes = {2: '-', 3: '--', 4: '-.', 6: ':'}

    with figsize(1, 1 / 1.6) as fig:
        ax = fig.add_subplot(1, 1, 1)

        for i, n in enumerate(ns):

            dataset = find(particles=n, quantity='N_total')
            mean = numpy.array(dataset['mean'])

            # normalize
            mean /= dataset['particles'] / 2.
            time = numpy.arange(mean.size)

            # filter near-zero parts, helps with readability
            indices = (mean > 0.05)
            mean = mean[indices]
            time = time[indices]

            ax.plot(time, mean, color=colors[n], linestyle=ndashes[n], dashes=dashes(ndashes[n]))

            xs = numpy.linspace(48, 60)
            ax.plot(xs, [0.9 - i * 0.12] * xs.size, color=colors[n],
                linestyle=ndashes[n], dashes=dashes(ndashes[n]))
            ax.text(63, 0.87 - i * 0.12, str(n) + " particles", fontsize=P_GLOBAL['font.size']-1)


        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_xlabel('$\\tau$')
        ax.set_ylabel('$F(\\tau)/F(0)$')

        fig.tight_layout(pad=0.3)
        fig.savefig('figures/ghz_decoherence.eps')



if __name__ == '__main__':

    with plot_params(**P_GLOBAL):
        buildGHZDistributions()
        buildCooperative()
        buildGHZCorrelations()
        buildGHZDecoherence()
