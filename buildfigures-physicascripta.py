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

FIG_EXT = 'eps'

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
    'axes.labelsize': 9,
    'axes.linewidth': 1,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.pad': 3,
    'xtick.minor.pad': 3,
    'ytick.major.pad': 3,
    'ytick.minor.pad': 3,

    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
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
        column_width_inches = 85 / 25.4 # 85 mm

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

    with open('data/ghz_binning_ardehali_2p_number.pickle') as f:
        pdata = pickle.load(f)

    p_s1x_s2x, p_s12x_s12y = pdata[1]

    p1, p1_edges = p_s1x_s2x
    p2, p2_edges = p_s12x_s12y

    X = (p2_edges[0][1:] + p2_edges[0][:-1]) / 2
    Y = (p2_edges[1][1:] + p2_edges[1][:-1]) / 2
    g1, g2 = numpy.meshgrid(X, Y)

    corrs = [
        dict(data=p1, edges=p1_edges, zmax=5.5, single=True),
        dict(data=p2, edges=p2_edges, zmax=3.5, single=False),
    ]

    for i, corr in enumerate(corrs):
        subfig = ('(a)', '(b)')[i]
        with figsize(1, 0.8) as fig:
            ax = fig.add_subplot(111, projection='3d')
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

            representation = 'pos-P'
            ax.set_zlabel('\n\nprobability ($\\times 10^{-2}$)')

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

            fig.text(0.05, 0.87, subfig, fontsize=P_GLOBAL['font.size'] + 2)

            fig.tight_layout(pad=1.2)
            fig.savefig('figures/ghz_distributions_' + subfig + '.' + FIG_EXT)


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

    with figsize(0.8, (5 ** 0.5 - 1) / 2) as fig:

        i = 0
        n = n1

        ax = fig.add_subplot(111)

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

        fig.tight_layout(pad=0.7)
        fig.savefig('figures/cooperative.' + FIG_EXT)


if __name__ == '__main__':

    with plot_params(**P_GLOBAL):
        #buildGHZDistributions()
        buildCooperative()
