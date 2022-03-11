import os
os.environ["MODIN_CPUS"] = "16"#max cpus. important to avoid overthreading 
os.environ["MODIN_OUT_OF_CORE"] = "true" #use disk when ram is full
import modin.pandas as pd


import sympy as sym
from sympy.physics.quantum import TensorProduct

import numpy as np
from datetime import datetime

pd.set_option('display.max_rows', None)
#os.environ["MODIN_ENGINE"] = "ray"


def double_dot(A, B):
    # Double-dot operation between two 3x3 matrices
    sum = 0
    for i in range(3):
        for j in range(3):
            sum = sum + A[i, j] * B[i, j]
    return sum


def theta_to_a01(theta):
    # defines a unit vector in the xy plane, theta degrees from the yy axis
    return [-np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0]


def HGO(params, stretch, load):
    # params = [c, κ, k1, k2, theta]
    # returns cauchy stress at yy direction if load == "uniaxial"
    # returns cauchy stress at xx and yy directions if load == "equibiaxial"

    if load != 'uniaxial' and load != 'equibiaxial':
        raise ValueError("Load string isn't acceptable")

    # 3x3 Identity Matrix
    I = sym.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    if load == "uniaxial":
        # Deformation Gradient assuming incompressibility and a uniaxial load
        F = sym.Matrix([[1 / (np.sqrt(stretch)), 0, 0], [0, stretch, 0], [0, 0, 1 / (np.sqrt(stretch))]])

    #  if load == "equibiaxial":
    # Deformation Gradient assuming incompressibility and a equibiaxial load
    #      F = sym.Matrix([[stretch,0,0], [0,stretch,0], [0,0,1/(stretch**2)]])

    Ft = sym.transpose(F)
    Jac = sym.det(F)

    # Modified Deformation Gradient
    Fm = Jac ** (-1 / 3) * I * F
    Fmt = sym.transpose(Fm)

    # Modified Right Cauchy-Green Deformation Tensor with values according to F: 'Cmv'
    Cmv = Fmt * Fm

    # Symbolic Modified Right Cauchy-Green Deformation Tensor 'Cm'
    Cm11 = sym.Symbol('Cm11')
    Cm12 = sym.Symbol('Cm12')
    Cm13 = sym.Symbol('Cm13')
    Cm21 = sym.Symbol('Cm21')
    Cm22 = sym.Symbol('Cm22')
    Cm23 = sym.Symbol('Cm23')
    Cm31 = sym.Symbol('Cm31')
    Cm32 = sym.Symbol('Cm32')
    Cm33 = sym.Symbol('Cm33')
    Cm = sym.Matrix([[Cm11, Cm12, Cm13], [Cm21, Cm22, Cm23], [Cm31, Cm32, Cm33]])

    # Compute the invariant im1  of the tensor Cm
    im1 = sym.trace(Cm)

    # symbolic Neo-Hookean parameter c
    c = sym.Symbol('c')

    # symbolic dispersion parameter κ (0 < κ < 1/3) (the symbol is the greek letter 'kappa')
    κ = sym.Symbol('κ')

    # symbolic material parameters k1 and k2 (k1>0; k2>0)
    k1 = sym.Symbol('k1')
    k2 = sym.Symbol('k2')

    # Unit vector representing the direction of the fibres in the stress free configuration
    # params[4] = angle theta between the mean orientation of the fibers and the yy axis.
    a01_list = theta_to_a01(params[4])
    a01 = sym.Matrix(a01_list)

    # Structure Tensors H1, which depend on κ and a01

    if double_dot(Cmv, TensorProduct(a01, sym.transpose(a01))) > 1:  # condition to only allow tensile stress
        H1 = κ * I + (1 - 3 * κ) * (TensorProduct(a01, sym.transpose(a01)))
    else:
        H1 = κ * I

    E1 = double_dot(H1, Cm) - 1

    # alternative: for debbugging
    # m0 =TensorProduct(a01,sym.transpose(a01))
    # im4 = double_dot(Cm,m0) #pseudo-invariant 4. very important measure. it tells us the squared stretch of the fibers
    # E1 = im4*(1-3*κ) + κ*im1-1

    # Generate SEF (Strain Energy Function)
    sef = 0.5 * c * (im1 - 3) + (k1 / (2 * k2)) * (sym.exp(k2 * E1 * E1) - 1)

    # Second Piola Kirchoff Stresses
    S11 = 2 * sym.diff(sef, Cm11)
    S12 = 2 * sym.diff(sef, Cm12)
    S13 = 2 * sym.diff(sef, Cm13)
    S21 = 2 * sym.diff(sef, Cm21)
    S22 = 2 * sym.diff(sef, Cm22)
    S23 = 2 * sym.diff(sef, Cm23)
    S31 = 2 * sym.diff(sef, Cm31)
    S32 = 2 * sym.diff(sef, Cm32)
    S33 = 2 * sym.diff(sef, Cm33)
    S = sym.Matrix([[S11, S12, S13], [S21, S22, S23], [S31, S32, S33]])

    T = (1 / Jac) * (F * S * Ft)  # cauchy stresses with no BCs
    T = T - (I * T[2, 2])  # imposing of boundary conditions

    T = T.subs([(Cm11, Cmv[0, 0]), (Cm12, Cmv[0, 1]),
                (Cm13, Cmv[0, 2]), (Cm21, Cmv[1, 0]),
                (Cm22, Cmv[1, 1]), (Cm23, Cmv[1, 2]),
                (Cm31, Cmv[2, 0]), (Cm32, Cmv[2, 1]),
                (Cm33, Cmv[2, 2]), (c, params[0]),
                (κ, params[1]), (k1, params[2]), (k2, params[3])])

    if load == 'uniaxial':
        return T[1, 1]


def get_curve(params, stretch_min, stretch_max, ninc, load):
    # stores HGO loading runs between a minimum and a maximum stretch

    if load == "uniaxial":
        stretches = np.linspace(stretch_min, stretch_max, ninc)
        stresses = [HGO(params, stretch, load) for stretch in stretches]
        return np.asarray(stresses)


# Initial data-----------------------------------------------------------------------------------------------------

c_min = 1.0
c_max = 40.0
n_c = 2

κ_min = 0.0
κ_max = 1 / 3
n_κ = 2

k1_min = 0.1
k1_max = 5.0
n_k1 = 2

k2_min = 0.1
k2_max = 5.0
n_k2 = 2

θ_min = 0  # in degrees
θ_max = 90  # in degrees
n_θ = 2

decimals = 2  # number of decimal cases for each parameter value

n = n_c * n_κ * n_k1 * n_k2 * n_θ  # total combinations

st_max = 1.6  # applied stretch
st_min = 1.0
ninc = 10  # number of stretch increments

# parameters grid
c_list = np.round(np.random.uniform(c_min, c_max, size=n_c), decimals)
κ_list = np.round(np.random.uniform(κ_min, κ_max, size=n_κ), decimals)
k1_list = np.round(np.random.uniform(k1_min, k1_max, size=n_k1), decimals)
k2_list = np.round(np.random.uniform(k2_min, k2_max, size=n_k2), decimals)
θ_list = np.round(np.random.uniform(θ_min, θ_max, size=n_θ), decimals)

# Computes the n combinations of params
params = []

for i in c_list:
    for j in κ_list:
        for k in k1_list:
            for l in k2_list:
                for m in θ_list:
                    params.append([i, j, k, l, m])

params = np.array(params, dtype=object)
params = params.reshape(-1, 5)
np.random.shuffle(params)

# Append the combinations to a DataFrame
df = pd.DataFrame(params[:, 0], columns=['c'])
df['κ'] = params[:, 1]
df['k1'] = params[:, 2]
df['k2'] = params[:, 3]
df['θ'] = params[:, 4]

now = datetime.now()
print("Started to generate curves at", now.strftime("%H:%M:%S"))

# generate uniaxial
# generate (x,y) data for each unique combination of params at the dataframe, for a uniaxial load
load = "uniaxial"
df_uniaxial = df.copy()
df_uniaxial['stretch'] = df.apply(lambda x: np.linspace(st_min, st_max, ninc), axis=1)
df_uniaxial['stress'] = df.apply(lambda x: get_curve(x, st_min, st_max, ninc, load), axis=1)

now = datetime.now()
print("Generated curves successfully. Ended at", now.strftime("%H:%M:%S"))

# get number of samples
nsamples = df_uniaxial.shape[0]
# get data channels. in this case, stretch and stress
channels = ['stretch', 'stress']
# number of channels
nchannels = len(channels)
# number of data points
npts = ninc
# data array
y = np.empty((nsamples, npts, nchannels))

for idx, signal in enumerate(channels):
    # convert signal to numpy
    s = df_uniaxial[signal].to_numpy()
    # flatten
    s = np.concatenate(s)
    # flatten then reshape ()
    s = s.reshape(df_uniaxial[signal].shape[0], npts)
    # append to data array
    y[:, :, idx] = s

# working with X like this will be easier to handle tensor shapes during model training
# X = np.asarray(X).astype('float32')
# X=np.asarray(X.tolist())
print(nsamples, npts, nchannels)

# features, i.e., material parameters
# features
features = ['c', 'κ', 'k1', 'k2', 'θ']
nfeatures = len(features)
# features array
X = np.empty((nsamples, nfeatures))

for idx, signal in enumerate(features):
    # convert signal to numpy
    s = df_uniaxial[signal].to_numpy()
    # append to features array
    X[:, idx] = s  # double check if it has the correct shape
# y=df1.ehertz.to_numpy()
# y=np.asarray(y.tolist())


print("X shape: ", np.shape(X))
print('\n')
print("y shape: ", np.shape(y))
print('\n')
print(X[0])
print('\n')
print(y[0])

np.save('X_run_5', X)
np.save('y_run_5', y)

print("X and y saved")
