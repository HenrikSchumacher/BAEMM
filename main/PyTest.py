import bempp.api
from bempp.api.operators.far_field import helmholtz as helmholtz_far
from bempp.api.operators.potential import helmholtz as helmholtz_pot
from bempp.api.operators.boundary import sparse,helmholtz
from bempp.core import opencl_kernels
import numpy as np
import json
import scipy as sp


def calc_FF(connectivity,vertices,wavenumber,incident_directions,measurement_directions):
    points = bempp.api.Grid(vertices,connectivity)
    space = bempp.api.function_space(points, "P", 1)
    
    SL = helmholtz.single_layer(space,space,space,wavenumber,precision = 'single').strong_form()
    DL = helmholtz.double_layer(space,space,space,wavenumber,precision = 'single').strong_form()
    I = sparse.identity(space,space,space,precision = 'single').strong_form()
    A = ((1/2)*I + DL - 1j*wavenumber*SL)

    bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = 'gpu'
    opencl_kernels.set_default_gpu_device(1,0)
    single_far = helmholtz_far.single_layer(space, measurement_directions, wavenumber,precision = 'single')
    double_far = helmholtz_far.double_layer(space, measurement_directions, wavenumber,precision = 'single')
    
    wave = -incWave(space,wavenumber,incident_directions)
    phi = solve(space,A,wave)
    d = len(wave)
    
    FF = []
    for i in range(d):
        u_inf = double_far * phi[i] - 1j * wavenumber * single_far * phi[i]
        FF.append(u_inf[0].tolist())
    return np.array(FF)

def calc_DFF_adj(connectivity,vertices,wavenumber,incident_directions,measurement_directions,y):
    points = bempp.api.Grid(vertices, connectivity)
    space = bempp.api.function_space(points, "P", 1)
    normals = vertex_normals(space)

    DL_H = helmholtz.adjoint_double_layer(space,space,space,wavenumber,precision = 'single').strong_form()
    SL = helmholtz.single_layer(space,space,space,wavenumber,precision = 'single').strong_form()
    I = sparse.identity(space,space,space,precision = 'single').strong_form()
    B = ((1/2)*I + DL_H - 1j * SL)

    wave = incWave_dnormal(space,normals,wavenumber,incident_directions) - 1j*incWave(space,wavenumber,incident_directions)
    dudn = solve(space,B,wave,"array")
    
    measurement_directions = np.transpose(measurement_directions)
    hwave = Herglotz_wave_dnormal(space,normals,wavenumber,measurement_directions,np.conjugate(y)) - 1j*Herglotz_wave(space,wavenumber,measurement_directions,np.conjugate(y))
    dvdn = solve(space,B,hwave,"array")

    DFF_adj = np.sum(np.multiply(dudn,dvdn),axis = 0)
    DFF_adj = -(1/incident_directions.shape[0])*np.multiply(normals,DFF_adj)
    return np.real(DFF_adj)

# def calc_DFF(connectivity,vertices,wavenumber,incident_directions,measurement_directions,v):
#     points = bempp.api.Grid(vertices, connectivity)
#     space = bempp.api.function_space(points, "P", 1)
#     normals = vertex_normals(space)

#     SL = helmholtz.single_layer(space,space,space,wavenumber,precision = 'single').strong_form()
#     DL = helmholtz.double_layer(space,space,space,wavenumber,precision = 'single').strong_form()
#     I = sparse.identity(space,space,space,precision = 'single').strong_form()
#     A = ((1/2)*I + DL - 1j*wavenumber*SL)

#     DL_H = helmholtz.adjoint_double_layer(space,space,space,wavenumber,precision = 'single').strong_form()
#     B = ((1/2)*I + DL_H - 1j * SL)

#     single_far = helmholtz_far.single_layer(space, measurement_directions, wavenumber)
#     double_far = helmholtz_far.double_layer(space, measurement_directions, wavenumber)

#     wave = incWave_dnormal(space,normals,wavenumber,incident_directions) - 1j*incWave(space,wavenumber,incident_directions)

#     dudn = solve(space,B,wave,"array")   
#     hnu = np.sum(np.multiply(normals,v),0)
#     boundary_condition = -np.multiply(dudn,hnu)
#     phi_d = solve(space,A,boundary_condition)

#     DFF = []
#     d = len(phi_d)
#     for i in range(d):
#         u_inf = double_far * phi_d[i] - 1j * wavenumber * single_far * phi_d[i]
#         DFF.append(u_inf[0].tolist())
#     return np.array(DFF)

def solve(space,A,wave,output = "wave_function"):
    d = len(wave[:,0])
    # I = sparse.identity(space,space,space).weak_form()
    phi=[]
    if output == "wave_function":
        for i in range(d):
            phi_coeff, info = sp.sparse.linalg.gmres(A,wave[i,:],tol = 1e-4)
            phi.append(bempp.api.GridFunction(space,coefficients = phi_coeff))
    elif output == "array":
        for i in range(d):
            phi_coeff, info = sp.sparse.linalg.gmres(A,wave[i,:],tol = 1e-4)
            phi.append(phi_coeff.tolist())
        phi = np.array(phi)
    else:
        pass
    return phi

def incWave(space,wavenumber,incident_directions):
    x = space.grid.vertices
        
    wave = np.exp(1j*wavenumber*np.matmul(incident_directions,x))  
    return wave

# introduce the normal derivative of the incident wave
def incWave_dnormal(space,normals,wavenumber,incident_directions):
    x = space.grid.vertices
        
    wave = 1j*wavenumber*np.matmul(incident_directions,normals)*np.exp(1j*wavenumber*np.matmul(incident_directions,x))
    return wave


# define the incident herglotz wave function
def Herglotz_wave(space,wavenumber,measurement_directions,g):
    x = space.grid.vertices

    wave = (1/(g.shape[1]))*np.matmul(g,np.exp(-1j*wavenumber*np.matmul(measurement_directions,x)))
    return wave

# define the normal derivative of the incident herglotz wave function
def Herglotz_wave_dnormal(space,normals,wavenumber,measurement_directions,g):
    x = space.grid.vertices

    factor = -1j*wavenumber*(1/g.shape[1])
    integrand = np.exp(-1j*wavenumber*np.matmul(measurement_directions,x))*np.matmul(measurement_directions,normals)
    wave  = factor*np.matmul(g,integrand)
    return wave

def vertex_normals(space):
    normals = []
    for i in range(3):
        normals.append(bempp.api.GridFunction(space,fun = normal_function(i)).coefficients.tolist())
    return np.array(normals)

def normal_function(i):
    @bempp.api.complex_callable
    def function(x,normal,domain_index,result):
        result[0] = normal[i]
    return function

# def vertex_normals(space):
#     dual = bempp.api.function_space(space.grid, "DP", 0)
#     M = sparse.identity(dual,dual,space).weak_form()
#     n = np.transpose(M * dual.grid.normals)
#     normals = np.multiply(n,1/np.linalg.norm(n,axis=0))
#     return normals

# f = open("C:\msys64\home\janni\github\BAEMM\src\Meshes\mesh.json")
# data = json.load(f)
# vertices = np.array(data["Vertices"])
# connectivity = np.array(data["Connectivity"])
# f.close()
vertices = np.loadtxt("/HOME1/users/guests/jannr/github/BAEMM/Meshes/Sphere_00162240T_V.txt").transpose()
connectivity = np.loadtxt("/HOME1/users/guests/jannr/github/BAEMM/Meshes/Sphere_00162240T_T.txt").transpose()
f = open("/HOME1/users/guests/jannr/github/BAEMM/main/meas.json")
data = json.load(f)
measurement_directions = np.array(data["Vertices"])
f.close()
# measurement_directions = np.transpose(measurement_directions)

points = bempp.api.Grid(vertices,connectivity)
space = bempp.api.function_space(points, "P", 1)

# ev_pts = points.centroids + 0.001*points.normals
# single_far = helmholtz_far.single_layer(space, measurement_directions, 2,precision = 'single')
# double_far = helmholtz_far.double_layer(space, measurement_directions, 2,precision = 'single')

# single_pot = helmholtz_pot.single_layer(space, ev_pts.transpose(), np.pi, precision = 'single')
# double_pot = helmholtz_pot.double_layer(space, ev_pts.transpose(), np.pi, precision = 'single')

# ret = ret[0]

DL = helmholtz.single_layer(space,space,space, 2*np.pi,precision = 'single').weak_form()

# g = np.ones((vertices.shape[1],1)) + 2j * np.ones((vertices.shape[1],1))
# ret = (1 - 4j) * incWave(space,2,incident_directions) + (-2 + 1j) * incWave_dnormal(space,normals,2,incident_directions)

# normals = vertex_normals(space)
incident_directions = np.array([[1,0,0],[0,1,0],[0,0,1],[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)]])
# incident_directions = np.array([[1,0,0]])
# ret = calc_FF(connectivity,vertices,2*np.pi,incident_directions,measurement_directions)

# ret = (1 - 4j) * incWave(space,2,incident_directions) + (-2 + 1j) * incWave_dnormal(space,normals,2,incident_directions)

g = incWave(space,2*np.pi,incident_directions)

# func = bempp.api.GridFunction(space,coefficients = g[0,:])
# ret = (2j) * single_pot * func + (1) * double_pot * func
ret = DL * g[0,:]
# normals = vertex_normals(space)
# ret = (1 - 4j) * Herglotz_wave(space,2,measurement_directions,g) + (-2 + 1j) * Herglotz_wave_dnormal(space,normals,2,measurement_directions,g)

test_real = np.loadtxt("/HOME1/users/guests/jannr/github/BAEMM/main/data_real.txt").transpose()
test_imag = np.loadtxt("/HOME1/users/guests/jannr/github/BAEMM/main/data_imag.txt").transpose()
# print(np.amax(np.abs(ret)))
res = ret - test_real[0,:] - 1j *test_imag[0,:]
# print(np.shape(res))
error = np.amax(np.divide(np.amax(np.abs(res)),np.amax(np.abs(ret))))
# error2 = np.amax(np.divide(np.amax(np.abs(res),axis = 0),np.amax(np.abs(ret),axis = 0)))
# error = np.divide(np.linalg.norm(res),np.linalg.norm(ret))

print(error)
# print(error2)