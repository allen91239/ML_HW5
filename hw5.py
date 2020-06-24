import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
def read_data():
    data = open('./input.data')
    #print(data)
    pts_list = list()
    for pts in data:
        #print(pts.split())
        pts_list.append(pts.split())
    #print(pts_list)
    return (np.array(pts_list)).astype(float)
def rational_quad(sigma, x, xp, a, l):
    return (sigma**2) * (1 + (((x - xp)**2)/(2 * a * (l**2))) ) ** (-1 * a)
def marg_likelihood(beta, sigma, points, a, l):
    cov_mat = list()
    for i in range(len(points)):
        cov = rational_quad(sigma, points[:, 0], points[i, 0], a, l)
        #print(cov)
        cov_mat.append(cov)
    cov_mat = np.array(cov_mat)
    cov_mat += np.identity(len(cov_mat))/beta
    #print(cov_mat)
    return cov_mat
def predict(c, beta, sigma, x, xp, a, l):
    result = list()
    for xs in xp:
        k = rational_quad(sigma, x[:,0], xs, a, l)
        k = k.reshape((34,1))
        u = np.dot(k.T, np.dot(np.linalg.inv(c), x[:, 1]))
        var = (rational_quad(sigma, xs, xs, a, l) + 1/beta - np.dot(k.T, np.dot(np.linalg.inv(c), k))).reshape((1))
        result.append((u, var))
        #print(k)
        #print(k.shape)
    #print(np.array(result))
    return np.array(result)
def plot(points, line, res, z, filename):
    plt.plot(points[:, 0], points[:, 1], 'ro')
    plt.fill_between(x=line, y1= res[:,0].flatten() + z*np.sqrt(res[:,1].flatten()), y2= res[:,0].flatten() - z*np.sqrt(res[:,1].flatten()), color='gray')
    plt.plot(line, res[:,0], linestyle='-', color='black')
    plt.savefig(f'{filename}.png')
    plt.close()
    return
def nll_fn(theta, points, beta):
    c = marg_likelihood(beta, theta[2], points, theta[0], theta[1])
    return 0.5 * np.log(np.linalg.det(c)) + \
               0.5 * points[:,1].T.dot(np.linalg.inv(c).dot(points[:,1])) + \
               0.5 * len(points) * np.log(2*np.pi)

    
if __name__ == '__main__':
    points = read_data()
    a = 1.0
    l = 1.0
    beta = 5.0
    sigma = 1.0
    z = 1.96 # 95%
    line = np.linspace(-60, 60, 300)
    c = marg_likelihood(beta, sigma, points, a, l)
    res = predict(c, beta, sigma, points, line, a, l)
    plot(points, line, res, z, 'gp')
    theta = [1, 1, 1]
    res = minimize(nll_fn, theta, args=(points, beta),
               method='Powell')
    print(res)
    a = res.x[0]
    l = res.x[1]
    sigma = res.x[2]
    
    c = marg_likelihood(beta, sigma, points, a, l)
    res = predict(c, beta, sigma, points, line, a, l)
    plot(points, line, res, z, 'opt')
    #print(res)
    