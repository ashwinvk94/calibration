import cv2
import os
import numpy as np
import math
from scipy.optimize import least_squares


def GetImages(pathIn):

    frame_array = []
    files = [f for f in os.listdir(
        pathIn) if os.path.isfile(os.path.join(pathIn, f))]

    for i in range(len(files)):
        filename = os.path.join(pathIn, files[i])
        # reading each files
        img = cv2.imread(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    return frame_array


def PlotCorners(img, corners):
    main_corners = []
    for i, corner in enumerate(corners):
        if(i == 0 or i == 8 or i == 45 or i == 53):
            x = corner[0][0]
            y = corner[0][1]
            cv2.circle(img, (x, y), 3, (0, 0, 255), 4)
            main_corners.append([x, y])
    return img, np.array(main_corners)


def getVij(H, i, j):
    vij = []
    vij.append(H[i, 0]*H[j, 0])
    vij.append(H[i, 0]*H[j, 1]+H[i, 1]*H[j, 0])
    vij.append(H[i, 1]*H[j, 1])
    vij.append(H[i, 2]*H[j, 0]+H[i, 0]*H[j, 2])
    vij.append(H[i, 2]*H[j, 1]+H[i, 1]*H[j, 2])
    vij.append(H[i, 2]*H[j, 2])
    return np.array(vij)


def findRT(K, H):
    Kinv = np.linalg.inv(K)
    Ah_hat = np.matmul(Kinv, H)  # K inv and columns of H
    lam = ((np.linalg.norm(np.matmul(
        Kinv, H[:, 0]))+(np.linalg.norm(np.matmul(Kinv, H[:, 1]))))/2)

    sgn = np.linalg.det(Ah_hat)
    if sgn < 0:
        s = Ah_hat*-1/lam
    elif sgn >= 0:
        s = Ah_hat/lam
    r1 = s[:, 0]
    r2 = s[:, 1]
    r3 = np.cross(r1, r2)
    t = s[:, 2]

    Q = np.array([r1, r2, r3]).T
    # Finding Rotation MAtrix from a 3x3 Matrix
    u, s, v = np.linalg.svd(Q)
    R = np.matmul(u, v)
    return np.hstack([R, np.reshape(t, (1, 3)).T])


def findV(HArr):
    tempV = np.zeros([1, 6])
    for i in range(HArr.shape[2]):

        V = np.vstack([transFunc(HArr[:, :, i], 0, 1).T,
                       (transFunc(HArr[:, :, i], 0, 0)-transFunc(HArr[:, :, i], 1, 1)).T])

        tempV = np.concatenate([tempV, V], axis=0)
    tempV = tempV[1:]
    return tempV


def transFunc(H, i, j):
    v = np.array([[H[0][i]*H[0][j]],
                  [H[0][i]*H[1][j] + H[0][j]*H[1][i]],
                  [H[1][i]*H[1][j]],
                  [H[2][i]*H[0][j] + H[0][i]*H[2][j]],
                  [H[2][i]*H[1][j] + H[1][i]*H[2][j]],
                  [H[2][i]*H[2][j]]])
    return np.reshape(v, (6, 1))


def findH(model_xy, main_corners):
    # cv2.findChessboardCorners finds corners indices interating thoruhg x first than then y axis
    H, _ = cv2.findHomography(model_xy, main_corners)
    return H


def reshapeB(B):
    reshapedB = np.zeros((3, 3))
    reshapedB[0, 0] = B[0]
    reshapedB[1, 0] = B[1]
    reshapedB[0, 1] = B[1]
    reshapedB[1, 1] = B[2]
    reshapedB[0, 2] = B[3]
    reshapedB[2, 0] = B[3]
    reshapedB[1, 2] = B[4]
    reshapedB[2, 1] = B[4]
    reshapedB[2, 2] = B[5]

    return reshapedB


def findK(B):
    v0 = (B[0][1]*B[0][2] - B[0][0]*B[1][2])/(B[0][0]*B[1][1]-B[0][1]**2)

    lam = (B[2][2]-((B[0][2]**2)+v0*(B[0][1]*B[0][2] - B[0][0]*B[1][2]))/B[0][0])
    alp = math.sqrt(lam/B[0][0])
    beta = math.sqrt(((lam*B[0][0])/(B[0][0]*B[1][1] - B[0][1]**2)))
    gamma = -(B[0][1])*(alp**2)*beta/lam
    u0 = (gamma*v0/beta) - (B[0][2]*alp**2)/lam

    K = np.array([[alp, gamma, u0],
                  [0,   beta,  v0],
                  [0,   0,      1]])

    return K


def converToArr(HList):
    HArr = np.array(HList[0])
    for k in range(len(HList)):
        if k == 0:
            continue
        HArr = np.dstack([HArr, HList[k]])
    return HArr


def findB(V_n):
    U, S, V = np.linalg.svd(V_n)
    b = V[:][5]
    B = np.zeros([3, 3])
    B[0][0] = b[0]
    B[0][1] = b[1]
    B[1][0] = b[1]
    B[0][2] = b[3]
    B[2][0] = b[3]
    B[1][1] = b[2]
    B[1][2] = b[4]
    B[2][1] = b[4]
    B[2][2] = b[5]

    return B


def restructure(A):
    a1 = np.reshape(np.array([A[0][0], 0, A[0][2], A[1][1], A[1][2]]), (5, 1))
    a3 = np.reshape(np.array([0, 0]), (2, 1))
    param = np.concatenate([a3, a1])
    return param


def fun(params, corners, Homographies):

    A = np.array([[params[2], 0, params[4]],
                  [0, params[5], params[6]],
                  [0, 0, 1]])
    K = np.reshape(params[0:2], (2, 1))
    w_xy = []
    for i in range(6):
        for j in range(9):
            w_xy.append([21.5*(j+1), 21.5*(i+1), 0, 1])
    w_xyz = np.array(w_xy)

    error = np.empty([54, 1])
    for i in range(Homographies.shape[2]):
        Rt = findRT(A, Homographies[:, :, i])
        norm_pts = np.matmul(Rt, w_xyz.T)
        norm_pts = norm_pts/norm_pts[2]
        P = np.matmul(A, Rt)
        pt = np.matmul(P, w_xyz.T)
        img_pts = pt/pt[2]

        u_hat = img_pts[0] + (img_pts[0] - A[0][2])*[(K[0]*((norm_pts[0])**2 + (norm_pts[1])**2)) +
                                                     (K[1]*((norm_pts[0])**2 + (norm_pts[1])**2)**2)]
        v_hat = img_pts[1] + (img_pts[1] - A[1][2])*[(K[0]*((norm_pts[0])**2 + (norm_pts[1])**2)) +
                                                     (K[1]*((norm_pts[0])**2 + (norm_pts[1])**2)**2)]

        proj = corners[i*54:(i+1)*54, 0:2]
        proj = np.reshape(proj, (-1, 2))
        reproj = np.reshape(np.array([u_hat, v_hat]), (2, 54)).T
        err = np.linalg.norm(np.subtract(proj, reproj), axis=1)**2
        error = np.vstack((error, err.reshape((54, 1))))

    error = error[54:]
    error = np.reshape(error, (702,))
    return error


def RMSerror(A, K, Homographies, corner_points):

    w_xy = []
    for i in range(6):
        for j in range(9):
            w_xy.append([21.5*(j+1), 21.5*(i+1), 0])
    w_xyz = np.array(w_xy)

    mean = 0
    error = np.zeros([2, 1])
    for i in range(Homographies.shape[2]):
        Rt = findRT(A, Homographies[:, :, i])
        img_points, _ = cv2.projectPoints(w_xyz, Rt[:, 0:3], Rt[:, 3], A, K)
        img_points = np.array(img_points)
        errors = np.linalg.norm(
            corner_points[i*54:(i+1)*54, 0, :]-img_points[:, 0, :], axis=1)
        error = np.concatenate(
            [error, np.reshape(errors, (errors.shape[0], 1))])
    mean_error = np.mean(error)
    return mean_error


images = GetImages('./input')

corners = []
corner_points = []
chessboard_flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

# calculating for first 3 iamges
HList = []
for i, img in enumerate(images):

    ret, corners = cv2.findChessboardCorners(img, (9, 6), chessboard_flags)
    corner_points.extend(corners)
    img, main_corners = PlotCorners(img, corners)
    model_xy = np.array([[21.5, 21.5],
                         [21.5*9, 21.5],
                         [21.5, 21.5*6],
                         [21.5*9, 21.5*6]], dtype='float32')
    # we could also use all the corners instead of onmly the main 4
    H = findH(model_xy, main_corners)

    # H verification
    # H_inv = np.linalg.inv(H)
    # warped_img = cv2.warpPerspective(
    # 	img, H_inv, dsize=(img.shape[0], img.shape[1]))
    # cv2.imwrite('warped_output/img'+str(i)+'.jpg', warped_img)
    # cv2.imwrite('output/img'+str(i)+'.jpg', img)

    HList.append(H)
corner_points = np.array(corner_points)
HArr = converToArr(HList)

V = findV(HArr)
B = findB(V)
K = findK(B)
print("The intial estimate of the calibration matrix:")
print(K)
print('\n')

params_init = restructure(K)

res = least_squares(fun, x0=np.squeeze(params_init),
                    method='lm', args=(corner_points, HArr))

# A = np.reshape(res.x[2:11],(3,3))
A = np.array([[res.x[2], res.x[3], res.x[4]],
              [0, res.x[5], res.x[6]],
              [0, 0, 1]])

print("The maximum likelihood intrisic calibration matrix")
print(A)
print('\n')
# finding rt for each image
# for H in HList:
#     r, t = findRT(K, H)
#     print r
#     print t


# The distortion parameters :
K = np.reshape(res.x[0:2], (2, 1))
print("The distortion parameters")
print(K)
print('\n')

distortion = np.array([K[0], K[1], 0, 0, 0], dtype=float)
undistorted_images = []
new_list = images

for i, image in enumerate(new_list):
    undist = cv2.undistort(image, A, distortion)
    undistorted_images.append(undist)
    cv2.imwrite('output_undist/'+str(i)+'.png', undist)

undist_corner_points = []
for i, img in enumerate(undistorted_images):
    ret, corners = cv2.findChessboardCorners(img, (9, 6), chessboard_flags)
    undist_corner_points.extend(corners)

reproj_error = RMSerror(A, distortion, HArr, corner_points)
print("RMS Error", reproj_error)
