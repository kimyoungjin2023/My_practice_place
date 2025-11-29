import numpy as np
import sklearn.svm as svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----- 데이터 2D → 3D로 확장 -----
X_2D = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [8, 7],
    [9, 8],
    [10, 8]
])
y = np.array([0, 0, 0, 1, 1, 1])

# 새로운 3차원 feature 추가 (예: X3 = X1 + X2)
X3 = (X_2D[:, 0] + X_2D[:, 1]).reshape(-1, 1)
X = np.hstack([X_2D, X3])   # shape = (6, 3)

# ----- SVM 모델 -----
svm_practice = svm.SVC(kernel='linear')
svm_practice.fit(X, y)

# ----- 테스트 포인트 -----
test_point = np.array([[4, 3]])
test_point_3D = np.array([[4, 3, 4+3]])  # X3 확장
prediction = svm_practice.predict(test_point_3D)[0]

# ----- 시각화 -----
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 클래스별 산점도
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], X[y == 0][:, 2],
           color='blue', label='Class 0')
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2],
           color='red', label='Class 1')

# 서포트 벡터
sv = svm_practice.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], sv[:, 2],
           s=150, edgecolors='black', facecolors='none', label='Support Vectors')

# ----- 결정 경계 평면 -----
w = svm_practice.coef_[0]
b = svm_practice.intercept_[0]

# 평면 그리기용 meshgrid
xx, yy = np.meshgrid(
    np.linspace(0, 12, 20),
    np.linspace(0, 12, 20)
)

# z 계산: w1*x + w2*y + w3*z + b = 0
zz = -(w[0] * xx + w[1] * yy + b) / w[2]

# 결정 경계 평면
ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

# ----- 마진 평면 -----
zz_up = -(w[0]*xx + w[1]*yy + b - 1) / w[2]
zz_down = -(w[0]*xx + w[1]*yy + b + 1) / w[2]

ax.plot_surface(xx, yy, zz_up, alpha=0.1, color='black')
ax.plot_surface(xx, yy, zz_down, alpha=0.1, color='black')

# ----- 테스트 포인트 -----
ax.scatter(test_point_3D[:, 0], test_point_3D[:, 1], test_point_3D[:, 2],
           color='green', s=200, marker='X', label='Test Point')

ax.set_title(f"3D SVM Classification (Predicted Class = {prediction})")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.legend()
plt.show()
