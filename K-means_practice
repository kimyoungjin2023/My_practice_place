import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터
X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [8, 7],
    [9, 8],
    [10, 8]
])

# K-Means 모델 (2개 클러스터)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# 클러스터 레이블
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 테스트 포인트
test_point = np.array([[4, 3]])
# 테스트 포인트가 어느 클러스터에 속하는지
test_label = kmeans.predict(test_point)[0]

# 시각화
plt.figure(figsize=(7, 6))

# 데이터 포인트 시각화 (클러스터별 색상)
plt.scatter(X[labels == 0][:, 0], X[labels == 0][:, 1], color='blue', label='Cluster 0')
plt.scatter(X[labels == 1][:, 0], X[labels == 1][:, 1], color='red', label='Cluster 1')

# 클러스터 센터 표시
plt.scatter(centers[:, 0], centers[:, 1], s=200, color='black', marker='X', label='Centroids')

# 테스트 포인트
plt.scatter(test_point[:, 0], test_point[:, 1], color='green', s=180, marker='D', label=f'Test Point (Cluster {test_label})')

plt.title("K-Means Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.show()
