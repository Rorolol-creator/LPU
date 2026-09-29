# %%
from skimage.feature import hog
from skimage import io, color
import matplotlib.pyplot as plt

# %%
# 1. Load and convert image
image = io.imread("butterfly.jpg")
gray = color.rgb2gray(image)

# %%
plt.imshow(gray, cmap='gray')
plt.show()

# %%
# 2. Extract HOG features
features, im = hog(
    gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True
)

# %%
print(len(features[features > 0]))
print(len(features))
plt.imshow(im)
plt.show()

# %% [markdown]
# 3. Train classifier (on many samples)
# clf = SVC(kernel='rbf')
# clf.fit(X_train_features, y_train_labels)

# %% [markdown]
# 4. Predict
# prediction = clf.predict([features])

