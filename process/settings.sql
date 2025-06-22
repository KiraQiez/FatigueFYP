model_selection=1
min_detection_confidence=0.5

cv2.resize(face, (256, 256))
cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)

static_image_mode=True
max_num_faces=1
refine_landmarks=True
dotmap size = 256×256
gaussian_filter(..., sigma=2)


np.save(..., landmark_all)
landmark_all.shape = (478, 2)

face_in.shape = (256, 256, 1)  
dot_in.shape = (256, 256, 1)  
lmk_in.shape = (478 * 2,)  
optimizer = 'adam'  
loss = 'binary_crossentropy'  
metrics = ['accuracy']  
Dropout rate = 0.4  
Epochs = 15
Model saved as = 'fatigue_fusion_model_generator_90.h5'
